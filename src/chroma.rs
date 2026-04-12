// Chroma filter bank projection kernels for iron-librosa.
//
// Provides efficient GEMM-based chroma feature extraction.
// The chroma filter bank maps n_fft spectrogram bins to n_chroma (typically 12) chroma bins.
//
// Follows the same faer-based GEMM pattern as mel-spectrogram projection (Phase 3).

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use std::sync::OnceLock;

use crate::backend::{resolved_rust_device, RustDevice};

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const DEFAULT_CHROMA_GPU_WORK_THRESHOLD: usize = 800_000_000;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn chroma_gpu_work_threshold() -> usize {
    static OVERRIDE: OnceLock<usize> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("IRON_LIBROSA_CHROMA_GPU_WORK_THRESHOLD")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_CHROMA_GPU_WORK_THRESHOLD)
    })
}

fn validate_shapes<T>(
    s: &ndarray::ArrayView2<'_, T>,
    chroma_basis: &ndarray::ArrayView2<'_, T>,
) -> PyResult<(usize, usize, usize)> {
    let n_fft_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let n_chroma = chroma_basis.shape()[0];

    if chroma_basis.shape()[1] != n_fft_bins {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: S is ({n_fft_bins}, {n_frames}), chroma_basis is \
             ({n_chroma}, {}). Expected chroma_basis.shape[1] == S.shape[0].",
            chroma_basis.shape()[1]
        )));
    }

    Ok((n_fft_bins, n_frames, n_chroma))
}

/// Reinterpret a C-contiguous `ArrayView2` as a **column-major** faer
/// `MatRef` of the *transposed* shape — zero copies.
///
/// A row-major matrix `A` (nrows×ncols, row_stride=ncols, col_stride=1)
/// in memory is *byte-identical* to its transpose `A^T` stored column-major
/// (ncols×nrows, row_stride=1, col_stride=nrows).  faer's GEMM is fastest
/// when all operands are column-major, so we expose them that way and adjust
/// the multiplication order: `C = A*B  ⟺  C^T = B^T * A^T`.
///
/// # Safety
/// `arr` must remain valid for the lifetime `'a`.
#[inline]
unsafe fn as_col_major_t<'a, T>(arr: &'a ndarray::ArrayView2<'a, T>) -> MatRef<'a, T> {
    let nrows = arr.shape()[0];
    let ncols = arr.shape()[1];
    let row_s = arr.strides()[0]; // = ncols for C-contiguous
    let col_s = arr.strides()[1]; // = 1   for C-contiguous
    // Transposed shape + swapped strides → column-major (row_stride=1).
    MatRef::from_raw_parts(arr.as_ptr(), ncols, nrows, col_s, row_s)
}

fn validate_chroma_params(sr: f64, n_fft: usize, n_chroma: usize) -> PyResult<()> {
    if !(sr.is_finite() && sr > 0.0) {
        return Err(PyValueError::new_err("sr must be positive and finite"));
    }
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be at least 2"));
    }
    if n_chroma == 0 {
        return Err(PyValueError::new_err("n_chroma must be positive"));
    }
    Ok(())
}

fn build_chroma_filter_f64(
    sr: f64,
    n_fft: usize,
    n_chroma: usize,
    tuning: f64,
    ctroct: f64,
    octwidth: Option<f64>,
    base_c: bool,
    norm: Option<u32>,  // None=no norm, 1=L1, 2=L2, 999=inf
) -> PyResult<Array2<f64>> {
    validate_chroma_params(sr, n_fft, n_chroma)?;

    let n_keep = 1 + n_fft / 2;
    let n_chroma_f = n_chroma as f64;
    let a440 = 440.0 * 2.0f64.powf(tuning / n_chroma_f);

    let mut frqbins = vec![0.0f64; n_fft];
    for k in 1..n_fft {
        let freq = sr * (k as f64) / (n_fft as f64);
        frqbins[k] = n_chroma_f * (freq / (a440 / 16.0)).log2();
    }
    frqbins[0] = frqbins[1] - 1.5 * n_chroma_f;

    let mut binwidthbins = vec![1.0f64; n_fft];
    for k in 0..(n_fft - 1) {
        binwidthbins[k] = (frqbins[k + 1] - frqbins[k]).max(1.0);
    }

    let n_chroma2 = (n_chroma_f / 2.0).round();
    let octave_weights = octwidth.map(|ow| {
        (0..n_keep)
            .map(|k| {
                let x = (frqbins[k] / n_chroma_f - ctroct) / ow;
                (-0.5 * x * x).exp()
            })
            .collect::<Vec<f64>>()
    });

    let cols: Vec<Vec<f64>> = (0..n_keep)
        .into_par_iter()
        .map(|k| {
            let bw = binwidthbins[k];
            let octave_scale = octave_weights.as_ref().map(|ow| ow[k]).unwrap_or(1.0);
            let mut col = vec![0.0f64; n_chroma];

            for i in 0..n_chroma {
                let mut d = frqbins[k] - (i as f64);
                d = (d + n_chroma2 + 10.0 * n_chroma_f).rem_euclid(n_chroma_f) - n_chroma2;
                let v = (-0.5 * (2.0 * d / bw).powi(2)).exp();
                col[i] = v;
            }

            // Apply normalization based on norm parameter
            if let Some(n) = norm {
                match n {
                    1 => {
                        // L1 norm: sum of absolute values
                        let l1 = col.iter().map(|v| v.abs()).sum::<f64>();
                        if l1 > f64::MIN_POSITIVE {
                            for v in &mut col {
                                *v = (*v / l1) * octave_scale;
                            }
                        } else if octave_scale != 1.0 {
                            for v in &mut col {
                                *v *= octave_scale;
                            }
                        }
                    }
                    2 => {
                        // L2 norm: sqrt of sum of squares (default)
                        let l2 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
                        if l2 > f64::MIN_POSITIVE {
                            for v in &mut col {
                                *v = (*v / l2) * octave_scale;
                            }
                        } else if octave_scale != 1.0 {
                            for v in &mut col {
                                *v *= octave_scale;
                            }
                        }
                    }
                    999 => {
                        // L-infinity norm: maximum absolute value
                        let linf = col.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                        if linf > f64::MIN_POSITIVE {
                            for v in &mut col {
                                *v = (*v / linf) * octave_scale;
                            }
                        } else if octave_scale != 1.0 {
                            for v in &mut col {
                                *v *= octave_scale;
                            }
                        }
                    }
                    _ => {
                        // Unknown norm code, just apply octave_scale
                        for v in &mut col {
                            *v *= octave_scale;
                        }
                    }
                }
            } else {
                // No normalization, just apply octave_scale
                for v in &mut col {
                    *v *= octave_scale;
                }
            }

            col
        })
        .collect();

    let mut out = Array2::<f64>::zeros((n_chroma, n_keep));
    for k in 0..n_keep {
        for i in 0..n_chroma {
            out[[i, k]] = cols[k][i];
        }
    }

    if base_c {
        let shift = 3 * (n_chroma / 12);
        if shift > 0 {
            let mut rolled = Array2::<f64>::zeros((n_chroma, n_keep));
            for i in 0..n_chroma {
                let src_i = (i + shift) % n_chroma;
                for k in 0..n_keep {
                    rolled[[i, k]] = out[[src_i, k]];
                }
            }
            Ok(rolled)
        } else {
            Ok(out)
        }
    } else {
        Ok(out)
    }
}

#[pyfunction]
#[pyo3(signature = (sr, n_fft, n_chroma=12, tuning=0.0, ctroct=5.0, octwidth=Some(2.0), base_c=true, norm=Some(2)))]
pub fn chroma_filter_f64<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_chroma: usize,
    tuning: f64,
    ctroct: f64,
    octwidth: Option<f64>,
    base_c: bool,
    norm: Option<u32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => chroma_filter_f64_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => chroma_filter_f64_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
        RustDevice::Auto => chroma_filter_f64_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => chroma_filter_f64_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
    }
}

fn chroma_filter_f64_cpu<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_chroma: usize,
    tuning: f64,
    ctroct: f64,
    octwidth: Option<f64>,
    base_c: bool,
    norm: Option<u32>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = build_chroma_filter_f64(sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm)?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (sr, n_fft, n_chroma=12, tuning=0.0, ctroct=5.0, octwidth=Some(2.0), base_c=true, norm=Some(2)))]
pub fn chroma_filter_f32<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_chroma: usize,
    tuning: f64,
    ctroct: f64,
    octwidth: Option<f64>,
    base_c: bool,
    norm: Option<u32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => chroma_filter_f32_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => chroma_filter_f32_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
        RustDevice::Auto => chroma_filter_f32_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => chroma_filter_f32_cpu(py, sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm),
    }
}

fn chroma_filter_f32_cpu<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_chroma: usize,
    tuning: f64,
    ctroct: f64,
    octwidth: Option<f64>,
    base_c: bool,
    norm: Option<u32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let out64 = build_chroma_filter_f64(sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c, norm)?;
    let out32 = out64.mapv(|x| x as f32);
    Ok(out32.into_pyarray_bound(py).to_owned())
}

/// Project a power spectrogram onto the chroma filter bank (f32 precision).
///
/// Computes: chroma = chroma_basis @ S  (matrix multiply)
/// where chroma_basis is (n_chroma, n_fft_bins) and S is (n_fft_bins, n_frames).
/// Output is (n_chroma, n_frames).
///
/// This matches the `np.einsum("cf,...ft->...ct", chromafb, S)` operation
/// used in `librosa.feature.chroma_stft()`.
#[pyfunction]
#[pyo3(signature = (s, chroma_basis))]
pub fn chroma_project_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    chroma_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => chroma_project_f32_cpu(py, s, chroma_basis),
        RustDevice::AppleGpu => chroma_project_f32_apple_gpu(py, s, chroma_basis),
        RustDevice::Auto => chroma_project_f32_cpu(py, s, chroma_basis),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => chroma_project_f32_cpu(py, s, chroma_basis),
    }
}

fn chroma_project_f32_apple_gpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    chroma_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    #[cfg(all(feature = "apple-gpu", target_os = "macos"))]
    {
        let s_view = s.as_array();
        let chroma_view = chroma_basis.as_array();
        let work = s_view
            .shape()[0]
            .saturating_mul(s_view.shape()[1])
            .saturating_mul(chroma_view.shape()[0]);

        if work >= chroma_gpu_work_threshold() {
            if let Some(out) = crate::mel::try_project_f32_apple_gpu(&s_view, &chroma_view) {
                return Ok(out.into_pyarray_bound(py).to_owned());
            }
        }
    }

    // Safe fallback policy: any unavailable/failed GPU path returns CPU output.
    chroma_project_f32_cpu(py, s, chroma_basis)
}

fn chroma_project_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    chroma_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let s = s.as_array();
    let chroma_basis = chroma_basis.as_array();
    let (_n_fft_bins, n_frames, n_chroma) = validate_shapes(&s, &chroma_basis)?;

    let mut out = ndarray::Array2::<f32>::zeros((n_chroma, n_frames));

    // SAFETY: inputs are C-contiguous (np.ascontiguousarray in Python caller)
    unsafe {
        let out_t = MatMut::<f32>::from_raw_parts_mut(
            out.as_mut_ptr(),
            n_frames, n_chroma,
            out.strides()[1],
            out.strides()[0],
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&s),
            as_col_major_t(&chroma_basis),
            1.0f32,
            Par::rayon(0),
        );
    }

    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Project a power spectrogram onto the chroma filter bank (f64 precision).
#[pyfunction]
#[pyo3(signature = (s, chroma_basis))]
pub fn chroma_project_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    chroma_basis: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => chroma_project_f64_cpu(py, s, chroma_basis),
        // Phase 1 scaffold: keep CPU path while Apple GPU kernels are pending.
        RustDevice::AppleGpu => chroma_project_f64_cpu(py, s, chroma_basis),
        RustDevice::Auto => chroma_project_f64_cpu(py, s, chroma_basis),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => chroma_project_f64_cpu(py, s, chroma_basis),
    }
}

fn chroma_project_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    chroma_basis: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_array();
    let chroma_basis = chroma_basis.as_array();
    let (_n_fft_bins, n_frames, n_chroma) = validate_shapes(&s, &chroma_basis)?;

    let mut out = ndarray::Array2::<f64>::zeros((n_chroma, n_frames));

    // SAFETY: inputs are C-contiguous (np.ascontiguousarray in Python caller)
    unsafe {
        let out_t = MatMut::<f64>::from_raw_parts_mut(
            out.as_mut_ptr(),
            n_frames, n_chroma,
            out.strides()[1],
            out.strides()[0],
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&s),
            as_col_major_t(&chroma_basis),
            1.0f64,
            Par::rayon(0),
        );
    }

    Ok(out.into_pyarray_bound(py).to_owned())
}


