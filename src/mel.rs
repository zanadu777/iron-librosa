use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_shapes<T>(
    s: &ndarray::ArrayView2<'_, T>,
    mel_basis: &ndarray::ArrayView2<'_, T>,
) -> PyResult<(usize, usize, usize)> {
    let n_fft_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let n_mels = mel_basis.shape()[0];

    if mel_basis.shape()[1] != n_fft_bins {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: S is ({n_fft_bins}, {n_frames}), mel_basis is \
             ({n_mels}, {}). Expected mel_basis.shape[1] == S.shape[0].",
            mel_basis.shape()[1]
        )));
    }

    Ok((n_fft_bins, n_frames, n_mels))
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

#[pyfunction]
#[pyo3(signature = (s, mel_basis))]
pub fn mel_project_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    mel_basis: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_array();
    let mel_basis = mel_basis.as_array();
    let (_n_fft_bins, n_frames, n_mels) = validate_shapes(&s, &mel_basis)?;

    let mut out = ndarray::Array2::<f64>::zeros((n_mels, n_frames));

    // SAFETY: inputs are C-contiguous (np.ascontiguousarray in Python caller)
    // and `out` is freshly allocated.
    //
    // Compute C^T = S^T × M^T  (all column-major) which is byte-identical to
    // C = M × S  (all row-major) stored in `out`.
    unsafe {
        // out^T : (n_frames × n_mels) column-major  (same bytes as out row-major)
        let out_t = MatMut::<f64>::from_raw_parts_mut(
            out.as_mut_ptr(),
            n_frames, n_mels,
            out.strides()[1], // row_stride = 1 (col stride of row-major out)
            out.strides()[0], // col_stride = n_frames (row stride of row-major out)
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&s),         // S^T : (n_frames × n_fft_bins) col-major
            as_col_major_t(&mel_basis), // M^T : (n_fft_bins × n_mels) col-major
            1.0f64,
            Par::rayon(0),
        );
    }

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, mel_basis))]
pub fn mel_project_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    mel_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let s = s.as_array();
    let mel_basis = mel_basis.as_array();
    let (_n_fft_bins, n_frames, n_mels) = validate_shapes(&s, &mel_basis)?;

    let mut out = ndarray::Array2::<f32>::zeros((n_mels, n_frames));

    unsafe {
        let out_t = MatMut::<f32>::from_raw_parts_mut(
            out.as_mut_ptr(),
            n_frames, n_mels,
            out.strides()[1],
            out.strides()[0],
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&s),
            as_col_major_t(&mel_basis),
            1.0f32,
            Par::rayon(0),
        );
    }

    Ok(out.into_pyarray_bound(py))
}

#[inline]
fn hz_to_mel_scalar(f: f64, htk: bool) -> f64 {
    if htk {
        2595.0 * (1.0 + f / 700.0).log10()
    } else {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4_f64.ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    }
}

#[inline]
fn mel_to_hz_scalar(m: f64, htk: bool) -> f64 {
    if htk {
        700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
    } else {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4_f64.ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    }
}

#[pyfunction]
#[pyo3(signature = (sr, n_fft, n_mels = 128, fmin = 0.0, fmax = None, htk = false, slaney_norm = true))]
pub fn mel_filter_f32<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    slaney_norm: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if sr <= 0.0 {
        return Err(PyValueError::new_err("sr must be > 0"));
    }
    if n_fft < 2 || n_mels == 0 {
        return Err(PyValueError::new_err("n_fft must be >=2 and n_mels > 0"));
    }

    let fmax = fmax.unwrap_or(sr / 2.0);
    let n_bins = 1 + n_fft / 2;

    // FFT bin centers in Hz
    let fftfreqs: Vec<f64> = (0..n_bins)
        .map(|i| (sr / 2.0) * (i as f64) / ((n_bins - 1) as f64))
        .collect();

    // Mel band edge frequencies in Hz (n_mels + 2 points)
    let m0 = hz_to_mel_scalar(fmin, htk);
    let m1 = hz_to_mel_scalar(fmax, htk);
    let mel_f: Vec<f64> = (0..(n_mels + 2))
        .map(|i| {
            let t = i as f64 / ((n_mels + 1) as f64);
            mel_to_hz_scalar(m0 + t * (m1 - m0), htk)
        })
        .collect();

    let mut weights = ndarray::Array2::<f32>::zeros((n_mels, n_bins));

    for i in 0..n_mels {
        let fdiff_l = mel_f[i + 1] - mel_f[i];
        let fdiff_u = mel_f[i + 2] - mel_f[i + 1];
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / fdiff_l;
            let upper = (mel_f[i + 2] - f) / fdiff_u;
            let v = lower.min(upper).max(0.0) as f32;
            weights[(i, j)] = v;
        }
    }

    if slaney_norm {
        for i in 0..n_mels {
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            let s = enorm as f32;
            let mut row = weights.row_mut(i);
            row *= s;
        }
    }

    Ok(weights.into_pyarray_bound(py))
}
