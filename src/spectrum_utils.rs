// Spectral feature reduction kernels for iron-librosa.
//
// Provides cache-friendly, optionally parallel implementations of:
//   - rms_spectrogram_f32 / f64  (frame-wise RMS from magnitude spectrogram)
//   - spectral_centroid_f32 / f64  (per-frame spectral centroid)
//
// Cache strategy: spectrogram S has C-order layout (n_bins, n_frames).
//   Iterating row-by-row (bin-by-bin) gives contiguous memory access per row.
//   Each function accumulates into a flat output buffer indexed by frame,
//   avoiding the column-strided access pattern of the original per-frame loop.
//
// Parallelism: uses rayon fold+reduce over bins when the total element count
//   exceeds PAR_THRESHOLD.  The threshold prevents rayon thread-pool overhead
//   from dominating on small arrays.

use ndarray::{s, Array1, Array2, Array3, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Minimum element count (n_bins × n_frames) before switching to parallel path.
const PAR_THRESHOLD: usize = 200_000; // ~800 KB for f32 / ~1.6 MB for f64

/// Batch-level parallelism thresholds for HPSS operations.
/// Tuned to avoid oversubscription and thread pool contention.
///
/// Strategy:
/// - At batch level: only parallelize if batch is large enough to justify thread pool overhead
///   AND the inner per-batch work is small enough to not compete with rayon's internal parallelism
/// - At element level: the inner median filter kernels use PAR_THRESHOLD to decide per-batch parallelism
///
/// The sweet spot is:
/// - batch_size >= 4 (justifies rayon overhead)
/// - per-batch elements < PAR_THRESHOLD (inner work stays sequential, avoiding nested contention)
const BATCH_PAR_SIZE_MIN: usize = 4; // minimum batch size to parallelize
const BATCH_ELEMENT_MAX: usize = 150_000; // max elements per batch for batch-level parallelism
                                           // (avoids nested rayon contention)

// ─────────────────────────────────────────────────────────────────────────────
// RMS kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulate weighted squared magnitudes into `sum_sq[n_frames]`
/// using a cache-friendly row-by-row traversal.
///
/// For C-order (n_bins, n_frames) arrays each row is contiguous;
/// iterating the inner loop over frames gives sequential reads.
#[inline]
fn accumulate_rms_row_f32(
    row: ndarray::ArrayView1<f32>,
    weight: f32,
    acc: &mut [f32],
) {
    if let Some(slice) = row.as_slice() {
        for (a, &v) in acc.iter_mut().zip(slice.iter()) {
            *a += weight * v * v;
        }
    } else {
        for (t, v) in row.iter().enumerate() {
            acc[t] += weight * v * v;
        }
    }
}

#[inline]
fn accumulate_rms_row_f64(
    row: ndarray::ArrayView1<f64>,
    weight: f64,
    acc: &mut [f64],
) {
    if let Some(slice) = row.as_slice() {
        for (a, &v) in acc.iter_mut().zip(slice.iter()) {
            *a += weight * v * v;
        }
    } else {
        for (t, v) in row.iter().enumerate() {
            acc[t] += weight * v * v;
        }
    }
}

/// Compute frame-wise RMS from a real-valued magnitude spectrogram (f32 precision).
///
/// Matches `librosa.feature.rms(S=...)`:
///   - DC bin (f=0) half-weighted
///   - Nyquist bin half-weighted when `frame_length` is even
///   - output shape (1, n_frames)
#[pyfunction]
#[pyo3(signature = (s, frame_length))]
pub fn rms_spectrogram_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    frame_length: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let s_view = s.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let frame_length_f = frame_length as f32;
    let even_fl = frame_length % 2 == 0;
    let scale = 2.0f32 / (frame_length_f * frame_length_f);

    let weight_for = |f: usize| -> f32 {
        if f == 0 || (even_fl && f == n_bins - 1) { 0.5 } else { 1.0 }
    };

    let sum_sq: Vec<f32> = if n_bins * n_frames >= PAR_THRESHOLD {
        // Parallel fold over rows (bins), then reduce partial accumulators.
        s_view
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .fold(
                || vec![0.0f32; n_frames],
                |mut acc, (f, row)| {
                    accumulate_rms_row_f32(row, weight_for(f), &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; n_frames],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += bi;
                    }
                    a
                },
            )
    } else {
        let mut acc = vec![0.0f32; n_frames];
        for (f, row) in s_view.axis_iter(Axis(0)).enumerate() {
            accumulate_rms_row_f32(row, weight_for(f), &mut acc);
        }
        acc
    };

    let out_data: Vec<f32> = sum_sq.iter().map(|&s| (scale * s).sqrt()).collect();
    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute frame-wise RMS from a real-valued magnitude spectrogram (f64 precision).
#[pyfunction]
#[pyo3(signature = (s, frame_length))]
pub fn rms_spectrogram_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    frame_length: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s_view = s.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let frame_length_f = frame_length as f64;
    let even_fl = frame_length % 2 == 0;
    let scale = 2.0f64 / (frame_length_f * frame_length_f);

    let weight_for = |f: usize| -> f64 {
        if f == 0 || (even_fl && f == n_bins - 1) { 0.5 } else { 1.0 }
    };

    let sum_sq: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        s_view
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .fold(
                || vec![0.0f64; n_frames],
                |mut acc, (f, row)| {
                    accumulate_rms_row_f64(row, weight_for(f), &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![0.0f64; n_frames],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += bi;
                    }
                    a
                },
            )
    } else {
        let mut acc = vec![0.0f64; n_frames];
        for (f, row) in s_view.axis_iter(Axis(0)).enumerate() {
            accumulate_rms_row_f64(row, weight_for(f), &mut acc);
        }
        acc
    };

    let out_data: Vec<f64> = sum_sq.iter().map(|&s| (scale * s).sqrt()).collect();
    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute frame-wise RMS from framed time-domain samples (f32 precision).
///
/// Input shape is (frame_length, n_frames), equivalent to
/// `util.frame(y, frame_length=..., hop_length=...)` per channel.
///
/// Returns shape (1, n_frames), matching `librosa.feature.rms(y=...)`.
#[pyfunction]
#[pyo3(signature = (x))]
pub fn rms_time_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x_view = x.as_array();
    let frame_length = x_view.shape()[0];
    let n_frames = x_view.shape()[1];

    if frame_length == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frame_length must be positive",
        ));
    }

    let frame_contiguous = x_view.strides()[0] == 1;
    let sum_sq: Vec<f32> = if frame_contiguous {
        // Common `util.frame` layout: x[:, t] is contiguous in memory.
        // Summing by frame avoids high-stride traversal across rows.
        if frame_length * n_frames >= PAR_THRESHOLD {
            (0..n_frames)
                .into_par_iter()
                .map(|t| {
                    let col = x_view.index_axis(Axis(1), t);
                    if let Some(slice) = col.as_slice() {
                        slice.iter().map(|&v| v * v).sum::<f32>()
                    } else {
                        col.iter().map(|&v| v * v).sum::<f32>()
                    }
                })
                .collect()
        } else {
            let mut out = vec![0.0f32; n_frames];
            for t in 0..n_frames {
                let col = x_view.index_axis(Axis(1), t);
                out[t] = if let Some(slice) = col.as_slice() {
                    slice.iter().map(|&v| v * v).sum::<f32>()
                } else {
                    col.iter().map(|&v| v * v).sum::<f32>()
                };
            }
            out
        }
    } else if frame_length * n_frames >= PAR_THRESHOLD {
        x_view
            .axis_iter(Axis(0))
            .into_par_iter()
            .fold(
                || vec![0.0f32; n_frames],
                |mut acc, row| {
                    if let Some(slice) = row.as_slice() {
                        for (a, &v) in acc.iter_mut().zip(slice.iter()) {
                            *a += v * v;
                        }
                    } else {
                        for (t, &v) in row.iter().enumerate() {
                            acc[t] += v * v;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f32; n_frames],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += bi;
                    }
                    a
                },
            )
    } else {
        let mut acc = vec![0.0f32; n_frames];
        for row in x_view.axis_iter(Axis(0)) {
            if let Some(slice) = row.as_slice() {
                for (a, &v) in acc.iter_mut().zip(slice.iter()) {
                    *a += v * v;
                }
            } else {
                for (t, &v) in row.iter().enumerate() {
                    acc[t] += v * v;
                }
            }
        }
        acc
    };

    let inv_len = 1.0f32 / (frame_length as f32);
    let out_data: Vec<f32> = sum_sq.iter().map(|&s| (s * inv_len).sqrt()).collect();
    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute frame-wise RMS from framed time-domain samples (f64 precision).
#[pyfunction]
#[pyo3(signature = (x))]
pub fn rms_time_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x_view = x.as_array();
    let frame_length = x_view.shape()[0];
    let n_frames = x_view.shape()[1];

    if frame_length == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frame_length must be positive",
        ));
    }

    let frame_contiguous = x_view.strides()[0] == 1;
    let sum_sq: Vec<f64> = if frame_contiguous {
        // Common `util.frame` layout: x[:, t] is contiguous in memory.
        // Summing by frame avoids high-stride traversal across rows.
        if frame_length * n_frames >= PAR_THRESHOLD {
            (0..n_frames)
                .into_par_iter()
                .map(|t| {
                    let col = x_view.index_axis(Axis(1), t);
                    if let Some(slice) = col.as_slice() {
                        slice.iter().map(|&v| v * v).sum::<f64>()
                    } else {
                        col.iter().map(|&v| v * v).sum::<f64>()
                    }
                })
                .collect()
        } else {
            let mut out = vec![0.0f64; n_frames];
            for t in 0..n_frames {
                let col = x_view.index_axis(Axis(1), t);
                out[t] = if let Some(slice) = col.as_slice() {
                    slice.iter().map(|&v| v * v).sum::<f64>()
                } else {
                    col.iter().map(|&v| v * v).sum::<f64>()
                };
            }
            out
        }
    } else if frame_length * n_frames >= PAR_THRESHOLD {
        x_view
            .axis_iter(Axis(0))
            .into_par_iter()
            .fold(
                || vec![0.0f64; n_frames],
                |mut acc, row| {
                    if let Some(slice) = row.as_slice() {
                        for (a, &v) in acc.iter_mut().zip(slice.iter()) {
                            *a += v * v;
                        }
                    } else {
                        for (t, &v) in row.iter().enumerate() {
                            acc[t] += v * v;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![0.0f64; n_frames],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += bi;
                    }
                    a
                },
            )
    } else {
        let mut acc = vec![0.0f64; n_frames];
        for row in x_view.axis_iter(Axis(0)) {
            if let Some(slice) = row.as_slice() {
                for (a, &v) in acc.iter_mut().zip(slice.iter()) {
                    *a += v * v;
                }
            } else {
                for (t, &v) in row.iter().enumerate() {
                    acc[t] += v * v;
                }
            }
        }
        acc
    };

    let inv_len = 1.0f64 / (frame_length as f64);
    let out_data: Vec<f64> = sum_sq.iter().map(|&s| (s * inv_len).sqrt()).collect();
    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral centroid kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Compute spectral centroid from a real, non-negative spectrogram (f32 input → f64 output).
///
/// Matches the `librosa.feature.spectral_centroid()` fast path:
///   centroid[t] = Σ_f S[f,t] · freq[f]  /  Σ_f S[f,t]
///
/// Uses cache-friendly row-by-row traversal and optional rayon parallelism.
/// The caller is responsible for ensuring S ≥ 0 (the Python dispatch reorders
/// the guard to fire this kernel before `np.any(S < 0)` for the fast path).
#[pyfunction]
#[pyo3(signature = (s, freq))]
pub fn spectral_centroid_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freq: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.len() != n_bins {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq length must match spectrogram bins",
        ));
    }

    let freq_slice = freq_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("freq must be contiguous")
    })?;

    // Each partial accumulator is (numer_vec, denom_vec) of length n_frames.
    let (numer, denom): (Vec<f64>, Vec<f64>) = if n_bins * n_frames >= PAR_THRESHOLD {
        s_view
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .fold(
                || (vec![0.0f64; n_frames], vec![0.0f64; n_frames]),
                |(mut num, mut den), (f, row)| {
                    let ff = freq_slice[f];
                    if let Some(slice) = row.as_slice() {
                        for (t, &v) in slice.iter().enumerate() {
                            let sv = f64::from(v);
                            num[t] += sv * ff;
                            den[t] += sv;
                        }
                    } else {
                        for (t, v) in row.iter().enumerate() {
                            let sv = f64::from(*v);
                            num[t] += sv * ff;
                            den[t] += sv;
                        }
                    }
                    (num, den)
                },
            )
            .reduce(
                || (vec![0.0f64; n_frames], vec![0.0f64; n_frames]),
                |(mut na, mut da), (nb, db)| {
                    for i in 0..n_frames {
                        na[i] += nb[i];
                        da[i] += db[i];
                    }
                    (na, da)
                },
            )
    } else {
        let mut num = vec![0.0f64; n_frames];
        let mut den = vec![0.0f64; n_frames];
        for (f, row) in s_view.axis_iter(Axis(0)).enumerate() {
            let ff = freq_slice[f];
            if let Some(slice) = row.as_slice() {
                for (t, &v) in slice.iter().enumerate() {
                    let sv = f64::from(v);
                    num[t] += sv * ff;
                    den[t] += sv;
                }
            } else {
                for (t, v) in row.iter().enumerate() {
                    let sv = f64::from(*v);
                    num[t] += sv * ff;
                    den[t] += sv;
                }
            }
        }
        (num, den)
    };

    let thresh = f64::from(f32::MIN_POSITIVE);
    let out_data: Vec<f64> = (0..n_frames)
        .map(|t| if denom[t] > thresh { numer[t] / denom[t] } else { 0.0 })
        .collect();

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral centroid from a real, non-negative spectrogram (f64 input → f64 output).
#[pyfunction]
#[pyo3(signature = (s, freq))]
pub fn spectral_centroid_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    freq: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.len() != n_bins {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq length must match spectrogram bins",
        ));
    }

    let freq_slice = freq_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("freq must be contiguous")
    })?;

    let (numer, denom): (Vec<f64>, Vec<f64>) = if n_bins * n_frames >= PAR_THRESHOLD {
        s_view
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .fold(
                || (vec![0.0f64; n_frames], vec![0.0f64; n_frames]),
                |(mut num, mut den), (f, row)| {
                    let ff = freq_slice[f];
                    if let Some(slice) = row.as_slice() {
                        for (t, &v) in slice.iter().enumerate() {
                            num[t] += v * ff;
                            den[t] += v;
                        }
                    } else {
                        for (t, &v) in row.iter().enumerate() {
                            num[t] += v * ff;
                            den[t] += v;
                        }
                    }
                    (num, den)
                },
            )
            .reduce(
                || (vec![0.0f64; n_frames], vec![0.0f64; n_frames]),
                |(mut na, mut da), (nb, db)| {
                    for i in 0..n_frames {
                        na[i] += nb[i];
                        da[i] += db[i];
                    }
                    (na, da)
                },
            )
    } else {
        let mut num = vec![0.0f64; n_frames];
        let mut den = vec![0.0f64; n_frames];
        for (f, row) in s_view.axis_iter(Axis(0)).enumerate() {
            let ff = freq_slice[f];
            if let Some(slice) = row.as_slice() {
                for (t, &v) in slice.iter().enumerate() {
                    num[t] += v * ff;
                    den[t] += v;
                }
            } else {
                for (t, &v) in row.iter().enumerate() {
                    num[t] += v * ff;
                    den[t] += v;
                }
            }
        }
        (num, den)
    };

    let out_data: Vec<f64> = (0..n_frames)
        .map(|t| if denom[t] > f64::MIN_POSITIVE { numer[t] / denom[t] } else { 0.0 })
        .collect();

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral centroid from real, non-negative spectrogram with a per-bin, per-frame
/// frequency grid (f32 input -> f64 output).
///
/// Shapes:
///   - s: (n_bins, n_frames)
///   - freq: (n_bins, n_frames)
/// Returns shape (1, n_frames).
#[pyfunction]
#[pyo3(signature = (s, freq))]
pub fn spectral_centroid_variable_freq_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freq: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.shape() != [n_bins, n_frames] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq must have shape (n_bins, n_frames) matching S",
        ));
    }

    let out_data: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut numer = 0.0f64;
                let mut denom = 0.0f64;
                for f in 0..n_bins {
                    let sv = f64::from(s_view[[f, t]]);
                    numer += sv * freq_view[[f, t]];
                    denom += sv;
                }
                if denom > f64::from(f32::MIN_POSITIVE) { numer / denom } else { 0.0 }
            })
            .collect()
    } else {
        let mut out = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            let mut numer = 0.0f64;
            let mut denom = 0.0f64;
            for f in 0..n_bins {
                let sv = f64::from(s_view[[f, t]]);
                numer += sv * freq_view[[f, t]];
                denom += sv;
            }
            out[t] = if denom > f64::from(f32::MIN_POSITIVE) { numer / denom } else { 0.0 };
        }
        out
    };

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral centroid from real, non-negative spectrogram with a per-bin, per-frame
/// frequency grid (f64 input -> f64 output).
#[pyfunction]
#[pyo3(signature = (s, freq))]
pub fn spectral_centroid_variable_freq_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    freq: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.shape() != [n_bins, n_frames] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq must have shape (n_bins, n_frames) matching S",
        ));
    }

    let out_data: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut numer = 0.0f64;
                let mut denom = 0.0f64;
                for f in 0..n_bins {
                    let sv = s_view[[f, t]];
                    numer += sv * freq_view[[f, t]];
                    denom += sv;
                }
                if denom > f64::MIN_POSITIVE { numer / denom } else { 0.0 }
            })
            .collect()
    } else {
        let mut out = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            let mut numer = 0.0f64;
            let mut denom = 0.0f64;
            for f in 0..n_bins {
                let sv = s_view[[f, t]];
                numer += sv * freq_view[[f, t]];
                denom += sv;
            }
            out[t] = if denom > f64::MIN_POSITIVE { numer / denom } else { 0.0 };
        }
        out
    };

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral rolloff from a real, non-negative spectrogram (f32 input -> f64 output).
///
/// Fast path equivalent to librosa's static-frequency-bin case:
///   total = cumsum(S, axis=-2)
///   threshold = roll_percent * total[-1]
///   rolloff[t] = min(freq[k] where total[k, t] >= threshold[t])
#[pyfunction]
#[pyo3(signature = (s, freq, roll_percent))]
pub fn spectral_rolloff_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freq: PyReadonlyArray1<'py, f64>,
    roll_percent: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !(0.0..1.0).contains(&roll_percent) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "roll_percent must lie in the range (0, 1)",
        ));
    }

    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.len() != n_bins {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq length must match spectrogram bins",
        ));
    }

    let freq_slice = freq_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("freq must be contiguous")
    })?;

    let out: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut total = 0.0f64;
                for f in 0..n_bins {
                    total += f64::from(s_view[[f, t]]);
                }
                let threshold = roll_percent * total;

                let mut cumsum = 0.0f64;
                let mut idx = 0usize;
                for f in 0..n_bins {
                    cumsum += f64::from(s_view[[f, t]]);
                    if cumsum >= threshold {
                        idx = f;
                        break;
                    }
                }
                freq_slice[idx]
            })
            .collect()
    } else {
        let mut out = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            let mut total = 0.0f64;
            for f in 0..n_bins {
                total += f64::from(s_view[[f, t]]);
            }
            let threshold = roll_percent * total;

            let mut cumsum = 0.0f64;
            let mut idx = 0usize;
            for f in 0..n_bins {
                cumsum += f64::from(s_view[[f, t]]);
                if cumsum >= threshold {
                    idx = f;
                    break;
                }
            }
            out[t] = freq_slice[idx];
        }
        out
    };

    let out = Array2::from_shape_vec((1, n_frames), out)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral rolloff from a real, non-negative spectrogram (f64 input -> f64 output).
#[pyfunction]
#[pyo3(signature = (s, freq, roll_percent))]
pub fn spectral_rolloff_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    freq: PyReadonlyArray1<'py, f64>,
    roll_percent: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !(0.0..1.0).contains(&roll_percent) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "roll_percent must lie in the range (0, 1)",
        ));
    }

    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.len() != n_bins {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq length must match spectrogram bins",
        ));
    }

    let freq_slice = freq_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("freq must be contiguous")
    })?;

    let out: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut total = 0.0f64;
                for f in 0..n_bins {
                    total += s_view[[f, t]];
                }
                let threshold = roll_percent * total;

                let mut cumsum = 0.0f64;
                let mut idx = 0usize;
                for f in 0..n_bins {
                    cumsum += s_view[[f, t]];
                    if cumsum >= threshold {
                        idx = f;
                        break;
                    }
                }
                freq_slice[idx]
            })
            .collect()
    } else {
        let mut out = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            let mut total = 0.0f64;
            for f in 0..n_bins {
                total += s_view[[f, t]];
            }
            let threshold = roll_percent * total;

            let mut cumsum = 0.0f64;
            let mut idx = 0usize;
            for f in 0..n_bins {
                cumsum += s_view[[f, t]];
                if cumsum >= threshold {
                    idx = f;
                    break;
                }
            }
            out[t] = freq_slice[idx];
        }
        out
    };

    let out = Array2::from_shape_vec((1, n_frames), out)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral rolloff for variable 2D frequency grids (f32 input).
#[pyfunction]
#[pyo3(signature = (s, freq, roll_percent))]
pub fn spectral_rolloff_variable_freq_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freq: PyReadonlyArray2<'py, f64>,
    roll_percent: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !(0.0..1.0).contains(&roll_percent) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "roll_percent must lie in the range (0, 1)",
        ));
    }

    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.shape() != [n_bins, n_frames] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq must have shape (n_bins, n_frames) matching S",
        ));
    }

    let out_data: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut total = 0.0f64;
                for f in 0..n_bins {
                    total += f64::from(s_view[[f, t]]);
                }
                let threshold = roll_percent * total;

                let mut cumsum = 0.0f64;
                let mut idx = 0usize;
                for f in 0..n_bins {
                    cumsum += f64::from(s_view[[f, t]]);
                    if cumsum >= threshold {
                        idx = f;
                        break;
                    }
                }
                freq_view[[idx, t]]
            })
            .collect()
    } else {
        let mut out = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            let mut total = 0.0f64;
            for f in 0..n_bins {
                total += f64::from(s_view[[f, t]]);
            }
            let threshold = roll_percent * total;

            let mut cumsum = 0.0f64;
            let mut idx = 0usize;
            for f in 0..n_bins {
                cumsum += f64::from(s_view[[f, t]]);
                if cumsum >= threshold {
                    idx = f;
                    break;
                }
            }
            out[t] = freq_view[[idx, t]];
        }
        out
    };

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral rolloff from real, non-negative spectrogram with a per-bin, per-frame
/// frequency grid (f64 input -> f64 output).
#[pyfunction]
#[pyo3(signature = (s, freq, roll_percent))]
pub fn spectral_rolloff_variable_freq_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    freq: PyReadonlyArray2<'py, f64>,
    roll_percent: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !(0.0..1.0).contains(&roll_percent) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "roll_percent must lie in the range (0, 1)",
        ));
    }

    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if freq_view.shape() != [n_bins, n_frames] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "freq must have shape (n_bins, n_frames) matching S",
        ));
    }

    let out_data: Vec<f64> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut total = 0.0f64;
                for f in 0..n_bins {
                    total += s_view[[f, t]];
                }
                let threshold = roll_percent * total;

                let mut cumsum = 0.0f64;
                let mut idx = 0usize;
                for f in 0..n_bins {
                    cumsum += s_view[[f, t]];
                    if cumsum >= threshold {
                        idx = f;
                        break;
                    }
                }
                freq_view[[idx, t]]
            })
            .collect()
    } else {
        let mut out = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            let mut total = 0.0f64;
            for f in 0..n_bins {
                total += s_view[[f, t]];
            }
            let threshold = roll_percent * total;

            let mut cumsum = 0.0f64;
            let mut idx = 0usize;
            for f in 0..n_bins {
                cumsum += s_view[[f, t]];
                if cumsum >= threshold {
                    idx = f;
                    break;
                }
            }
            out[t] = freq_view[[idx, t]];
        }
        out
    };

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Convert power spectrogram to dB scale (f32 precision).
///
/// Computes: dB = 10 * log10(S / ref_power)
/// with optional thresholding and clipping.
///
/// Parameters:
/// - S: Power spectrogram (float32), any shape
/// - ref_power: Reference power level (default 1.0)
/// - amin: Minimum power threshold to avoid log(0) (default 1e-10)
/// - top_db: Clip output to this range above minimum (default None = no clipping)
///
/// Returns: dB-scaled spectrogram (float32)
#[pyfunction]
#[pyo3(signature = (S, ref_power = 1.0, amin = 1e-10, top_db = None))]
pub fn power_to_db_f32<'py>(
    py: Python<'py>,
    S: PyReadonlyArray1<'py, f32>,
    ref_power: f32,
    amin: f32,
    top_db: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let S_slice = S.as_slice()?;

    // Compute dB with thresholding
    let mut result = Vec::with_capacity(S_slice.len());
    for &s in S_slice {
        let s_clipped = s.max(amin);
        let db = 10.0 * (s_clipped / ref_power).log10();
        result.push(db);
    }

    // Apply optional top_db clipping below the peak, matching librosa.
    if let Some(top) = top_db {
        let max_db = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_db - top;
        for db in &mut result {
            if *db < threshold {
                *db = threshold;
            }
        }
    }

    // Convert to ndarray and return as PyArray
    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert power spectrogram to dB scale (f64 precision).
#[pyfunction]
#[pyo3(signature = (S, ref_power = 1.0, amin = 1e-10, top_db = None))]
pub fn power_to_db_f64<'py>(
    py: Python<'py>,
    S: PyReadonlyArray1<'py, f64>,
    ref_power: f64,
    amin: f64,
    top_db: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let S_slice = S.as_slice()?;

    let mut result = Vec::with_capacity(S_slice.len());
    for &s in S_slice {
        let s_clipped = s.max(amin);
        let db = 10.0 * (s_clipped / ref_power).log10();
        result.push(db);
    }

    if let Some(top) = top_db {
        let max_db = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_db - top;
        for db in &mut result {
            if *db < threshold {
                *db = threshold;
            }
        }
    }

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert amplitude spectrogram to dB scale (f32 precision).
///
/// Computes: dB = 20 * log10(A / ref_amplitude)
/// with optional thresholding and clipping.
///
/// Parameters:
/// - A: Amplitude spectrogram (float32), any shape
/// - ref_amplitude: Reference amplitude level (default 1.0)
/// - amin: Minimum amplitude threshold to avoid log(0) (default 1e-5)
/// - top_db: Clip output to this range above minimum (default None)
///
/// Returns: dB-scaled spectrogram (float32)
#[pyfunction]
#[pyo3(signature = (A, ref_amplitude = 1.0, amin = 1e-5, top_db = None))]
pub fn amplitude_to_db_f32<'py>(
    py: Python<'py>,
    A: PyReadonlyArray1<'py, f32>,
    ref_amplitude: f32,
    amin: f32,
    top_db: Option<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let A_slice = A.as_slice()?;

    let mut result = Vec::with_capacity(A_slice.len());
    for &a in A_slice {
        let a_clipped = a.max(amin);
        let db = 20.0 * (a_clipped / ref_amplitude).log10();
        result.push(db);
    }

    if let Some(top) = top_db {
        let max_db = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let threshold = max_db - top;
        for db in &mut result {
            if *db < threshold {
                *db = threshold;
            }
        }
    }

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert amplitude spectrogram to dB scale (f64 precision).
#[pyfunction]
#[pyo3(signature = (A, ref_amplitude = 1.0, amin = 1e-5, top_db = None))]
pub fn amplitude_to_db_f64<'py>(
    py: Python<'py>,
    A: PyReadonlyArray1<'py, f64>,
    ref_amplitude: f64,
    amin: f64,
    top_db: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let A_slice = A.as_slice()?;

    let mut result = Vec::with_capacity(A_slice.len());
    for &a in A_slice {
        let a_clipped = a.max(amin);
        let db = 20.0 * (a_clipped / ref_amplitude).log10();
        result.push(db);
    }

    if let Some(top) = top_db {
        let max_db = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_db - top;
        for db in &mut result {
            if *db < threshold {
                *db = threshold;
            }
        }
    }

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert dB to power scale (f32 precision).
///
/// Inverse of power_to_db: Power = ref_power * 10^(dB / 10)
#[pyfunction]
#[pyo3(signature = (S_db, ref_power = 1.0))]
pub fn db_to_power_f32<'py>(
    py: Python<'py>,
    S_db: PyReadonlyArray1<'py, f32>,
    ref_power: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let S_db_slice = S_db.as_slice()?;

    let result: Vec<f32> = S_db_slice
        .iter()
        .map(|&db| ref_power * (10.0f32.powf(db / 10.0)))
        .collect();

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert dB to power scale (f64 precision).
#[pyfunction]
#[pyo3(signature = (S_db, ref_power = 1.0))]
pub fn db_to_power_f64<'py>(
    py: Python<'py>,
    S_db: PyReadonlyArray1<'py, f64>,
    ref_power: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let S_db_slice = S_db.as_slice()?;

    let result: Vec<f64> = S_db_slice
        .iter()
        .map(|&db| ref_power * (10.0f64.powf(db / 10.0)))
        .collect();

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert dB to amplitude scale (f32 precision).
///
/// Inverse of amplitude_to_db: Amplitude = ref_amplitude * 10^(dB / 20)
#[pyfunction]
#[pyo3(signature = (A_db, ref_amplitude = 1.0))]
pub fn db_to_amplitude_f32<'py>(
    py: Python<'py>,
    A_db: PyReadonlyArray1<'py, f32>,
    ref_amplitude: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let A_db_slice = A_db.as_slice()?;

    let result: Vec<f32> = A_db_slice
        .iter()
        .map(|&db| ref_amplitude * (10.0f32.powf(db / 20.0)))
        .collect();

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

/// Convert dB to amplitude scale (f64 precision).
#[pyfunction]
#[pyo3(signature = (A_db, ref_amplitude = 1.0))]
pub fn db_to_amplitude_f64<'py>(
    py: Python<'py>,
    A_db: PyReadonlyArray1<'py, f64>,
    ref_amplitude: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let A_db_slice = A_db.as_slice()?;

    let result: Vec<f64> = A_db_slice
        .iter()
        .map(|&db| ref_amplitude * (10.0f64.powf(db / 20.0)))
        .collect();

    let arr = Array1::from_vec(result);
    Ok(arr.into_pyarray_bound(py).to_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral contrast kernels
// ─────────────────────────────────────────────────────────────────────────────

/// Per-channel scheduling thresholds for fused spectral contrast.
///
/// Small stereo jobs can lose to rayon overhead, while quad+ tends to benefit
/// from channel-level parallelism even at moderate frame counts.
const CONTRAST_TASK_PAR_MIN_WORK_STEREO: usize = 48_000;
const CONTRAST_TASK_PAR_MIN_WORK_TRI: usize = 36_000;

#[inline]
fn use_task_parallel_for_contrast(n_channels: usize, n_bands: usize, n_frames: usize) -> bool {
    let work = n_channels * n_bands * n_frames;
    if n_channels <= 2 {
        return work >= CONTRAST_TASK_PAR_MIN_WORK_STEREO;
    }
    if n_channels == 3 {
        return work >= CONTRAST_TASK_PAR_MIN_WORK_TRI;
    }
    true
}

/// Compute peak and valley energies for a spectrogram sub-band (f32).
///
/// Operates on a 2D array of shape (n_bins_in_band, n_frames):
///   - Sorts along axis 0 (bins) for each frame
///   - Extracts quantile-based indices:
///     - valley: bottom `idx` entries
///     - peak: top `idx` entries
///   - Computes mean energy for each (peak, valley) across frames
///
/// Returns (peak[1, n_frames], valley[1, n_frames])
#[inline]
fn spectral_contrast_band_core_f32(
    s_view: ndarray::ArrayView2<'_, f32>,
    idx: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let mut peak_vec = vec![0.0f32; n_frames];
    let mut valley_vec = vec![0.0f32; n_frames];

    if n_bins * n_frames >= PAR_THRESHOLD {
        let chunks: Vec<_> = (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut col: Vec<f32> = s_view.column(t).iter().cloned().collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let valley = col[0..idx].iter().sum::<f32>() / idx as f32;
                let peak = col[n_bins - idx..n_bins].iter().sum::<f32>() / idx as f32;
                (peak, valley)
            })
            .collect();

        for (t, (peak, valley)) in chunks.into_iter().enumerate() {
            peak_vec[t] = peak;
            valley_vec[t] = valley;
        }
    } else {
        for t in 0..n_frames {
            let mut col: Vec<f32> = s_view.column(t).iter().cloned().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            valley_vec[t] = col[0..idx].iter().sum::<f32>() / idx as f32;
            peak_vec[t] = col[n_bins - idx..n_bins].iter().sum::<f32>() / idx as f32;
        }
    }

    (peak_vec, valley_vec)
}

#[inline]
fn spectral_contrast_band_write_f32(
    s_view: ndarray::ArrayView2<'_, f32>,
    idx: usize,
    peak_out: &mut [f32],
    valley_out: &mut [f32],
) {
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    for t in 0..n_frames {
        let mut col: Vec<f32> = s_view.column(t).iter().cloned().collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        valley_out[t] = col[0..idx].iter().sum::<f32>() / idx as f32;
        peak_out[t] = col[n_bins - idx..n_bins].iter().sum::<f32>() / idx as f32;
    }
}

#[pyfunction]
#[pyo3(signature = (s_band, quantile))]
pub fn spectral_contrast_band_f32<'py>(
    py: Python<'py>,
    s_band: PyReadonlyArray2<'py, f32>,
    quantile: f64,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    if !(0.0 < quantile && quantile < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quantile must lie in (0, 1)",
        ));
    }

    let s_view = s_band.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if n_bins == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "band must have at least one bin",
        ));
    }

    let idx = std::cmp::max(1usize, (quantile * n_bins as f64).round() as usize);
    let idx = std::cmp::min(idx, n_bins - 1);
    let (peak_vec, valley_vec) = spectral_contrast_band_core_f32(s_view, idx);

    let peak_out = Array2::from_shape_vec((1, n_frames), peak_vec)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let valley_out = Array2::from_shape_vec((1, n_frames), valley_vec)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        peak_out.into_pyarray_bound(py).to_owned(),
        valley_out.into_pyarray_bound(py).to_owned(),
    ))
}

/// Compute peak and valley energies for a spectrogram sub-band (f64).
#[inline]
fn spectral_contrast_band_core_f64(
    s_view: ndarray::ArrayView2<'_, f64>,
    idx: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let mut peak_vec = vec![0.0f64; n_frames];
    let mut valley_vec = vec![0.0f64; n_frames];

    if n_bins * n_frames >= PAR_THRESHOLD {
        let chunks: Vec<_> = (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut col: Vec<f64> = s_view.column(t).iter().cloned().collect();
                col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let valley = col[0..idx].iter().sum::<f64>() / idx as f64;
                let peak = col[n_bins - idx..n_bins].iter().sum::<f64>() / idx as f64;

                (peak, valley)
            })
            .collect();

        for (t, (peak, valley)) in chunks.into_iter().enumerate() {
            peak_vec[t] = peak;
            valley_vec[t] = valley;
        }
    } else {
        for t in 0..n_frames {
            let mut col: Vec<f64> = s_view.column(t).iter().cloned().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            valley_vec[t] = col[0..idx].iter().sum::<f64>() / idx as f64;
            peak_vec[t] = col[n_bins - idx..n_bins].iter().sum::<f64>() / idx as f64;
        }
    }

    (peak_vec, valley_vec)
}

#[inline]
fn spectral_contrast_band_write_f64(
    s_view: ndarray::ArrayView2<'_, f64>,
    idx: usize,
    peak_out: &mut [f64],
    valley_out: &mut [f64],
) {
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    for t in 0..n_frames {
        let mut col: Vec<f64> = s_view.column(t).iter().cloned().collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        valley_out[t] = col[0..idx].iter().sum::<f64>() / idx as f64;
        peak_out[t] = col[n_bins - idx..n_bins].iter().sum::<f64>() / idx as f64;
    }
}

#[pyfunction]
#[pyo3(signature = (s_band, quantile))]
pub fn spectral_contrast_band_f64<'py>(
    py: Python<'py>,
    s_band: PyReadonlyArray2<'py, f64>,
    quantile: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    if !(0.0 < quantile && quantile < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "quantile must lie in (0, 1)",
        ));
    }

    let s_view = s_band.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    if n_bins == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "band must have at least one bin",
        ));
    }

    let idx = std::cmp::max(1usize, (quantile * n_bins as f64).round() as usize);
    let idx = std::cmp::min(idx, n_bins - 1);
    let (peak_vec, valley_vec) = spectral_contrast_band_core_f64(s_view, idx);

    let peak_out = Array2::from_shape_vec((1, n_frames), peak_vec)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let valley_out = Array2::from_shape_vec((1, n_frames), valley_vec)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        peak_out.into_pyarray_bound(py).to_owned(),
        valley_out.into_pyarray_bound(py).to_owned(),
    ))
}

#[pyfunction]
#[pyo3(signature = (s_batch, band_starts, band_stops, idx_qs))]
pub fn spectral_contrast_fused_f32<'py>(
    py: Python<'py>,
    s_batch: PyReadonlyArray3<'py, f32>,
    band_starts: PyReadonlyArray1<'py, i64>,
    band_stops: PyReadonlyArray1<'py, i64>,
    idx_qs: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
    let s_view = s_batch.as_array();
    let starts = band_starts.as_slice()?;
    let stops = band_stops.as_slice()?;
    let idxs = idx_qs.as_slice()?;

    if starts.len() != stops.len() || starts.len() != idxs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "band metadata arrays must have equal length",
        ));
    }

    let (n_channels, n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1], s_view.shape()[2]);
    let n_bands = starts.len();
    let mut band_meta = Vec::with_capacity(n_bands);

    for ((&start_i64, &stop_i64), &idx_i64) in starts.iter().zip(stops.iter()).zip(idxs.iter()) {
        let start = usize::try_from(start_i64).map_err(|_| pyo3::exceptions::PyValueError::new_err("band_starts must be non-negative"))?;
        let stop = usize::try_from(stop_i64).map_err(|_| pyo3::exceptions::PyValueError::new_err("band_stops must be non-negative"))?;
        let idx = usize::try_from(idx_i64).map_err(|_| pyo3::exceptions::PyValueError::new_err("idx_qs must be non-negative"))?;

        if !(start < stop && stop <= n_bins) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid band range in spectral_contrast_fused_f32",
            ));
        }
        let n_sub = stop - start;
        if idx == 0 || idx >= n_sub {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "idx_q must satisfy 1 <= idx_q < band width",
            ));
        }
        band_meta.push((start, stop, idx));
    }

    let n_tasks = n_channels * n_bands;
    let mut peak_data = vec![0.0f32; n_tasks * n_frames];
    let mut valley_data = vec![0.0f32; n_tasks * n_frames];

    if n_channels == 1 {
        for band in 0..n_bands {
            let (start, stop, idx) = band_meta[band];
            let sub = s_view.slice(s![0, start..stop, ..]);
            let (pk, vl) = spectral_contrast_band_core_f32(sub, idx);
            let row_start = band * n_frames;
            peak_data[row_start..row_start + n_frames].copy_from_slice(&pk);
            valley_data[row_start..row_start + n_frames].copy_from_slice(&vl);
        }
    } else if n_channels >= 4 {
        // Heavy multichannel workloads scale better with channel-level tasks.
        let ch_span = n_bands * n_frames;
        peak_data
            .par_chunks_mut(ch_span)
            .zip(valley_data.par_chunks_mut(ch_span))
            .enumerate()
            .for_each(|(ch, (peak_ch, valley_ch))| {
                for band in 0..n_bands {
                    let (start, stop, idx) = band_meta[band];
                    let sub = s_view.slice(s![ch, start..stop, ..]);
                    let row_start = band * n_frames;
                    spectral_contrast_band_write_f32(
                        sub,
                        idx,
                        &mut peak_ch[row_start..row_start + n_frames],
                        &mut valley_ch[row_start..row_start + n_frames],
                    );
                }
            });
    } else if use_task_parallel_for_contrast(n_channels, n_bands, n_frames) {
        peak_data
            .par_chunks_mut(n_frames)
            .zip(valley_data.par_chunks_mut(n_frames))
            .enumerate()
            .for_each(|(task_idx, (peak_row, valley_row))| {
                let ch = task_idx / n_bands;
                let band = task_idx % n_bands;
                let (start, stop, idx) = band_meta[band];
                let sub = s_view.slice(s![ch, start..stop, ..]);
                spectral_contrast_band_write_f32(sub, idx, peak_row, valley_row);
            });
    } else {
        for ch in 0..n_channels {
            for band in 0..n_bands {
                let (start, stop, idx) = band_meta[band];
                let sub = s_view.slice(s![ch, start..stop, ..]);
                let row_start = (ch * n_bands + band) * n_frames;
                spectral_contrast_band_write_f32(
                    sub,
                    idx,
                    &mut peak_data[row_start..row_start + n_frames],
                    &mut valley_data[row_start..row_start + n_frames],
                );
            }
        }
    }

    let peak = Array3::from_shape_vec((n_channels, n_bands, n_frames), peak_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let valley = Array3::from_shape_vec((n_channels, n_bands, n_frames), valley_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        peak.into_pyarray_bound(py).to_owned(),
        valley.into_pyarray_bound(py).to_owned(),
    ))
}

#[pyfunction]
#[pyo3(signature = (s_batch, band_starts, band_stops, idx_qs))]
pub fn spectral_contrast_fused_f64<'py>(
    py: Python<'py>,
    s_batch: PyReadonlyArray3<'py, f64>,
    band_starts: PyReadonlyArray1<'py, i64>,
    band_stops: PyReadonlyArray1<'py, i64>,
    idx_qs: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>)> {
    let s_view = s_batch.as_array();
    let starts = band_starts.as_slice()?;
    let stops = band_stops.as_slice()?;
    let idxs = idx_qs.as_slice()?;

    if starts.len() != stops.len() || starts.len() != idxs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "band metadata arrays must have equal length",
        ));
    }

    let (n_channels, n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1], s_view.shape()[2]);
    let n_bands = starts.len();
    let mut band_meta = Vec::with_capacity(n_bands);

    for ((&start_i64, &stop_i64), &idx_i64) in starts.iter().zip(stops.iter()).zip(idxs.iter()) {
        let start = usize::try_from(start_i64).map_err(|_| pyo3::exceptions::PyValueError::new_err("band_starts must be non-negative"))?;
        let stop = usize::try_from(stop_i64).map_err(|_| pyo3::exceptions::PyValueError::new_err("band_stops must be non-negative"))?;
        let idx = usize::try_from(idx_i64).map_err(|_| pyo3::exceptions::PyValueError::new_err("idx_qs must be non-negative"))?;

        if !(start < stop && stop <= n_bins) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid band range in spectral_contrast_fused_f64",
            ));
        }
        let n_sub = stop - start;
        if idx == 0 || idx >= n_sub {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "idx_q must satisfy 1 <= idx_q < band width",
            ));
        }
        band_meta.push((start, stop, idx));
    }

    let n_tasks = n_channels * n_bands;
    let mut peak_data = vec![0.0f64; n_tasks * n_frames];
    let mut valley_data = vec![0.0f64; n_tasks * n_frames];

    if n_channels == 1 {
        for band in 0..n_bands {
            let (start, stop, idx) = band_meta[band];
            let sub = s_view.slice(s![0, start..stop, ..]);
            let (pk, vl) = spectral_contrast_band_core_f64(sub, idx);
            let row_start = band * n_frames;
            peak_data[row_start..row_start + n_frames].copy_from_slice(&pk);
            valley_data[row_start..row_start + n_frames].copy_from_slice(&vl);
        }
    } else if n_channels >= 4 {
        let ch_span = n_bands * n_frames;
        peak_data
            .par_chunks_mut(ch_span)
            .zip(valley_data.par_chunks_mut(ch_span))
            .enumerate()
            .for_each(|(ch, (peak_ch, valley_ch))| {
                for band in 0..n_bands {
                    let (start, stop, idx) = band_meta[band];
                    let sub = s_view.slice(s![ch, start..stop, ..]);
                    let row_start = band * n_frames;
                    spectral_contrast_band_write_f64(
                        sub,
                        idx,
                        &mut peak_ch[row_start..row_start + n_frames],
                        &mut valley_ch[row_start..row_start + n_frames],
                    );
                }
            });
    } else if use_task_parallel_for_contrast(n_channels, n_bands, n_frames) {
        peak_data
            .par_chunks_mut(n_frames)
            .zip(valley_data.par_chunks_mut(n_frames))
            .enumerate()
            .for_each(|(task_idx, (peak_row, valley_row))| {
                let ch = task_idx / n_bands;
                let band = task_idx % n_bands;
                let (start, stop, idx) = band_meta[band];
                let sub = s_view.slice(s![ch, start..stop, ..]);
                spectral_contrast_band_write_f64(sub, idx, peak_row, valley_row);
            });
    } else {
        for ch in 0..n_channels {
            for band in 0..n_bands {
                let (start, stop, idx) = band_meta[band];
                let sub = s_view.slice(s![ch, start..stop, ..]);
                let row_start = (ch * n_bands + band) * n_frames;
                spectral_contrast_band_write_f64(
                    sub,
                    idx,
                    &mut peak_data[row_start..row_start + n_frames],
                    &mut valley_data[row_start..row_start + n_frames],
                );
            }
        }
    }

    let peak = Array3::from_shape_vec((n_channels, n_bands, n_frames), peak_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let valley = Array3::from_shape_vec((n_channels, n_bands, n_frames), valley_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        peak.into_pyarray_bound(py).to_owned(),
        valley.into_pyarray_bound(py).to_owned(),
    ))
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral flatness kernels
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (s, amin, power))]
pub fn spectral_flatness_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    amin: f64,
    power: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if amin <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("amin must be strictly positive"));
    }

    let s_view = s.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    let amin_f = amin as f32;
    let power_f = power as f32;
    let inv_n = 1.0f32 / n_bins as f32;

    let (sum_log, sum_val): (Vec<f32>, Vec<f32>) = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut sl = 0.0f32;
                let mut sv = 0.0f32;
                for f in 0..n_bins {
                    let mut x = s_view[[f, t]];
                    x = if (power - 2.0).abs() < 1e-12 { x * x } else if (power - 1.0).abs() < 1e-12 { x } else { x.powf(power_f) };
                    let v = x.max(amin_f);
                    sl += v.ln();
                    sv += v;
                }
                (sl, sv)
            })
            .unzip()
    } else {
        let mut sl = vec![0.0f32; n_frames];
        let mut sv = vec![0.0f32; n_frames];
        for t in 0..n_frames {
            for f in 0..n_bins {
                let mut x = s_view[[f, t]];
                x = if (power - 2.0).abs() < 1e-12 { x * x } else if (power - 1.0).abs() < 1e-12 { x } else { x.powf(power_f) };
                let v = x.max(amin_f);
                sl[t] += v.ln();
                sv[t] += v;
            }
        }
        (sl, sv)
    };

    let out_data: Vec<f32> = (0..n_frames)
        .map(|t| ((sum_log[t] * inv_n).exp()) / (sum_val[t] * inv_n))
        .collect();

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (s, amin, power))]
pub fn spectral_flatness_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    amin: f64,
    power: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if amin <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("amin must be strictly positive"));
    }

    let s_view = s.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let inv_n = 1.0f64 / n_bins as f64;

    let (sum_log, sum_val): (Vec<f64>, Vec<f64>) = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_frames)
            .into_par_iter()
            .map(|t| {
                let mut sl = 0.0f64;
                let mut sv = 0.0f64;
                for f in 0..n_bins {
                    let mut x = s_view[[f, t]];
                    x = if (power - 2.0).abs() < 1e-12 { x * x } else if (power - 1.0).abs() < 1e-12 { x } else { x.powf(power) };
                    let v = x.max(amin);
                    sl += v.ln();
                    sv += v;
                }
                (sl, sv)
            })
            .unzip()
    } else {
        let mut sl = vec![0.0f64; n_frames];
        let mut sv = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            for f in 0..n_bins {
                let mut x = s_view[[f, t]];
                x = if (power - 2.0).abs() < 1e-12 { x * x } else if (power - 1.0).abs() < 1e-12 { x } else { x.powf(power) };
                let v = x.max(amin);
                sl[t] += v.ln();
                sv[t] += v;
            }
        }
        (sl, sv)
    };

    let out_data: Vec<f64> = (0..n_frames)
        .map(|t| ((sum_log[t] * inv_n).exp()) / (sum_val[t] * inv_n))
        .collect();

    let out = Array2::from_shape_vec((1, n_frames), out_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral bandwidth from a real, non-negative spectrogram (f32 input → f64 output).
///
/// Matches the `librosa.feature.spectral_bandwidth()` fast path:
///   bandwidth[t] = Σ_f S[f,t] · |freq[f] - centroid[t]|  /  Σ_f S[f,t]
///
/// Uses cache-friendly row-by-row traversal and optional rayon parallelism.
/// The caller is responsible for ensuring S ≥ 0 (the Python dispatch reorders
/// the guard to fire this kernel before `np.any(S < 0)` for the fast path).
#[pyfunction]
#[pyo3(signature = (s, freq, centroid, norm=true, p=2.0))]
pub fn spectral_bandwidth_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freq: PyReadonlyArray1<'py, f64>,
    centroid: PyReadonlyArray2<'py, f64>,
    norm: bool,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if p <= 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("p must be strictly positive")); }
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let c_view = centroid.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    if freq_view.len() != n_bins { return Err(pyo3::exceptions::PyValueError::new_err("freq length must match spectrogram bins")); }
    if c_view.shape() != [1, n_frames] { return Err(pyo3::exceptions::PyValueError::new_err("centroid must have shape (1, n_frames)")); }
    let freq_slice = freq_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("freq must be contiguous"))?;
    let inv_p = 1.0 / p;
    let use_p1 = (p - 1.0).abs() < 1e-12;
    let use_p2 = (p - 2.0).abs() < 1e-12;
    let out: Vec<f64> = (0..n_frames).map(|t| {
        let c = c_view[[0, t]];
        let mut den = 1.0f64;
        if norm {
            den = 0.0;
            for f in 0..n_bins { den += f64::from(s_view[[f, t]]); }
            if den <= f64::MIN_POSITIVE { return 0.0; }
        }
        let mut accum = 0.0f64;
        for f in 0..n_bins {
            let s_val = f64::from(s_view[[f, t]]) / den;
            let d = (freq_slice[f] - c).abs();
            let dev = if use_p2 { d * d } else if use_p1 { d } else { d.powf(p) };
            accum += s_val * dev;
        }
        if use_p2 { accum.sqrt() } else if use_p1 { accum } else { accum.powf(inv_p) }
    }).collect();
    let out = Array2::from_shape_vec((1, n_frames), out).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral bandwidth from a real, non-negative spectrogram (f64 input → f64 output).
#[pyfunction]
#[pyo3(signature = (s, freq, centroid, norm=true, p=2.0))]
pub fn spectral_bandwidth_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    freq: PyReadonlyArray1<'py, f64>,
    centroid: PyReadonlyArray2<'py, f64>,
    norm: bool,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if p <= 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("p must be strictly positive")); }
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let c_view = centroid.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    if freq_view.len() != n_bins { return Err(pyo3::exceptions::PyValueError::new_err("freq length must match spectrogram bins")); }
    if c_view.shape() != [1, n_frames] { return Err(pyo3::exceptions::PyValueError::new_err("centroid must have shape (1, n_frames)")); }
    let freq_slice = freq_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("freq must be contiguous"))?;
    let inv_p = 1.0 / p;
    let use_p1 = (p - 1.0).abs() < 1e-12;
    let use_p2 = (p - 2.0).abs() < 1e-12;
    let out: Vec<f64> = (0..n_frames).map(|t| {
        let c = c_view[[0, t]];
        let mut den = 1.0f64;
        if norm {
            den = 0.0;
            for f in 0..n_bins { den += s_view[[f, t]]; }
            if den <= f64::MIN_POSITIVE { return 0.0; }
        }
        let mut accum = 0.0f64;
        for f in 0..n_bins {
            let s_val = s_view[[f, t]] / den;
            let d = (freq_slice[f] - c).abs();
            let dev = if use_p2 { d * d } else if use_p1 { d } else { d.powf(p) };
            accum += s_val * dev;
        }
        if use_p2 { accum.sqrt() } else if use_p1 { accum } else { accum.powf(inv_p) }
    }).collect();
    let out = Array2::from_shape_vec((1, n_frames), out).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral bandwidth from a real, non-negative spectrogram (f32 input → f64 output)
/// with automatic centroid calculation.
///
/// Norm options:
/// - If `norm=True`, each frequency bin is normalized by the sum of magnitudes in the frame,
///   ensuring the bandwidth is computed from the proportionate energy in each bin.
/// - If `norm=False`, no normalization is applied (equivalent to uniform bin weighting).
#[pyfunction]
#[pyo3(signature = (s, freq, norm=true, p=2.0))]
pub fn spectral_bandwidth_auto_centroid_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    freq: PyReadonlyArray1<'py, f64>,
    norm: bool,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if p <= 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("p must be strictly positive")); }
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    if freq_view.len() != n_bins { return Err(pyo3::exceptions::PyValueError::new_err("freq length must match spectrogram bins")); }
    let freq_slice = freq_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("freq must be contiguous"))?;
    let inv_p = 1.0 / p;
    let use_p1 = (p - 1.0).abs() < 1e-12;
    let use_p2 = (p - 2.0).abs() < 1e-12;
    let out: Vec<f64> = (0..n_frames).map(|t| {
        let mut den = 0.0f64;
        let mut numer = 0.0f64;
        for f in 0..n_bins { let sv = f64::from(s_view[[f, t]]); den += sv; numer += sv * freq_slice[f]; }
        if den <= f64::MIN_POSITIVE { return 0.0; }
        let c = numer / den;
        let norm_den = if norm { den } else { 1.0 };
        let mut accum = 0.0f64;
        for f in 0..n_bins {
            let s_val = f64::from(s_view[[f, t]]) / norm_den;
            let d = (freq_slice[f] - c).abs();
            let dev = if use_p2 { d * d } else if use_p1 { d } else { d.powf(p) };
            accum += s_val * dev;
        }
        if use_p2 { accum.sqrt() } else if use_p1 { accum } else { accum.powf(inv_p) }
    }).collect();
    let out = Array2::from_shape_vec((1, n_frames), out).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

/// Compute spectral bandwidth from a real, non-negative spectrogram (f64 input → f64 output)
/// with automatic centroid calculation.
#[pyfunction]
#[pyo3(signature = (s, freq, norm=true, p=2.0))]
pub fn spectral_bandwidth_auto_centroid_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    freq: PyReadonlyArray1<'py, f64>,
    norm: bool,
    p: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if p <= 0.0 { return Err(pyo3::exceptions::PyValueError::new_err("p must be strictly positive")); }
    let s_view = s.as_array();
    let freq_view = freq.as_array();
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    if freq_view.len() != n_bins { return Err(pyo3::exceptions::PyValueError::new_err("freq length must match spectrogram bins")); }
    let freq_slice = freq_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("freq must be contiguous"))?;
    let inv_p = 1.0 / p;
    let use_p1 = (p - 1.0).abs() < 1e-12;
    let use_p2 = (p - 2.0).abs() < 1e-12;
    let out: Vec<f64> = (0..n_frames).map(|t| {
        let mut den = 0.0f64;
        let mut numer = 0.0f64;
        for f in 0..n_bins { let sv = s_view[[f, t]]; den += sv; numer += sv * freq_slice[f]; }
        if den <= f64::MIN_POSITIVE { return 0.0; }
        let c = numer / den;
        let norm_den = if norm { den } else { 1.0 };
        let mut accum = 0.0f64;
        for f in 0..n_bins {
            let s_val = s_view[[f, t]] / norm_den;
            let d = (freq_slice[f] - c).abs();
            let dev = if use_p2 { d * d } else if use_p1 { d } else { d.powf(p) };
            accum += s_val * dev;
        }
        if use_p2 { accum.sqrt() } else if use_p1 { accum } else { accum.powf(inv_p) }
    }).collect();
    let out = Array2::from_shape_vec((1, n_frames), out).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 10A: HPSS Median Filter Kernels
// ─────────────────────────────────────────────────────────────────────────────
// Optimized 2D median filtering for harmonic/percussive source separation.
//
// - Harmonic filter: vertical kernel (size along freq bins, 1 along time)
// - Percussive filter: horizontal kernel (1 along freq bins, size along time)
// - Padding: reflect mode (mirror at boundaries)
// - Strategy: separate passes per frame/bin to maximize cache locality

/// Compute the median of a slice (assumes contiguous storage).
#[inline]
fn median_window_f32(vals: &mut [f32]) -> f32 {
    let mid = vals.len() / 2;
    let (_, nth, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *nth
}

#[inline]
fn median_window_f64(vals: &mut [f64]) -> f64 {
    let mid = vals.len() / 2;
    let (_, nth, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *nth
}

/// Map an arbitrary signed index onto `[0, len)` using scipy/ndimage
/// `mode="reflect"` semantics as observed for median_filter: mirror including
/// the edge pixel.
#[inline]
fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }

    let period = 2 * len as isize;
    let folded = idx.rem_euclid(period);
    if folded >= len as isize {
        (period - 1 - folded) as usize
    } else {
        folded as usize
    }
}

/// 2D median filter for HPSS harmonic component (horizontal/time kernel).
/// kernel_size: size along time, 1 along frequency.
fn compute_median_harmonic_f32(s_view: ndarray::ArrayView2<'_, f32>, kernel_size: usize) -> Array2<f32> {
    let (n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1]);
    let pad = kernel_size as isize / 2;
    let rows: Vec<Vec<f32>> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_bins)
            .into_par_iter()
            .map(|f| {
                let mut row_out = vec![0.0f32; n_frames];
                let mut window = vec![0.0f32; kernel_size];
                for t in 0..n_frames {
                    for k in 0..kernel_size {
                        let src_t = reflect_index(t as isize + k as isize - pad, n_frames);
                        window[k] = s_view[[f, src_t]];
                    }
                    row_out[t] = median_window_f32(&mut window);
                }
                row_out
            })
            .collect()
    } else {
        let mut rows = Vec::with_capacity(n_bins);
        for f in 0..n_bins {
            let mut row_out = vec![0.0f32; n_frames];
            let mut window = vec![0.0f32; kernel_size];
            for t in 0..n_frames {
                for k in 0..kernel_size {
                    let src_t = reflect_index(t as isize + k as isize - pad, n_frames);
                    window[k] = s_view[[f, src_t]];
                }
                row_out[t] = median_window_f32(&mut window);
            }
            rows.push(row_out);
        }
        rows
    };

    let mut out = Array2::<f32>::zeros((n_bins, n_frames));
    for (f, row_out) in rows.into_iter().enumerate() {
        for (t, v) in row_out.into_iter().enumerate() {
            out[[f, t]] = v;
        }
    }
    out
}

fn compute_median_harmonic_f64(s_view: ndarray::ArrayView2<'_, f64>, kernel_size: usize) -> Array2<f64> {
    let (n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1]);
    let pad = kernel_size as isize / 2;
    let rows: Vec<Vec<f64>> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_bins)
            .into_par_iter()
            .map(|f| {
                let mut row_out = vec![0.0f64; n_frames];
                let mut window = vec![0.0f64; kernel_size];
                for t in 0..n_frames {
                    for k in 0..kernel_size {
                        let src_t = reflect_index(t as isize + k as isize - pad, n_frames);
                        window[k] = s_view[[f, src_t]];
                    }
                    row_out[t] = median_window_f64(&mut window);
                }
                row_out
            })
            .collect()
    } else {
        let mut rows = Vec::with_capacity(n_bins);
        for f in 0..n_bins {
            let mut row_out = vec![0.0f64; n_frames];
            let mut window = vec![0.0f64; kernel_size];
            for t in 0..n_frames {
                for k in 0..kernel_size {
                    let src_t = reflect_index(t as isize + k as isize - pad, n_frames);
                    window[k] = s_view[[f, src_t]];
                }
                row_out[t] = median_window_f64(&mut window);
            }
            rows.push(row_out);
        }
        rows
    };

    let mut out = Array2::<f64>::zeros((n_bins, n_frames));
    for (f, row_out) in rows.into_iter().enumerate() {
        for (t, v) in row_out.into_iter().enumerate() {
            out[[f, t]] = v;
        }
    }
    out
}

fn compute_median_percussive_f32(s_view: ndarray::ArrayView2<'_, f32>, kernel_size: usize) -> Array2<f32> {
    let (n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1]);
    let pad = kernel_size as isize / 2;
    let rows: Vec<Vec<f32>> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_bins)
            .into_par_iter()
            .map(|f| {
                let mut row_out = vec![0.0f32; n_frames];
                let mut window = vec![0.0f32; kernel_size];
                for t in 0..n_frames {
                    for k in 0..kernel_size {
                        let src_f = reflect_index(f as isize + k as isize - pad, n_bins);
                        window[k] = s_view[[src_f, t]];
                    }
                    row_out[t] = median_window_f32(&mut window);
                }
                row_out
            })
            .collect()
    } else {
        let mut rows = Vec::with_capacity(n_bins);
        for f in 0..n_bins {
            let mut row_out = vec![0.0f32; n_frames];
            let mut window = vec![0.0f32; kernel_size];
            for t in 0..n_frames {
                for k in 0..kernel_size {
                    let src_f = reflect_index(f as isize + k as isize - pad, n_bins);
                    window[k] = s_view[[src_f, t]];
                }
                row_out[t] = median_window_f32(&mut window);
            }
            rows.push(row_out);
        }
        rows
    };

    let mut out = Array2::<f32>::zeros((n_bins, n_frames));
    for (f, row_out) in rows.into_iter().enumerate() {
        for (t, v) in row_out.into_iter().enumerate() {
            out[[f, t]] = v;
        }
    }
    out
}

fn compute_median_percussive_f64(s_view: ndarray::ArrayView2<'_, f64>, kernel_size: usize) -> Array2<f64> {
    let (n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1]);
    let pad = kernel_size as isize / 2;
    let rows: Vec<Vec<f64>> = if n_bins * n_frames >= PAR_THRESHOLD {
        (0..n_bins)
            .into_par_iter()
            .map(|f| {
                let mut row_out = vec![0.0f64; n_frames];
                let mut window = vec![0.0f64; kernel_size];
                for t in 0..n_frames {
                    for k in 0..kernel_size {
                        let src_f = reflect_index(f as isize + k as isize - pad, n_bins);
                        window[k] = s_view[[src_f, t]];
                    }
                    row_out[t] = median_window_f64(&mut window);
                }
                row_out
            })
            .collect()
    } else {
        let mut rows = Vec::with_capacity(n_bins);
        for f in 0..n_bins {
            let mut row_out = vec![0.0f64; n_frames];
            let mut window = vec![0.0f64; kernel_size];
            for t in 0..n_frames {
                for k in 0..kernel_size {
                    let src_f = reflect_index(f as isize + k as isize - pad, n_bins);
                    window[k] = s_view[[src_f, t]];
                }
                row_out[t] = median_window_f64(&mut window);
            }
            rows.push(row_out);
        }
        rows
    };

    let mut out = Array2::<f64>::zeros((n_bins, n_frames));
    for (f, row_out) in rows.into_iter().enumerate() {
        for (t, v) in row_out.into_iter().enumerate() {
            out[[f, t]] = v;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (s, kernel_size))]
pub fn median_filter_harmonic_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    kernel_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if kernel_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernel_size must be positive"));
    }

    let out = compute_median_harmonic_f32(s.as_array(), kernel_size);

    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (s, kernel_size))]
pub fn median_filter_harmonic_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    kernel_size: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if kernel_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernel_size must be positive"));
    }

    let out = compute_median_harmonic_f64(s.as_array(), kernel_size);

    Ok(out.into_pyarray_bound(py).to_owned())
}

/// 2D median filter for HPSS percussive component (vertical/frequency kernel).
/// kernel_size: size along frequency, 1 along time.
#[pyfunction]
#[pyo3(signature = (s, kernel_size))]
pub fn median_filter_percussive_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    kernel_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if kernel_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernel_size must be positive"));
    }

    let out = compute_median_percussive_f32(s.as_array(), kernel_size);

    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (s, kernel_size))]
pub fn median_filter_percussive_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    kernel_size: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if kernel_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernel_size must be positive"));
    }

    let out = compute_median_percussive_f64(s.as_array(), kernel_size);

    Ok(out.into_pyarray_bound(py).to_owned())
}

#[inline]
fn softmask_pair_f32(x: f32, x_ref: f32, power: f32, split_zeros: bool) -> f32 {
    let z = x.max(x_ref);
    if z < f32::MIN_POSITIVE {
        return if split_zeros { 0.5 } else { 0.0 };
    }
    let mask = (x / z).powf(power);
    let ref_mask = (x_ref / z).powf(power);
    mask / (mask + ref_mask)
}

#[inline]
fn softmask_pair_f64(x: f64, x_ref: f64, power: f64, split_zeros: bool) -> f64 {
    let z = x.max(x_ref);
    if z < f64::MIN_POSITIVE {
        return if split_zeros { 0.5 } else { 0.0 };
    }
    let mask = (x / z).powf(power);
    let ref_mask = (x_ref / z).powf(power);
    mask / (mask + ref_mask)
}

fn hpss_fused_core_2d_f32(
    s_view: ndarray::ArrayView2<'_, f32>,
    win_harm: usize,
    win_perc: usize,
    power: f64,
    margin_harm: f64,
    margin_perc: f64,
    return_mask: bool,
) -> PyResult<(Array2<f32>, Array2<f32>)> {
    if win_harm == 0 || win_perc == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernel sizes must be positive"));
    }
    if power <= 0.0 || !power.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err("power must be finite and strictly positive"));
    }
    if margin_harm < 1.0 || margin_perc < 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("margins must be >= 1.0"));
    }

    let (n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1]);
    let power_f = power as f32;
    let margin_harm_f = margin_harm as f32;
    let margin_perc_f = margin_perc as f32;
    let split_zeros = (margin_harm_f == 1.0) && (margin_perc_f == 1.0);

    let harm = compute_median_harmonic_f32(s_view, win_harm);
    let perc = compute_median_percussive_f32(s_view, win_perc);

    // Parallelize over frames (axis 1) for masking computation
    // This avoids nested sequential loops and enables frame-level parallelism
    let total_elements = n_bins * n_frames;
    let use_frame_parallelism = total_elements >= PAR_THRESHOLD;

    if use_frame_parallelism {
        // Parallel masking: process each frame independently
        let frame_results: Vec<Vec<(f32, f32)>> = (0..n_frames)
            .into_par_iter()
            .map(|t| {
                (0..n_bins)
                    .map(|f| {
                        let harm_val = harm[[f, t]];
                        let perc_val = perc[[f, t]];
                        let mh = softmask_pair_f32(harm_val, perc_val * margin_harm_f, power_f, split_zeros);
                        let mp = softmask_pair_f32(perc_val, harm_val * margin_perc_f, power_f, split_zeros);
                        (mh, mp)
                    })
                    .collect()
            })
            .collect();

        let mut out_h = Array2::<f32>::zeros((n_bins, n_frames));
        let mut out_p = Array2::<f32>::zeros((n_bins, n_frames));

        for (t, frame_masks) in frame_results.into_iter().enumerate() {
            for (f, (mh, mp)) in frame_masks.into_iter().enumerate() {
                if return_mask {
                    out_h[[f, t]] = mh;
                    out_p[[f, t]] = mp;
                } else {
                    let s_val = s_view[[f, t]];
                    out_h[[f, t]] = s_val * mh;
                    out_p[[f, t]] = s_val * mp;
                }
            }
        }
        Ok((out_h, out_p))
    } else {
        // Sequential masking: for small inputs, avoid parallelism overhead
        let mut out_h = Array2::<f32>::zeros((n_bins, n_frames));
        let mut out_p = Array2::<f32>::zeros((n_bins, n_frames));

        for f in 0..n_bins {
            for t in 0..n_frames {
                let harm_val = harm[[f, t]];
                let perc_val = perc[[f, t]];
                let mh = softmask_pair_f32(harm_val, perc_val * margin_harm_f, power_f, split_zeros);
                let mp = softmask_pair_f32(perc_val, harm_val * margin_perc_f, power_f, split_zeros);
                if return_mask {
                    out_h[[f, t]] = mh;
                    out_p[[f, t]] = mp;
                } else {
                    let s_val = s_view[[f, t]];
                    out_h[[f, t]] = s_val * mh;
                    out_p[[f, t]] = s_val * mp;
                }
            }
        }
        Ok((out_h, out_p))
    }
}

fn hpss_fused_core_2d_f64(
    s_view: ndarray::ArrayView2<'_, f64>,
    win_harm: usize,
    win_perc: usize,
    power: f64,
    margin_harm: f64,
    margin_perc: f64,
    return_mask: bool,
) -> PyResult<(Array2<f64>, Array2<f64>)> {
    if win_harm == 0 || win_perc == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("kernel sizes must be positive"));
    }
    if power <= 0.0 || !power.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err("power must be finite and strictly positive"));
    }
    if margin_harm < 1.0 || margin_perc < 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("margins must be >= 1.0"));
    }

    let (n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1]);
    let split_zeros = (margin_harm == 1.0) && (margin_perc == 1.0);

    let harm = compute_median_harmonic_f64(s_view, win_harm);
    let perc = compute_median_percussive_f64(s_view, win_perc);

    // Parallelize over frames (axis 1) for masking computation
    let total_elements = n_bins * n_frames;
    let use_frame_parallelism = total_elements >= PAR_THRESHOLD;

    if use_frame_parallelism {
        // Parallel masking: process each frame independently
        let frame_results: Vec<Vec<(f64, f64)>> = (0..n_frames)
            .into_par_iter()
            .map(|t| {
                (0..n_bins)
                    .map(|f| {
                        let harm_val = harm[[f, t]];
                        let perc_val = perc[[f, t]];
                        let mh = softmask_pair_f64(harm_val, perc_val * margin_harm, power, split_zeros);
                        let mp = softmask_pair_f64(perc_val, harm_val * margin_perc, power, split_zeros);
                        (mh, mp)
                    })
                    .collect()
            })
            .collect();

        let mut out_h = Array2::<f64>::zeros((n_bins, n_frames));
        let mut out_p = Array2::<f64>::zeros((n_bins, n_frames));

        for (t, frame_masks) in frame_results.into_iter().enumerate() {
            for (f, (mh, mp)) in frame_masks.into_iter().enumerate() {
                if return_mask {
                    out_h[[f, t]] = mh;
                    out_p[[f, t]] = mp;
                } else {
                    let s_val = s_view[[f, t]];
                    out_h[[f, t]] = s_val * mh;
                    out_p[[f, t]] = s_val * mp;
                }
            }
        }
        Ok((out_h, out_p))
    } else {
        // Sequential masking: for small inputs, avoid parallelism overhead
        let mut out_h = Array2::<f64>::zeros((n_bins, n_frames));
        let mut out_p = Array2::<f64>::zeros((n_bins, n_frames));

        for f in 0..n_bins {
            for t in 0..n_frames {
                let harm_val = harm[[f, t]];
                let perc_val = perc[[f, t]];
                let mh = softmask_pair_f64(harm_val, perc_val * margin_harm, power, split_zeros);
                let mp = softmask_pair_f64(perc_val, harm_val * margin_perc, power, split_zeros);
                if return_mask {
                    out_h[[f, t]] = mh;
                    out_p[[f, t]] = mp;
                } else {
                    let s_val = s_view[[f, t]];
                    out_h[[f, t]] = s_val * mh;
                    out_p[[f, t]] = s_val * mp;
                }
            }
        }
        Ok((out_h, out_p))
    }
}

#[pyfunction]
#[pyo3(signature = (s, win_harm, win_perc, power, margin_harm, margin_perc, return_mask=false))]
pub fn hpss_fused_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    win_harm: usize,
    win_perc: usize,
    power: f64,
    margin_harm: f64,
    margin_perc: f64,
    return_mask: bool,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let (out_h, out_p) = hpss_fused_core_2d_f32(
        s.as_array(),
        win_harm,
        win_perc,
        power,
        margin_harm,
        margin_perc,
        return_mask,
    )?;

    Ok((
        out_h.into_pyarray_bound(py).to_owned(),
        out_p.into_pyarray_bound(py).to_owned(),
    ))
}

#[pyfunction]
#[pyo3(signature = (s, win_harm, win_perc, power, margin_harm, margin_perc, return_mask=false))]
pub fn hpss_fused_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    win_harm: usize,
    win_perc: usize,
    power: f64,
    margin_harm: f64,
    margin_perc: f64,
    return_mask: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (out_h, out_p) = hpss_fused_core_2d_f64(
        s.as_array(),
        win_harm,
        win_perc,
        power,
        margin_harm,
        margin_perc,
        return_mask,
    )?;

    Ok((
        out_h.into_pyarray_bound(py).to_owned(),
        out_p.into_pyarray_bound(py).to_owned(),
    ))
}

#[pyfunction]
#[pyo3(signature = (s, win_harm, win_perc, power, margin_harm, margin_perc, return_mask=false))]
pub fn hpss_fused_batch_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray3<'py, f32>,
    win_harm: usize,
    win_perc: usize,
    power: f64,
    margin_harm: f64,
    margin_perc: f64,
    return_mask: bool,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
    let s_view = s.as_array();
    let (batch, n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1], s_view.shape()[2]);
    let per_batch_elements = n_bins * n_frames;

    // Adaptive dispatch heuristic:
    // Use batch-level parallelism ONLY when:
    //   1. Batch is large enough to justify thread pool overhead (>= 4)
    //   2. Per-batch work is small (< 150K elements) so inner kernels stay sequential
    //      This avoids nested rayon contention that slows down performance
    let use_batch_parallelism = batch >= BATCH_PAR_SIZE_MIN && per_batch_elements < BATCH_ELEMENT_MAX;

    if use_batch_parallelism {
        // Parallel batch processing: each batch runs sequentially (no inner parallelism)
        let results: Vec<_> = (0..batch)
            .into_par_iter()
            .map(|b| {
                hpss_fused_core_2d_f32(
                    s_view.index_axis(Axis(0), b),
                    win_harm,
                    win_perc,
                    power,
                    margin_harm,
                    margin_perc,
                    return_mask,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut out_h = Array3::<f32>::zeros((batch, n_bins, n_frames));
        let mut out_p = Array3::<f32>::zeros((batch, n_bins, n_frames));

        for (b, (harm_b, perc_b)) in results.into_iter().enumerate() {
            out_h.index_axis_mut(Axis(0), b).assign(&harm_b);
            out_p.index_axis_mut(Axis(0), b).assign(&perc_b);
        }

        Ok((
            out_h.into_pyarray_bound(py).to_owned(),
            out_p.into_pyarray_bound(py).to_owned(),
        ))
    } else {
        // Sequential batch processing: allows inner kernels to parallelize if work is large
        let mut out_h = Array3::<f32>::zeros((batch, n_bins, n_frames));
        let mut out_p = Array3::<f32>::zeros((batch, n_bins, n_frames));

        for b in 0..batch {
            let (harm_b, perc_b) = hpss_fused_core_2d_f32(
                s_view.index_axis(Axis(0), b),
                win_harm,
                win_perc,
                power,
                margin_harm,
                margin_perc,
                return_mask,
            )?;
            out_h.index_axis_mut(Axis(0), b).assign(&harm_b);
            out_p.index_axis_mut(Axis(0), b).assign(&perc_b);
        }

        Ok((
            out_h.into_pyarray_bound(py).to_owned(),
            out_p.into_pyarray_bound(py).to_owned(),
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (s, win_harm, win_perc, power, margin_harm, margin_perc, return_mask=false))]
pub fn hpss_fused_batch_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray3<'py, f64>,
    win_harm: usize,
    win_perc: usize,
    power: f64,
    margin_harm: f64,
    margin_perc: f64,
    return_mask: bool,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>)> {
    let s_view = s.as_array();
    let (batch, n_bins, n_frames) = (s_view.shape()[0], s_view.shape()[1], s_view.shape()[2]);
    let per_batch_elements = n_bins * n_frames;

    // Adaptive dispatch heuristic: same as f32 version
    let use_batch_parallelism = batch >= BATCH_PAR_SIZE_MIN && per_batch_elements < BATCH_ELEMENT_MAX;

    if use_batch_parallelism {
        // Parallel batch processing
        let results: Vec<_> = (0..batch)
            .into_par_iter()
            .map(|b| {
                hpss_fused_core_2d_f64(
                    s_view.index_axis(Axis(0), b),
                    win_harm,
                    win_perc,
                    power,
                    margin_harm,
                    margin_perc,
                    return_mask,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut out_h = Array3::<f64>::zeros((batch, n_bins, n_frames));
        let mut out_p = Array3::<f64>::zeros((batch, n_bins, n_frames));

        for (b, (harm_b, perc_b)) in results.into_iter().enumerate() {
            out_h.index_axis_mut(Axis(0), b).assign(&harm_b);
            out_p.index_axis_mut(Axis(0), b).assign(&perc_b);
        }

        Ok((
            out_h.into_pyarray_bound(py).to_owned(),
            out_p.into_pyarray_bound(py).to_owned(),
        ))
    } else {
        // Sequential batch processing
        let mut out_h = Array3::<f64>::zeros((batch, n_bins, n_frames));
        let mut out_p = Array3::<f64>::zeros((batch, n_bins, n_frames));

        for b in 0..batch {
            let (harm_b, perc_b) = hpss_fused_core_2d_f64(
                s_view.index_axis(Axis(0), b),
                win_harm,
                win_perc,
                power,
                margin_harm,
                margin_perc,
                return_mask,
            )?;
            out_h.index_axis_mut(Axis(0), b).assign(&harm_b);
            out_p.index_axis_mut(Axis(0), b).assign(&perc_b);
        }

        Ok((
            out_h.into_pyarray_bound(py).to_owned(),
            out_p.into_pyarray_bound(py).to_owned(),
        ))
    }
}
