/// Phase 15: Parallel tempogram autocorrelation using rustfft + rayon.
///
/// `tempogram_ac_f32/f64` replaces the `autocorrelate(windowed, axis=-2)` call
/// inside `librosa.feature.tempogram` for the 2-D (mono) case.
///
/// Algorithm (per time frame, rayon-parallel):
///   1. Copy win_length real samples into a zero-padded complex buffer of n_pad.
///   2. Forward complex FFT.
///   3. Replace each element with its power (|z|²).
///   4. Inverse complex FFT.
///   5. Normalize by n_pad, discard imaginary noise, truncate to win_length.
///
/// This matches scipy's rfft + irfft autocorrelation exactly (up to f32/f64
/// floating-point rounding) because for a real signal the complex and
/// real-to-complex power spectra are numerically identical.
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::Arc;

// ── f32 ──────────────────────────────────────────────────────────────────────

fn tempogram_ac_impl_f32(
    windowed: ndarray::ArrayView2<'_, f32>,
    n_pad: usize,
) -> PyResult<Array2<f32>> {
    let (win_len, n_frames) = windowed.dim();

    if win_len == 0 || n_frames == 0 {
        return Ok(Array2::zeros((win_len, n_frames)));
    }
    if n_pad < win_len {
        return Err(PyValueError::new_err(
            "n_pad must be >= win_length",
        ));
    }

    // Build shared FFT plans once; Arc<dyn Fft<f32>> is Send+Sync.
    let mut planner = FftPlanner::<f32>::new();
    let fft_fwd: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_forward(n_pad);
    let fft_inv: Arc<dyn rustfft::Fft<f32>> = planner.plan_fft_inverse(n_pad);

    let scratch_len = fft_fwd
        .get_inplace_scratch_len()
        .max(fft_inv.get_inplace_scratch_len())
        .max(1);

    let scale = 1.0f32 / n_pad as f32;

    // Parallel per-frame autocorrelation.
    // map_init creates one (buf, scratch) pair per Rayon worker thread.
    let cols: Vec<Vec<f32>> = (0..n_frames)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![Complex::<f32>::new(0.0, 0.0); n_pad],
                    vec![Complex::<f32>::new(0.0, 0.0); scratch_len],
                )
            },
            |(buf, scratch), t| {
                // Zero-pad the entire buffer first.
                for x in buf.iter_mut() {
                    *x = Complex::new(0.0, 0.0);
                }
                // Copy this frame's data.
                for i in 0..win_len {
                    buf[i] = Complex::new(windowed[[i, t]], 0.0);
                }
                // Forward FFT → power spectrum → inverse FFT.
                fft_fwd.process_with_scratch(buf, scratch);
                for x in buf.iter_mut() {
                    let p = x.re * x.re + x.im * x.im;
                    *x = Complex::new(p, 0.0);
                }
                fft_inv.process_with_scratch(buf, scratch);
                // Normalize (rustfft is unnormalized), keep win_len lags.
                buf[..win_len].iter().map(|x| x.re * scale).collect()
            },
        )
        .collect();

    // Assemble column-major output (win_len × n_frames).
    let mut out = Array2::<f32>::zeros((win_len, n_frames));
    for (t, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            out[[i, t]] = v;
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn tempogram_ac_f32<'py>(
    py: Python<'py>,
    windowed: PyReadonlyArray2<'py, f32>,
    n_pad: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let out = tempogram_ac_impl_f32(windowed.as_array(), n_pad)?;
    Ok(out.into_pyarray_bound(py))
}

// ── f64 ──────────────────────────────────────────────────────────────────────

fn tempogram_ac_impl_f64(
    windowed: ndarray::ArrayView2<'_, f64>,
    n_pad: usize,
) -> PyResult<Array2<f64>> {
    let (win_len, n_frames) = windowed.dim();

    if win_len == 0 || n_frames == 0 {
        return Ok(Array2::zeros((win_len, n_frames)));
    }
    if n_pad < win_len {
        return Err(PyValueError::new_err(
            "n_pad must be >= win_length",
        ));
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_forward(n_pad);
    let fft_inv: Arc<dyn rustfft::Fft<f64>> = planner.plan_fft_inverse(n_pad);

    let scratch_len = fft_fwd
        .get_inplace_scratch_len()
        .max(fft_inv.get_inplace_scratch_len())
        .max(1);

    let scale = 1.0f64 / n_pad as f64;

    let cols: Vec<Vec<f64>> = (0..n_frames)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![Complex::<f64>::new(0.0, 0.0); n_pad],
                    vec![Complex::<f64>::new(0.0, 0.0); scratch_len],
                )
            },
            |(buf, scratch), t| {
                for x in buf.iter_mut() {
                    *x = Complex::new(0.0, 0.0);
                }
                for i in 0..win_len {
                    buf[i] = Complex::new(windowed[[i, t]], 0.0);
                }
                fft_fwd.process_with_scratch(buf, scratch);
                for x in buf.iter_mut() {
                    let p = x.re * x.re + x.im * x.im;
                    *x = Complex::new(p, 0.0);
                }
                fft_inv.process_with_scratch(buf, scratch);
                buf[..win_len].iter().map(|x| x.re * scale).collect()
            },
        )
        .collect();

    let mut out = Array2::<f64>::zeros((win_len, n_frames));
    for (t, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            out[[i, t]] = v;
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn tempogram_ac_f64<'py>(
    py: Python<'py>,
    windowed: PyReadonlyArray2<'py, f64>,
    n_pad: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = tempogram_ac_impl_f64(windowed.as_array(), n_pad)?;
    Ok(out.into_pyarray_bound(py))
}

