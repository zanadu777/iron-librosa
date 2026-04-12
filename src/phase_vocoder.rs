// Phase vocoder inner-loop kernel for iron-librosa.
//
// Replaces the Python `for t, idx in enumerate(step_int):` loop in
// `librosa.phase_vocoder`, which iterates over output time-frames and
// accumulates the per-bin phase state sequentially.
//
// The sequential frame dependency (phase_acc[t] depends on phase_acc[t-1])
// is preserved — no cross-frame parallelism is possible.  The speedup
// comes from eliminating Python interpreter overhead over the frame loop
// (typically hundreds to thousands of iterations) and from better SIMD
// code generation for the inner-loop arithmetic.
//
// Layout convention:
//   d_phase_t / d_mag_t are passed *transposed* from Python:
//     shape (n_padded_frames, n_bins), C-contiguous
//   This makes each frame's bin data a contiguous memory row, giving
//   cache-friendly sequential reads inside the inner bin loop.
//
// Output:
//   shape (n_bins, n_out_frames), C-contiguous — matches Python d_stretch.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex;

use crate::backend::{resolved_rust_device, RustDevice};

// ── f32 / complex64 path ─────────────────────────────────────────────────────

#[pyfunction]
pub fn phase_vocoder_f32<'py>(
    py: Python<'py>,
    d_phase_t: PyReadonlyArray2<'py, f32>,
    d_mag_t: PyReadonlyArray2<'py, f32>,
    phi_advance: PyReadonlyArray1<'py, f64>,
    step_int: PyReadonlyArray1<'py, i64>,
    step_alpha: PyReadonlyArray1<'py, f64>,
    phase_acc_init: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<Complex<f32>>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => phase_vocoder_f32_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => phase_vocoder_f32_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
        RustDevice::Auto => phase_vocoder_f32_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => phase_vocoder_f32_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
    }
}

fn phase_vocoder_f32_cpu<'py>(
    py: Python<'py>,
    d_phase_t: PyReadonlyArray2<'py, f32>,
    d_mag_t: PyReadonlyArray2<'py, f32>,
    phi_advance: PyReadonlyArray1<'py, f64>,
    step_int: PyReadonlyArray1<'py, i64>,
    step_alpha: PyReadonlyArray1<'py, f64>,
    phase_acc_init: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<Complex<f32>>>> {
    let phase_t = d_phase_t.as_array();
    let mag_t = d_mag_t.as_array();
    let phi = phi_advance.as_slice()?;
    let s_int = step_int.as_slice()?;
    let s_alpha = step_alpha.as_slice()?;
    let pa_init = phase_acc_init.as_slice()?;

    let n_padded = phase_t.shape()[0];
    let n_bins = phase_t.shape()[1];
    let n_out = s_int.len();

    // ── validation ────────────────────────────────────────────────────────────
    if mag_t.shape()[0] != n_padded || mag_t.shape()[1] != n_bins {
        return Err(PyValueError::new_err(format!(
            "d_mag_t shape {:?} must match d_phase_t shape {:?}",
            mag_t.shape(), phase_t.shape()
        )));
    }
    if phi.len() != n_bins {
        return Err(PyValueError::new_err(format!(
            "phi_advance length {} must equal n_bins {}",
            phi.len(), n_bins
        )));
    }
    if s_alpha.len() != n_out {
        return Err(PyValueError::new_err(
            "step_alpha length must equal step_int length",
        ));
    }
    if pa_init.len() != n_bins {
        return Err(PyValueError::new_err(format!(
            "phase_acc_init length {} must equal n_bins {}",
            pa_init.len(), n_bins
        )));
    }
    if let Some(&max_idx) = s_int.iter().max() {
        let max_i = max_idx as usize;
        if max_i + 1 >= n_padded {
            return Err(PyValueError::new_err(format!(
                "step_int max ({}) + 1 >= n_padded_frames ({}); ensure D is padded by ≥ 2",
                max_i, n_padded
            )));
        }
    }

    // ── inner loop ────────────────────────────────────────────────────────────
    let mut out = ndarray::Array2::<Complex<f32>>::zeros((n_bins, n_out));
    let mut phase_acc = pa_init.to_vec();
    let two_pi = 2.0_f64 * std::f64::consts::PI;

    for t in 0..n_out {
        let idx = s_int[t] as usize;
        let alpha = s_alpha[t];
        let oma = 1.0_f64 - alpha;

        for b in 0..n_bins {
            // Match Python dtype promotion: alpha is float64, so interpolation happens in f64.
            let mg = oma * (mag_t[[idx, b]] as f64) + alpha * (mag_t[[idx + 1, b]] as f64);

            // Keep phase math in f64 (Python parity), cast only at store.
            let pa = phase_acc[b];
            out[[b, t]] = Complex::new((mg * pa.cos()) as f32, (mg * pa.sin()) as f32);

            // Phase advance: dphase = Δangle - phi_advance, wrapped to [-π, π]
            // Use ties-to-even rounding (NumPy semantics) instead of Rust default.
            let mut dp = (phase_t[[idx + 1, b]] as f64) - (phase_t[[idx, b]] as f64) - phi[b];
            dp -= two_pi * (dp / two_pi).round_ties_even();
            phase_acc[b] = pa + phi[b] + dp;
        }
    }

    Ok(out.into_pyarray_bound(py))
}

// ── f64 / complex128 path ────────────────────────────────────────────────────

#[pyfunction]
pub fn phase_vocoder_f64<'py>(
    py: Python<'py>,
    d_phase_t: PyReadonlyArray2<'py, f64>,
    d_mag_t: PyReadonlyArray2<'py, f64>,
    phi_advance: PyReadonlyArray1<'py, f64>,
    step_int: PyReadonlyArray1<'py, i64>,
    step_alpha: PyReadonlyArray1<'py, f64>,
    phase_acc_init: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => phase_vocoder_f64_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => phase_vocoder_f64_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
        RustDevice::Auto => phase_vocoder_f64_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => phase_vocoder_f64_cpu(py, d_phase_t, d_mag_t, phi_advance, step_int, step_alpha, phase_acc_init),
    }
}

fn phase_vocoder_f64_cpu<'py>(
    py: Python<'py>,
    d_phase_t: PyReadonlyArray2<'py, f64>,
    d_mag_t: PyReadonlyArray2<'py, f64>,
    phi_advance: PyReadonlyArray1<'py, f64>,
    step_int: PyReadonlyArray1<'py, i64>,
    step_alpha: PyReadonlyArray1<'py, f64>,
    phase_acc_init: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
    let phase_t = d_phase_t.as_array();
    let mag_t = d_mag_t.as_array();
    let phi = phi_advance.as_slice()?;
    let s_int = step_int.as_slice()?;
    let s_alpha = step_alpha.as_slice()?;
    let pa_init = phase_acc_init.as_slice()?;

    let n_padded = phase_t.shape()[0];
    let n_bins = phase_t.shape()[1];
    let n_out = s_int.len();

    if mag_t.shape()[0] != n_padded || mag_t.shape()[1] != n_bins {
        return Err(PyValueError::new_err(format!(
            "d_mag_t shape {:?} must match d_phase_t shape {:?}",
            mag_t.shape(), phase_t.shape()
        )));
    }
    if phi.len() != n_bins {
        return Err(PyValueError::new_err(format!(
            "phi_advance length {} must equal n_bins {}",
            phi.len(), n_bins
        )));
    }
    if s_alpha.len() != n_out {
        return Err(PyValueError::new_err(
            "step_alpha length must equal step_int length",
        ));
    }
    if pa_init.len() != n_bins {
        return Err(PyValueError::new_err(format!(
            "phase_acc_init length {} must equal n_bins {}",
            pa_init.len(), n_bins
        )));
    }
    if let Some(&max_idx) = s_int.iter().max() {
        let max_i = max_idx as usize;
        if max_i + 1 >= n_padded {
            return Err(PyValueError::new_err(format!(
                "step_int max ({}) + 1 >= n_padded_frames ({}); ensure D is padded by ≥ 2",
                max_i, n_padded
            )));
        }
    }

    let mut out = ndarray::Array2::<Complex<f64>>::zeros((n_bins, n_out));
    let mut phase_acc = pa_init.to_vec();
    let two_pi = 2.0_f64 * std::f64::consts::PI;

    for t in 0..n_out {
        let idx = s_int[t] as usize;
        let alpha = s_alpha[t];
        let oma = 1.0_f64 - alpha;

        for b in 0..n_bins {
            let mg = oma * mag_t[[idx, b]] + alpha * mag_t[[idx + 1, b]];

            let pa = phase_acc[b];
            out[[b, t]] = Complex::new(mg * pa.cos(), mg * pa.sin());

            let mut dp = phase_t[[idx + 1, b]] - phase_t[[idx, b]] - phi[b];
            // Use ties-to-even rounding (NumPy semantics) for phase wrapping.
            dp -= two_pi * (dp / two_pi).round_ties_even();
            phase_acc[b] = pa + phi[b] + dp;
        }
    }

    Ok(out.into_pyarray_bound(py))
}

