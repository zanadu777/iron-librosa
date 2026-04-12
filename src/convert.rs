// Rust implementations of librosa.core.convert functions.
//
// Design contract:
//   - Each function accepts a 1-D f64 NumPy array (already created by
//     `np.asanyarray(x, dtype=float)` on the Python side).
//   - Scalars are handled entirely in Python; only ndim >= 1 inputs reach
//     these functions.
//   - Numerical results must be bit-for-bit identical (within float64
//     rounding) to the Python/NumPy equivalents.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::backend::{resolved_rust_device, RustDevice};

// ---------------------------------------------------------------------------
// hz_to_mel
// ---------------------------------------------------------------------------

/// Convert an array of Hz values to the Mel scale.
///
/// Matches ``librosa.hz_to_mel`` for array inputs.
/// Two scales are supported:
///   - Slaney (default, ``htk=False``): piecewise linear/log
///   - HTK (``htk=True``): pure log
///
/// Python signature (mirrored):
///     hz_to_mel(frequencies: np.ndarray, *, htk: bool = False) -> np.ndarray
#[pyfunction]
#[pyo3(signature = (frequencies, *, htk = false))]
pub fn hz_to_mel<'py>(
    py: Python<'py>,
    frequencies: PyReadonlyArray1<'py, f64>,
    htk: bool,
) -> Bound<'py, PyArray1<f64>> {
    match resolved_rust_device() {
        RustDevice::Cpu => hz_to_mel_cpu(py, frequencies, htk),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => hz_to_mel_cpu(py, frequencies, htk),
        RustDevice::Auto => hz_to_mel_cpu(py, frequencies, htk),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => hz_to_mel_cpu(py, frequencies, htk),
    }
}

fn hz_to_mel_cpu<'py>(
    py: Python<'py>,
    frequencies: PyReadonlyArray1<'py, f64>,
    htk: bool,
) -> Bound<'py, PyArray1<f64>> {
    let freqs = frequencies.as_array();

    let mels: Array1<f64> = if htk {
        // HTK formula: 2595 * log10(1 + f/700)
        freqs.mapv(|f| 2595.0 * (1.0_f64 + f / 700.0).log10())
    } else {
        // Slaney mel scale
        let f_min: f64 = 0.0;
        let f_sp: f64 = 200.0 / 3.0;
        let min_log_hz: f64 = 1000.0; // start of log region (Hz)
        let min_log_mel: f64 = (min_log_hz - f_min) / f_sp;
        let logstep: f64 = 6.4_f64.ln() / 27.0; // log step size

        freqs.mapv(|f| {
            if f >= min_log_hz {
                min_log_mel + (f / min_log_hz).ln() / logstep
            } else {
                (f - f_min) / f_sp
            }
        })
    };

    mels.into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// mel_to_hz
// ---------------------------------------------------------------------------

/// Convert an array of Mel values back to Hz.
///
/// Matches ``librosa.mel_to_hz`` for array inputs.
///
/// Python signature (mirrored):
///     mel_to_hz(mels: np.ndarray, *, htk: bool = False) -> np.ndarray
#[pyfunction]
#[pyo3(signature = (mels, *, htk = false))]
pub fn mel_to_hz<'py>(
    py: Python<'py>,
    mels: PyReadonlyArray1<'py, f64>,
    htk: bool,
) -> Bound<'py, PyArray1<f64>> {
    match resolved_rust_device() {
        RustDevice::Cpu => mel_to_hz_cpu(py, mels, htk),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => mel_to_hz_cpu(py, mels, htk),
        RustDevice::Auto => mel_to_hz_cpu(py, mels, htk),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => mel_to_hz_cpu(py, mels, htk),
    }
}

fn mel_to_hz_cpu<'py>(
    py: Python<'py>,
    mels: PyReadonlyArray1<'py, f64>,
    htk: bool,
) -> Bound<'py, PyArray1<f64>> {
    let mel_arr = mels.as_array();

    let freqs: Array1<f64> = if htk {
        // HTK inverse: 700 * (10^(m/2595) - 1)
        mel_arr.mapv(|m| 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0))
    } else {
        // Slaney inverse
        let f_min: f64 = 0.0;
        let f_sp: f64 = 200.0 / 3.0;
        let min_log_hz: f64 = 1000.0;
        let min_log_mel: f64 = (min_log_hz - f_min) / f_sp;
        let logstep: f64 = 6.4_f64.ln() / 27.0;

        mel_arr.mapv(|m| {
            if m >= min_log_mel {
                min_log_hz * (logstep * (m - min_log_mel)).exp()
            } else {
                f_min + f_sp * m
            }
        })
    };

    freqs.into_pyarray_bound(py)
}
