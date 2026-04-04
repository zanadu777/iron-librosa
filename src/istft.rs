// Inverse Short-Time Fourier Transform (ISTFT) kernels for iron-librosa.
//
// Converts complex STFT matrix back to time-domain audio using inverse FFT
// with proper windowing and overlap-add reconstruction.
//
// Matches librosa.istft() behavior for float32 and float64 precision.

use ndarray::{s, Array1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::sync::Arc;

// ── per-thread state ──────────────────────────────────────────────────────────
// Thread-local FFT plan and scratch buffers for f32 and f64

struct IstftPlanF32 {
    planner: FftPlanner<f32>,
    fft: Option<Arc<dyn rustfft::Fft<f32>>>,
    last_n_fft: usize,
}

struct IstftPlanF64 {
    planner: FftPlanner<f64>,
    fft: Option<Arc<dyn rustfft::Fft<f64>>>,
    last_n_fft: usize,
}

thread_local! {
    static TL_ISTFT_PLAN_F32: RefCell<IstftPlanF32> = RefCell::new(IstftPlanF32 {
        planner: FftPlanner::new(),
        fft: None,
        last_n_fft: 0,
    });
    static TL_ISTFT_BUF_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    static TL_ISTFT_SCRATCH_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };

    static TL_ISTFT_PLAN_F64: RefCell<IstftPlanF64> = RefCell::new(IstftPlanF64 {
        planner: FftPlanner::new(),
        fft: None,
        last_n_fft: 0,
    });
    static TL_ISTFT_BUF_F64: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static TL_ISTFT_SCRATCH_F64: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Compute inverse STFT from complex spectrogram (f32 precision).
///
/// Converts complex STFT matrix back to time-domain audio using inverse FFT
/// with Hann window overlap-add reconstruction.
///
/// Parameters:
/// - stft_matrix: 2D complex64 array, shape (n_fft//2+1, n_frames)
/// - n_fft: FFT window size
/// - hop_length: Frame hop length
/// - win_length: Window length (if None, defaults to n_fft)
/// - window: Precomputed window array of length n_fft (f32)
///
/// Returns: f32 array shape (n_samples,), time-domain audio reconstructed from STFT.
#[pyfunction]
#[pyo3(signature = (stft_matrix, n_fft, hop_length, win_length = None, window = None))]
pub fn istft_f32<'py>(
    py: Python<'py>,
    stft_matrix: PyReadonlyArray2<'py, Complex<f32>>,
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    window: Option<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let stft_slice = stft_matrix.as_array();
    let n_bins = stft_slice.shape()[0];
    let n_frames = stft_slice.shape()[1];

    // Validate n_bins matches n_fft//2+1
    if n_bins != n_fft / 2 + 1 {
        return Err(PyValueError::new_err(format!(
            "STFT bins {} != n_fft//2+1 {}",
            n_bins,
            n_fft / 2 + 1
        )));
    }

    let win_length = win_length.unwrap_or(n_fft);
    if win_length > n_fft {
        return Err(PyValueError::new_err("win_length must be <= n_fft"));
    }

    // Get or build window
    let window_vec: Vec<f32> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(format!(
                "Window length {} != n_fft {}",
                w_slice.len(),
                n_fft
            )));
        }
        w_slice.to_vec()
    } else {
        hann_window_f32(n_fft)
    };

    // Compute expected output length (assuming center=False, no trimming)
    let expected_len = n_fft + hop_length * (n_frames.saturating_sub(1));

    // Allocate output and overlap buffer
    let mut y = Array1::<f32>::zeros(expected_len);
    let mut overlap_buf = Array1::<f32>::zeros(expected_len);

    // Process each frame via IFFT
    for frame_idx in 0..n_frames {
        // Extract this frame's complex spectrum
        let frame = stft_slice.slice(s![.., frame_idx]);

        // Perform inverse FFT (in-place in buffer)
        let ifft_frame = inverse_fft_f32(py, &frame, n_fft)?;

        // Window and overlap-add
        let start_sample = frame_idx * hop_length;
        for (i, sample) in ifft_frame.iter().enumerate() {
            if start_sample + i < expected_len {
                overlap_buf[start_sample + i] += sample * window_vec[i];
            }
        }
    }

    // Normalize by window sum-of-squares
    let window_sum_sq = compute_window_sumsquare_f32(&window_vec, n_frames, hop_length);
    for i in 0..expected_len {
        if window_sum_sq[i] > 1e-8 {
            y[i] = overlap_buf[i] / window_sum_sq[i];
        }
    }

    Ok(y.into_pyarray_bound(py).to_owned())
}

/// Compute inverse STFT from complex spectrogram (f64 precision).
#[pyfunction]
#[pyo3(signature = (stft_matrix, n_fft, hop_length, win_length = None, window = None))]
pub fn istft_f64<'py>(
    py: Python<'py>,
    stft_matrix: PyReadonlyArray2<'py, Complex<f64>>,
    n_fft: usize,
    hop_length: usize,
    win_length: Option<usize>,
    window: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let stft_slice = stft_matrix.as_array();
    let n_bins = stft_slice.shape()[0];
    let n_frames = stft_slice.shape()[1];

    if n_bins != n_fft / 2 + 1 {
        return Err(PyValueError::new_err(format!(
            "STFT bins {} != n_fft//2+1 {}",
            n_bins,
            n_fft / 2 + 1
        )));
    }

    let win_length = win_length.unwrap_or(n_fft);
    if win_length > n_fft {
        return Err(PyValueError::new_err("win_length must be <= n_fft"));
    }

    let window_vec: Vec<f64> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(format!(
                "Window length {} != n_fft {}",
                w_slice.len(),
                n_fft
            )));
        }
        w_slice.to_vec()
    } else {
        hann_window_f64(n_fft)
    };

    let expected_len = n_fft + hop_length * (n_frames.saturating_sub(1));

    let mut y = Array1::<f64>::zeros(expected_len);
    let mut overlap_buf = Array1::<f64>::zeros(expected_len);

    for frame_idx in 0..n_frames {
        let frame = stft_slice.slice(s![.., frame_idx]);
        let ifft_frame = inverse_fft_f64(py, &frame, n_fft)?;

        let start_sample = frame_idx * hop_length;
        for (i, sample) in ifft_frame.iter().enumerate() {
            if start_sample + i < expected_len {
                overlap_buf[start_sample + i] += sample * window_vec[i];
            }
        }
    }

    let window_sum_sq = compute_window_sumsquare_f64(&window_vec, n_frames, hop_length);
    for i in 0..expected_len {
        if window_sum_sq[i] > 1e-10 {
            y[i] = overlap_buf[i] / window_sum_sq[i];
        }
    }

    Ok(y.into_pyarray_bound(py).to_owned())
}

// ── Helper functions ──────────────────────────────────────────────────────────

/// Compute Hann window for ISTFT (periodic, matching scipy)
fn hann_window_f32(n: usize) -> Vec<f32> {
    use std::f64::consts::TAU;
    (0..n)
        .map(|k| (0.5 * (1.0 - (TAU * k as f64 / n as f64).cos())) as f32)
        .collect()
}

fn hann_window_f64(n: usize) -> Vec<f64> {
    use std::f64::consts::TAU;
    (0..n)
        .map(|k| 0.5 * (1.0 - (TAU * k as f64 / n as f64).cos()))
        .collect()
}

/// Inverse FFT for f32, using thread-local cached planner
fn inverse_fft_f32(
    _py: Python,
    stft_frame: &ndarray::ArrayView1<Complex<f32>>,
    n_fft: usize,
) -> PyResult<Vec<f32>> {
    let spectrum: Vec<Complex<f32>> = stft_frame.to_vec();

    // Mirror the negative frequencies for real-valued output
    // STFT only gives positive frequencies up to n_fft//2
    // We need to reconstruct full spectrum for inverse FFT
    let mut full_spectrum = vec![Complex::new(0.0, 0.0); n_fft];

    // Copy positive frequencies
    for i in 0..spectrum.len() {
        full_spectrum[i] = spectrum[i];
    }

    // Mirror negative frequencies (complex conjugate)
    if n_fft % 2 == 0 {
        // Even n_fft: Nyquist is at index n_fft//2 and is real
        for i in 1..n_fft / 2 {
            full_spectrum[n_fft - i] = full_spectrum[i].conj();
        }
    } else {
        // Odd n_fft
        for i in 1..(n_fft + 1) / 2 {
            full_spectrum[n_fft - i] = full_spectrum[i].conj();
        }
    }

    // Get or build FFT plan for this thread
    let fft = TL_ISTFT_PLAN_F32.with(|cell| {
        let mut p = cell.borrow_mut();
        if p.last_n_fft != n_fft || p.fft.is_none() {
            p.fft = Some(p.planner.plan_fft_inverse(n_fft));
            p.last_n_fft = n_fft;
        }
        p.fft.clone().unwrap()
    });

    // Prepare buffers
    let mut buf = TL_ISTFT_BUF_F32.with(|cell| {
        let mut b = cell.borrow_mut();
        b.clear();
        b.extend_from_slice(&full_spectrum);
        if b.len() < n_fft {
            b.resize(n_fft, Complex::new(0.0, 0.0));
        }
        b.to_vec()
    });

    let mut scratch = TL_ISTFT_SCRATCH_F32.with(|cell| {
        let mut s = cell.borrow_mut();
        s.resize(fft.get_inplace_scratch_len(), Complex::new(0.0, 0.0));
        s.clone()
    });

    // Execute inverse FFT
    fft.process_with_scratch(&mut buf, &mut scratch);

    // Convert complex to real by taking real part, scaled by 1/n_fft
    let result: Vec<f32> = buf.iter().map(|c| c.re / n_fft as f32).collect();

    Ok(result)
}

/// Inverse FFT for f64
fn inverse_fft_f64(
    _py: Python,
    stft_frame: &ndarray::ArrayView1<Complex<f64>>,
    n_fft: usize,
) -> PyResult<Vec<f64>> {
    let spectrum: Vec<Complex<f64>> = stft_frame.to_vec();

    let mut full_spectrum = vec![Complex::new(0.0, 0.0); n_fft];

    for i in 0..spectrum.len() {
        full_spectrum[i] = spectrum[i];
    }

    if n_fft % 2 == 0 {
        for i in 1..n_fft / 2 {
            full_spectrum[n_fft - i] = full_spectrum[i].conj();
        }
    } else {
        for i in 1..(n_fft + 1) / 2 {
            full_spectrum[n_fft - i] = full_spectrum[i].conj();
        }
    }

    let fft = TL_ISTFT_PLAN_F64.with(|cell| {
        let mut p = cell.borrow_mut();
        if p.last_n_fft != n_fft || p.fft.is_none() {
            p.fft = Some(p.planner.plan_fft_inverse(n_fft));
            p.last_n_fft = n_fft;
        }
        p.fft.clone().unwrap()
    });

    let mut buf = TL_ISTFT_BUF_F64.with(|cell| {
        let mut b = cell.borrow_mut();
        b.clear();
        b.extend_from_slice(&full_spectrum);
        if b.len() < n_fft {
            b.resize(n_fft, Complex::new(0.0, 0.0));
        }
        b.to_vec()
    });

    let mut scratch = TL_ISTFT_SCRATCH_F64.with(|cell| {
        let mut s = cell.borrow_mut();
        s.resize(fft.get_inplace_scratch_len(), Complex::new(0.0, 0.0));
        s.clone()
    });

    fft.process_with_scratch(&mut buf, &mut scratch);

    let result: Vec<f64> = buf.iter().map(|c| c.re / n_fft as f64).collect();

    Ok(result)
}

/// Compute window sum-of-squares for normalization in ISTFT
fn compute_window_sumsquare_f32(
    window: &[f32],
    n_frames: usize,
    hop_length: usize,
) -> Vec<f32> {
    let n_fft = window.len();
    let total_len = n_fft + hop_length * (n_frames.saturating_sub(1));

    let mut window_sq = vec![0.0f32; total_len];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        for (i, &w) in window.iter().enumerate() {
            if start + i < total_len {
                window_sq[start + i] += w * w;
            }
        }
    }

    window_sq
}

fn compute_window_sumsquare_f64(
    window: &[f64],
    n_frames: usize,
    hop_length: usize,
) -> Vec<f64> {
    let n_fft = window.len();
    let total_len = n_fft + hop_length * (n_frames.saturating_sub(1));

    let mut window_sq = vec![0.0f64; total_len];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        for (i, &w) in window.iter().enumerate() {
            if start + i < total_len {
                window_sq[start + i] += w * w;
            }
        }
    }

    window_sq
}







