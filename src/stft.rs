// Parallel STFT / power-spectrogram kernel for iron-librosa.
//
// Matches librosa.stft() + np.abs()**2 with parallel Rayon FFT across frames.
// Defaults: center=True (zero-pad n_fft//2 each side), Hann window, power=2.

use ndarray::parallel::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::sync::Arc;
use std::option::Option as StdOption;

use crate::backend::{requested_rust_device, resolved_rust_device, RustDevice};

fn fft_timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("IRON_LIBROSA_FFT_TIMING")
            .ok()
            .map(|s| {
                matches!(
                    s.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

/// Workload size threshold for GPU FFT dispatch in Auto mode.
/// When n_frames * n_fft * log2(n_fft) >= this value, GPU dispatch is used.
fn fft_gpu_work_threshold() -> usize {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(100_000_000)
    })
}

/// Optional minimum frame count for STFT GPU dispatch.
/// When set (>0), workloads with fewer frames run on CPU even on apple-gpu mode.
/// Phase 19 A/B testing showed min_frames=200 reduces regressions and improves score (0.887 vs 0.767).
/// Default changed to 200 in Phase 20 based on empirical optimization.
fn fft_gpu_min_frames() -> usize {
    static MIN_FRAMES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_FRAMES.get_or_init(|| {
        std::env::var("IRON_LIBROSA_FFT_GPU_MIN_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(200)  // Phase 20: Changed from 0 to 200 (Phase 19 A/B winner)
    })
}

// ── per-thread state ──────────────────────────────────────────────────────────
// Each Rayon worker thread owns its own FftPlanner (plan cache), FFT scratch
// buffer, and windowed-frame buffer.  This eliminates per-frame heap allocation
// which was the main overhead in the naive implementation.
// ── per-thread state: three separate RefCells so the borrow checker sees them
// as independent borrows (avoids E0499 from simultaneous disjoint-field use) ──

struct TlPlan {
    planner: FftPlanner<f32>,
    fft: Option<Arc<dyn rustfft::Fft<f32>>>,
    last_n_fft: usize,
}

struct TlPlanF64 {
    planner: FftPlanner<f64>,
    fft: Option<Arc<dyn rustfft::Fft<f64>>>,
    last_n_fft: usize,
}

thread_local! {
    static TL_PLAN: RefCell<TlPlan> = RefCell::new(TlPlan {
        planner: FftPlanner::new(),
        fft: None,
        last_n_fft: 0,
    });
    static TL_BUF:     RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    static TL_SCRATCH: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };

    static TL_PLAN_F64: RefCell<TlPlanF64> = RefCell::new(TlPlanF64 {
        planner: FftPlanner::new(),
        fft: None,
        last_n_fft: 0,
    });
    static TL_BUF_F64:     RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
    static TL_SCRATCH_F64: RefCell<Vec<Complex<f64>>> = const { RefCell::new(Vec::new()) };
}

// ── Hann window ───────────────────────────────────────────────────────────────
// Periodic Hann (fftbins=True), matching scipy.signal.get_window('hann', n, fftbins=True).
// w[k] = 0.5 * (1 - cos(2π·k / n))   for k in 0..n
fn hann_window(n: usize) -> Vec<f32> {
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

/// Compute power spectrogram (|STFT|^2) from raw f32 audio.
///
/// Matches `np.abs(librosa.stft(y, n_fft, hop_length, center=True))**2`.
/// Uses Rayon-parallel FFT with zero per-frame heap allocation.
///
/// Parameters:
/// - y: 1D float32 audio array
/// - n_fft: FFT window size
/// - hop_length: Frame hop length
/// - center: Whether to center-pad the signal
/// - window: Optional precomputed window array of length n_fft.
///   If None, uses Hann window. If provided, must be float32 of correct length.
///
/// Returns f32 array shape (n_fft//2+1, n_frames), C-contiguous.
#[pyfunction]
#[pyo3(signature = (y, n_fft = 2048, hop_length = 512, center = true, window = None))]
pub fn stft_power<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: StdOption<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let y_slice = y.as_slice()?;

    // center-pad with zeros (librosa default pad_mode='constant')
    let y_padded: Vec<f32> = if center {
        let pad = n_fft / 2;
        let mut v = Vec::with_capacity(y_slice.len() + 2 * pad);
        v.extend(std::iter::repeat(0.0f32).take(pad));
        v.extend_from_slice(y_slice);
        v.extend(std::iter::repeat(0.0f32).take(pad));
        v
    } else {
        y_slice.to_vec()
    };

    let n_samples = y_padded.len();
    if n_samples < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    let n_bins = n_fft / 2 + 1;

    // Use provided window or fall back to Hann window.
    let window: Arc<Vec<f32>> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(
                format!("Window length {} != n_fft {}", w_slice.len(), n_fft),
            ));
        }
        Arc::new(w_slice.to_vec())
    } else {
        Arc::new(hann_window(n_fft))
    };

    // Output (n_bins × n_frames) C-order; each Axis-1 column = one frame.
    let mut out = ndarray::Array2::<f32>::zeros((n_bins, n_frames));

    out.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame_idx, mut col)| {
            let start = frame_idx * hop_length;
            let win = &*window;

            // ── Step 1: get (or build) the cached FFT plan for this thread ──
            let fft: Arc<dyn rustfft::Fft<f32>> = TL_PLAN.with(|cell| {
                let mut p = cell.borrow_mut();
                if p.last_n_fft != n_fft || p.fft.is_none() {
                    p.fft = Some(p.planner.plan_fft_forward(n_fft));
                    p.last_n_fft = n_fft;
                }
                p.fft.clone().unwrap() // Arc clone — no borrow retained
            });

            let scratch_len = fft.get_inplace_scratch_len();

            // ── Step 2: borrow buf + scratch independently, fill, FFT, write ─
            TL_BUF.with(|bc| {
                let mut buf = bc.borrow_mut();
                if buf.len() != n_fft {
                    buf.resize(n_fft, Complex::default());
                }

                TL_SCRATCH.with(|sc| {
                    let mut scratch = sc.borrow_mut();
                    if scratch.len() < scratch_len {
                        scratch.resize(scratch_len, Complex::default());
                    }

                    // fill windowed frame (zero alloc — reuses buf)
                    for k in 0..n_fft {
                        buf[k] = Complex::new(y_padded[start + k] * win[k], 0.0);
                    }

                    // in-place forward FFT
                    fft.process_with_scratch(&mut buf, &mut scratch);
                });

                // write |z|^2 for positive frequencies
                for (bin, val) in col.iter_mut().enumerate() {
                    let z = buf[bin];
                    *val = z.re * z.re + z.im * z.im;
                }
            });
        });

    Ok(out.into_pyarray_bound(py))
}

/// Compute complex STFT from raw f32 audio.
///
/// Matches `librosa.stft(y, n_fft, hop_length, center=True)` with window.
/// Uses Rayon-parallel FFT with zero per-frame heap allocation.
///
/// Parameters:
/// - y: 1D float32 audio array
/// - n_fft: FFT window size
/// - hop_length: Frame hop length
/// - center: Whether to center-pad the signal
/// - window: Optional precomputed window array of length n_fft.
///   If None, uses Hann window. If provided, must be float32 of correct length.
///
/// Returns complex array shape (n_fft//2+1, n_frames), C-contiguous.
#[pyfunction]
#[pyo3(signature = (y, n_fft = 2048, hop_length = 512, center = true, window = None))]
pub fn stft_complex<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: StdOption<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray2<Complex<f32>>>> {
    let timing_enabled = fft_timing_enabled();
    let total_start = std::time::Instant::now();

    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let y_slice = y.as_slice()?;

    // center-pad with zeros (librosa default pad_mode='constant')
    let y_padded: Vec<f32> = if center {
        let pad = n_fft / 2;
        let mut v = Vec::with_capacity(y_slice.len() + 2 * pad);
        v.extend(std::iter::repeat(0.0f32).take(pad));
        v.extend_from_slice(y_slice);
        v.extend(std::iter::repeat(0.0f32).take(pad));
        v
    } else {
        y_slice.to_vec()
    };

    let n_samples = y_padded.len();
    if n_samples < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    let n_bins = n_fft / 2 + 1;

    // Use provided window or fall back to Hann window.
    let window: Arc<Vec<f32>> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(
                format!("Window length {} != n_fft {}", w_slice.len(), n_fft),
            ));
        }
        Arc::new(w_slice.to_vec())
    } else {
        Arc::new(hann_window(n_fft))
    };

    // ── GPU dispatch (AppleGpu) ───────────────────────────────────────────────
    // When apple-gpu device is requested, use Metal FFT (if available / enabled)
    // with an automatic CPU fallback via fft_forward_with_fallback.
    // Falls back to Rayon CPU path when Auto mode doesn't meet work threshold.
    // Phase 20: Added min_frames gate (default 200) to reduce small-batch dispatch overhead.
    let requested_device = requested_rust_device();
    let resolved_device = resolved_rust_device();
    let work_size = n_frames
        .saturating_mul(n_fft)
        .saturating_mul((n_fft as f32).log2() as usize);
    let use_gpu = match requested_device {
        RustDevice::AppleGpu => resolved_device == RustDevice::AppleGpu && n_frames >= fft_gpu_min_frames(),
        RustDevice::Auto => {
            resolved_device == RustDevice::AppleGpu && work_size >= fft_gpu_work_threshold() && n_frames >= fft_gpu_min_frames()
        }
        RustDevice::Cpu => false,
        // Phase 21 stub: CUDA GPU dispatch not yet implemented; route to CPU.
        RustDevice::CudaGpu => false,
    };

    if use_gpu {
        // Batched path: process all frames in one GPU dispatch to amortize launch/copy overhead.
        let gpu_start = std::time::Instant::now();
        let mut out = ndarray::Array2::<Complex<f32>>::zeros((n_bins, n_frames));
        let win = &*window;
        let mut batch_buf = vec![Complex::<f32>::default(); n_frames * n_fft];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let frame_off = frame_idx * n_fft;
            for k in 0..n_fft {
                batch_buf[frame_off + k] = Complex::new(y_padded[start + k] * win[k], 0.0);
            }
        }

        // Try batched Metal GPU; fall back to CPU rustfft on any error.
        let _ = crate::metal_fft::fft_forward_batched_chunked_with_fallback(
            &mut batch_buf,
            n_fft,
            n_frames,
        );

        for frame_idx in 0..n_frames {
            let frame_off = frame_idx * n_fft;
            for (bin, val) in out.index_axis_mut(ndarray::Axis(1), frame_idx).iter_mut().enumerate() {
                // Conjugate to match NumPy/scipy rfft sign convention.
                *val = batch_buf[frame_off + bin].conj();
            }
        }

        if timing_enabled {
            eprintln!(
                "[iron-librosa][stft_complex] mode=gpu requested={:?} resolved={:?} n_fft={} n_frames={} work={} total_ms={:.3} gpu_ms={:.3}",
                requested_device,
                resolved_device,
                n_fft,
                n_frames,
                work_size,
                total_start.elapsed().as_secs_f64() * 1000.0,
                gpu_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(out.into_pyarray_bound(py));
    }

    // ── CPU path (Rayon parallel) ────────────────────────────────────────────
    // Output (n_bins × n_frames) C-order; each column = one frame.
    let mut out = ndarray::Array2::<Complex<f32>>::zeros((n_bins, n_frames));

    out.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame_idx, mut col)| {
            let start = frame_idx * hop_length;
            let win = &*window;

            // ── Step 1: get (or build) the cached FFT plan for this thread ──
            let fft: Arc<dyn rustfft::Fft<f32>> = TL_PLAN.with(|cell| {
                let mut p = cell.borrow_mut();
                if p.last_n_fft != n_fft || p.fft.is_none() {
                    p.fft = Some(p.planner.plan_fft_forward(n_fft));
                    p.last_n_fft = n_fft;
                }
                p.fft.clone().unwrap() // Arc clone — no borrow retained
            });

            let scratch_len = fft.get_inplace_scratch_len();

            // ── Step 2: borrow buf + scratch independently, fill, FFT, write ─
            TL_BUF.with(|bc| {
                let mut buf = bc.borrow_mut();
                if buf.len() != n_fft {
                    buf.resize(n_fft, Complex::default());
                }

                TL_SCRATCH.with(|sc| {
                    let mut scratch = sc.borrow_mut();
                    if scratch.len() < scratch_len {
                        scratch.resize(scratch_len, Complex::default());
                    }

                    // fill windowed frame (zero alloc — reuses buf)
                    for k in 0..n_fft {
                        buf[k] = Complex::new(y_padded[start + k] * win[k], 0.0);
                    }

                    // in-place forward FFT
                    fft.process_with_scratch(&mut buf, &mut scratch);
                });

                // Match numpy/scipy forward-FFT sign convention.
                // rustfft's forward output is conjugated relative to numpy rfft.
                for (bin, val) in col.iter_mut().enumerate() {
                    *val = buf[bin].conj();
                }
            });
        });

    if timing_enabled {
        eprintln!(
            "[iron-librosa][stft_complex] mode=cpu requested={:?} resolved={:?} n_fft={} n_frames={} work={} total_ms={:.3}",
            requested_device,
            resolved_device,
            n_fft,
            n_frames,
            work_size,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Ok(out.into_pyarray_bound(py))
}

/// Compute power spectrogram (|STFT|^2) from raw f64 audio.
#[pyfunction]
#[pyo3(signature = (y, n_fft = 2048, hop_length = 512, center = true, window = None))]
pub fn stft_power_f64<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: StdOption<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let y_slice = y.as_slice()?;
    let y_padded: Vec<f64> = if center {
        let pad = n_fft / 2;
        let mut v = Vec::with_capacity(y_slice.len() + 2 * pad);
        v.extend(std::iter::repeat(0.0f64).take(pad));
        v.extend_from_slice(y_slice);
        v.extend(std::iter::repeat(0.0f64).take(pad));
        v
    } else {
        y_slice.to_vec()
    };

    let n_samples = y_padded.len();
    if n_samples < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    let n_bins = n_fft / 2 + 1;

    let window: Arc<Vec<f64>> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(
                format!("Window length {} != n_fft {}", w_slice.len(), n_fft),
            ));
        }
        Arc::new(w_slice.to_vec())
    } else {
        Arc::new(hann_window_f64(n_fft))
    };

    let mut out = ndarray::Array2::<f64>::zeros((n_bins, n_frames));

    out.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame_idx, mut col)| {
            let start = frame_idx * hop_length;
            let win = &*window;

            let fft: Arc<dyn rustfft::Fft<f64>> = TL_PLAN_F64.with(|cell| {
                let mut p = cell.borrow_mut();
                if p.last_n_fft != n_fft || p.fft.is_none() {
                    p.fft = Some(p.planner.plan_fft_forward(n_fft));
                    p.last_n_fft = n_fft;
                }
                p.fft.clone().unwrap()
            });

            let scratch_len = fft.get_inplace_scratch_len();

            TL_BUF_F64.with(|bc| {
                let mut buf = bc.borrow_mut();
                if buf.len() != n_fft {
                    buf.resize(n_fft, Complex::default());
                }

                TL_SCRATCH_F64.with(|sc| {
                    let mut scratch = sc.borrow_mut();
                    if scratch.len() < scratch_len {
                        scratch.resize(scratch_len, Complex::default());
                    }

                    for k in 0..n_fft {
                        buf[k] = Complex::new(y_padded[start + k] * win[k], 0.0);
                    }

                    fft.process_with_scratch(&mut buf, &mut scratch);
                });

                for (bin, val) in col.iter_mut().enumerate() {
                    let z = buf[bin];
                    *val = z.re * z.re + z.im * z.im;
                }
            });
        });

    Ok(out.into_pyarray_bound(py))
}

/// Compute complex STFT from raw f64 audio.
#[pyfunction]
#[pyo3(signature = (y, n_fft = 2048, hop_length = 512, center = true, window = None))]
pub fn stft_complex_f64<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: StdOption<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let y_slice = y.as_slice()?;
    let y_padded: Vec<f64> = if center {
        let pad = n_fft / 2;
        let mut v = Vec::with_capacity(y_slice.len() + 2 * pad);
        v.extend(std::iter::repeat(0.0f64).take(pad));
        v.extend_from_slice(y_slice);
        v.extend(std::iter::repeat(0.0f64).take(pad));
        v
    } else {
        y_slice.to_vec()
    };

    let n_samples = y_padded.len();
    if n_samples < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    let n_bins = n_fft / 2 + 1;

    let window: Arc<Vec<f64>> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(
                format!("Window length {} != n_fft {}", w_slice.len(), n_fft),
            ));
        }
        Arc::new(w_slice.to_vec())
    } else {
        Arc::new(hann_window_f64(n_fft))
    };

    let mut out = ndarray::Array2::<Complex<f64>>::zeros((n_bins, n_frames));

    out.axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(frame_idx, mut col)| {
            let start = frame_idx * hop_length;
            let win = &*window;

            let fft: Arc<dyn rustfft::Fft<f64>> = TL_PLAN_F64.with(|cell| {
                let mut p = cell.borrow_mut();
                if p.last_n_fft != n_fft || p.fft.is_none() {
                    p.fft = Some(p.planner.plan_fft_forward(n_fft));
                    p.last_n_fft = n_fft;
                }
                p.fft.clone().unwrap()
            });

            let scratch_len = fft.get_inplace_scratch_len();

            TL_BUF_F64.with(|bc| {
                let mut buf = bc.borrow_mut();
                if buf.len() != n_fft {
                    buf.resize(n_fft, Complex::default());
                }

                TL_SCRATCH_F64.with(|sc| {
                    let mut scratch = sc.borrow_mut();
                    if scratch.len() < scratch_len {
                        scratch.resize(scratch_len, Complex::default());
                    }

                    for k in 0..n_fft {
                        buf[k] = Complex::new(y_padded[start + k] * win[k], 0.0);
                    }

                    fft.process_with_scratch(&mut buf, &mut scratch);
                });

                for (bin, val) in col.iter_mut().enumerate() {
                    *val = buf[bin].conj();
                }
            });
        });

    Ok(out.into_pyarray_bound(py))
}

/// Compute complex STFT for batched f32 audio, shape=(channels, n_samples).
#[pyfunction]
#[pyo3(signature = (y, n_fft = 2048, hop_length = 512, center = true, window = None))]
pub fn stft_complex_batch<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: StdOption<PyReadonlyArray1<'py, f32>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f32>>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let y_arr = y.as_array();
    let n_channels = y_arr.shape()[0];
    let n_src_samples = y_arr.shape()[1];

    let y_padded: ndarray::Array2<f32> = if center {
        let pad = n_fft / 2;
        let mut out = ndarray::Array2::<f32>::zeros((n_channels, n_src_samples + 2 * pad));
        out.slice_mut(ndarray::s![.., pad..(pad + n_src_samples)])
            .assign(&y_arr);
        out
    } else {
        y_arr.to_owned()
    };

    let n_samples = y_padded.shape()[1];
    if n_samples < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    let n_bins = n_fft / 2 + 1;

    let window: Arc<Vec<f32>> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(
                format!("Window length {} != n_fft {}", w_slice.len(), n_fft),
            ));
        }
        Arc::new(w_slice.to_vec())
    } else {
        Arc::new(hann_window(n_fft))
    };

    let mut out = ndarray::Array3::<Complex<f32>>::zeros((n_channels, n_bins, n_frames));

    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(ch_idx, mut ch_out)| {
            let y_ch = y_padded.row(ch_idx);
            let win = &*window;

            let fft: Arc<dyn rustfft::Fft<f32>> = TL_PLAN.with(|cell| {
                let mut p = cell.borrow_mut();
                if p.last_n_fft != n_fft || p.fft.is_none() {
                    p.fft = Some(p.planner.plan_fft_forward(n_fft));
                    p.last_n_fft = n_fft;
                }
                p.fft.clone().unwrap()
            });

            let scratch_len = fft.get_inplace_scratch_len();

            for frame_idx in 0..n_frames {
                let start = frame_idx * hop_length;

                TL_BUF.with(|bc| {
                    let mut buf = bc.borrow_mut();
                    if buf.len() != n_fft {
                        buf.resize(n_fft, Complex::default());
                    }

                    TL_SCRATCH.with(|sc| {
                        let mut scratch = sc.borrow_mut();
                        if scratch.len() < scratch_len {
                            scratch.resize(scratch_len, Complex::default());
                        }

                        for k in 0..n_fft {
                            buf[k] = Complex::new(y_ch[start + k] * win[k], 0.0);
                        }

                        fft.process_with_scratch(&mut buf, &mut scratch);
                    });

                    for (bin, val) in ch_out.index_axis_mut(ndarray::Axis(1), frame_idx).iter_mut().enumerate() {
                        *val = buf[bin].conj();
                    }
                });
            }
        });

    Ok(out.into_pyarray_bound(py))
}

/// Compute complex STFT for batched f64 audio, shape=(channels, n_samples).
#[pyfunction]
#[pyo3(signature = (y, n_fft = 2048, hop_length = 512, center = true, window = None))]
pub fn stft_complex_f64_batch<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: StdOption<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f64>>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }

    let y_arr = y.as_array();
    let n_channels = y_arr.shape()[0];
    let n_src_samples = y_arr.shape()[1];

    let y_padded: ndarray::Array2<f64> = if center {
        let pad = n_fft / 2;
        let mut out = ndarray::Array2::<f64>::zeros((n_channels, n_src_samples + 2 * pad));
        out.slice_mut(ndarray::s![.., pad..(pad + n_src_samples)])
            .assign(&y_arr);
        out
    } else {
        y_arr.to_owned()
    };

    let n_samples = y_padded.shape()[1];
    if n_samples < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (n_samples - n_fft) / hop_length;
    let n_bins = n_fft / 2 + 1;

    let window: Arc<Vec<f64>> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != n_fft {
            return Err(PyValueError::new_err(
                format!("Window length {} != n_fft {}", w_slice.len(), n_fft),
            ));
        }
        Arc::new(w_slice.to_vec())
    } else {
        Arc::new(hann_window_f64(n_fft))
    };

    let mut out = ndarray::Array3::<Complex<f64>>::zeros((n_channels, n_bins, n_frames));

    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(ch_idx, mut ch_out)| {
            let y_ch = y_padded.row(ch_idx);
            let win = &*window;

            let fft: Arc<dyn rustfft::Fft<f64>> = TL_PLAN_F64.with(|cell| {
                let mut p = cell.borrow_mut();
                if p.last_n_fft != n_fft || p.fft.is_none() {
                    p.fft = Some(p.planner.plan_fft_forward(n_fft));
                    p.last_n_fft = n_fft;
                }
                p.fft.clone().unwrap()
            });

            let scratch_len = fft.get_inplace_scratch_len();

            for frame_idx in 0..n_frames {
                let start = frame_idx * hop_length;

                TL_BUF_F64.with(|bc| {
                    let mut buf = bc.borrow_mut();
                    if buf.len() != n_fft {
                        buf.resize(n_fft, Complex::default());
                    }

                    TL_SCRATCH_F64.with(|sc| {
                        let mut scratch = sc.borrow_mut();
                        if scratch.len() < scratch_len {
                            scratch.resize(scratch_len, Complex::default());
                        }

                        for k in 0..n_fft {
                            buf[k] = Complex::new(y_ch[start + k] * win[k], 0.0);
                        }

                        fft.process_with_scratch(&mut buf, &mut scratch);
                    });

                    for (bin, val) in ch_out.index_axis_mut(ndarray::Axis(1), frame_idx).iter_mut().enumerate() {
                        *val = buf[bin].conj();
                    }
                });
            }
        });

    Ok(out.into_pyarray_bound(py))
}

