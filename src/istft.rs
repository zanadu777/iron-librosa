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

use crate::backend::{requested_rust_device, resolved_rust_device, RustDevice};

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

/// Optional minimum frame count for iSTFT GPU dispatch.
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

fn fft_cuda_min_work_threshold() -> usize {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("IRON_LIBROSA_CUDA_FFT_MIN_WORK_THRESHOLD")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            // Phase 21 CUD-006 safety retune:
            // Keep auto mode conservative until CUDA path is consistently faster.
            // Phase 21 CUD-007 safety retune to keep Auto conservative until
            // CUDA iSTFT shows stable wins across medium/long workloads.
            .unwrap_or(30_000_000)
    })
}

fn fft_cuda_min_frames() -> usize {
    static MIN_FRAMES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_FRAMES.get_or_init(|| {
        std::env::var("IRON_LIBROSA_CUDA_FFT_MIN_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            // Safety gate for auto mode.
            .unwrap_or(1024)
    })
}

/// Debug-only override for local benchmarking.
/// Normal drop-in dispatch must not require this to use CUDA.
fn cuda_debug_force_on() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| {
        matches!(
            std::env::var("IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "force-on" | "1" | "true" | "yes" | "on"
        )
    })
}

fn cuda_c2r_istft_experimental() -> bool {
    static FORCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCE.get_or_init(|| {
        matches!(
            std::env::var("IRON_LIBROSA_CUDA_C2R_ISTFT_EXPERIMENTAL")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn fft_cuda_max_work() -> Option<usize> {
    static MAX_WORK: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *MAX_WORK.get_or_init(|| {
        std::env::var("IRON_LIBROSA_CUDA_MAX_WORK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
    })
}

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
    static TL_ISTFT_GPU_FULL_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    static TL_ISTFT_GPU_HALF_F32: RefCell<Vec<Complex<f32>>> = const { RefCell::new(Vec::new()) };
    static TL_ISTFT_GPU_REAL_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };

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
    let timing_enabled = fft_timing_enabled();
    let total_start = std::time::Instant::now();

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
    if win_length == 0 {
        return Err(PyValueError::new_err("win_length must be >= 1"));
    }
    if win_length > n_fft {
        return Err(PyValueError::new_err("win_length must be <= n_fft"));
    }

    // Build librosa-compatible synthesis window: length win_length, centered/padded to n_fft.
    let window_vec = build_istft_window_f32(n_fft, win_length, window)?;

    // Compute expected output length (assuming center=False, no trimming)
    let expected_len = n_fft + hop_length * (n_frames.saturating_sub(1));

    // Allocate overlap-add numerator and normalization denominator.
    // These are later fused into the final output with the same thresholding as before.
    let mut overlap_buf = vec![0.0f32; expected_len];
    let mut norm_buf = vec![0.0f32; expected_len];
    let window_sq: Vec<f32> = window_vec.iter().map(|w| w * w).collect();

    // Dispatch to GPU (if requested) or CPU per-frame IFFT.
    let requested_device = requested_rust_device();
    let resolved_device = resolved_rust_device();
    let work_size = n_frames
        .saturating_mul(n_fft)
        .saturating_mul((n_fft as f32).log2() as usize);
    let cuda_allowed = fft_cuda_max_work().map(|mx| work_size <= mx).unwrap_or(true);
    let cuda_force = cuda_debug_force_on();
    let use_gpu = match requested_device {
        RustDevice::AppleGpu => {
            resolved_device == RustDevice::AppleGpu
        }
        RustDevice::Auto => match resolved_device {
            RustDevice::AppleGpu => {
                work_size >= fft_gpu_work_threshold() && n_frames >= fft_gpu_min_frames()
            }
            RustDevice::CudaGpu => {
                (cuda_force || (work_size >= fft_cuda_min_work_threshold()
                    && n_frames >= fft_cuda_min_frames()))
                    && cuda_allowed
            }
            _ => false,
        },
        RustDevice::Cpu => false,
        RustDevice::CudaGpu => {
            resolved_device == RustDevice::CudaGpu
                && cuda_allowed
        }
    };

    let mut ifft_ms = 0.0f64;
    let mut overlap_add_ms = 0.0f64;
    let mut ifft_frame_cpu = vec![0.0f32; n_fft];

    let mut ifft_gpu_batch: Option<Vec<Complex<f32>>> = None;
    let mut ifft_gpu_real_batch: Option<Vec<f32>> = None;
    let mut gpu_half_spectrum: Option<Vec<Complex<f32>>> = None;
    if use_gpu {
        let ifft_start = std::time::Instant::now();
        if resolved_device == RustDevice::CudaGpu && cuda_c2r_istft_experimental() {
            let half_needed = n_frames * n_bins;
            let mut half_spectrum = TL_ISTFT_GPU_HALF_F32.with(|cell| {
                let mut cached = cell.borrow_mut();
                let mut buf = std::mem::take(&mut *cached);
                if buf.len() != half_needed {
                    buf.resize(half_needed, Complex::new(0.0f32, 0.0f32));
                }
                buf
            });
            for frame_idx in 0..n_frames {
                let off = frame_idx * n_bins;
                for i in 0..n_bins {
                    half_spectrum[off + i] = stft_slice[(i, frame_idx)];
                }
            }

            let real_needed = n_frames * n_fft;
            let mut real_batch = TL_ISTFT_GPU_REAL_F32.with(|cell| {
                let mut cached = cell.borrow_mut();
                let mut buf = std::mem::take(&mut *cached);
                if buf.len() != real_needed {
                    buf.resize(real_needed, 0.0f32);
                }
                buf
            });

            let _ = crate::cuda_fft::fft_inverse_real_batched_chunked_with_fallback(
                &half_spectrum,
                &mut real_batch,
                n_fft,
                n_frames,
            );
            gpu_half_spectrum = Some(half_spectrum);
            ifft_gpu_real_batch = Some(real_batch);
        } else {
            let needed = n_frames * n_fft;
            let mut full_spectrum = TL_ISTFT_GPU_FULL_F32.with(|cell| {
                let mut cached = cell.borrow_mut();
                let mut buf = std::mem::take(&mut *cached);
                if buf.len() != needed {
                    buf.resize(needed, Complex::new(0.0f32, 0.0f32));
                }
                buf
            });
            inverse_fft_f32_gpu_batch_complex_into(
                &stft_slice,
                n_fft,
                resolved_device,
                &mut full_spectrum,
            );
            ifft_gpu_batch = Some(full_spectrum);
        }
        ifft_ms += ifft_start.elapsed().as_secs_f64() * 1000.0;
    }

    // Process each frame via IFFT
    for frame_idx in 0..n_frames {
        if !use_gpu {
            let frame = stft_slice.slice(s![.., frame_idx]);
            let ifft_start = std::time::Instant::now();
            inverse_fft_f32_into(py, &frame, n_fft, &mut ifft_frame_cpu)?;
            ifft_ms += ifft_start.elapsed().as_secs_f64() * 1000.0;
        }

        // Window and overlap-add
        let overlap_start = std::time::Instant::now();
        let start_sample = frame_idx * hop_length;
        if let Some(gpu_real) = ifft_gpu_real_batch.as_ref() {
            let frame_off = frame_idx * n_fft;
            let scale = 1.0f32 / n_fft as f32;
            for i in 0..n_fft {
                let idx = start_sample + i;
                overlap_buf[idx] += gpu_real[frame_off + i] * scale * window_vec[i];
                norm_buf[idx] += window_sq[i];
            }
        } else if let Some(gpu_batch) = ifft_gpu_batch.as_ref() {
            let frame_off = frame_idx * n_fft;
            let scale = 1.0f32 / n_fft as f32;
            for i in 0..n_fft {
                let idx = start_sample + i;
                overlap_buf[idx] += gpu_batch[frame_off + i].re * scale * window_vec[i];
                norm_buf[idx] += window_sq[i];
            }
        } else {
            for (i, sample) in ifft_frame_cpu.iter().enumerate() {
                let idx = start_sample + i;
                overlap_buf[idx] += sample * window_vec[i];
                norm_buf[idx] += window_sq[i];
            }
        }
        overlap_add_ms += overlap_start.elapsed().as_secs_f64() * 1000.0;
    }

    // Normalize by window sum-of-squares
    let normalize_start = std::time::Instant::now();
    for i in 0..expected_len {
        if norm_buf[i] > f32::MIN_POSITIVE {
            overlap_buf[i] /= norm_buf[i];
        }
    }
    let normalize_ms = normalize_start.elapsed().as_secs_f64() * 1000.0;

    if timing_enabled {
        eprintln!(
            "[iron-librosa][istft_f32] mode={} requested={:?} resolved={:?} n_fft={} n_frames={} work={} total_ms={:.3} ifft_ms={:.3} overlap_add_ms={:.3} normalize_ms={:.3}",
            if use_gpu { "gpu" } else { "cpu" },
            requested_device,
            resolved_device,
            n_fft,
            n_frames,
            work_size,
            total_start.elapsed().as_secs_f64() * 1000.0,
            ifft_ms,
            overlap_add_ms,
            normalize_ms,
        );
    }

    if use_gpu {
        if let Some(half) = gpu_half_spectrum.take() {
            TL_ISTFT_GPU_HALF_F32.with(|cell| {
                *cell.borrow_mut() = half;
            });
        }
        if let Some(real) = ifft_gpu_real_batch.take() {
            TL_ISTFT_GPU_REAL_F32.with(|cell| {
                *cell.borrow_mut() = real;
            });
        }
        if let Some(full_spectrum) = ifft_gpu_batch.take() {
            TL_ISTFT_GPU_FULL_F32.with(|cell| {
                *cell.borrow_mut() = full_spectrum;
            });
        }
    }

    Ok(Array1::from_vec(overlap_buf).into_pyarray_bound(py).to_owned())
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
    if win_length == 0 {
        return Err(PyValueError::new_err("win_length must be >= 1"));
    }
    if win_length > n_fft {
        return Err(PyValueError::new_err("win_length must be <= n_fft"));
    }

    // Build librosa-compatible synthesis window: length win_length, centered/padded to n_fft.
    let window_vec = build_istft_window_f64(n_fft, win_length, window)?;

    let expected_len = n_fft + hop_length * (n_frames.saturating_sub(1));

    let mut overlap_buf = vec![0.0f64; expected_len];
    let mut norm_buf = vec![0.0f64; expected_len];
    let window_sq: Vec<f64> = window_vec.iter().map(|w| w * w).collect();

    // Reuse one frame buffer to avoid per-frame time-domain allocations.
    let mut ifft_frame = vec![0.0f64; n_fft];
    for frame_idx in 0..n_frames {
        let frame = stft_slice.slice(s![.., frame_idx]);
        inverse_fft_f64_into(py, &frame, n_fft, &mut ifft_frame)?;

        let start_sample = frame_idx * hop_length;
        for (i, sample) in ifft_frame.iter().enumerate() {
            let idx = start_sample + i;
            overlap_buf[idx] += sample * window_vec[i];
            norm_buf[idx] += window_sq[i];
        }
    }

    for i in 0..expected_len {
        if norm_buf[i] > f64::MIN_POSITIVE {
            overlap_buf[i] /= norm_buf[i];
        }
    }

    Ok(Array1::from_vec(overlap_buf).into_pyarray_bound(py).to_owned())
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

fn build_istft_window_f32(
    n_fft: usize,
    win_length: usize,
    window: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<Vec<f32>> {
    let base_window: Vec<f32> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != win_length {
            return Err(PyValueError::new_err(format!(
                "Window length {} != win_length {}",
                w_slice.len(),
                win_length
            )));
        }
        w_slice.to_vec()
    } else {
        hann_window_f32(win_length)
    };

    Ok(pad_center_f32(&base_window, n_fft))
}

fn build_istft_window_f64(
    n_fft: usize,
    win_length: usize,
    window: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Vec<f64>> {
    let base_window: Vec<f64> = if let Some(w) = window {
        let w_slice = w.as_slice()?;
        if w_slice.len() != win_length {
            return Err(PyValueError::new_err(format!(
                "Window length {} != win_length {}",
                w_slice.len(),
                win_length
            )));
        }
        w_slice.to_vec()
    } else {
        hann_window_f64(win_length)
    };

    Ok(pad_center_f64(&base_window, n_fft))
}

fn pad_center_f32(window: &[f32], size: usize) -> Vec<f32> {
    if window.len() == size {
        return window.to_vec();
    }
    let mut out = vec![0.0f32; size];
    let start = (size - window.len()) / 2;
    out[start..start + window.len()].copy_from_slice(window);
    out
}

fn pad_center_f64(window: &[f64], size: usize) -> Vec<f64> {
    if window.len() == size {
        return window.to_vec();
    }
    let mut out = vec![0.0f64; size];
    let start = (size - window.len()) / 2;
    out[start..start + window.len()].copy_from_slice(window);
    out
}

/// Inverse FFT for f32 using Metal GPU with CPU fallback.
/// Mirrors negative frequencies for all frames and runs batched inverse FFT.
/// Returns complex time-domain frames (unscaled, rustfft inverse semantics).
fn inverse_fft_f32_gpu_batch_complex_into(
    stft_matrix: &ndarray::ArrayView2<Complex<f32>>,
    n_fft: usize,
    resolved_device: RustDevice,
    full_spectrum: &mut Vec<Complex<f32>>,
) {
    let n_bins = stft_matrix.shape()[0];
    let n_frames = stft_matrix.shape()[1];
    let needed = n_frames * n_fft;
    if full_spectrum.len() != needed {
        full_spectrum.resize(needed, Complex::new(0.0f32, 0.0f32));
    }

    for frame_idx in 0..n_frames {
        let frame_off = frame_idx * n_fft;
        for i in 0..n_bins {
            full_spectrum[frame_off + i] = stft_matrix[(i, frame_idx)];
        }

        if n_fft % 2 == 0 {
            for i in 1..n_fft / 2 {
                full_spectrum[frame_off + (n_fft - i)] = full_spectrum[frame_off + i].conj();
            }
        } else {
            for i in 1..(n_fft + 1) / 2 {
                full_spectrum[frame_off + (n_fft - i)] = full_spectrum[frame_off + i].conj();
            }
        }
    }

    // Try backend GPU inverse FFT; each backend wrapper falls back to CPU.
    match resolved_device {
        RustDevice::AppleGpu => {
            let _ = crate::metal_fft::fft_inverse_batched_chunked_with_fallback(
                full_spectrum,
                n_fft,
                n_frames,
            );
        }
        RustDevice::CudaGpu => {
            let _ = crate::cuda_fft::fft_inverse_batched_chunked_with_fallback(
                full_spectrum,
                n_fft,
                n_frames,
            );
        }
        _ => {}
    }
}

fn inverse_fft_f32_into(
    _py: Python,
    stft_frame: &ndarray::ArrayView1<Complex<f32>>,
    n_fft: usize,
    out: &mut [f32],
) -> PyResult<()> {
    if out.len() < n_fft {
        return Err(PyValueError::new_err("output buffer too small for inverse_fft_f32_into"));
    }

    let fft = TL_ISTFT_PLAN_F32.with(|cell| {
        let mut p = cell.borrow_mut();
        if p.last_n_fft != n_fft || p.fft.is_none() {
            p.fft = Some(p.planner.plan_fft_inverse(n_fft));
            p.last_n_fft = n_fft;
        }
        p.fft.clone().unwrap()
    });

    let scratch_len = fft.get_inplace_scratch_len();
    TL_ISTFT_BUF_F32.with(|bc| {
        let mut buf = bc.borrow_mut();
        if buf.len() != n_fft {
            buf.resize(n_fft, Complex::new(0.0, 0.0));
        }
        buf.fill(Complex::new(0.0, 0.0));

        for i in 0..stft_frame.len() {
            buf[i] = stft_frame[i];
        }
        if n_fft % 2 == 0 {
            for i in 1..n_fft / 2 {
                buf[n_fft - i] = buf[i].conj();
            }
        } else {
            for i in 1..(n_fft + 1) / 2 {
                buf[n_fft - i] = buf[i].conj();
            }
        }

        TL_ISTFT_SCRATCH_F32.with(|sc| {
            let mut scratch = sc.borrow_mut();
            if scratch.len() < scratch_len {
                scratch.resize(scratch_len, Complex::new(0.0, 0.0));
            }
            fft.process_with_scratch(&mut buf[..], &mut scratch[..scratch_len]);
        });

        let scale = 1.0f32 / n_fft as f32;
        for i in 0..n_fft {
            out[i] = buf[i].re * scale;
        }
    });

    Ok(())
}

fn inverse_fft_f64_into(
    _py: Python,
    stft_frame: &ndarray::ArrayView1<Complex<f64>>,
    n_fft: usize,
    out: &mut [f64],
) -> PyResult<()> {
    if out.len() < n_fft {
        return Err(PyValueError::new_err("output buffer too small for inverse_fft_f64_into"));
    }

    let fft = TL_ISTFT_PLAN_F64.with(|cell| {
        let mut p = cell.borrow_mut();
        if p.last_n_fft != n_fft || p.fft.is_none() {
            p.fft = Some(p.planner.plan_fft_inverse(n_fft));
            p.last_n_fft = n_fft;
        }
        p.fft.clone().unwrap()
    });

    let scratch_len = fft.get_inplace_scratch_len();
    TL_ISTFT_BUF_F64.with(|bc| {
        let mut buf = bc.borrow_mut();
        if buf.len() != n_fft {
            buf.resize(n_fft, Complex::new(0.0, 0.0));
        }
        buf.fill(Complex::new(0.0, 0.0));

        for i in 0..stft_frame.len() {
            buf[i] = stft_frame[i];
        }
        if n_fft % 2 == 0 {
            for i in 1..n_fft / 2 {
                buf[n_fft - i] = buf[i].conj();
            }
        } else {
            for i in 1..(n_fft + 1) / 2 {
                buf[n_fft - i] = buf[i].conj();
            }
        }

        TL_ISTFT_SCRATCH_F64.with(|sc| {
            let mut scratch = sc.borrow_mut();
            if scratch.len() < scratch_len {
                scratch.resize(scratch_len, Complex::new(0.0, 0.0));
            }
            fft.process_with_scratch(&mut buf[..], &mut scratch[..scratch_len]);
        });

        let scale = 1.0f64 / n_fft as f64;
        for i in 0..n_fft {
            out[i] = buf[i].re * scale;
        }
    });

    Ok(())
}








