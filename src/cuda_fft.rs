// Phase 21 — CUDA FFT wrapper stub for STFT/iSTFT.
//
// This module mirrors the public API contract of src/metal_fft.rs so that
// src/stft.rs and src/istft.rs can dispatch to it with identical call sites.
//
// CURRENT STATE: stub — all functions return Err() and fall back to CPU.
// The actual cuFFT FFI implementation will be filled in during Phase 21.
//
// Integration strategy: Option C (hybrid spike → abstraction)
//   1. Spike cuFFT FFI in fft_forward_batched_gpu / fft_inverse_batched_gpu
//   2. Validate parity + speedup against phase21_cuda_baseline benchmark
//   3. Extract abstraction once overhead profile is understood
//
// Environment:
//   IRON_LIBROSA_RUST_DEVICE=cuda-gpu  — enable CUDA dispatch (opt-in)
//
// Feature gate:
//   Cargo.toml: features = ["cuda-gpu"]
//   Build:      maturin develop --release --features cuda-gpu

use rustfft::num_complex::Complex;

/// Phase 21: CUDA FFT context — not yet implemented.
/// Will wrap cuFFT handle and device/stream state once cuFFT FFI is active.
#[cfg(feature = "cuda-gpu")]
pub mod cuda_fft_impl {
    use rustfft::num_complex::Complex;

    fn cuda_fft_enabled() -> bool {
        std::env::var("IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL")
            .ok()
            .map(|v| v.trim() == "force-on")
            .unwrap_or(false)
    }

    /// Forward batched FFT via CUDA — stub, always returns Err.
    /// Replace body with cuFFT FFI call once implementation is ready.
    pub fn fft_forward_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        if !cuda_fft_enabled() {
            return Err(
                "CUDA FFT disabled (set IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on)"
                    .to_string(),
            );
        }
        // TODO Phase 21: cuFFT FFI — fwd batched
        Err("CUDA FFT not yet implemented — Phase 21 stub".to_string())
    }

    /// Inverse batched FFT via CUDA — stub, always returns Err.
    /// Replace body with cuFFT FFI call once implementation is ready.
    pub fn fft_inverse_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        if !cuda_fft_enabled() {
            return Err(
                "CUDA FFT disabled (set IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on)"
                    .to_string(),
            );
        }
        // TODO Phase 21: cuFFT FFI — inv batched
        Err("CUDA FFT not yet implemented — Phase 21 stub".to_string())
    }
}

/// Non-CUDA builds: all functions return Err, never called in practice
/// because cuda_gpu_runtime_available() returns false.
#[cfg(not(feature = "cuda-gpu"))]
pub mod cuda_fft_impl {
    use rustfft::num_complex::Complex;

    pub fn fft_forward_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("CUDA FFT not available (build without cuda-gpu feature)".to_string())
    }

    pub fn fft_inverse_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("CUDA FFT not available (build without cuda-gpu feature)".to_string())
    }
}

// ── High-level fallback API ────────────────────────────────────────────────
// Mirrors the metal_fft.rs public API shape:
//   fft_forward_batched_with_fallback(buf, n, n_frames) -> Result<(), String>
//   fft_inverse_batched_with_fallback(buf, n, n_frames) -> Result<(), String>
//   fft_forward_batched_chunked_with_fallback(buf, n, n_frames) -> Result<(), String>
//   fft_inverse_batched_chunked_with_fallback(buf, n, n_frames) -> Result<(), String>
//
// All variants try CUDA first, then fall back to rustfft CPU path.

fn cuda_chunk_size(n_fft: usize, n_frames: usize) -> usize {
    // Reuse the same adaptive heuristic established in Phase 20 for Metal.
    // Will be tuned with empirical CUDA data in Phase 21 implementation phase.
    if let Ok(s) = std::env::var("IRON_LIBROSA_CUDA_FFT_BATCH_CHUNK_SIZE") {
        if let Ok(v) = s.trim().parse::<usize>() {
            if v > 0 {
                return v;
            }
        }
    }
    let total_work = n_fft.saturating_mul(n_frames);
    if total_work <= 65536 {
        return n_frames;
    }
    let chunk_divisor = std::cmp::max(4, n_fft / 256);
    let recommended = std::cmp::max(64, n_frames / chunk_divisor);
    std::cmp::min(recommended, 512)
}

/// Attempt batched forward FFT on CUDA; fall back to CPU on any error.
pub fn fft_forward_batched_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let total = n
        .checked_mul(n_frames)
        .ok_or_else(|| "batched FFT size overflow".to_string())?;
    if buffer.len() < total {
        return Err("batched FFT buffer is smaller than n * n_frames".to_string());
    }

    if cuda_fft_impl::fft_forward_batched_gpu(buffer, n, n_frames).is_ok() {
        return Ok(());
    }

    // CPU fallback via rustfft
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    for frame in 0..n_frames {
        let start = frame * n;
        fft.process(&mut buffer[start..start + n]);
    }
    Ok(())
}

/// Attempt batched inverse FFT on CUDA; fall back to CPU on any error.
pub fn fft_inverse_batched_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let total = n
        .checked_mul(n_frames)
        .ok_or_else(|| "batched FFT size overflow".to_string())?;
    if buffer.len() < total {
        return Err("batched FFT buffer is smaller than n * n_frames".to_string());
    }

    if cuda_fft_impl::fft_inverse_batched_gpu(buffer, n, n_frames).is_ok() {
        return Ok(());
    }

    // CPU fallback via rustfft
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    for frame in 0..n_frames {
        let start = frame * n;
        fft.process(&mut buffer[start..start + n]);
    }
    Ok(())
}

/// Chunked batched forward FFT on CUDA with adaptive chunk sizing.
/// Mirrors fft_forward_batched_chunked_with_fallback in metal_fft.rs.
pub fn fft_forward_batched_chunked_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);
    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let start = frame_start * n;
        let end = start + frames * n;
        fft_forward_batched_with_fallback(&mut buffer[start..end], n, frames)?;
        frame_start += frames;
    }
    Ok(())
}

/// Chunked batched inverse FFT on CUDA with adaptive chunk sizing.
/// Mirrors fft_inverse_batched_chunked_with_fallback in metal_fft.rs.
pub fn fft_inverse_batched_chunked_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);
    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let start = frame_start * n;
        let end = start + frames * n;
        fft_inverse_batched_with_fallback(&mut buffer[start..end], n, frames)?;
        frame_start += frames;
    }
    Ok(())
}

