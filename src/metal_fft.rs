// Metal FFT wrapper for STFT/iSTFT.
//
// Uses a real Metal compute path when apple-gpu feature is enabled on macOS.
// On all other configurations, this module cleanly falls back to rustfft.

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
pub mod metal_fft_impl {
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
        MTLResourceOptions, MTLSize,
    };
    use rustfft::num_complex::Complex;
    use std::cell::RefCell;

    const MAX_GPU_N: usize = 1024;

    fn metal_fft_enabled() -> bool {
        std::env::var("IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL")
            .ok()
            .map(|v| v.trim() == "force-on")
            .unwrap_or(false)
    }

    thread_local! {
        static METAL_FFT_CONTEXT: RefCell<Option<MetalFftContext>> = RefCell::new(build_context());
    }

    struct MetalFftContext {
        device: Device,
        queue: CommandQueue,
        fwd_pipeline: ComputePipelineState,
        inv_pipeline: ComputePipelineState,
        data_buf: Buffer,
        n_buf: Buffer,
        data_cap_bytes: usize,
    }

    fn build_context() -> Option<MetalFftContext> {
        let device = Device::system_default()?;
        let options = CompileOptions::new();
        let lib = device.new_library_with_source(include_str!("metal_fft.metal"), &options).ok()?;
        let fwd_fn = lib.get_function("fft_forward_frame", None).ok()?;
        let inv_fn = lib.get_function("fft_inverse_frame", None).ok()?;
        let fwd_pipeline = device.new_compute_pipeline_state_with_function(&fwd_fn).ok()?;
        let inv_pipeline = device.new_compute_pipeline_state_with_function(&inv_fn).ok()?;
        let queue = device.new_command_queue();

        // Seed tiny shared buffers, grown on demand.
        let data_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
        let n_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        Some(MetalFftContext {
            device,
            queue,
            fwd_pipeline,
            inv_pipeline,
            data_buf,
            n_buf,
            data_cap_bytes: 8,
        })
    }

    fn ensure_data_capacity(ctx: &mut MetalFftContext, n_complex: usize) {
        let need = n_complex.saturating_mul(std::mem::size_of::<Complex<f32>>());
        if need > ctx.data_cap_bytes {
            let grow = (need as f32 * 1.5) as usize;
            ctx.data_buf = ctx
                .device
                .new_buffer(grow as u64, MTLResourceOptions::StorageModeShared);
            ctx.data_cap_bytes = grow;
        }
    }

    fn dispatch_fft_batched(
        ctx: &mut MetalFftContext,
        buffer: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
        inverse: bool,
    ) -> Result<(), String> {
        if n_frames == 0 {
            return Ok(());
        }
        let total = n
            .checked_mul(n_frames)
            .ok_or_else(|| "batched FFT size overflow".to_string())?;
        if buffer.len() < total {
            return Err("batched FFT buffer is smaller than n * n_frames".to_string());
        }

        ensure_data_capacity(ctx, total);

        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.as_ptr() as *const u8,
                ctx.data_buf.contents() as *mut u8,
                total * std::mem::size_of::<Complex<f32>>(),
            );
            *(ctx.n_buf.contents() as *mut u32) = n as u32;
        }

        let cmd = ctx.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let pipeline = if inverse {
            &ctx.inv_pipeline
        } else {
            &ctx.fwd_pipeline
        };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(&ctx.data_buf), 0);
        enc.set_buffer(1, Some(&ctx.n_buf), 0);

        // One threadgroup processes one frame. Dispatch one group per frame.
        enc.dispatch_thread_groups(
            MTLSize {
                width: n_frames as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            },
        );

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        if cmd.status() != MTLCommandBufferStatus::Completed {
            return Err("Metal FFT command buffer did not complete successfully".to_string());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.data_buf.contents() as *const u8,
                buffer.as_mut_ptr() as *mut u8,
                total * std::mem::size_of::<Complex<f32>>(),
            );
        }

        Ok(())
    }

    fn dispatch_fft(
        ctx: &mut MetalFftContext,
        buffer: &mut [Complex<f32>],
        n: usize,
        inverse: bool,
    ) -> Result<(), String> {
        dispatch_fft_batched(ctx, buffer, n, 1, inverse)
    }

    pub fn fft_forward_gpu(buffer: &mut [Complex<f32>], n: usize) -> Result<(), String> {
        if !metal_fft_enabled() {
            return Err("Metal FFT disabled (set IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on to enable)".to_string());
        }
        if n == 0 || (n & (n - 1)) != 0 {
            return Err("FFT size must be a power of 2".to_string());
        }
        if n > MAX_GPU_N {
            return Err(format!("Metal FFT currently supports n <= {}", MAX_GPU_N));
        }

        METAL_FFT_CONTEXT.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let ctx = ctx
                .as_mut()
                .ok_or_else(|| "Metal FFT context unavailable".to_string())?;
            dispatch_fft(ctx, buffer, n, false)
        })
    }

    pub fn fft_inverse_gpu(buffer: &mut [Complex<f32>], n: usize) -> Result<(), String> {
        if !metal_fft_enabled() {
            return Err("Metal FFT disabled (set IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on to enable)".to_string());
        }
        if n == 0 || (n & (n - 1)) != 0 {
            return Err("FFT size must be a power of 2".to_string());
        }
        if n > MAX_GPU_N {
            return Err(format!("Metal FFT currently supports n <= {}", MAX_GPU_N));
        }

        METAL_FFT_CONTEXT.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let ctx = ctx
                .as_mut()
                .ok_or_else(|| "Metal FFT context unavailable".to_string())?;
            dispatch_fft(ctx, buffer, n, true)
        })
    }

    pub fn fft_forward_batched_gpu(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        if !metal_fft_enabled() {
            return Err("Metal FFT disabled (set IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on to enable)".to_string());
        }
        if n == 0 || (n & (n - 1)) != 0 {
            return Err("FFT size must be a power of 2".to_string());
        }
        if n > MAX_GPU_N {
            return Err(format!("Metal FFT currently supports n <= {}", MAX_GPU_N));
        }
        if n_frames == 0 {
            return Ok(());
        }

        METAL_FFT_CONTEXT.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let ctx = ctx
                .as_mut()
                .ok_or_else(|| "Metal FFT context unavailable".to_string())?;
            dispatch_fft_batched(ctx, buffer, n, n_frames, false)
        })
    }

    pub fn fft_inverse_batched_gpu(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        if !metal_fft_enabled() {
            return Err("Metal FFT disabled (set IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on to enable)".to_string());
        }
        if n == 0 || (n & (n - 1)) != 0 {
            return Err("FFT size must be a power of 2".to_string());
        }
        if n > MAX_GPU_N {
            return Err(format!("Metal FFT currently supports n <= {}", MAX_GPU_N));
        }
        if n_frames == 0 {
            return Ok(());
        }

        METAL_FFT_CONTEXT.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let ctx = ctx
                .as_mut()
                .ok_or_else(|| "Metal FFT context unavailable".to_string())?;
            dispatch_fft_batched(ctx, buffer, n, n_frames, true)
        })
    }
}

#[cfg(not(all(feature = "apple-gpu", target_os = "macos")))]
pub mod metal_fft_impl {
    use rustfft::num_complex::Complex;

    pub fn fft_forward_gpu(_buffer: &mut [Complex<f32>], _n: usize) -> Result<(), String> {
        Err("Metal FFT not available on non-macOS platforms".to_string())
    }

    pub fn fft_inverse_gpu(_buffer: &mut [Complex<f32>], _n: usize) -> Result<(), String> {
        Err("Metal FFT not available on non-macOS platforms".to_string())
    }

    pub fn fft_forward_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("Metal FFT not available on non-macOS platforms".to_string())
    }

    pub fn fft_inverse_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("Metal FFT not available on non-macOS platforms".to_string())
    }
}

// High-level API: try GPU first, fall back to CPU on any error or unavailability
use rustfft::num_complex::Complex;

fn metal_fft_batch_chunk_size() -> Option<usize> {
    static CHUNK: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .and_then(|v| if v > 0 { Some(v) } else { None })
    })
}

/// Phase 20: Adaptive chunk size calculator for GPU FFT batches.
/// Reduces dispatch overhead for smaller workloads by intelligently chunking.
/// When env var IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE is set, uses that value.
/// Otherwise, computes adaptive size based on n_fft and n_frames:
/// - If n_fft * n_frames ≤ 65536 (2^16): use CPU (too small for GPU amortization)
/// - Otherwise: chunk = max(64, n_frames / max(4, n_fft / 256))
///   This creates chunks sized for ~64-512 frames depending on FFT size.
fn adaptive_chunk_size(n_fft: usize, n_frames: usize) -> usize {
    if let Some(explicit) = metal_fft_batch_chunk_size() {
        return explicit;
    }

    let total_work = n_fft.saturating_mul(n_frames);
    // If very small, don't chunk (will likely fall back to CPU anyway)
    if total_work <= 65536 {
        return n_frames; // Return full batch; fallback path will handle
    }

    // Adaptive: larger FFTs -> smaller chunks to amortize setup cost more frequently
    let chunk_divisor = std::cmp::max(4, n_fft / 256);
    let recommended = std::cmp::max(64, n_frames / chunk_divisor);
    std::cmp::min(recommended, 512) // Cap at 512 frames per chunk
}

/// Attempt FFT on GPU; fall back to CPU on any error.
pub fn fft_forward_with_fallback(buffer: &mut [Complex<f32>], n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }

    // Try GPU path first
    if metal_fft_impl::fft_forward_gpu(buffer, n).is_ok() {
        return Ok(());
    }

    // Fall back to CPU via rustfft
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer[..n]);
    Ok(())
}

/// Attempt inverse FFT on GPU; fall back to CPU on any error.
pub fn fft_inverse_with_fallback(buffer: &mut [Complex<f32>], n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }

    // Try GPU path first
    if metal_fft_impl::fft_inverse_gpu(buffer, n).is_ok() {
        return Ok(());
    }

    // Fall back to CPU via rustfft
    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut buffer[..n]);
    Ok(())
}

/// Attempt batched FFT on GPU; fall back to CPU on any error.
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

    if metal_fft_impl::fft_forward_batched_gpu(buffer, n, n_frames).is_ok() {
        return Ok(());
    }

    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    for frame in 0..n_frames {
        let start = frame * n;
        fft.process(&mut buffer[start..start + n]);
    }
    Ok(())
}

/// Attempt batched inverse FFT on GPU; fall back to CPU on any error.
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

    if metal_fft_impl::fft_inverse_batched_gpu(buffer, n, n_frames).is_ok() {
        return Ok(());
    }

    use rustfft::FftPlanner;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    for frame in 0..n_frames {
        let start = frame * n;
        fft.process(&mut buffer[start..start + n]);
    }
    Ok(())
}

/// Attempt batched FFT on GPU in chunked dispatches; falls back to CPU on any error.
/// Phase 20: Uses adaptive chunk sizing to balance overhead vs GPU utilization.
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

    let chunk = adaptive_chunk_size(n, n_frames).min(n_frames).max(1);
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

/// Attempt batched inverse FFT on GPU in chunked dispatches; falls back to CPU on any error.
/// Phase 20: Uses adaptive chunk sizing to balance overhead vs GPU utilization.
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

    let chunk = adaptive_chunk_size(n, n_frames).min(n_frames).max(1);
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

