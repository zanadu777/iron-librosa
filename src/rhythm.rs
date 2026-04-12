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

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use std::{cell::RefCell, sync::OnceLock};

use crate::backend::{resolved_rust_device, RustDevice};

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const TEMPOGRAM_TILE_SIZE: u64 = 16;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const DEFAULT_TEMPOGRAM_GPU_WORK_THRESHOLD: usize = 300_000_000;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const TEMPOGRAM_AC_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    // out[lag, t] = sum_{i=0}^{win_len-lag-1} w[i, t] * w[i+lag, t]
    kernel void tempogram_ac_f32_kernel(
        const device float* windowed [[buffer(0)]],
        device float* out [[buffer(1)]],
        const device uint* dims [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint t = gid.x;
        uint lag = gid.y;
        uint win_len = dims[0];
        uint n_frames = dims[1];

        if (lag >= win_len || t >= n_frames) {
            return;
        }

        float acc = 0.0f;
        uint stop = win_len - lag;
        for (uint i = 0; i < stop; ++i) {
            float a = windowed[i * n_frames + t];
            float b = windowed[(i + lag) * n_frames + t];
            acc += a * b;
        }
        out[lag * n_frames + t] = acc;
    }
"#;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
struct TempogramAcF32Context {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    windowed_buf: Buffer,
    out_buf: Buffer,
    dims_buf: Buffer,
    in_cap_bytes: usize,
    out_cap_bytes: usize,
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
thread_local! {
    static TEMPOGRAM_AC_F32_CONTEXT: RefCell<Option<TempogramAcF32Context>> =
        RefCell::new(build_tempogram_ac_f32_context());
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn build_tempogram_ac_f32_context() -> Option<TempogramAcF32Context> {
    let device = Device::system_default()?;
    let options = CompileOptions::new();
    let library = device.new_library_with_source(TEMPOGRAM_AC_MSL, &options).ok()?;
    let function = library.get_function("tempogram_ac_f32_kernel", None).ok()?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .ok()?;
    let queue = device.new_command_queue();

    let windowed_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let out_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let dims_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);

    Some(TempogramAcF32Context {
        device,
        queue,
        pipeline,
        windowed_buf,
        out_buf,
        dims_buf,
        in_cap_bytes: 4,
        out_cap_bytes: 4,
    })
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn tempogram_gpu_work_threshold() -> usize {
    static OVERRIDE: OnceLock<usize> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("IRON_LIBROSA_TEMPOGRAM_GPU_WORK_THRESHOLD")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_TEMPOGRAM_GPU_WORK_THRESHOLD)
    })
}

// ── f32 ──────────────────────────────────────────────────────────────────────

fn tempogram_ac_impl_f32_cpu(
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
    let out = match resolved_rust_device() {
        RustDevice::Cpu => tempogram_ac_impl_f32_cpu(windowed.as_array(), n_pad)?,
        RustDevice::AppleGpu => tempogram_ac_impl_f32_apple_gpu(windowed.as_array(), n_pad)?,
        RustDevice::Auto => tempogram_ac_impl_f32_cpu(windowed.as_array(), n_pad)?,
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => tempogram_ac_impl_f32_cpu(windowed.as_array(), n_pad)?,
    };
    Ok(out.into_pyarray_bound(py))
}

fn tempogram_ac_impl_f32_apple_gpu(
    windowed: ndarray::ArrayView2<'_, f32>,
    n_pad: usize,
) -> PyResult<Array2<f32>> {
    let (win_len, n_frames) = windowed.dim();
    if win_len == 0 || n_frames == 0 {
        return Ok(Array2::zeros((win_len, n_frames)));
    }
    if n_pad < win_len {
        return Err(PyValueError::new_err("n_pad must be >= win_length"));
    }

    #[cfg(all(feature = "apple-gpu", target_os = "macos"))]
    {
        if let Some(out) = try_tempogram_ac_f32_apple_gpu(&windowed) {
            return Ok(out);
        }
    }

    // Safe fallback policy: any unavailable/failed GPU path returns CPU output.
    tempogram_ac_impl_f32_cpu(windowed, n_pad)
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn try_tempogram_ac_f32_apple_gpu(
    windowed: &ndarray::ArrayView2<'_, f32>,
) -> Option<Array2<f32>> {
    if !windowed.is_standard_layout() {
        return None;
    }

    let win_len = windowed.shape()[0];
    let n_frames = windowed.shape()[1];

    let work = win_len
        .saturating_mul(win_len)
        .saturating_mul(n_frames);
    if work < tempogram_gpu_work_threshold() {
        return None;
    }

    let out_len = win_len.checked_mul(n_frames)?;
    let in_slice = windowed.as_slice()?;
    let in_bytes = std::mem::size_of_val(in_slice);
    let out_bytes = out_len * std::mem::size_of::<f32>();
    let dims = [win_len as u32, n_frames as u32];

    TEMPOGRAM_AC_F32_CONTEXT.with(|slot| {
        let mut ctx_ref = slot.borrow_mut();
        let ctx = ctx_ref.as_mut()?;

        if in_bytes > ctx.in_cap_bytes {
            ctx.windowed_buf = ctx
                .device
                .new_buffer(in_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.in_cap_bytes = in_bytes;
        }
        if out_bytes > ctx.out_cap_bytes {
            ctx.out_buf = ctx
                .device
                .new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.out_cap_bytes = out_bytes;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                in_slice.as_ptr(),
                ctx.windowed_buf.contents().cast::<f32>(),
                in_slice.len(),
            );
            std::ptr::copy_nonoverlapping(
                dims.as_ptr(),
                ctx.dims_buf.contents().cast::<u32>(),
                dims.len(),
            );
        }

        let command_buffer = ctx.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        encoder.set_buffer(0, Some(&ctx.windowed_buf), 0);
        encoder.set_buffer(1, Some(&ctx.out_buf), 0);
        encoder.set_buffer(2, Some(&ctx.dims_buf), 0);

        let threadgroups = MTLSize {
            width: (n_frames as u64 + TEMPOGRAM_TILE_SIZE - 1) / TEMPOGRAM_TILE_SIZE,
            height: (win_len as u64 + TEMPOGRAM_TILE_SIZE - 1) / TEMPOGRAM_TILE_SIZE,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: TEMPOGRAM_TILE_SIZE,
            height: TEMPOGRAM_TILE_SIZE,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return None;
        }

        let out_slice = unsafe {
            std::slice::from_raw_parts(ctx.out_buf.contents().cast::<f32>(), out_len)
        };
        Array2::from_shape_vec((win_len, n_frames), out_slice.to_vec()).ok()
    })
}

// ── f64 ──────────────────────────────────────────────────────────────────────

fn tempogram_ac_impl_f64_cpu(
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
    let out = match resolved_rust_device() {
        RustDevice::Cpu => tempogram_ac_impl_f64_cpu(windowed.as_array(), n_pad)?,
        // Phase 1 scaffold: keep CPU path while Apple GPU kernels are pending.
        RustDevice::AppleGpu => tempogram_ac_impl_f64_cpu(windowed.as_array(), n_pad)?,
        RustDevice::Auto => tempogram_ac_impl_f64_cpu(windowed.as_array(), n_pad)?,
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => tempogram_ac_impl_f64_cpu(windowed.as_array(), n_pad)?,
    };
    Ok(out.into_pyarray_bound(py))
}

