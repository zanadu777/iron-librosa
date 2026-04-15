use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use std::{cell::RefCell, sync::OnceLock};

use crate::backend::{resolved_rust_device, RustDevice};

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const TILE_SIZE: u64 = 32;

// Only dispatch GPU for workloads where kernel + buffer overhead is justified.
// Below this threshold the faer CPU GEMM is consistently faster.
#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const DEFAULT_GPU_WORK_THRESHOLD: usize = 200_000_000; // 200 M multiply-adds

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const MEL_PROJECT_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    // Tiled GEMM: out[m][t] = sum_k mel_basis[m][k] * s[k][t]
    //
    // Each 32x32 threadgroup (1024 threads, Apple Silicon max) loads tiles of
    // mel_basis and S into threadgroup (shared) memory, accumulates partial
    // dot products, then writes out[m][t].  Each global value is reused 32×,
    // reducing bandwidth vs the naive scalar kernel by 32×.
    kernel void mel_project_f32_kernel(
        const device float* s          [[buffer(0)]],
        const device float* mel_basis  [[buffer(1)]],
        device       float* out        [[buffer(2)]],
        const device uint*  dims       [[buffer(3)]],
        uint2 gid  [[thread_position_in_grid]],
        uint2 tid  [[thread_position_in_threadgroup]])
    {
        constexpr uint TS = 32;
        threadgroup float tileA[TS][TS]; // mel_basis tile: [mel_row][k_col]
        threadgroup float tileB[TS][TS]; // S tile:         [k_row][frame_col]

        const uint t          = gid.x;
        const uint m          = gid.y;
        const uint n_fft_bins = dims[0];
        const uint n_frames   = dims[1];
        const uint n_mels     = dims[2];

        float acc = 0.0f;
        const uint n_tiles = (n_fft_bins + TS - 1) / TS;

        for (uint tile = 0; tile < n_tiles; ++tile) {
            const uint k = tile * TS;

            // Cooperatively load mel_basis[m][k + tid.x] into shared tileA
            tileA[tid.y][tid.x] = (m < n_mels && (k + tid.x) < n_fft_bins)
                ? mel_basis[m * n_fft_bins + k + tid.x]
                : 0.0f;

            // Cooperatively load s[k + tid.y][t] into shared tileB
            tileB[tid.y][tid.x] = ((k + tid.y) < n_fft_bins && t < n_frames)
                ? s[(k + tid.y) * n_frames + t]
                : 0.0f;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate partial dot product from the tile
            for (uint i = 0; i < TS; ++i) {
                acc += tileA[tid.y][i] * tileB[i][tid.x];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (m < n_mels && t < n_frames) {
            out[m * n_frames + t] = acc;
        }
    }
"#;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
struct MetalMelContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    // Persistent shared-memory buffers — grown only when workload exceeds capacity.
    // Avoids per-call Metal buffer allocation (malloc + driver bookkeeping overhead).
    s_buf: Buffer,
    mel_buf: Buffer,
    out_buf: Buffer,
    dims_buf: Buffer,     // always 16 bytes (4 × u32)
    s_cap_bytes: usize,
    mel_cap_bytes: usize,
    out_cap_bytes: usize,
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
thread_local! {
    static MEL_METAL_CONTEXT: RefCell<Option<MetalMelContext>> =
        RefCell::new(build_metal_mel_context());
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn build_metal_mel_context() -> Option<MetalMelContext> {
    let device = Device::system_default()?;
    let options = CompileOptions::new();
    let library = device.new_library_with_source(MEL_PROJECT_MSL, &options).ok()?;
    let function = library.get_function("mel_project_f32_kernel", None).ok()?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .ok()?;
    let queue = device.new_command_queue();

    // Seed tiny buffers; they will grow lazily on the first real workload.
    let s_buf   = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let mel_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let out_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    // dims is always 3 u32 = 12 bytes; allocate 16 for alignment.
    let dims_buf = device.new_buffer(16, MTLResourceOptions::StorageModeShared);

    Some(MetalMelContext {
        device, queue, pipeline,
        s_buf, mel_buf, out_buf, dims_buf,
        s_cap_bytes: 4,
        mel_cap_bytes: 4,
        out_cap_bytes: 4,
    })
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn gpu_work_threshold() -> usize {
    static OVERRIDE: OnceLock<usize> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("IRON_LIBROSA_GPU_WORK_THRESHOLD")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_GPU_WORK_THRESHOLD)
    })
}

fn validate_shapes<T>(
    s: &ndarray::ArrayView2<'_, T>,
    mel_basis: &ndarray::ArrayView2<'_, T>,
) -> PyResult<(usize, usize, usize)> {
    let n_fft_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let n_mels = mel_basis.shape()[0];

    if mel_basis.shape()[1] != n_fft_bins {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: S is ({n_fft_bins}, {n_frames}), mel_basis is \
             ({n_mels}, {}). Expected mel_basis.shape[1] == S.shape[0].",
            mel_basis.shape()[1]
        )));
    }

    Ok((n_fft_bins, n_frames, n_mels))
}

fn fused_mel_single_channel(
    y_slice: &[f32],
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window_slice: &[f32],
    mel_basis_slice: &[f32],
    n_mels: usize,
) -> PyResult<(usize, Vec<f32>)> {
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

    if y_padded.len() < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }

    let n_frames = 1 + (y_padded.len() - n_fft) / hop_length;
    let mut out = vec![0.0f32; n_mels * n_frames];

    crate::cuda_fft::fused_stft_mel_power_f32_gpu(
        &y_padded,
        window_slice,
        hop_length,
        mel_basis_slice,
        &mut out,
        n_fft,
        n_frames,
        n_mels,
    )
    .map_err(PyRuntimeError::new_err)?;

    Ok((n_frames, out))
}

#[pyfunction]
#[pyo3(signature = (y, n_fft, hop_length, center, window, mel_basis))]
pub fn melspectrogram_fused_f32<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: PyReadonlyArray1<'py, f32>,
    mel_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }
    if resolved_rust_device() != RustDevice::CudaGpu {
        return Err(PyRuntimeError::new_err(
            "CUDA fused mel path is only available when resolved device is cuda-gpu",
        ));
    }

    let y_slice = y.as_slice()?;
    let window_slice = window.as_slice()?;
    if window_slice.len() != n_fft {
        return Err(PyValueError::new_err(format!(
            "Window length {} != n_fft {}",
            window_slice.len(),
            n_fft
        )));
    }

    let mel_basis_view = mel_basis.as_array();
    let n_bins = n_fft / 2 + 1;
    if mel_basis_view.shape()[1] != n_bins {
        return Err(PyValueError::new_err(format!(
            "mel_basis.shape[1] {} != n_fft//2+1 {}",
            mel_basis_view.shape()[1],
            n_bins
        )));
    }
    let n_mels = mel_basis_view.shape()[0];

    let mel_basis_slice = mel_basis_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mel_basis must be C-contiguous"))?;

    let (n_frames, out) = fused_mel_single_channel(
        y_slice,
        n_fft,
        hop_length,
        center,
        window_slice,
        mel_basis_slice,
        n_mels,
    )?;

    let out_arr = ndarray::Array2::from_shape_vec((n_mels, n_frames), out)
        .map_err(|e| PyRuntimeError::new_err(format!("mel output shape error: {}", e)))?;
    Ok(out_arr.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (y, n_fft, hop_length, center, window, mel_basis))]
pub fn melspectrogram_fused_batch_f32<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    window: PyReadonlyArray1<'py, f32>,
    mel_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    if n_fft < 2 {
        return Err(PyValueError::new_err("n_fft must be >= 2"));
    }
    if hop_length == 0 {
        return Err(PyValueError::new_err("hop_length must be > 0"));
    }
    if resolved_rust_device() != RustDevice::CudaGpu {
        return Err(PyRuntimeError::new_err(
            "CUDA fused mel path is only available when resolved device is cuda-gpu",
        ));
    }

    let y_view = y.as_array();
    let n_channels = y_view.shape()[0];
    let n_src_samples = y_view.shape()[1];
    if n_channels == 0 {
        return Err(PyValueError::new_err("batch dimension must be > 0"));
    }

    let window_slice = window.as_slice()?;
    if window_slice.len() != n_fft {
        return Err(PyValueError::new_err(format!(
            "Window length {} != n_fft {}",
            window_slice.len(),
            n_fft
        )));
    }

    let mel_basis_view = mel_basis.as_array();
    let n_bins = n_fft / 2 + 1;
    if mel_basis_view.shape()[1] != n_bins {
        return Err(PyValueError::new_err(format!(
            "mel_basis.shape[1] {} != n_fft//2+1 {}",
            mel_basis_view.shape()[1],
            n_bins
        )));
    }
    let n_mels = mel_basis_view.shape()[0];
    let mel_basis_slice = mel_basis_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("mel_basis must be C-contiguous"))?;

    let y_padded = if center {
        let pad = n_fft / 2;
        let n_padded = n_src_samples + 2 * pad;
        let mut out = vec![0.0f32; n_channels * n_padded];
        for ch in 0..n_channels {
            let src = y_view.row(ch);
            let dst_off = ch * n_padded + pad;
            for (i, val) in src.iter().enumerate() {
                out[dst_off + i] = *val;
            }
        }
        out
    } else {
        let mut out = Vec::with_capacity(n_channels * n_src_samples);
        for ch in 0..n_channels {
            out.extend(y_view.row(ch).iter().copied());
        }
        out
    };

    let n_samples_per_channel = if center {
        n_src_samples + n_fft
    } else {
        n_src_samples
    };
    if n_samples_per_channel < n_fft {
        return Err(PyValueError::new_err("Audio too short for given n_fft."));
    }
    let n_frames = 1 + (n_samples_per_channel - n_fft) / hop_length;
    let mut out_all = vec![0.0f32; n_channels * n_mels * n_frames];

    crate::cuda_fft::fused_stft_mel_power_batch_f32_gpu(
        &y_padded,
        n_channels,
        n_samples_per_channel,
        window_slice,
        hop_length,
        mel_basis_slice,
        &mut out_all,
        n_fft,
        n_frames,
        n_mels,
    )
    .map_err(PyRuntimeError::new_err)?;

    let out_arr = ndarray::Array3::from_shape_vec((n_channels, n_mels, n_frames), out_all)
        .map_err(|e| PyRuntimeError::new_err(format!("mel output shape error: {}", e)))?;
    Ok(out_arr.into_pyarray_bound(py))
}

/// Reinterpret a C-contiguous `ArrayView2` as a **column-major** faer
/// `MatRef` of the *transposed* shape — zero copies.
///
/// A row-major matrix `A` (nrows×ncols, row_stride=ncols, col_stride=1)
/// in memory is *byte-identical* to its transpose `A^T` stored column-major
/// (ncols×nrows, row_stride=1, col_stride=nrows).  faer's GEMM is fastest
/// when all operands are column-major, so we expose them that way and adjust
/// the multiplication order: `C = A*B  ⟺  C^T = B^T * A^T`.
///
/// # Safety
/// `arr` must remain valid for the lifetime `'a`.
#[inline]
unsafe fn as_col_major_t<'a, T>(arr: &'a ndarray::ArrayView2<'a, T>) -> MatRef<'a, T> {
    let nrows = arr.shape()[0];
    let ncols = arr.shape()[1];
    let row_s = arr.strides()[0]; // = ncols for C-contiguous
    let col_s = arr.strides()[1]; // = 1   for C-contiguous
    // Transposed shape + swapped strides → column-major (row_stride=1).
    MatRef::from_raw_parts(arr.as_ptr(), ncols, nrows, col_s, row_s)
}

#[pyfunction]
#[pyo3(signature = (s, mel_basis))]
pub fn mel_project_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    mel_basis: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => mel_project_f64_cpu(py, s, mel_basis),
        // Phase 1 scaffold: keep CPU path while Apple GPU kernels are pending.
        RustDevice::AppleGpu => mel_project_f64_cpu(py, s, mel_basis),
        RustDevice::Auto => mel_project_f64_cpu(py, s, mel_basis),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => mel_project_f64_cpu(py, s, mel_basis),
    }
}

fn mel_project_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    mel_basis: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_array();
    let mel_basis = mel_basis.as_array();
    let (_n_fft_bins, n_frames, n_mels) = validate_shapes(&s, &mel_basis)?;

    let mut out = ndarray::Array2::<f64>::zeros((n_mels, n_frames));

    // SAFETY: inputs are C-contiguous (np.ascontiguousarray in Python caller)
    // and `out` is freshly allocated.
    //
    // Compute C^T = S^T × M^T  (all column-major) which is byte-identical to
    // C = M × S  (all row-major) stored in `out`.
    unsafe {
        // out^T : (n_frames × n_mels) column-major  (same bytes as out row-major)
        let out_t = MatMut::<f64>::from_raw_parts_mut(
            out.as_mut_ptr(),
            n_frames, n_mels,
            out.strides()[1], // row_stride = 1 (col stride of row-major out)
            out.strides()[0], // col_stride = n_frames (row stride of row-major out)
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&s),         // S^T : (n_frames × n_fft_bins) col-major
            as_col_major_t(&mel_basis), // M^T : (n_fft_bins × n_mels) col-major
            1.0f64,
            Par::rayon(0),
        );
    }

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, mel_basis))]
pub fn mel_project_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    mel_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => mel_project_f32_cpu(py, s, mel_basis),
        RustDevice::AppleGpu => mel_project_f32_apple_gpu(py, s, mel_basis),
        RustDevice::Auto => mel_project_f32_cpu(py, s, mel_basis),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => mel_project_f32_cpu(py, s, mel_basis),
    }
}

fn mel_project_f32_apple_gpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    mel_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    #[cfg(all(feature = "apple-gpu", target_os = "macos"))]
    {
        let s_view = s.as_array();
        let mel_view = mel_basis.as_array();
        if let Some(out) = try_mel_project_f32_apple_gpu(&s_view, &mel_view) {
            return Ok(out.into_pyarray_bound(py));
        }
    }

    // Safe fallback policy: any unavailable/failed GPU path returns CPU output.
    mel_project_f32_cpu(py, s, mel_basis)
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn try_mel_project_f32_apple_gpu(
    s: &ndarray::ArrayView2<'_, f32>,
    mel_basis: &ndarray::ArrayView2<'_, f32>,
) -> Option<ndarray::Array2<f32>> {
    try_project_f32_apple_gpu(s, mel_basis)
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
pub(crate) fn try_project_f32_apple_gpu(
    s: &ndarray::ArrayView2<'_, f32>,
    basis: &ndarray::ArrayView2<'_, f32>,
) -> Option<ndarray::Array2<f32>> {
    if !s.is_standard_layout() || !basis.is_standard_layout() {
        return None;
    }

    let n_fft_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let n_rows = basis.shape()[0];
    if basis.shape()[1] != n_fft_bins {
        return None;
    }

    // Skip GPU for small workloads where command-buffer and buffer-allocation
    // overhead dominates; the CPU faer GEMM is faster below this threshold.
    let total_ops = n_fft_bins.saturating_mul(n_rows).saturating_mul(n_frames);
    if total_ops < gpu_work_threshold() {
        return None;
    }

    let out_len = n_frames.checked_mul(n_rows)?;

    let s_slice = s.as_slice()?;
    let basis_slice = basis.as_slice()?;
    let dims = [n_fft_bins as u32, n_frames as u32, n_rows as u32];

    let s_bytes   = std::mem::size_of_val(s_slice);
    let basis_bytes = std::mem::size_of_val(basis_slice);
    let out_bytes = out_len * std::mem::size_of::<f32>();

    MEL_METAL_CONTEXT.with(|slot| {
        let mut ctx_ref = slot.borrow_mut();
        let ctx = ctx_ref.as_mut()?;

        // Grow persistent buffers only when this call is larger than any prior call.
        // On Apple Silicon StorageModeShared == unified memory; no GPU-side copy needed.
        if s_bytes > ctx.s_cap_bytes {
            ctx.s_buf = ctx.device.new_buffer(s_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.s_cap_bytes = s_bytes;
        }
        if basis_bytes > ctx.mel_cap_bytes {
            ctx.mel_buf = ctx.device.new_buffer(basis_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.mel_cap_bytes = basis_bytes;
        }
        if out_bytes > ctx.out_cap_bytes {
            ctx.out_buf = ctx.device.new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.out_cap_bytes = out_bytes;
        }

        // SAFETY: buffers are pre-allocated StorageModeShared with at least the required
        // capacity; contents() returns a valid CPU-accessible pointer for the full buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                s_slice.as_ptr(),
                ctx.s_buf.contents().cast::<f32>(),
                s_slice.len(),
            );
            std::ptr::copy_nonoverlapping(
                basis_slice.as_ptr(),
                ctx.mel_buf.contents().cast::<f32>(),
                basis_slice.len(),
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
        encoder.set_buffer(0, Some(&ctx.s_buf), 0);
        encoder.set_buffer(1, Some(&ctx.mel_buf), 0);
        encoder.set_buffer(2, Some(&ctx.out_buf), 0);
        encoder.set_buffer(3, Some(&ctx.dims_buf), 0);

        // Tiled GEMM requires exactly TILE_SIZE × TILE_SIZE threads per group
        // so that threadgroup_barrier synchronises the full tile load.
        //
        // dispatch_thread_groups (not dispatch_threads) ensures every threadgroup is
        // complete, so all 32×32 threads always reach threadgroup_barrier together.
        // OOB threads are handled by the bounds checks inside the kernel itself.
        let threadgroups = MTLSize {
            width:  (n_frames as u64 + TILE_SIZE - 1) / TILE_SIZE,
            height: (n_rows   as u64 + TILE_SIZE - 1) / TILE_SIZE,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width:  TILE_SIZE,
            height: TILE_SIZE,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let status = command_buffer.status();
        if status != metal::MTLCommandBufferStatus::Completed {
            return None;
        }

        // SAFETY: out_buf is StorageModeShared, contents() is valid for out_len f32s.
        let out_slice = unsafe {
            std::slice::from_raw_parts(ctx.out_buf.contents().cast::<f32>(), out_len)
        };
        ndarray::Array2::from_shape_vec((n_rows, n_frames), out_slice.to_vec()).ok()
    })
}

fn mel_project_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    mel_basis: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let s = s.as_array();
    let mel_basis = mel_basis.as_array();
    let (_n_fft_bins, n_frames, n_mels) = validate_shapes(&s, &mel_basis)?;

    let mut out = ndarray::Array2::<f32>::zeros((n_mels, n_frames));

    unsafe {
        let out_t = MatMut::<f32>::from_raw_parts_mut(
            out.as_mut_ptr(),
            n_frames, n_mels,
            out.strides()[1],
            out.strides()[0],
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&s),
            as_col_major_t(&mel_basis),
            1.0f32,
            Par::rayon(0),
        );
    }

    Ok(out.into_pyarray_bound(py))
}

#[inline]
fn hz_to_mel_scalar(f: f64, htk: bool) -> f64 {
    if htk {
        2595.0 * (1.0 + f / 700.0).log10()
    } else {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4_f64.ln() / 27.0;
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    }
}

#[inline]
fn mel_to_hz_scalar(m: f64, htk: bool) -> f64 {
    if htk {
        700.0 * (10.0_f64.powf(m / 2595.0) - 1.0)
    } else {
        let f_sp = 200.0 / 3.0;
        let min_log_hz = 1000.0;
        let min_log_mel = min_log_hz / f_sp;
        let logstep = 6.4_f64.ln() / 27.0;
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    }
}

#[pyfunction]
#[pyo3(signature = (sr, n_fft, n_mels = 128, fmin = 0.0, fmax = None, htk = false, slaney_norm = true))]
pub fn mel_filter_f32<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    slaney_norm: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => mel_filter_f32_cpu(py, sr, n_fft, n_mels, fmin, fmax, htk, slaney_norm),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => mel_filter_f32_cpu(py, sr, n_fft, n_mels, fmin, fmax, htk, slaney_norm),
        RustDevice::Auto => mel_filter_f32_cpu(py, sr, n_fft, n_mels, fmin, fmax, htk, slaney_norm),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => mel_filter_f32_cpu(py, sr, n_fft, n_mels, fmin, fmax, htk, slaney_norm),
    }
}

fn mel_filter_f32_cpu<'py>(
    py: Python<'py>,
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
    htk: bool,
    slaney_norm: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if sr <= 0.0 {
        return Err(PyValueError::new_err("sr must be > 0"));
    }
    if n_fft < 2 || n_mels == 0 {
        return Err(PyValueError::new_err("n_fft must be >=2 and n_mels > 0"));
    }

    let fmax = fmax.unwrap_or(sr / 2.0);
    let n_bins = 1 + n_fft / 2;

    // FFT bin centers in Hz
    let fftfreqs: Vec<f64> = (0..n_bins)
        .map(|i| (sr / 2.0) * (i as f64) / ((n_bins - 1) as f64))
        .collect();

    // Mel band edge frequencies in Hz (n_mels + 2 points)
    let m0 = hz_to_mel_scalar(fmin, htk);
    let m1 = hz_to_mel_scalar(fmax, htk);
    let mel_f: Vec<f64> = (0..(n_mels + 2))
        .map(|i| {
            let t = i as f64 / ((n_mels + 1) as f64);
            mel_to_hz_scalar(m0 + t * (m1 - m0), htk)
        })
        .collect();

    let mut weights = ndarray::Array2::<f32>::zeros((n_mels, n_bins));

    for i in 0..n_mels {
        let fdiff_l = mel_f[i + 1] - mel_f[i];
        let fdiff_u = mel_f[i + 2] - mel_f[i + 1];
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / fdiff_l;
            let upper = (mel_f[i + 2] - f) / fdiff_u;
            let v = lower.min(upper).max(0.0) as f32;
            weights[(i, j)] = v;
        }
    }

    if slaney_norm {
        for i in 0..n_mels {
            let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
            let s = enorm as f32;
            let mut row = weights.row_mut(i);
            row *= s;
        }
    }

    Ok(weights.into_pyarray_bound(py))
}
