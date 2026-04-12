// Tuning-estimation helper kernels for iron-librosa.
//
// These kernels accelerate the post-processing stage of librosa.estimate_tuning:
// thresholding piptrack output and running pitch_tuning histogram voting.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
use std::{cell::RefCell, sync::OnceLock};

use crate::backend::{resolved_rust_device, RustDevice};

const PAR_THRESHOLD: usize = 200_000;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const PIPTRACK_TILE_SIZE: u64 = 16;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const DEFAULT_PIPTRACK_GPU_WORK_THRESHOLD: usize = 5_000_000;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
const PIPTRACK_MSL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void piptrack_f32_kernel(
        const device float* s         [[buffer(0)]],
        const device float* shift     [[buffer(1)]],
        const device float* dskew     [[buffer(2)]],
        const device float* ref_vals  [[buffer(3)]],
        device float* pitch_out       [[buffer(4)]],
        device float* mag_out         [[buffer(5)]],
        const device uint* params_u32 [[buffer(6)]],
        const device float* params_f32[[buffer(7)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint t = gid.x;
        uint f = gid.y;
        uint n_bins = params_u32[0];
        uint n_frames = params_u32[1];
        uint start_bin = params_u32[2];
        uint stop_bin = params_u32[3];
        float sr_over_nfft = params_f32[0];

        if (f >= n_bins || t >= n_frames) {
            return;
        }

        uint idx = f * n_frames + t;
        pitch_out[idx] = 0.0f;
        mag_out[idx] = 0.0f;

        if (n_bins < 3 || f == 0 || f >= (n_bins - 1) || f < start_bin || f >= stop_bin) {
            return;
        }

        float threshold = ref_vals[t];
        float left_raw = s[(f - 1) * n_frames + t];
        float center_raw = s[idx];
        float right_raw = s[(f + 1) * n_frames + t];

        float left = left_raw > threshold ? left_raw : 0.0f;
        float center = center_raw > threshold ? center_raw : 0.0f;
        float right = right_raw > threshold ? right_raw : 0.0f;

        if (center > left && center >= right) {
            pitch_out[idx] = (float(f) + shift[idx]) * sr_over_nfft;
            mag_out[idx] = center_raw + dskew[idx];
        }
    }
"#;

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
struct PiptrackF32Context {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    s_buf: Buffer,
    shift_buf: Buffer,
    dskew_buf: Buffer,
    ref_buf: Buffer,
    pitch_buf: Buffer,
    mag_buf: Buffer,
    params_u32_buf: Buffer,
    params_f32_buf: Buffer,
    s_cap_bytes: usize,
    shift_cap_bytes: usize,
    dskew_cap_bytes: usize,
    ref_cap_bytes: usize,
    out_cap_bytes: usize,
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
thread_local! {
    static PIPTRACK_F32_CONTEXT: RefCell<Option<PiptrackF32Context>> =
        RefCell::new(build_piptrack_f32_context());
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn build_piptrack_f32_context() -> Option<PiptrackF32Context> {
    let device = Device::system_default()?;
    let options = CompileOptions::new();
    let library = device.new_library_with_source(PIPTRACK_MSL, &options).ok()?;
    let function = library.get_function("piptrack_f32_kernel", None).ok()?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .ok()?;
    let queue = device.new_command_queue();

    let s_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let shift_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let dskew_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let ref_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let pitch_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let mag_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let params_u32_buf = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
    let params_f32_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    Some(PiptrackF32Context {
        device,
        queue,
        pipeline,
        s_buf,
        shift_buf,
        dskew_buf,
        ref_buf,
        pitch_buf,
        mag_buf,
        params_u32_buf,
        params_f32_buf,
        s_cap_bytes: 4,
        shift_cap_bytes: 4,
        dskew_cap_bytes: 4,
        ref_cap_bytes: 4,
        out_cap_bytes: 4,
    })
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn piptrack_gpu_work_threshold() -> usize {
    static OVERRIDE: OnceLock<usize> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("IRON_LIBROSA_PIPTRACK_GPU_WORK_THRESHOLD")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_PIPTRACK_GPU_WORK_THRESHOLD)
    })
}

#[inline]
fn piptrack_from_spectrogram_core_f32(
    s_view: ndarray::ArrayView2<'_, f32>,
    shift_view: ndarray::ArrayView2<'_, f32>,
    dskew_view: ndarray::ArrayView2<'_, f32>,
    ref_values: &[f32],
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let mut pitch_data = vec![0.0f32; n_bins * n_frames];
    let mut mag_data = vec![0.0f32; n_bins * n_frames];

    if n_bins < 3 || n_frames == 0 {
        return (pitch_data, mag_data);
    }

    let start = start_bin.max(1).min(n_bins - 1);
    let stop = stop_bin.min(n_bins - 1);
    if start >= stop {
        return (pitch_data, mag_data);
    }

    let write_row = |f: usize, pitch_row: &mut [f32], mag_row: &mut [f32]| {
        for t in 0..n_frames {
            let threshold = ref_values[t];
            let left = if s_view[[f - 1, t]] > threshold {
                s_view[[f - 1, t]]
            } else {
                0.0
            };
            let center_raw = s_view[[f, t]];
            let center = if center_raw > threshold { center_raw } else { 0.0 };
            let right = if s_view[[f + 1, t]] > threshold {
                s_view[[f + 1, t]]
            } else {
                0.0
            };

            if center > left && center >= right {
                pitch_row[t] = (f as f32 + shift_view[[f, t]]) * sr_over_nfft;
                mag_row[t] = center_raw + dskew_view[[f, t]];
            }
        }
    };

    if (stop - start) * n_frames >= PAR_THRESHOLD {
        pitch_data
            .par_chunks_mut(n_frames)
            .zip(mag_data.par_chunks_mut(n_frames))
            .enumerate()
            .for_each(|(f, (pitch_row, mag_row))| {
                if f >= start && f < stop {
                    write_row(f, pitch_row, mag_row);
                }
            });
    } else {
        for f in start..stop {
            let row_start = f * n_frames;
            write_row(
                f,
                &mut pitch_data[row_start..row_start + n_frames],
                &mut mag_data[row_start..row_start + n_frames],
            );
        }
    }

    (pitch_data, mag_data)
}

#[inline]
fn piptrack_from_spectrogram_core_f64(
    s_view: ndarray::ArrayView2<'_, f64>,
    shift_view: ndarray::ArrayView2<'_, f64>,
    dskew_view: ndarray::ArrayView2<'_, f64>,
    ref_values: &[f64],
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];
    let mut pitch_data = vec![0.0f64; n_bins * n_frames];
    let mut mag_data = vec![0.0f64; n_bins * n_frames];

    if n_bins < 3 || n_frames == 0 {
        return (pitch_data, mag_data);
    }

    let start = start_bin.max(1).min(n_bins - 1);
    let stop = stop_bin.min(n_bins - 1);
    if start >= stop {
        return (pitch_data, mag_data);
    }

    let write_row = |f: usize, pitch_row: &mut [f64], mag_row: &mut [f64]| {
        for t in 0..n_frames {
            let threshold = ref_values[t];
            let left = if s_view[[f - 1, t]] > threshold {
                s_view[[f - 1, t]]
            } else {
                0.0
            };
            let center_raw = s_view[[f, t]];
            let center = if center_raw > threshold { center_raw } else { 0.0 };
            let right = if s_view[[f + 1, t]] > threshold {
                s_view[[f + 1, t]]
            } else {
                0.0
            };

            if center > left && center >= right {
                pitch_row[t] = (f as f64 + shift_view[[f, t]]) * sr_over_nfft;
                mag_row[t] = center_raw + dskew_view[[f, t]];
            }
        }
    };

    if (stop - start) * n_frames >= PAR_THRESHOLD {
        pitch_data
            .par_chunks_mut(n_frames)
            .zip(mag_data.par_chunks_mut(n_frames))
            .enumerate()
            .for_each(|(f, (pitch_row, mag_row))| {
                if f >= start && f < stop {
                    write_row(f, pitch_row, mag_row);
                }
            });
    } else {
        for f in start..stop {
            let row_start = f * n_frames;
            write_row(
                f,
                &mut pitch_data[row_start..row_start + n_frames],
                &mut mag_data[row_start..row_start + n_frames],
            );
        }
    }

    (pitch_data, mag_data)
}

#[inline]
fn median_from_vec(mut vals: Vec<f64>) -> f64 {
    let n = vals.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        vals[mid]
    } else {
        let hi = vals[mid];
        vals.select_nth_unstable_by(mid - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        0.5 * (vals[mid - 1] + hi)
    }
}

#[inline]
fn estimate_tuning_impl<I>(
    iter_pitch_mag: I,
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64>
where
    I: Clone + Iterator<Item = (f64, f64)>,
{
    if !(0.0 < resolution && resolution < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resolution must be in the range (0, 1)",
        ));
    }
    if bins_per_octave <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bins_per_octave must be positive",
        ));
    }

    // threshold = median(mag[pitch > 0])
    let mags_pos: Vec<f64> = iter_pitch_mag
        .clone()
        .filter_map(|(p, m)| if p > 0.0 { Some(m) } else { None })
        .collect();

    let threshold = median_from_vec(mags_pos);

    // residual = mod(bpo * hz_to_octs(freq), 1.0), then fold to [-0.5, 0.5)
    let bpo = bins_per_octave as f64;
    let n_bins = (1.0 / resolution).ceil() as usize;
    let bin_width = 1.0 / (n_bins as f64);
    let mut counts = vec![0usize; n_bins];

    for (pitch, mag) in iter_pitch_mag {
        if pitch <= 0.0 || mag < threshold {
            continue;
        }

        // hz_to_octs(pitch) with tuning=0, bins_per_octave default in convert.hz_to_octs
        let oct = (pitch / 27.5).log2();
        let mut residual = (bpo * oct).rem_euclid(1.0);
        if residual >= 0.5 {
            residual -= 1.0;
        }

        let mut idx = ((residual + 0.5) / bin_width).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= n_bins {
            idx = (n_bins as isize) - 1;
        }
        counts[idx as usize] += 1;
    }

    // If no pitched values survive thresholding, match pitch_tuning behavior (return 0.0)
    if counts.iter().all(|&c| c == 0) {
        return Ok(0.0);
    }

    let mut best_i = 0usize;
    let mut best_c = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > best_c {
            best_c = c;
            best_i = i;
        }
    }

    Ok(-0.5 + (best_i as f64) * bin_width)
}

#[inline]
fn histogram_tuning_from_slices_f64(
    pitch: &[f64],
    mag: &[f64],
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64> {
    if !(0.0 < resolution && resolution < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resolution must be in the range (0, 1)",
        ));
    }
    if bins_per_octave <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bins_per_octave must be positive",
        ));
    }

    let mut mags_pos = Vec::<f64>::new();
    mags_pos.reserve(mag.len() / 4);
    for i in 0..pitch.len() {
        if pitch[i] > 0.0 {
            mags_pos.push(mag[i]);
        }
    }
    let threshold = median_from_vec(mags_pos);

    let bpo = bins_per_octave as f64;
    let n_bins = (1.0 / resolution).ceil() as usize;
    let bin_width = 1.0 / (n_bins as f64);

    let counts = if pitch.len() >= PAR_THRESHOLD {
        (0..pitch.len())
            .into_par_iter()
            .fold(
                || vec![0usize; n_bins],
                |mut local, i| {
                    let p = pitch[i];
                    let m = mag[i];
                    if p > 0.0 && m >= threshold {
                        let oct = (p / 27.5).log2();
                        let mut residual = (bpo * oct).rem_euclid(1.0);
                        if residual >= 0.5 {
                            residual -= 1.0;
                        }
                        let mut idx = ((residual + 0.5) / bin_width).floor() as isize;
                        if idx < 0 {
                            idx = 0;
                        }
                        if idx as usize >= n_bins {
                            idx = (n_bins as isize) - 1;
                        }
                        local[idx as usize] += 1;
                    }
                    local
                },
            )
            .reduce(
                || vec![0usize; n_bins],
                |mut a, b| {
                    for i in 0..n_bins {
                        a[i] += b[i];
                    }
                    a
                },
            )
    } else {
        let mut counts = vec![0usize; n_bins];
        for i in 0..pitch.len() {
            let p = pitch[i];
            let m = mag[i];
            if p <= 0.0 || m < threshold {
                continue;
            }
            let oct = (p / 27.5).log2();
            let mut residual = (bpo * oct).rem_euclid(1.0);
            if residual >= 0.5 {
                residual -= 1.0;
            }
            let mut idx = ((residual + 0.5) / bin_width).floor() as isize;
            if idx < 0 {
                idx = 0;
            }
            if idx as usize >= n_bins {
                idx = (n_bins as isize) - 1;
            }
            counts[idx as usize] += 1;
        }
        counts
    };

    if counts.iter().all(|&c| c == 0) {
        return Ok(0.0);
    }
    let mut best_i = 0usize;
    let mut best_c = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > best_c {
            best_c = c;
            best_i = i;
        }
    }
    Ok(-0.5 + (best_i as f64) * bin_width)
}

#[inline]
fn histogram_tuning_from_slices_f32(
    pitch: &[f32],
    mag: &[f32],
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64> {
    if !(0.0 < resolution && resolution < 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "resolution must be in the range (0, 1)",
        ));
    }
    if bins_per_octave <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bins_per_octave must be positive",
        ));
    }

    let mut mags_pos = Vec::<f64>::new();
    mags_pos.reserve(mag.len() / 4);
    for i in 0..pitch.len() {
        if pitch[i] > 0.0 {
            mags_pos.push(f64::from(mag[i]));
        }
    }
    let threshold = median_from_vec(mags_pos);

    let bpo = bins_per_octave as f64;
    let n_bins = (1.0 / resolution).ceil() as usize;
    let bin_width = 1.0 / (n_bins as f64);

    let counts = if pitch.len() >= PAR_THRESHOLD {
        (0..pitch.len())
            .into_par_iter()
            .fold(
                || vec![0usize; n_bins],
                |mut local, i| {
                    let p = f64::from(pitch[i]);
                    let m = f64::from(mag[i]);
                    if p > 0.0 && m >= threshold {
                        let oct = (p / 27.5).log2();
                        let mut residual = (bpo * oct).rem_euclid(1.0);
                        if residual >= 0.5 {
                            residual -= 1.0;
                        }
                        let mut idx = ((residual + 0.5) / bin_width).floor() as isize;
                        if idx < 0 {
                            idx = 0;
                        }
                        if idx as usize >= n_bins {
                            idx = (n_bins as isize) - 1;
                        }
                        local[idx as usize] += 1;
                    }
                    local
                },
            )
            .reduce(
                || vec![0usize; n_bins],
                |mut a, b| {
                    for i in 0..n_bins {
                        a[i] += b[i];
                    }
                    a
                },
            )
    } else {
        let mut counts = vec![0usize; n_bins];
        for i in 0..pitch.len() {
            let p = f64::from(pitch[i]);
            let m = f64::from(mag[i]);
            if p <= 0.0 || m < threshold {
                continue;
            }
            let oct = (p / 27.5).log2();
            let mut residual = (bpo * oct).rem_euclid(1.0);
            if residual >= 0.5 {
                residual -= 1.0;
            }
            let mut idx = ((residual + 0.5) / bin_width).floor() as isize;
            if idx < 0 {
                idx = 0;
            }
            if idx as usize >= n_bins {
                idx = (n_bins as isize) - 1;
            }
            counts[idx as usize] += 1;
        }
        counts
    };

    if counts.iter().all(|&c| c == 0) {
        return Ok(0.0);
    }
    let mut best_i = 0usize;
    let mut best_c = counts[0];
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > best_c {
            best_c = c;
            best_i = i;
        }
    }
    Ok(-0.5 + (best_i as f64) * bin_width)
}

#[pyfunction]
#[pyo3(signature = (pitch, mag, resolution = 0.01, bins_per_octave = 12))]
pub fn estimate_tuning_from_piptrack_f32(
    pitch: PyReadonlyArrayDyn<'_, f32>,
    mag: PyReadonlyArrayDyn<'_, f32>,
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64> {
    match resolved_rust_device() {
        RustDevice::Cpu => estimate_tuning_from_piptrack_f32_cpu(pitch, mag, resolution, bins_per_octave),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => estimate_tuning_from_piptrack_f32_cpu(pitch, mag, resolution, bins_per_octave),
        RustDevice::Auto => estimate_tuning_from_piptrack_f32_cpu(pitch, mag, resolution, bins_per_octave),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => estimate_tuning_from_piptrack_f32_cpu(pitch, mag, resolution, bins_per_octave),
    }
}

fn estimate_tuning_from_piptrack_f32_cpu(
    pitch: PyReadonlyArrayDyn<'_, f32>,
    mag: PyReadonlyArrayDyn<'_, f32>,
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64> {
    let p = pitch.as_array();
    let m = mag.as_array();

    if p.shape() != m.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pitch and mag must have the same shape",
        ));
    }

    if let (Some(ps), Some(ms)) = (p.as_slice_memory_order(), m.as_slice_memory_order()) {
        return histogram_tuning_from_slices_f32(ps, ms, resolution, bins_per_octave);
    }

    let iter = p.iter().zip(m.iter()).map(|(&pp, &mm)| (f64::from(pp), f64::from(mm)));
    estimate_tuning_impl(iter, resolution, bins_per_octave)
}

#[pyfunction]
#[pyo3(signature = (pitch, mag, resolution = 0.01, bins_per_octave = 12))]
pub fn estimate_tuning_from_piptrack_f64(
    pitch: PyReadonlyArrayDyn<'_, f64>,
    mag: PyReadonlyArrayDyn<'_, f64>,
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64> {
    match resolved_rust_device() {
        RustDevice::Cpu => estimate_tuning_from_piptrack_f64_cpu(pitch, mag, resolution, bins_per_octave),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => estimate_tuning_from_piptrack_f64_cpu(pitch, mag, resolution, bins_per_octave),
        RustDevice::Auto => estimate_tuning_from_piptrack_f64_cpu(pitch, mag, resolution, bins_per_octave),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => estimate_tuning_from_piptrack_f64_cpu(pitch, mag, resolution, bins_per_octave),
    }
}

fn estimate_tuning_from_piptrack_f64_cpu(
    pitch: PyReadonlyArrayDyn<'_, f64>,
    mag: PyReadonlyArrayDyn<'_, f64>,
    resolution: f64,
    bins_per_octave: i32,
) -> PyResult<f64> {
    let p = pitch.as_array();
    let m = mag.as_array();

    if p.shape() != m.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pitch and mag must have the same shape",
        ));
    }

    if let (Some(ps), Some(ms)) = (p.as_slice_memory_order(), m.as_slice_memory_order()) {
        return histogram_tuning_from_slices_f64(ps, ms, resolution, bins_per_octave);
    }

    let iter = p.iter().zip(m.iter()).map(|(&pp, &mm)| (pp, mm));
    estimate_tuning_impl(iter, resolution, bins_per_octave)
}

#[pyfunction]
#[pyo3(signature = (s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft))]
pub fn piptrack_from_spectrogram_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    shift: PyReadonlyArray2<'py, f32>,
    dskew: PyReadonlyArray2<'py, f32>,
    ref_values: PyReadonlyArray1<'py, f32>,
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    match resolved_rust_device() {
        RustDevice::Cpu => piptrack_from_spectrogram_f32_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
        RustDevice::AppleGpu => piptrack_from_spectrogram_f32_apple_gpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
        RustDevice::Auto => piptrack_from_spectrogram_f32_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => piptrack_from_spectrogram_f32_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
    }
}

fn piptrack_from_spectrogram_f32_apple_gpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    shift: PyReadonlyArray2<'py, f32>,
    dskew: PyReadonlyArray2<'py, f32>,
    ref_values: PyReadonlyArray1<'py, f32>,
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    #[cfg(all(feature = "apple-gpu", target_os = "macos"))]
    {
        let s_view = s.as_array();
        let shift_view = shift.as_array();
        let dskew_view = dskew.as_array();
        let ref_slice = ref_values.as_slice()?;
        let shape = s_view.shape();

        if shift_view.shape() != shape || dskew_view.shape() != shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "s, shift, and dskew must have the same shape",
            ));
        }
        if ref_slice.len() != shape[1] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ref_values length must match frame count",
            ));
        }
        if sr_over_nfft <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sr_over_nfft must be positive",
            ));
        }

        if let Some((pitch, mag)) = try_piptrack_from_spectrogram_f32_apple_gpu(
            &s_view,
            &shift_view,
            &dskew_view,
            ref_slice,
            start_bin,
            stop_bin,
            sr_over_nfft,
        ) {
            return Ok((
                pitch.into_pyarray_bound(py).to_owned(),
                mag.into_pyarray_bound(py).to_owned(),
            ));
        }
    }

    // Safe fallback policy: any unavailable/failed GPU path returns CPU output.
    piptrack_from_spectrogram_f32_cpu(
        py,
        s,
        shift,
        dskew,
        ref_values,
        start_bin,
        stop_bin,
        sr_over_nfft,
    )
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn try_piptrack_from_spectrogram_f32_apple_gpu(
    s_view: &ndarray::ArrayView2<'_, f32>,
    shift_view: &ndarray::ArrayView2<'_, f32>,
    dskew_view: &ndarray::ArrayView2<'_, f32>,
    ref_slice: &[f32],
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f32,
) -> Option<(Array2<f32>, Array2<f32>)> {
    if !(s_view.is_standard_layout() && shift_view.is_standard_layout() && dskew_view.is_standard_layout()) {
        return None;
    }

    let n_bins = s_view.shape()[0];
    let n_frames = s_view.shape()[1];

    let start = start_bin.max(1).min(n_bins.saturating_sub(1));
    let stop = stop_bin.min(n_bins.saturating_sub(1));
    if start >= stop {
        return None;
    }

    let work = (stop - start).saturating_mul(n_frames);
    if work < piptrack_gpu_work_threshold() {
        return None;
    }

    let out_len = n_bins.checked_mul(n_frames)?;

    let s_slice = s_view.as_slice()?;
    let shift_slice = shift_view.as_slice()?;
    let dskew_slice = dskew_view.as_slice()?;

    let s_bytes = std::mem::size_of_val(s_slice);
    let shift_bytes = std::mem::size_of_val(shift_slice);
    let dskew_bytes = std::mem::size_of_val(dskew_slice);
    let ref_bytes = std::mem::size_of_val(ref_slice);
    let out_bytes = out_len * std::mem::size_of::<f32>();

    let params_u32 = [n_bins as u32, n_frames as u32, start_bin as u32, stop_bin as u32];
    let params_f32 = [sr_over_nfft];

    PIPTRACK_F32_CONTEXT.with(|slot| {
        let mut ctx_ref = slot.borrow_mut();
        let ctx = ctx_ref.as_mut()?;

        if s_bytes > ctx.s_cap_bytes {
            ctx.s_buf = ctx.device.new_buffer(s_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.s_cap_bytes = s_bytes;
        }
        if shift_bytes > ctx.shift_cap_bytes {
            ctx.shift_buf = ctx.device.new_buffer(shift_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.shift_cap_bytes = shift_bytes;
        }
        if dskew_bytes > ctx.dskew_cap_bytes {
            ctx.dskew_buf = ctx.device.new_buffer(dskew_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.dskew_cap_bytes = dskew_bytes;
        }
        if ref_bytes > ctx.ref_cap_bytes {
            ctx.ref_buf = ctx.device.new_buffer(ref_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.ref_cap_bytes = ref_bytes;
        }
        if out_bytes > ctx.out_cap_bytes {
            ctx.pitch_buf = ctx.device.new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.mag_buf = ctx.device.new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared);
            ctx.out_cap_bytes = out_bytes;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(s_slice.as_ptr(), ctx.s_buf.contents().cast::<f32>(), s_slice.len());
            std::ptr::copy_nonoverlapping(shift_slice.as_ptr(), ctx.shift_buf.contents().cast::<f32>(), shift_slice.len());
            std::ptr::copy_nonoverlapping(dskew_slice.as_ptr(), ctx.dskew_buf.contents().cast::<f32>(), dskew_slice.len());
            std::ptr::copy_nonoverlapping(ref_slice.as_ptr(), ctx.ref_buf.contents().cast::<f32>(), ref_slice.len());
            std::ptr::copy_nonoverlapping(params_u32.as_ptr(), ctx.params_u32_buf.contents().cast::<u32>(), params_u32.len());
            std::ptr::copy_nonoverlapping(params_f32.as_ptr(), ctx.params_f32_buf.contents().cast::<f32>(), params_f32.len());
        }

        let command_buffer = ctx.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&ctx.pipeline);
        encoder.set_buffer(0, Some(&ctx.s_buf), 0);
        encoder.set_buffer(1, Some(&ctx.shift_buf), 0);
        encoder.set_buffer(2, Some(&ctx.dskew_buf), 0);
        encoder.set_buffer(3, Some(&ctx.ref_buf), 0);
        encoder.set_buffer(4, Some(&ctx.pitch_buf), 0);
        encoder.set_buffer(5, Some(&ctx.mag_buf), 0);
        encoder.set_buffer(6, Some(&ctx.params_u32_buf), 0);
        encoder.set_buffer(7, Some(&ctx.params_f32_buf), 0);

        let threadgroups = MTLSize {
            width: (n_frames as u64 + PIPTRACK_TILE_SIZE - 1) / PIPTRACK_TILE_SIZE,
            height: (n_bins as u64 + PIPTRACK_TILE_SIZE - 1) / PIPTRACK_TILE_SIZE,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: PIPTRACK_TILE_SIZE,
            height: PIPTRACK_TILE_SIZE,
            depth: 1,
        };

        encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return None;
        }

        let pitch_slice = unsafe {
            std::slice::from_raw_parts(ctx.pitch_buf.contents().cast::<f32>(), out_len)
        };
        let mag_slice = unsafe {
            std::slice::from_raw_parts(ctx.mag_buf.contents().cast::<f32>(), out_len)
        };

        let pitch = Array2::from_shape_vec((n_bins, n_frames), pitch_slice.to_vec()).ok()?;
        let mag = Array2::from_shape_vec((n_bins, n_frames), mag_slice.to_vec()).ok()?;
        Some((pitch, mag))
    })
}

fn piptrack_from_spectrogram_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    shift: PyReadonlyArray2<'py, f32>,
    dskew: PyReadonlyArray2<'py, f32>,
    ref_values: PyReadonlyArray1<'py, f32>,
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let s_view = s.as_array();
    let shift_view = shift.as_array();
    let dskew_view = dskew.as_array();
    let ref_slice = ref_values.as_slice()?;
    let shape = s_view.shape();

    if shift_view.shape() != shape || dskew_view.shape() != shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s, shift, and dskew must have the same shape",
        ));
    }
    if ref_slice.len() != shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ref_values length must match frame count",
        ));
    }
    if sr_over_nfft <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sr_over_nfft must be positive",
        ));
    }

    let (pitch_data, mag_data) = piptrack_from_spectrogram_core_f32(
        s_view, shift_view, dskew_view, ref_slice, start_bin, stop_bin, sr_over_nfft,
    );

    let pitch = Array2::from_shape_vec((shape[0], shape[1]), pitch_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mag = Array2::from_shape_vec((shape[0], shape[1]), mag_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        pitch.into_pyarray_bound(py).to_owned(),
        mag.into_pyarray_bound(py).to_owned(),
    ))
}

#[pyfunction]
#[pyo3(signature = (s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft))]
pub fn piptrack_from_spectrogram_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    shift: PyReadonlyArray2<'py, f64>,
    dskew: PyReadonlyArray2<'py, f64>,
    ref_values: PyReadonlyArray1<'py, f64>,
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    match resolved_rust_device() {
        RustDevice::Cpu => piptrack_from_spectrogram_f64_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => piptrack_from_spectrogram_f64_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
        RustDevice::Auto => piptrack_from_spectrogram_f64_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => piptrack_from_spectrogram_f64_cpu(py, s, shift, dskew, ref_values, start_bin, stop_bin, sr_over_nfft),
    }
}

fn piptrack_from_spectrogram_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    shift: PyReadonlyArray2<'py, f64>,
    dskew: PyReadonlyArray2<'py, f64>,
    ref_values: PyReadonlyArray1<'py, f64>,
    start_bin: usize,
    stop_bin: usize,
    sr_over_nfft: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let s_view = s.as_array();
    let shift_view = shift.as_array();
    let dskew_view = dskew.as_array();
    let ref_slice = ref_values.as_slice()?;
    let shape = s_view.shape();

    if shift_view.shape() != shape || dskew_view.shape() != shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "s, shift, and dskew must have the same shape",
        ));
    }
    if ref_slice.len() != shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ref_values length must match frame count",
        ));
    }
    if sr_over_nfft <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sr_over_nfft must be positive",
        ));
    }

    let (pitch_data, mag_data) = piptrack_from_spectrogram_core_f64(
        s_view, shift_view, dskew_view, ref_slice, start_bin, stop_bin, sr_over_nfft,
    );

    let pitch = Array2::from_shape_vec((shape[0], shape[1]), pitch_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let mag = Array2::from_shape_vec((shape[0], shape[1]), mag_data)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok((
        pitch.into_pyarray_bound(py).to_owned(),
        mag.into_pyarray_bound(py).to_owned(),
    ))
}

