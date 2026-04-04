// Tuning-estimation helper kernels for iron-librosa.
//
// These kernels accelerate the post-processing stage of librosa.estimate_tuning:
// thresholding piptrack output and running pitch_tuning histogram voting.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 200_000;

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
        s_view,
        shift_view,
        dskew_view,
        ref_slice,
        start_bin,
        stop_bin,
        sr_over_nfft,
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
        s_view,
        shift_view,
        dskew_view,
        ref_slice,
        start_bin,
        stop_bin,
        sr_over_nfft,
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

