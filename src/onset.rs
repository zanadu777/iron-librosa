use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::backend::{resolved_rust_device, RustDevice};

fn reflect_index(mut idx: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n_i = n as isize;
    while idx < 0 || idx >= n_i {
        if idx < 0 {
            idx = -idx - 1;
        }
        if idx >= n_i {
            idx = 2 * n_i - idx - 1;
        }
    }
    idx as usize
}

fn onset_flux_mean_ref_impl(
    s: ndarray::ArrayView2<'_, f32>,
    ref_s: ndarray::ArrayView2<'_, f32>,
    lag: usize,
) -> PyResult<ndarray::Array2<f32>> {
    if lag == 0 {
        return Err(PyValueError::new_err("lag must be a positive integer"));
    }

    if s.shape() != ref_s.shape() {
        return Err(PyValueError::new_err(
            "reference spectrum must match input spectrum shape",
        ));
    }

    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let out_frames = n_frames.saturating_sub(lag);

    let mut out = ndarray::Array2::<f32>::zeros((1, out_frames));

    if n_bins == 0 || out_frames == 0 {
        return Ok(out);
    }

    let inv_bins = 1.0f32 / n_bins as f32;

    for t in 0..out_frames {
        let mut acc = 0.0f32;
        for f in 0..n_bins {
            let diff = s[(f, t + lag)] - ref_s[(f, t)];
            if diff > 0.0 {
                acc += diff;
            }
        }
        out[(0, t)] = acc * inv_bins;
    }

    Ok(out)
}

fn onset_flux_mean_ref_impl_f64(
    s: ndarray::ArrayView2<'_, f64>,
    ref_s: ndarray::ArrayView2<'_, f64>,
    lag: usize,
) -> PyResult<ndarray::Array2<f64>> {
    if lag == 0 {
        return Err(PyValueError::new_err("lag must be a positive integer"));
    }

    if s.shape() != ref_s.shape() {
        return Err(PyValueError::new_err(
            "reference spectrum must match input spectrum shape",
        ));
    }

    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let out_frames = n_frames.saturating_sub(lag);

    let mut out = ndarray::Array2::<f64>::zeros((1, out_frames));

    if n_bins == 0 || out_frames == 0 {
        return Ok(out);
    }

    let inv_bins = 1.0f64 / n_bins as f64;

    for t in 0..out_frames {
        let mut acc = 0.0f64;
        for f in 0..n_bins {
            let diff = s[(f, t + lag)] - ref_s[(f, t)];
            if diff > 0.0 {
                acc += diff;
            }
        }
        out[(0, t)] = acc * inv_bins;
    }

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (s, lag = 1))]
pub fn onset_flux_mean_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_mean_f32_cpu(py, s, lag),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_mean_f32_cpu(py, s, lag),
        RustDevice::Auto => onset_flux_mean_f32_cpu(py, s, lag),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_mean_f32_cpu(py, s, lag),
    }
}

fn onset_flux_mean_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let s = s.as_array();
    let out = onset_flux_mean_ref_impl(s, s, lag)?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, ref_s, lag = 1))]
pub fn onset_flux_mean_ref_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    ref_s: PyReadonlyArray2<'py, f32>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_mean_ref_f32_cpu(py, s, ref_s, lag),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_mean_ref_f32_cpu(py, s, ref_s, lag),
        RustDevice::Auto => onset_flux_mean_ref_f32_cpu(py, s, ref_s, lag),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_mean_ref_f32_cpu(py, s, ref_s, lag),
    }
}

fn onset_flux_mean_ref_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    ref_s: PyReadonlyArray2<'py, f32>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let out = onset_flux_mean_ref_impl(s.as_array(), ref_s.as_array(), lag)?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, lag = 1))]
pub fn onset_flux_mean_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_mean_f64_cpu(py, s, lag),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_mean_f64_cpu(py, s, lag),
        RustDevice::Auto => onset_flux_mean_f64_cpu(py, s, lag),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_mean_f64_cpu(py, s, lag),
    }
}

fn onset_flux_mean_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let s = s.as_array();
    let out = onset_flux_mean_ref_impl_f64(s, s, lag)?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, ref_s, lag = 1))]
pub fn onset_flux_mean_ref_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    ref_s: PyReadonlyArray2<'py, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_mean_ref_f64_cpu(py, s, ref_s, lag),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_mean_ref_f64_cpu(py, s, ref_s, lag),
        RustDevice::Auto => onset_flux_mean_ref_f64_cpu(py, s, ref_s, lag),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_mean_ref_f64_cpu(py, s, ref_s, lag),
    }
}

fn onset_flux_mean_ref_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    ref_s: PyReadonlyArray2<'py, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = onset_flux_mean_ref_impl_f64(s.as_array(), ref_s.as_array(), lag)?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, lag = 1, max_size = 3))]
pub fn onset_flux_mean_maxfilter_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    lag: usize,
    max_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_mean_maxfilter_f32_cpu(py, s, lag, max_size),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_mean_maxfilter_f32_cpu(py, s, lag, max_size),
        RustDevice::Auto => onset_flux_mean_maxfilter_f32_cpu(py, s, lag, max_size),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_mean_maxfilter_f32_cpu(py, s, lag, max_size),
    }
}

fn onset_flux_mean_maxfilter_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    lag: usize,
    max_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if lag == 0 {
        return Err(PyValueError::new_err("lag must be a positive integer"));
    }
    if max_size <= 1 || max_size % 2 == 0 {
        return Err(PyValueError::new_err(
            "max_size must be an odd integer greater than 1",
        ));
    }

    let s = s.as_array();
    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let out_frames = n_frames.saturating_sub(lag);
    let mut out = ndarray::Array2::<f32>::zeros((1, out_frames));

    if n_bins == 0 || out_frames == 0 {
        return Ok(out.into_pyarray_bound(py));
    }

    let inv_bins = 1.0f32 / n_bins as f32;
    let radius = (max_size / 2) as isize;

    for t in 0..out_frames {
        let mut acc = 0.0f32;
        for f in 0..n_bins {
            let mut local_max = f32::NEG_INFINITY;
            for k in 0..max_size {
                let idx = f as isize + k as isize - radius;
                let rf = reflect_index(idx, n_bins);
                let v = s[(rf, t)];
                if v > local_max {
                    local_max = v;
                }
            }

            let diff = s[(f, t + lag)] - local_max;
            if diff > 0.0 {
                acc += diff;
            }
        }
        out[(0, t)] = acc * inv_bins;
    }

    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, lag = 1, max_size = 3))]
pub fn onset_flux_mean_maxfilter_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    lag: usize,
    max_size: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_mean_maxfilter_f64_cpu(py, s, lag, max_size),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_mean_maxfilter_f64_cpu(py, s, lag, max_size),
        RustDevice::Auto => onset_flux_mean_maxfilter_f64_cpu(py, s, lag, max_size),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_mean_maxfilter_f64_cpu(py, s, lag, max_size),
    }
}

fn onset_flux_mean_maxfilter_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    lag: usize,
    max_size: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if lag == 0 {
        return Err(PyValueError::new_err("lag must be a positive integer"));
    }
    if max_size <= 1 || max_size % 2 == 0 {
        return Err(PyValueError::new_err(
            "max_size must be an odd integer greater than 1",
        ));
    }

    let s = s.as_array();
    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let out_frames = n_frames.saturating_sub(lag);
    let mut out = ndarray::Array2::<f64>::zeros((1, out_frames));

    if n_bins == 0 || out_frames == 0 {
        return Ok(out.into_pyarray_bound(py));
    }

    let inv_bins = 1.0f64 / n_bins as f64;
    let radius = (max_size / 2) as isize;

    for t in 0..out_frames {
        let mut acc = 0.0f64;
        for f in 0..n_bins {
            let mut local_max = f64::NEG_INFINITY;
            for k in 0..max_size {
                let idx = f as isize + k as isize - radius;
                let rf = reflect_index(idx, n_bins);
                let v = s[(rf, t)];
                if v > local_max {
                    local_max = v;
                }
            }

            let diff = s[(f, t + lag)] - local_max;
            if diff > 0.0 {
                acc += diff;
            }
        }
        out[(0, t)] = acc * inv_bins;
    }

    Ok(out.into_pyarray_bound(py))
}

// ── Median variants ──────────────────────────────────────────────────────────
// Mirror onset_flux_mean_ref_* but aggregate with median; rayon-parallel over frames.

fn onset_flux_median_ref_impl_f32(
    s: ndarray::ArrayView2<'_, f32>,
    ref_s: ndarray::ArrayView2<'_, f32>,
    lag: usize,
) -> PyResult<ndarray::Array2<f32>> {
    if lag == 0 {
        return Err(PyValueError::new_err("lag must be a positive integer"));
    }
    if s.shape() != ref_s.shape() {
        return Err(PyValueError::new_err(
            "reference spectrum shape must match input spectrum shape",
        ));
    }
    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let out_frames = n_frames.saturating_sub(lag);
    if n_bins == 0 || out_frames == 0 {
        return Ok(ndarray::Array2::<f32>::zeros((1, out_frames)));
    }
    let medians: Vec<f32> = (0..out_frames)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; n_bins],
            |flux, t| {
                if flux.len() != n_bins {
                    flux.resize(n_bins, 0.0);
                }
                for f in 0..n_bins {
                    let diff = s[(f, t + lag)] - ref_s[(f, t)];
                    flux[f] = if diff.is_nan() { f32::NAN } else { diff.max(0.0) };
                }

                if flux.iter().any(|v| v.is_nan()) {
                    return f32::NAN;
                }

                let cmp = |a: &f32, b: &f32| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                };
                let mid = n_bins / 2;

                if n_bins % 2 == 1 {
                    let (_, median, _) = flux.select_nth_unstable_by(mid, cmp);
                    *median
                } else {
                    let (_, lower, right) = flux.select_nth_unstable_by(mid - 1, cmp);
                    let upper = right
                        .iter()
                        .min_by(|a, b| cmp(a, b))
                        .copied()
                        .unwrap_or(*lower);
                    (*lower + upper) * 0.5
                }
            },
        )
        .collect();
    let mut out = ndarray::Array2::<f32>::zeros((1, out_frames));
    for (t, &v) in medians.iter().enumerate() {
        out[(0, t)] = v;
    }
    Ok(out)
}

fn onset_flux_median_ref_impl_f64(
    s: ndarray::ArrayView2<'_, f64>,
    ref_s: ndarray::ArrayView2<'_, f64>,
    lag: usize,
) -> PyResult<ndarray::Array2<f64>> {
    if lag == 0 {
        return Err(PyValueError::new_err("lag must be a positive integer"));
    }
    if s.shape() != ref_s.shape() {
        return Err(PyValueError::new_err(
            "reference spectrum shape must match input spectrum shape",
        ));
    }
    let n_bins = s.shape()[0];
    let n_frames = s.shape()[1];
    let out_frames = n_frames.saturating_sub(lag);
    if n_bins == 0 || out_frames == 0 {
        return Ok(ndarray::Array2::<f64>::zeros((1, out_frames)));
    }
    let medians: Vec<f64> = (0..out_frames)
        .into_par_iter()
        .map_init(
            || vec![0.0f64; n_bins],
            |flux, t| {
                if flux.len() != n_bins {
                    flux.resize(n_bins, 0.0);
                }
                for f in 0..n_bins {
                    let diff = s[(f, t + lag)] - ref_s[(f, t)];
                    flux[f] = if diff.is_nan() { f64::NAN } else { diff.max(0.0) };
                }

                if flux.iter().any(|v| v.is_nan()) {
                    return f64::NAN;
                }

                let cmp = |a: &f64, b: &f64| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                };
                let mid = n_bins / 2;

                if n_bins % 2 == 1 {
                    let (_, median, _) = flux.select_nth_unstable_by(mid, cmp);
                    *median
                } else {
                    let (_, lower, right) = flux.select_nth_unstable_by(mid - 1, cmp);
                    let upper = right
                        .iter()
                        .min_by(|a, b| cmp(a, b))
                        .copied()
                        .unwrap_or(*lower);
                    (*lower + upper) * 0.5
                }
            },
        )
        .collect();
    let mut out = ndarray::Array2::<f64>::zeros((1, out_frames));
    for (t, &v) in medians.iter().enumerate() {
        out[(0, t)] = v;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (s, ref_s, lag = 1))]
pub fn onset_flux_median_ref_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    ref_s: PyReadonlyArray2<'py, f32>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_median_ref_f32_cpu(py, s, ref_s, lag),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_median_ref_f32_cpu(py, s, ref_s, lag),
        RustDevice::Auto => onset_flux_median_ref_f32_cpu(py, s, ref_s, lag),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_median_ref_f32_cpu(py, s, ref_s, lag),
    }
}

fn onset_flux_median_ref_f32_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    ref_s: PyReadonlyArray2<'py, f32>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let out = onset_flux_median_ref_impl_f32(s.as_array(), ref_s.as_array(), lag)?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (s, ref_s, lag = 1))]
pub fn onset_flux_median_ref_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    ref_s: PyReadonlyArray2<'py, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    match resolved_rust_device() {
        RustDevice::Cpu => onset_flux_median_ref_f64_cpu(py, s, ref_s, lag),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => onset_flux_median_ref_f64_cpu(py, s, ref_s, lag),
        RustDevice::Auto => onset_flux_median_ref_f64_cpu(py, s, ref_s, lag),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => onset_flux_median_ref_f64_cpu(py, s, ref_s, lag),
    }
}

fn onset_flux_median_ref_f64_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    ref_s: PyReadonlyArray2<'py, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let out = onset_flux_median_ref_impl_f64(s.as_array(), ref_s.as_array(), lag)?;
    Ok(out.into_pyarray_bound(py))
}
