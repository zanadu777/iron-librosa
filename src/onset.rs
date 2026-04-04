use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

