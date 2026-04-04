use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// DCT-II with ortho normalization along axis 0 for a 2D (n_mels, n_frames) array.
/// Returns shape (min(n_out, n_mels), n_frames).
#[pyfunction]
#[pyo3(signature = (s, n_out))]
pub fn dct2_ortho_f32<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f32>,
    n_out: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = s.as_array();
    let n_mels = x.shape()[0];
    let n_frames = x.shape()[1];

    if n_mels == 0 {
        return Err(PyValueError::new_err("Input has zero mel bins"));
    }

    let k_max = n_out.min(n_mels);
    let n = n_mels as f32;
    let scale0 = (1.0f32 / n).sqrt();
    let scale = (2.0f32 / n).sqrt();

    let mut out = ndarray::Array2::<f32>::zeros((k_max, n_frames));

    for k in 0..k_max {
        let alpha = if k == 0 { scale0 } else { scale };
        for t in 0..n_frames {
            let mut acc = 0.0f32;
            for n_idx in 0..n_mels {
                let theta = std::f32::consts::PI * (n_idx as f32 + 0.5f32) * (k as f32) / n;
                acc += x[(n_idx, t)] * theta.cos();
            }
            out[(k, t)] = alpha * acc;
        }
    }

    Ok(out.into_pyarray_bound(py))
}

/// DCT-II with ortho normalization along axis 0 for a 2D (n_mels, n_frames) array.
/// Returns shape (min(n_out, n_mels), n_frames).
#[pyfunction]
#[pyo3(signature = (s, n_out))]
pub fn dct2_ortho_f64<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    n_out: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let x = s.as_array();
    let n_mels = x.shape()[0];
    let n_frames = x.shape()[1];

    if n_mels == 0 {
        return Err(PyValueError::new_err("Input has zero mel bins"));
    }

    let k_max = n_out.min(n_mels);
    let n = n_mels as f64;
    let scale0 = (1.0f64 / n).sqrt();
    let scale = (2.0f64 / n).sqrt();

    let mut out = ndarray::Array2::<f64>::zeros((k_max, n_frames));

    for k in 0..k_max {
        let alpha = if k == 0 { scale0 } else { scale };
        for t in 0..n_frames {
            let mut acc = 0.0f64;
            for n_idx in 0..n_mels {
                let theta = std::f64::consts::PI * (n_idx as f64 + 0.5f64) * (k as f64) / n;
                acc += x[(n_idx, t)] * theta.cos();
            }
            out[(k, t)] = alpha * acc;
        }
    }

    Ok(out.into_pyarray_bound(py))
}

