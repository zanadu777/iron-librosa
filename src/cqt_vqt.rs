use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::num_complex::Complex;

fn validate_shapes_f32(
    d: &ArrayView3<'_, Complex<f32>>,
    fft_basis: &ArrayView2<'_, Complex<f32>>,
) -> PyResult<(usize, usize, usize)> {
    let batch = d.shape()[0];
    let n_fft_bins = d.shape()[1];
    let n_frames = d.shape()[2];
    let n_bins = fft_basis.shape()[0];

    if fft_basis.shape()[1] != n_fft_bins {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: D is ({batch}, {n_fft_bins}, {n_frames}), fft_basis is ({n_bins}, {}). Expected fft_basis.shape[1] == D.shape[1].",
            fft_basis.shape()[1]
        )));
    }

    Ok((batch, n_bins, n_frames))
}

fn validate_shapes_f64(
    d: &ArrayView3<'_, Complex<f64>>,
    fft_basis: &ArrayView2<'_, Complex<f64>>,
) -> PyResult<(usize, usize, usize)> {
    let batch = d.shape()[0];
    let n_fft_bins = d.shape()[1];
    let n_frames = d.shape()[2];
    let n_bins = fft_basis.shape()[0];

    if fft_basis.shape()[1] != n_fft_bins {
        return Err(PyValueError::new_err(format!(
            "Incompatible shapes: D is ({batch}, {n_fft_bins}, {n_frames}), fft_basis is ({n_bins}, {}). Expected fft_basis.shape[1] == D.shape[1].",
            fft_basis.shape()[1]
        )));
    }

    Ok((batch, n_bins, n_frames))
}

fn project_f32(
    d: ArrayView3<'_, Complex<f32>>,
    fft_basis: ArrayView2<'_, Complex<f32>>,
) -> PyResult<Array3<Complex<f32>>> {
    let (batch, n_bins, n_frames) = validate_shapes_f32(&d, &fft_basis)?;

    let mut out = Array3::<Complex<f32>>::zeros((batch, n_bins, n_frames));

    if batch < 4 {
        for batch_idx in 0..batch {
            let plane = fft_basis.dot(&d.index_axis(Axis(0), batch_idx));
            out.index_axis_mut(Axis(0), batch_idx).assign(&plane);
        }

        return Ok(out);
    }

    let planes: Vec<Array2<Complex<f32>>> = (0..batch)
        .into_par_iter()
        .map(|batch_idx| fft_basis.dot(&d.index_axis(Axis(0), batch_idx)))
        .collect();

    for (batch_idx, plane) in planes.into_iter().enumerate() {
        out.index_axis_mut(Axis(0), batch_idx).assign(&plane);
    }

    Ok(out)
}

fn project_f64(
    d: ArrayView3<'_, Complex<f64>>,
    fft_basis: ArrayView2<'_, Complex<f64>>,
) -> PyResult<Array3<Complex<f64>>> {
    let (batch, n_bins, n_frames) = validate_shapes_f64(&d, &fft_basis)?;

    let mut out = Array3::<Complex<f64>>::zeros((batch, n_bins, n_frames));

    if batch < 4 {
        for batch_idx in 0..batch {
            let plane = fft_basis.dot(&d.index_axis(Axis(0), batch_idx));
            out.index_axis_mut(Axis(0), batch_idx).assign(&plane);
        }

        return Ok(out);
    }

    let planes: Vec<Array2<Complex<f64>>> = (0..batch)
        .into_par_iter()
        .map(|batch_idx| fft_basis.dot(&d.index_axis(Axis(0), batch_idx)))
        .collect();

    for (batch_idx, plane) in planes.into_iter().enumerate() {
        out.index_axis_mut(Axis(0), batch_idx).assign(&plane);
    }

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (d, fft_basis))]
pub fn cqt_project_f32<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, Complex<f32>>,
    fft_basis: PyReadonlyArray2<'py, Complex<f32>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f32>>>> {
    let out = project_f32(d.as_array(), fft_basis.as_array())?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (d, fft_basis))]
pub fn cqt_project_f64<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, Complex<f64>>,
    fft_basis: PyReadonlyArray2<'py, Complex<f64>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f64>>>> {
    let out = project_f64(d.as_array(), fft_basis.as_array())?;
    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (d, fft_basis))]
pub fn vqt_project_f32<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, Complex<f32>>,
    fft_basis: PyReadonlyArray2<'py, Complex<f32>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f32>>>> {
    cqt_project_f32(py, d, fft_basis)
}

#[pyfunction]
#[pyo3(signature = (d, fft_basis))]
pub fn vqt_project_f64<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, Complex<f64>>,
    fft_basis: PyReadonlyArray2<'py, Complex<f64>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f64>>>> {
    cqt_project_f64(py, d, fft_basis)
}


