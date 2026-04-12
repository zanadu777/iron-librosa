use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, Axis};
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustfft::num_complex::Complex;

use crate::backend::{resolved_rust_device, RustDevice};

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

#[inline]
unsafe fn as_col_major_t<'a, T>(arr: &'a ArrayView2<'_, T>) -> MatRef<'a, T> {
    let nrows = arr.shape()[0];
    let ncols = arr.shape()[1];
    let row_s = arr.strides()[0];
    let col_s = arr.strides()[1];
    MatRef::from_raw_parts(arr.as_ptr(), ncols, nrows, col_s, row_s)
}

fn project_plane_f32_into(
    fft_basis: ArrayView2<'_, Complex<f32>>,
    d_plane: ArrayView2<'_, Complex<f32>>,
    mut out_plane: ArrayViewMut2<'_, Complex<f32>>,
) {
    let n_frames = out_plane.shape()[1];
    let n_bins = out_plane.shape()[0];

    // SAFETY: inputs are contiguous and out is newly allocated.
    unsafe {
        let out_t = MatMut::<Complex<f32>>::from_raw_parts_mut(
            out_plane.as_mut_ptr(),
            n_frames,
            n_bins,
            out_plane.strides()[1],
            out_plane.strides()[0],
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&d_plane),
            as_col_major_t(&fft_basis),
            Complex::new(1.0f32, 0.0f32),
            Par::Seq,
        );
    }
}

fn project_plane_f64_into(
    fft_basis: ArrayView2<'_, Complex<f64>>,
    d_plane: ArrayView2<'_, Complex<f64>>,
    mut out_plane: ArrayViewMut2<'_, Complex<f64>>,
) {
    let n_frames = out_plane.shape()[1];
    let n_bins = out_plane.shape()[0];

    // SAFETY: inputs are contiguous and out is newly allocated.
    unsafe {
        let out_t = MatMut::<Complex<f64>>::from_raw_parts_mut(
            out_plane.as_mut_ptr(),
            n_frames,
            n_bins,
            out_plane.strides()[1],
            out_plane.strides()[0],
        );
        matmul(
            out_t,
            Accum::Replace,
            as_col_major_t(&d_plane),
            as_col_major_t(&fft_basis),
            Complex::new(1.0f64, 0.0f64),
            Par::Seq,
        );
    }
}

fn project_plane_f32_ndarray(
    fft_basis: ArrayView2<'_, Complex<f32>>,
    d_plane: ArrayView2<'_, Complex<f32>>,
) -> Array2<Complex<f32>> {
    fft_basis.dot(&d_plane)
}

fn project_plane_f64_ndarray(
    fft_basis: ArrayView2<'_, Complex<f64>>,
    d_plane: ArrayView2<'_, Complex<f64>>,
) -> Array2<Complex<f64>> {
    fft_basis.dot(&d_plane)
}

fn project_f32_cpu(
    d: ArrayView3<'_, Complex<f32>>,
    fft_basis: ArrayView2<'_, Complex<f32>>,
) -> PyResult<Array3<Complex<f32>>> {
    let (batch, n_bins, n_frames) = validate_shapes_f32(&d, &fft_basis)?;

    let mut out = Array3::<Complex<f32>>::zeros((batch, n_bins, n_frames));

    // Heuristic: faer performs best for mono projection on this host.
    // For multichannel projection, ndarray::dot is typically faster.
    if batch == 1 {
        for batch_idx in 0..batch {
            project_plane_f32_into(
                fft_basis,
                d.index_axis(Axis(0), batch_idx),
                out.index_axis_mut(Axis(0), batch_idx),
            );
        }

        return Ok(out);
    }

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch_idx, mut out_plane)| {
            let plane = project_plane_f32_ndarray(fft_basis, d.index_axis(Axis(0), batch_idx));
            out_plane.assign(&plane);
        });

    Ok(out)
}

fn project_f32_apple_gpu(
    d: ArrayView3<'_, Complex<f32>>,
    fft_basis: ArrayView2<'_, Complex<f32>>,
) -> PyResult<Array3<Complex<f32>>> {
    #[cfg(all(feature = "apple-gpu", target_os = "macos"))]
    {
        if let Some(out) = try_project_f32_apple_gpu(d, fft_basis) {
            return Ok(out);
        }
    }

    // Safe fallback policy: any unavailable/failed GPU path returns CPU output.
    project_f32_cpu(d, fft_basis)
}

#[cfg(all(feature = "apple-gpu", target_os = "macos"))]
fn try_project_f32_apple_gpu(
    d: ArrayView3<'_, Complex<f32>>,
    fft_basis: ArrayView2<'_, Complex<f32>>,
) -> Option<Array3<Complex<f32>>> {
    let (batch, n_bins, n_frames) = validate_shapes_f32(&d, &fft_basis).ok()?;

    let basis_re = fft_basis.mapv(|v| v.re);
    let basis_im = fft_basis.mapv(|v| v.im);
    let basis_re_v = basis_re.view();
    let basis_im_v = basis_im.view();

    let mut out = Array3::<Complex<f32>>::zeros((batch, n_bins, n_frames));

    for batch_idx in 0..batch {
        let d_plane = d.index_axis(Axis(0), batch_idx);
        let d_re = d_plane.mapv(|v| v.re);
        let d_im = d_plane.mapv(|v| v.im);

        let br_dr = crate::mel::try_project_f32_apple_gpu(&d_re.view(), &basis_re_v)?;
        let bi_di = crate::mel::try_project_f32_apple_gpu(&d_im.view(), &basis_im_v)?;
        let br_di = crate::mel::try_project_f32_apple_gpu(&d_im.view(), &basis_re_v)?;
        let bi_dr = crate::mel::try_project_f32_apple_gpu(&d_re.view(), &basis_im_v)?;

        let mut out_plane = out.index_axis_mut(Axis(0), batch_idx);
        for bin_idx in 0..n_bins {
            for frame_idx in 0..n_frames {
                let re = br_dr[(bin_idx, frame_idx)] - bi_di[(bin_idx, frame_idx)];
                let im = br_di[(bin_idx, frame_idx)] + bi_dr[(bin_idx, frame_idx)];
                out_plane[(bin_idx, frame_idx)] = Complex::new(re, im);
            }
        }
    }

    Some(out)
}

fn project_f64_cpu(
    d: ArrayView3<'_, Complex<f64>>,
    fft_basis: ArrayView2<'_, Complex<f64>>,
) -> PyResult<Array3<Complex<f64>>> {
    let (batch, n_bins, n_frames) = validate_shapes_f64(&d, &fft_basis)?;

    let mut out = Array3::<Complex<f64>>::zeros((batch, n_bins, n_frames));

    // Heuristic: faer performs best for mono projection on this host.
    // For multichannel projection, ndarray::dot is typically faster.
    if batch == 1 {
        for batch_idx in 0..batch {
            project_plane_f64_into(
                fft_basis,
                d.index_axis(Axis(0), batch_idx),
                out.index_axis_mut(Axis(0), batch_idx),
            );
        }

        return Ok(out);
    }

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(batch_idx, mut out_plane)| {
            let plane = project_plane_f64_ndarray(fft_basis, d.index_axis(Axis(0), batch_idx));
            out_plane.assign(&plane);
        });

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (d, fft_basis))]
pub fn cqt_project_f32<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, Complex<f32>>,
    fft_basis: PyReadonlyArray2<'py, Complex<f32>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f32>>>> {
    let out = match resolved_rust_device() {
        RustDevice::Cpu => project_f32_cpu(d.as_array(), fft_basis.as_array())?,
        RustDevice::AppleGpu => project_f32_apple_gpu(d.as_array(), fft_basis.as_array())?,
        RustDevice::Auto => project_f32_cpu(d.as_array(), fft_basis.as_array())?,
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => project_f32_cpu(d.as_array(), fft_basis.as_array())?,
    };
    Ok(out.into_pyarray_bound(py).to_owned())
}

#[pyfunction]
#[pyo3(signature = (d, fft_basis))]
pub fn cqt_project_f64<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, Complex<f64>>,
    fft_basis: PyReadonlyArray2<'py, Complex<f64>>,
) -> PyResult<Bound<'py, PyArray3<Complex<f64>>>> {
    let out = match resolved_rust_device() {
        RustDevice::Cpu => project_f64_cpu(d.as_array(), fft_basis.as_array())?,
        // Phase 1 scaffold: keep CPU path while Apple GPU kernels are pending.
        RustDevice::AppleGpu => project_f64_cpu(d.as_array(), fft_basis.as_array())?,
        RustDevice::Auto => project_f64_cpu(d.as_array(), fft_basis.as_array())?,
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => project_f64_cpu(d.as_array(), fft_basis.as_array())?,
    };
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


