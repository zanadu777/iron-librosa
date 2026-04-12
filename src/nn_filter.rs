// Rust implementation of librosa.decompose.nn_filter (mean and weighted average)
// Accepts data array S, recurrence matrix in CSC/CSR format (data, indices, indptr),
// and aggregation type.
//
// Parallelism strategy:
//   - Rows are processed in parallel via Rayon when the total work is large
//     enough to amortise the thread-pool overhead.
//   - Below PAR_THRESHOLD (n_frames × n_cols) we fall back to a sequential
//     loop.  In benchmarks the crossover sits around ~10 000 elements; we use
//     16 000 to stay comfortably on the right side.

use ndarray::parallel::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::backend::{resolved_rust_device, RustDevice};

/// Minimum total-element count (n_frames × n_cols) at which Rayon parallelism
/// is beneficial.  Below this threshold the sequential path is faster because
/// thread-pool overhead dominates.
const PAR_THRESHOLD: usize = 16_000;

#[pyfunction]
#[pyo3(signature = (s, rec_data, rec_indices, rec_indptr, weighted = false))]
pub fn nn_filter<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    rec_data: PyReadonlyArray1<'py, f64>,
    rec_indices: PyReadonlyArray1<'py, i32>,
    rec_indptr: PyReadonlyArray1<'py, i32>,
    weighted: bool,
) -> Bound<'py, PyArray2<f64>> {
    match resolved_rust_device() {
        RustDevice::Cpu => nn_filter_cpu(py, s, rec_data, rec_indices, rec_indptr, weighted),
        // GPU stub: fallback to CPU until Metal kernel is implemented.
        RustDevice::AppleGpu => nn_filter_cpu(py, s, rec_data, rec_indices, rec_indptr, weighted),
        RustDevice::Auto => nn_filter_cpu(py, s, rec_data, rec_indices, rec_indptr, weighted),
        // Phase 21 stub: CUDA not yet implemented; route to CPU.
        RustDevice::CudaGpu => nn_filter_cpu(py, s, rec_data, rec_indices, rec_indptr, weighted),
    }
}

fn nn_filter_cpu<'py>(
    py: Python<'py>,
    s: PyReadonlyArray2<'py, f64>,
    rec_data: PyReadonlyArray1<'py, f64>,
    rec_indices: PyReadonlyArray1<'py, i32>,
    rec_indptr: PyReadonlyArray1<'py, i32>,
    weighted: bool,
) -> Bound<'py, PyArray2<f64>> {
    let s = s.as_array();
    let rec_data = rec_data.as_slice().unwrap();
    let rec_indices = rec_indices.as_slice().unwrap();
    let rec_indptr = rec_indptr.as_slice().unwrap();

    let n_frames = rec_indptr.len() - 1;
    let n_cols = s.shape()[1];

    let mut s_out = ndarray::Array2::<f64>::zeros((n_frames, n_cols));

    // Shared inner logic for a single output row.
    let process = |i: usize, mut row_out: ndarray::ArrayViewMut1<f64>| {
        let start = rec_indptr[i] as usize;
        let end = rec_indptr[i + 1] as usize;
        let targets = &rec_indices[start..end];
        let n_neighbors = targets.len();

        // No neighbours → copy the original frame unchanged.
        if n_neighbors == 0 {
            row_out.assign(&s.row(i));
            return;
        }

        if weighted {
            let weights = &rec_data[start..end];
            let sum_weights: f64 = weights.iter().sum();
            if sum_weights <= 0.0 {
                row_out.assign(&s.row(i));
                return;
            }
            for (k, &idx) in targets.iter().enumerate() {
                let w = weights[k];
                let src = s.row(idx as usize);
                for (o, &v) in row_out.iter_mut().zip(src.iter()) {
                    *o += v * w;
                }
            }
            let inv = 1.0 / sum_weights;
            row_out.iter_mut().for_each(|v| *v *= inv);
        } else {
            for &idx in targets {
                let src = s.row(idx as usize);
                for (o, &v) in row_out.iter_mut().zip(src.iter()) {
                    *o += v;
                }
            }
            let inv = 1.0 / n_neighbors as f64;
            row_out.iter_mut().for_each(|v| *v *= inv);
        }
    };

    if n_frames * n_cols >= PAR_THRESHOLD {
        // Large workload: process rows in parallel.
        s_out
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, row_out)| process(i, row_out));
    } else {
        // Small workload: sequential — avoids Rayon thread-pool overhead.
        s_out
            .axis_iter_mut(ndarray::Axis(0))
            .enumerate()
            .for_each(|(i, row_out)| process(i, row_out));
    }

    s_out.into_pyarray_bound(py)
}
