use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_shapes(n: usize, fpb_len: usize) -> PyResult<()> {
    if n == 0 {
        return Ok(());
    }
    if fpb_len != 1 && fpb_len != n {
        return Err(PyValueError::new_err(
            "frames_per_beat must have length 1 or match localscore length",
        ));
    }
    Ok(())
}

fn beat_track_dp_impl_f32(
    localscore: ndarray::ArrayView1<'_, f32>,
    frames_per_beat: ndarray::ArrayView1<'_, f32>,
    tightness: f32,
) -> PyResult<(Array1<i32>, Array1<f32>)> {
    if !tightness.is_finite() || tightness <= 0.0 {
        return Err(PyValueError::new_err("tightness must be a positive finite value"));
    }

    let n = localscore.len();
    validate_shapes(n, frames_per_beat.len())?;

    let mut backlink = Array1::<i32>::from_elem(n, -1);
    let mut cumscore = Array1::<f32>::zeros(n);

    if n == 0 {
        return Ok((backlink, cumscore));
    }

    let mut local_max = f32::NEG_INFINITY;
    for &v in localscore.iter() {
        if v > local_max {
            local_max = v;
        }
    }
    let score_thresh = 0.01f32 * local_max;

    let tv = frames_per_beat.len() > 1;
    let mut first_beat = true;

    for i in 0..n {
        let score_i = localscore[i];
        let fpb = if tv { frames_per_beat[i] } else { frames_per_beat[0] };

        if !fpb.is_finite() || fpb <= 0.0 {
            return Err(PyValueError::new_err(
                "frames_per_beat must contain positive finite values",
            ));
        }

        let start = i as isize - (fpb / 2.0).round() as isize;
        let stop_exclusive = i as isize - (2.0 * fpb + 1.0) as isize;

        let mut best_score = f32::NEG_INFINITY;
        let mut beat_location: i32 = -1;

        let mut loc = start;
        while loc > stop_exclusive {
            if loc < 0 {
                break;
            }
            let d = (i as isize - loc) as f32;
            if d > 0.0 {
                let penalty = tightness * (d.ln() - fpb.ln()).powi(2);
                let score = cumscore[loc as usize] - penalty;
                if score > best_score {
                    best_score = score;
                    beat_location = loc as i32;
                }
            }
            loc -= 1;
        }

        cumscore[i] = if beat_location >= 0 {
            score_i + best_score
        } else {
            score_i
        };

        if first_beat && score_i < score_thresh {
            backlink[i] = -1;
        } else {
            backlink[i] = beat_location;
            first_beat = false;
        }
    }

    Ok((backlink, cumscore))
}

fn beat_track_dp_impl_f64(
    localscore: ndarray::ArrayView1<'_, f64>,
    frames_per_beat: ndarray::ArrayView1<'_, f64>,
    tightness: f64,
) -> PyResult<(Array1<i32>, Array1<f64>)> {
    if !tightness.is_finite() || tightness <= 0.0 {
        return Err(PyValueError::new_err("tightness must be a positive finite value"));
    }

    let n = localscore.len();
    validate_shapes(n, frames_per_beat.len())?;

    let mut backlink = Array1::<i32>::from_elem(n, -1);
    let mut cumscore = Array1::<f64>::zeros(n);

    if n == 0 {
        return Ok((backlink, cumscore));
    }

    let mut local_max = f64::NEG_INFINITY;
    for &v in localscore.iter() {
        if v > local_max {
            local_max = v;
        }
    }
    let score_thresh = 0.01f64 * local_max;

    let tv = frames_per_beat.len() > 1;
    let mut first_beat = true;

    for i in 0..n {
        let score_i = localscore[i];
        let fpb = if tv { frames_per_beat[i] } else { frames_per_beat[0] };

        if !fpb.is_finite() || fpb <= 0.0 {
            return Err(PyValueError::new_err(
                "frames_per_beat must contain positive finite values",
            ));
        }

        let start = i as isize - (fpb / 2.0).round() as isize;
        let stop_exclusive = i as isize - (2.0 * fpb + 1.0) as isize;

        let mut best_score = f64::NEG_INFINITY;
        let mut beat_location: i32 = -1;

        let mut loc = start;
        while loc > stop_exclusive {
            if loc < 0 {
                break;
            }
            let d = (i as isize - loc) as f64;
            if d > 0.0 {
                let penalty = tightness * (d.ln() - fpb.ln()).powi(2);
                let score = cumscore[loc as usize] - penalty;
                if score > best_score {
                    best_score = score;
                    beat_location = loc as i32;
                }
            }
            loc -= 1;
        }

        cumscore[i] = if beat_location >= 0 {
            score_i + best_score
        } else {
            score_i
        };

        if first_beat && score_i < score_thresh {
            backlink[i] = -1;
        } else {
            backlink[i] = beat_location;
            first_beat = false;
        }
    }

    Ok((backlink, cumscore))
}

#[pyfunction]
pub fn beat_track_dp_f32<'py>(
    py: Python<'py>,
    localscore: PyReadonlyArray1<'py, f32>,
    frames_per_beat: PyReadonlyArray1<'py, f32>,
    tightness: f64,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
    let (backlink, cumscore) = beat_track_dp_impl_f32(
        localscore.as_array(),
        frames_per_beat.as_array(),
        tightness as f32,
    )?;
    Ok((
        backlink.into_pyarray_bound(py),
        cumscore.into_pyarray_bound(py),
    ))
}

#[pyfunction]
pub fn beat_track_dp_f64<'py>(
    py: Python<'py>,
    localscore: PyReadonlyArray1<'py, f64>,
    frames_per_beat: PyReadonlyArray1<'py, f64>,
    tightness: f64,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f64>>)> {
    let (backlink, cumscore) = beat_track_dp_impl_f64(
        localscore.as_array(),
        frames_per_beat.as_array(),
        tightness,
    )?;
    Ok((
        backlink.into_pyarray_bound(py),
        cumscore.into_pyarray_bound(py),
    ))
}
