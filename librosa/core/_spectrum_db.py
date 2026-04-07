#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decibel and power scale conversion utilities"""
from __future__ import annotations
import warnings
import os

import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate

from numba import jit

from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from .._rust_bridge import _rust_ext, RUST_AVAILABLE, FORCE_NUMPY_STFT, FORCE_RUST_STFT
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import (
    _WindowSpec,
    _PadMode,
    _PadModeSTFT,
    _SequenceLike,
    _ScalarOrSequence,
    _ComplexLike_co,
    _FloatLike_co
)

# Cache periodic Hann windows used by Rust STFT fast-path.
_RUST_HANN_WINDOW_CACHE: dict[tuple[int, np.dtype], np.ndarray] = {}

@overload
def power_to_db(
    S: _ComplexLike_co,
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.floating[Any]:
    ...

@overload
def power_to_db(
    S: _SequenceLike[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.ndarray:
    ...

@overload
def power_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def power_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``::

            10 * log10(S / ref)

        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``abs(S)`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(10 * log10(S/ref)) - top_db``

    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809],
           ...,
           [-41.809, -41.809, ..., -41.809, -41.809],
           [-41.809, -41.809, ..., -41.809, -41.809]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.],
           ...,
           [-80., -80., ..., -80., -80.],
           [-80., -80., ..., -80., -80.]], dtype=float32)

    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578],
           ...,
           [16.578, 16.578, ..., 16.578, 16.578],
           [16.578, 16.578, ..., 16.578, 16.578]], dtype=float32)

    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(S**2, sr=sr, y_axis='log', x_axis='time',
    ...                                   ax=ax[0])
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> imgdb = librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                                  sr=sr, y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Log-Power spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(imgdb, ax=ax[1], format="%+2.0f dB")
    """
    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    if top_db is not None and top_db < 0:
        raise ParameterError("top_db must be non-negative")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "power_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    # --- iron-librosa: Rust acceleration ---
    if (
        RUST_AVAILABLE
        and isinstance(magnitude, np.ndarray)
        and magnitude.ndim
        and not callable(ref)
        and np.isscalar(ref_value)
        and magnitude.dtype in (np.float32, np.float64)
    ):
        _rust_power_to_db = (
            getattr(_rust_ext, "power_to_db_f32", None)
            if magnitude.dtype == np.float32
            else getattr(_rust_ext, "power_to_db_f64", None)
        )

        if _rust_power_to_db is not None:
            flat = np.ravel(np.ascontiguousarray(magnitude))
            ref_scalar = float(np.asarray(ref_value).item())
            return _rust_power_to_db(
                flat,
                ref_power=ref_scalar,
                amin=float(amin),
                top_db=None if top_db is None else float(top_db),
            ).reshape(magnitude.shape)
    # --- end Rust acceleration ---

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


@overload
def db_to_power(
    S_db: _FloatLike_co,
    *,
    ref: float = ...,
) -> np.floating[Any]:
    ...

@overload
def db_to_power(
        S_db: np.ndarray,
    *,
    ref: float = ...,
) -> np.ndarray:
    ...

@overload
def db_to_power(
    S_db: Union[_FloatLike_co, np.ndarray],
    *,
    ref: float = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def db_to_power(S_db: Union[_FloatLike_co, np.ndarray], *, ref: float = 1.0) -> Union[np.floating[Any], np.ndarray]:
    """Convert dB-scale values to a power values.

    This effectively inverts ``power_to_db``::

        db_to_power(S_db) ~= ref * 10.0**(S_db / 10)

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power values

    Notes
    -----
    This function caches at level 30.
    """
    S_db = np.asarray(S_db)

    # --- iron-librosa: Rust acceleration ---
    if (
        RUST_AVAILABLE
        and S_db.ndim
        and S_db.dtype in (np.float32, np.float64)
    ):
        _rust_db_to_power = (
            getattr(_rust_ext, "db_to_power_f32", None)
            if S_db.dtype == np.float32
            else getattr(_rust_ext, "db_to_power_f64", None)
        )

        if _rust_db_to_power is not None:
            flat = np.ravel(np.ascontiguousarray(S_db))
            return _rust_db_to_power(flat, ref_power=float(ref)).reshape(S_db.shape)
    # --- end Rust acceleration ---

    return ref * np.power(10.0, S_db * 0.1)


@overload
def amplitude_to_db(
    S: _ComplexLike_co,
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.floating[Any]:
    ...

@overload
def amplitude_to_db(
    S: _SequenceLike[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> np.ndarray:
    ...

@overload
def amplitude_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = ...,
    amin: float = ...,
    top_db: Optional[float] = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def amplitude_to_db(
    S: _ScalarOrSequence[_ComplexLike_co],
    *,
    ref: Union[float, Callable] = 1.0,
    amin: float = 1e-5,
    top_db: Optional[float] = 80.0,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2, ref=ref**2, amin=amin**2, top_db=top_db)``,
    but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude ``abs(S)`` is scaled relative to ``ref``:
        ``20 * log10(S / ref)``.
        Zeros in the output correspond to positions where ``S == ref``.

        If callable, the reference value is computed as ``ref(S)``.

    amin : float > 0 [scalar]
        minimum threshold for ``S`` and ``ref``

    top_db : float >= 0 [scalar]
        threshold the output at ``top_db`` below the peak:
        ``max(20 * log10(S/ref)) - top_db``

    Returns
    -------
    S_db : np.ndarray
        ``S`` measured in dB

    See Also
    --------
    power_to_db, db_to_amplitude

    Notes
    -----
    This function caches at level 30.
    """
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(S)) instead.",
            stacklevel=2,
        )

    magnitude = np.abs(S)

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    out_array = magnitude if isinstance(magnitude, np.ndarray) else None
    power = np.square(magnitude, out=out_array)

    db: np.ndarray = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
    return db


@overload
def db_to_amplitude(
    S_db: _FloatLike_co,
    *,
    ref: float = ...,
) -> np.floating[Any]:
    ...

@overload
def db_to_amplitude(
    S_db: np.ndarray,
    *,
    ref: float = ...,
) -> np.ndarray:
    ...

@overload
def db_to_amplitude(
    S_db: Union[_FloatLike_co, np.ndarray],
    *,
    ref: float = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...

@cache(level=30)
def db_to_amplitude(S_db: Union[_FloatLike_co, np.ndarray], *, ref: float = 1.0) -> Union[np.floating[Any], np.ndarray]:
    """Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`::

        db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db/10 + log10(ref))

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled values
    ref : number > 0
        Optional reference amplitude.

    Returns
    -------
    S : np.ndarray
        Linear magnitude values

    Notes
    -----
    This function caches at level 30.
    """
    return db_to_power(S_db, ref=ref**2) ** 0.5


@cache(level=30)
def perceptual_weighting(
    S: np.ndarray, frequencies: np.ndarray, *, kind: str = "A", **kwargs: Any
) -> np.ndarray:
    """Perceptual weighting of a power spectrogram::

        S_p[..., f, :] = frequency_weighting(f, 'A') + 10*log(S[..., f, :] / ref)

    Parameters
    ----------
    S : np.ndarray [shape=(..., d, t)]
        Power spectrogram
    frequencies : np.ndarray [shape=(d,)]
        Center frequency for each row of` `S``
    kind : str
        The frequency weighting curve to use.
        e.g. `'A'`, `'B'`, `'C'`, `'D'`, `None or 'Z'`
    **kwargs : additional keyword arguments
        Additional keyword arguments to `power_to_db`.

    Returns
    -------
    S_p : np.ndarray [shape=(..., d, t)]
        perceptually weighted version of ``S``

    See Also
    --------
    power_to_db

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    Re-weight a CQT power spectrum, using peak power as reference

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1')))
    >>> freqs = librosa.cqt_frequencies(C.shape[0],
    ...                                 fmin=librosa.note_to_hz('A1'))
    >>> perceptual_CQT = librosa.perceptual_weighting(C**2,
    ...                                               freqs,
    ...                                               ref=np.max)
    >>> perceptual_CQT
    array([[ -96.528,  -97.101, ..., -108.561, -108.561],
           [ -95.88 ,  -96.479, ..., -107.551, -107.551],
           ...,
           [ -65.142,  -53.256, ...,  -80.098,  -80.098],
           [ -71.542,  -53.197, ...,  -80.311,  -80.311]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C,
    ...                                                        ref=np.max),
    ...                                fmin=librosa.note_to_hz('A1'),
    ...                                y_axis='cqt_hz', x_axis='time',
    ...                                ax=ax[0])
    >>> ax[0].set(title='Log CQT power')
    >>> ax[0].label_outer()
    >>> imgp = librosa.display.specshow(perceptual_CQT, y_axis='cqt_hz',
    ...                                 fmin=librosa.note_to_hz('A1'),
    ...                                 x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Perceptually weighted log CQT')
    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    >>> fig.colorbar(imgp, ax=ax[1], format="%+2.0f dB")
    """
    offset = convert.frequency_weighting(frequencies, kind=kind).reshape((-1, 1))

    result: np.ndarray = offset + power_to_db(S, **kwargs)
    return result


@cache(level=30)
def fmt(
    y: np.ndarray,
    *,
    t_min: float = 0.5,
    n_fmt: Optional[int] = None,
    kind: str = "cubic",
    beta: float = 0.5,
    over_sample: float = 1,
    axis: int = -1,
) -> np.ndarray:
    """Fast Mellin transform (FMT)

    The Mellin of a signal `y` is performed by interpolating `y` on an exponential time
    axis, applying a polynomial window, and then taking the discrete Fourier transform.

    When the Mellin parameter (beta) is 1/2, it is also known as the scale transform. [#]_
    The scale transform can be useful for audio analysis because its magnitude is invariant
    to scaling of the domain (e.g., time stretching or compression).  This is analogous
    to the magnitude of the Fourier transform being invariant to shifts in the input domain.

    .. [#] De Sena, Antonio, and Davide Rocchesso.
        "A fast Mellin and scale transform."
        EURASIP Journal on Applied Signal Processing 2007.1 (2007): 75-75.

    .. [#] Cohen, L.
        "The scale representation."
        IEEE Transactions on Signal Processing 41, no. 12 (1993): 3275-3292.

    Parameters
    ----------
    y : np.ndarray, real-valued
        The input signal(s).  Can be multidimensional.
        The target axis must contain at least 3 samples.

    t_min : float > 0
        The minimum time spacing (in samples).
        This value should generally be less than 1 to preserve as much information as
        possible.

    n_fmt : int > 2 or None
        The number of scale transform bins to use.
        If None, then ``n_bins = over_sample * ceil(n * log((n-1)/t_min))`` is taken,
        where ``n = y.shape[axis]``

    kind : str
        The type of interpolation to use when re-sampling the input.
        See `scipy.interpolate.interp1d` for possible values.

        Note that the default is to use high-precision (cubic) interpolation.
        This can be slow in practice; if speed is preferred over accuracy,
        then consider using ``kind='linear'``.

    beta : float
        The Mellin parameter.  ``beta=0.5`` provides the scale transform.

    over_sample : float >= 1
        Over-sampling factor for exponential resampling.

    axis : int
        The axis along which to transform ``y``

    Returns
    -------
    x_scale : np.ndarray [dtype=complex]
        The scale transform of ``y`` along the ``axis`` dimension.

    Raises
    ------
    ParameterError
        if ``n_fmt < 2`` or ``t_min <= 0``
        or if ``y`` is not finite
        or if ``y.shape[axis] < 3``.

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> # Generate a signal and time-stretch it (with energy normalization)
    >>> scale = 1.25
    >>> freq = 3.0
    >>> x1 = np.linspace(0, 1, num=1024, endpoint=False)
    >>> x2 = np.linspace(0, 1, num=int(scale * len(x1)), endpoint=False)
    >>> y1 = np.sin(2 * np.pi * freq * x1)
    >>> y2 = np.sin(2 * np.pi * freq * x2) / np.sqrt(scale)
    >>> # Verify that the two signals have the same energy
    >>> np.sum(np.abs(y1)**2), np.sum(np.abs(y2)**2)
        (255.99999999999997, 255.99999999999969)
    >>> scale1 = librosa.fmt(y1, n_fmt=512)
    >>> scale2 = librosa.fmt(y2, n_fmt=512)

    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2)
    >>> ax[0].plot(y1, label='Original')
    >>> ax[0].plot(y2, linestyle='--', label='Stretched')
    >>> ax[0].set(xlabel='time (samples)', title='Input signals')
    >>> ax[0].legend()
    >>> ax[1].semilogy(np.abs(scale1), label='Original')
    >>> ax[1].semilogy(np.abs(scale2), linestyle='--', label='Stretched')
    >>> ax[1].set(xlabel='scale coefficients', title='Scale transform magnitude')
    >>> ax[1].legend()

    >>> # Plot the scale transform of an onset strength autocorrelation
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr)
    >>> # Auto-correlate with up to 10 seconds lag
    >>> odf_ac = librosa.autocorrelate(odf, max_size=10 * sr // 512)
    >>> # Normalize
    >>> odf_ac_norm = librosa.util.normalize(odf_ac, norm=np.inf)
    >>> # Compute the scale transform
    >>> odf_ac_scale = librosa.fmt(odf_ac_norm, n_fmt=512)
    >>> # Plot the results
    >>> fig, ax = plt.subplots(nrows=3)
    >>> ax[0].plot(odf, label='Onset strength')
    >>> ax[0].set(xlabel='Time (frames)', title='Onset strength')
    >>> ax[0].legend()
    >>> ax[1].plot(odf_ac_norm, label='Onset autocorrelation')
    >>> ax[1].set(xlabel='Lag (frames)', title='Onset autocorrelation')
    >>> ax[1].legend()
    >>> ax[2].semilogy(np.abs(odf_ac_scale), label='Scale transform magnitude')
    >>> ax[2].set(xlabel='scale coefficients')
    """
    n = y.shape[axis]

    if n < 3:
        raise ParameterError(f"y.shape[{axis}]=={n} < 3")

    if t_min <= 0:
        raise ParameterError(f"t_min={t_min} must be a positive number")

    if n_fmt is None:
        if over_sample < 1:
            raise ParameterError(f"over_sample={over_sample} must be >= 1")

        # The base is the maximum ratio between adjacent samples
        # Since the sample spacing is increasing, this is simply the
        # ratio between the positions of the last two samples: (n-1)/(n-2)
        log_base = np.log(n - 1) - np.log(n - 2)

        n_fmt = int(np.ceil(over_sample * (np.log(n - 1) - np.log(t_min)) / log_base))

    elif n_fmt < 3:
        raise ParameterError(f"n_fmt=={n_fmt} < 3")
    else:
        log_base = (np.log(n_fmt - 1) - np.log(n_fmt - 2)) / over_sample

    if not np.all(np.isfinite(y)):
        raise ParameterError("y must be finite everywhere")

    base = np.exp(log_base)
    # original grid: signal covers [0, 1).  This range is arbitrary, but convenient.
    # The final sample is positioned at (n-1)/n, so we omit the endpoint
    x = np.linspace(0, 1, num=n, endpoint=False)

    # build the interpolator
    f_interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=axis)

    # build the new sampling grid
    # exponentially spaced between t_min/n and 1 (exclusive)
    # we'll go one past where we need, and drop the last sample
    # When over-sampling, the last input sample contributions n_over samples.
    # To keep the spacing consistent, we over-sample by n_over, and then
    # trim the final samples.
    n_over = int(np.ceil(over_sample))
    x_exp = np.logspace(
        (np.log(t_min) - np.log(n)) / log_base,
        0,
        num=n_fmt + n_over,
        endpoint=False,
        base=base,
    )[:-n_over]

    # Clean up any rounding errors at the boundaries of the interpolation
    # The interpolator gets angry if we try to extrapolate, so clipping is necessary here.
    if x_exp[0] < t_min or x_exp[-1] > float(n - 1.0) / n:
        x_exp = np.clip(x_exp, float(t_min) / n, x[-1])

    # Make sure that all sample points are unique
    # This should never happen!
    if len(np.unique(x_exp)) != len(x_exp):
        raise ParameterError("Redundant sample positions in Mellin transform")

    # Resample the signal
    y_res = f_interp(x_exp)

    # Broadcast the window correctly
    shape = [1] * y_res.ndim
    shape[axis] = -1

    # Apply the window and fft
    # Normalization is absorbed into the window here for expedience
    fft = get_fftlib()
    result: np.ndarray = fft.rfft(
        y_res * ((x_exp**beta).reshape(shape) * np.sqrt(n) / n_fmt), axis=axis
    )
    return result


