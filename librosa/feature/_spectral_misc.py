#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral flatness, RMS, polynomial features, zero-crossing rate"""

import os
import json
import numpy as np
import scipy
import scipy.signal

from .. import util
from .. import filters
from ..util.exceptions import ParameterError

from ..core.fft import get_fftlib
from ..core.convert import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt, vqt
from ..core.pitch import estimate_tuning
from .._rust_bridge import (
    _rust_ext,
    RUST_AVAILABLE,
    FORCE_RUST_MEL,
    FORCE_NUMPY_MEL,
)
from typing import Any, Optional, Union, Collection, Dict
from typing_extensions import Literal
from numpy.typing import DTypeLike
from .._typing import _FloatLike_co, _WindowSpec, _PadMode, _PadModeSTFT

try:
    from ._mel_threshold_registry import MEL_WORK_THRESHOLDS
except Exception:
    MEL_WORK_THRESHOLDS = {}


_ENABLE_RUST_RMS_TIME = os.getenv("IRON_LIBROSA_ENABLE_RUST_RMS_TIME", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


def spectral_flatness(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    amin: float = 1e-10,
    power: float = 2.0,
) -> np.ndarray:
    """Compute spectral flatness

    Spectral flatness (or tonality coefficient) is a measure to
    quantify how much noise-like a sound is, as opposed to being
    tone-like [#]_. A high spectral flatness (closer to 1.0)
    indicates the spectrum is similar to white noise.
    It is often converted to decibel.

    .. [#] Dubnov, Shlomo  "Generalization of spectral flatness
           measure for non-gaussian linear processes"
           IEEE Signal Processing Letters, 2004, Vol. 11.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) pre-computed spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If `False`, then frame `t` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    amin : float > 0 [scalar]
        minimum threshold for ``S`` (=added noise floor for numerical stability)
    power : float > 0 [scalar]
        Exponent for the magnitude spectrogram.
        e.g., 1 for energy, 2 for power, etc.
        Power spectrogram is usually used for computing spectral flatness.

    Returns
    -------
    flatness : np.ndarray [shape=(..., 1, t)]
        spectral flatness for each frame.
        The returned value is in [0, 1] and often converted to dB scale.

    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> flatness = librosa.feature.spectral_flatness(y=y)
    >>> flatness
    array([[0.001, 0.   , ..., 0.218, 0.184]], dtype=float32)

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_flatness(S=S)
    array([[0.001, 0.   , ..., 0.218, 0.184]], dtype=float32)

    From power spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> S_power = S ** 2
    >>> librosa.feature.spectral_flatness(S=S_power, power=1.0)
    array([[0.001, 0.   , ..., 0.218, 0.184]], dtype=float32)

    """
    if amin <= 0:
        raise ParameterError("amin must be strictly positive")

    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    if not np.isrealobj(S):
        raise ParameterError(
            "Spectral flatness is only defined " "with real-valued input"
        )
    elif np.any(S < 0):
        raise ParameterError(
            "Spectral flatness is only defined " "with non-negative energies"
        )

    # ── Rust fast path ──────────────────────────────────────────────────────
    # Guard: real, non-negative, 2-D+ float32/float64, 1-D static-freq bin
    # dimension (S.ndim >= 2 satisfied by _spectrogram contract).
    if (
        RUST_AVAILABLE
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
    ):
        _flat_name = (
            "spectral_flatness_f32" if S.dtype == np.float32 else "spectral_flatness_f64"
        )
        _flat_kernel = getattr(_rust_ext, _flat_name, None)

        if _flat_kernel is not None:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
            )
            results = [_flat_kernel(channel, float(amin), float(power)) for channel in s_flat]

            if S.ndim == 2:
                return results[0]

            return np.stack(results, axis=0).reshape(*S.shape[:-2], 1, S.shape[-1])
    # ── Python fallback ─────────────────────────────────────────────────────

    S_thresh = np.maximum(amin, S**power)
    gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))
    amean = np.mean(S_thresh, axis=-2, keepdims=True)
    flatness: np.ndarray = gmean / amean
    return flatness



def rms(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: _PadMode = "constant",
    dtype: DTypeLike = np.float32,
) -> np.ndarray:
    """Compute root-mean-square (RMS) value for each frame, either from the
    audio samples ``y`` or from a spectrogram ``S``.

    Computing the RMS value from audio samples is faster as it doesn't require
    a STFT calculation. However, using a spectrogram will give a more accurate
    representation of energy over time because its frames can be windowed,
    thus prefer using ``S`` if it's already available.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        (optional) audio time series. Required if ``S`` is not input.
        Multi-channel is supported.
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude. Required if ``y`` is not input.
    frame_length : int > 0 [scalar]
        length of analysis frame (in samples) for energy calculation
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    center : bool
        If `True` and operating on time-domain input (``y``), pad the signal
        by ``frame_length//2`` on either side.
        If operating on spectrogram input, this has no effect.
    pad_mode : str
        Padding mode for centered analysis.  See `numpy.pad` for valid
        values.
    dtype : np.dtype, optional
        Data type of the output array.  Defaults to float32.

    Returns
    -------
    rms : np.ndarray [shape=(..., 1, t)]
        RMS value for each frame

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.rms(y=y)
    array([[1.248e-01, 1.259e-01, ..., 1.845e-05, 1.796e-05]],
          dtype=float32)

    Or from spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> rms = librosa.feature.rms(S=S)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(rms)
    >>> ax[0].semilogy(times, rms[0], label='RMS Energy')
    >>> ax[0].set(xticks=[])
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='log Power spectrogram')

    Use a STFT window of constant ones and no frame centering to get consistent
    results with the RMS computed from the audio samples ``y``

    >>> S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    >>> librosa.feature.rms(S=S)
    >>> plt.show()

    """
    if y is not None:
        if center:
            padding = [(0, 0) for _ in range(y.ndim)]
            padding[-1] = (int(frame_length // 2), int(frame_length // 2))
            y = np.pad(y, padding, mode=pad_mode)

        x = util.frame(y, frame_length=frame_length, hop_length=hop_length)

        target_dtype = np.dtype(dtype)
        if (
            _ENABLE_RUST_RMS_TIME
            and
            RUST_AVAILABLE
            and np.isrealobj(x)
            and target_dtype in (np.float32, np.float64)
        ):
            _rms_time_name = (
                "rms_time_f32" if target_dtype == np.float32 else "rms_time_f64"
            )
            _rms_time_kernel = getattr(_rust_ext, _rms_time_name, None)

            if _rms_time_kernel is not None:
                # Preserve framed stride layout when dtype already matches;
                # avoid forcing a large contiguous copy on the hot path.
                x_flat = np.reshape(
                    np.asarray(x, dtype=target_dtype),
                    (-1, x.shape[-2], x.shape[-1]),
                )
                rms_frames = [_rms_time_kernel(channel) for channel in x_flat]

                if x.ndim == 2:
                    return rms_frames[0]

                return np.stack(rms_frames, axis=0).reshape(
                    *x.shape[:-2], 1, x.shape[-1]
                )

        # Calculate power
        power = np.mean(util.abs2(x, dtype=dtype), axis=-2, keepdims=True)
    elif S is not None:
        # Check the frame length
        if S.shape[-2] != frame_length // 2 + 1:
            raise ParameterError(
                "Since S.shape[-2] is {}, "
                "frame_length is expected to be {} or {}; "
                "found {}".format(
                    S.shape[-2], S.shape[-2] * 2 - 2, S.shape[-2] * 2 - 1, frame_length
                )
            )

        # Rust fast path for the real-valued spectrogram path when output dtype
        # matches the spectrogram precision.
        target_dtype = np.dtype(dtype)
        if (
            RUST_AVAILABLE
            and np.isrealobj(S)
            and S.dtype in (np.float32, np.float64)
            and S.dtype == target_dtype
        ):
            _rms_name = (
                "rms_spectrogram_f32" if S.dtype == np.float32 else "rms_spectrogram_f64"
            )
            _rms_kernel = getattr(_rust_ext, _rms_name, None)

            if _rms_kernel is not None:
                s_flat = np.reshape(
                    np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
                )
                rms_frames = [_rms_kernel(channel, int(frame_length)) for channel in s_flat]

                if S.ndim == 2:
                    return rms_frames[0]

                return np.stack(rms_frames, axis=0).reshape(
                    *S.shape[:-2], 1, S.shape[-1]
                )

        # power spectrogram
        x = util.abs2(S, dtype=dtype)

        # Adjust the DC and sr/2 component
        x[..., 0, :] *= 0.5
        if frame_length % 2 == 0:
            x[..., -1, :] *= 0.5

        # Calculate power
        power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length**2
    else:
        raise ParameterError("Either `y` or `S` must be input.")

    rms_result: np.ndarray = np.sqrt(power)
    return rms_result



def poly_features(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    order: int = 1,
    freq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get coefficients of fitting an nth-order polynomial to the columns
    of a spectrogram.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.
        If unspecified, defaults to ``win_length = n_fft``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``
        .. see also:: `librosa.filters.get_window`
    center : boolean
        - If `True`, the signal ``y`` is padded so that frame
          `t` is centered at ``y[t * hop_length]``.
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    order : int > 0
        order of the polynomial to fit
    freq : None or np.ndarray [shape=(d,) or shape=(..., d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`

    Returns
    -------
    coefficients : np.ndarray [shape=(..., order+1, t)]
        polynomial coefficients for each frame.

        ``coefficients[..., 0, :]`` corresponds to the highest degree (``order``),

        ``coefficients[..., 1, :]`` corresponds to the next highest degree (``order-1``),

        down to the constant term ``coefficients[..., order, :]``.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))

    Fit a degree-0 polynomial (constant) to each frame

    >>> p0 = librosa.feature.poly_features(S=S, order=0)

    Fit a linear polynomial to each frame

    >>> p1 = librosa.feature.poly_features(S=S, order=1)

    Fit a quadratic to each frame

    >>> p2 = librosa.feature.poly_features(S=S, order=2)

    Plot the results for comparison

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
    >>> times = librosa.times_like(p0)
    >>> ax[0].plot(times, p0[0], label='order=0', alpha=0.8)
    >>> ax[0].plot(times, p1[1], label='order=1', alpha=0.8)
    >>> ax[0].plot(times, p2[2], label='order=2', alpha=0.8)
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> ax[0].set(ylabel='Constant term ')
    >>> ax[1].plot(times, p1[0], label='order=1', alpha=0.8)
    >>> ax[1].plot(times, p2[1], label='order=2', alpha=0.8)
    >>> ax[1].set(ylabel='Linear term')
    >>> ax[1].label_outer()
    >>> ax[1].legend()
    >>> ax[2].plot(times, p2[0], label='order=2', alpha=0.8)
    >>> ax[2].set(ylabel='Quadratic term')
    >>> ax[2].legend()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[3])
    """
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    coefficients: np.ndarray

    if freq.ndim == 1:
        # If frequencies are constant over frames, then we only need to fit once
        fitter = np.vectorize(
            lambda y: np.polyfit(freq, y, order), signature="(f,t)->(d,t)"
        )
        coefficients = fitter(S)
    else:
        # Otherwise, we have variable frequencies, and need to fit independently
        fitter = np.vectorize(
            lambda x, y: np.polyfit(x, y, order), signature="(f),(f)->(d)"
        )

        # We have to do some axis swapping to preserve layout
        # otherwise, the new dimension gets put at the end instead of the penultimate position
        coefficients = fitter(freq.swapaxes(-2, -1), S.swapaxes(-2, -1)).swapaxes(
            -2, -1
        )

    return coefficients



def zero_crossing_rate(
    y: np.ndarray,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the zero-crossing rate of an audio time series.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        Audio time series. Multi-channel is supported.
    frame_length : int > 0
        Length of the frame over which to compute zero crossing rates
    hop_length : int > 0
        Number of samples to advance for each frame
    center : bool
        If `True`, frames are centered by padding the edges of ``y``.
        This is similar to the padding in `librosa.stft`,
        but uses edge-value copies instead of zero-padding.
    **kwargs : additional keyword arguments to pass to `librosa.zero_crossings`
    threshold : float >= 0
        If specified, values where ``-threshold <= y <= threshold`` are
        clipped to 0.
    ref_magnitude : float > 0 or callable
        If numeric, the threshold is scaled relative to ``ref_magnitude``.
        If callable, the threshold is scaled relative to
        ``ref_magnitude(np.abs(y))``.
    zero_pos : boolean
        If ``True`` then the value 0 is interpreted as having positive sign.
        If ``False``, then 0, -1, and +1 all have distinct signs.
    axis : int
        Axis along which to compute zero-crossings.
        .. note:: By default, the ``pad`` parameter is set to `False`, which
        differs from the default specified by
        `librosa.zero_crossings`.

    Returns
    -------
    zcr : np.ndarray [shape=(..., 1, t)]
        ``zcr[..., 0, i]`` is the fraction of zero crossings in frame ``i``

    See Also
    --------
    librosa.zero_crossings : Compute zero-crossings in a time-series

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.zero_crossing_rate(y)
    array([[0.044, 0.074, ..., 0.488, 0.355]])
    """
    # check if audio is valid
    util.valid_audio(y)

    if center:
        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode="edge")

    y_framed = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    kwargs["axis"] = -2
    kwargs.setdefault("pad", False)

    crossings = zero_crossings(y_framed, **kwargs)

    zcrate: np.ndarray = np.mean(crossings, axis=-2, keepdims=True)
    return zcrate


# -- Chroma --#
