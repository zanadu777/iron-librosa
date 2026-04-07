#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral centroid and bandwidth features"""

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


def spectral_centroid(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    freq: Optional[np.ndarray] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> np.ndarray:
    """Compute the spectral centroid.

    Each frame of a magnitude spectrogram is normalized and treated as a
    distribution over frequency bins, from which the mean (centroid) is
    extracted per frame.

    More precisely, the centroid at frame ``t`` is defined as [#]_::

        centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    where ``S`` is a magnitude spectrogram, and ``freq`` is the array of
    frequencies (e.g., FFT frequencies in Hz) of the rows of ``S``.

    .. [#] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing
        methods for music transcription, chapter 5.
        Springer Science & Business Media.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    freq : None or np.ndarray [shape=(d,) or shape=(d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`
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

    Returns
    -------
    centroid : np.ndarray [shape=(..., 1, t)]
        centroid frequencies

    See Also
    --------
    librosa.stft : Short-time Fourier Transform
    librosa.reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    From time-series input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    >>> cent
    array([[1768.888, 1921.774, ..., 5663.477, 5813.683]])

    From spectrogram input:

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_centroid(S=S)
    array([[1768.888, 1921.774, ..., 5663.477, 5813.683]])

    Using variable bin center frequencies:

    >>> freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    >>> librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)
    array([[1768.838, 1921.801, ..., 5663.513, 5813.747]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(cent)
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax)
    >>> ax.plot(times, cent.T, label='Spectral centroid', color='w')
    >>> ax.legend(loc='upper right')
    >>> ax.set(title='log Power spectrogram')
    """
    # input is time domain:y or spectrogram:s
    #

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

    if not np.isrealobj(S):
        raise ParameterError(
            "Spectral centroid is only defined " "with real-valued input"
        )

    # Compute the center frequencies of each bin (needed by both paths).
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Keep 2D variable-frequency validation explicit for clearer errors.
    if isinstance(freq, np.ndarray) and S.ndim == 2 and freq.ndim == 2 and freq.shape != S.shape:
        raise ParameterError(
            f"freq.shape mismatch: expected {S.shape}, found {freq.shape}"
        )

    # ── Rust fast path ──────────────────────────────────────────────────────
    # Placed before np.any(S < 0) to avoid an O(n_bins×n_frames) Python scan
    # on the hot path.  The Rust kernel assumes non-negative magnitudes per
    # the documented function contract; callers who pass negative values and
    # need the ParameterError will still get it via the Python fallback below.
    if (
        RUST_AVAILABLE
        and np.isrealobj(S)
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
        and isinstance(freq, np.ndarray)
        and freq.ndim == 1
        and freq.dtype == np.float64
    ):
        _centroid_name = (
            "spectral_centroid_f32" if S.dtype == np.float32 else "spectral_centroid_f64"
        )
        _centroid_kernel = getattr(_rust_ext, _centroid_name, None)

        if _centroid_kernel is not None:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
            )
            freq_c = np.ascontiguousarray(freq)
            centroids = [_centroid_kernel(channel, freq_c) for channel in s_flat]

            if S.ndim == 2:
                return centroids[0]

            return np.stack(centroids, axis=0).reshape(*S.shape[:-2], 1, S.shape[-1])

    # Pilot: variable-frequency fast path for 2-D inputs.
    if (
        RUST_AVAILABLE
        and np.isrealobj(S)
        and S.dtype in (np.float32, np.float64)
        and S.ndim == 2
        and isinstance(freq, np.ndarray)
        and freq.ndim == 2
        and freq.shape == S.shape
        and freq.dtype == np.float64
    ):
        _var_name = (
            "spectral_centroid_variable_freq_f32"
            if S.dtype == np.float32
            else "spectral_centroid_variable_freq_f64"
        )
        _var_kernel = getattr(_rust_ext, _var_name, None)
        if _var_kernel is not None:
            return _var_kernel(np.ascontiguousarray(S), np.ascontiguousarray(freq))

    # ── Python fallback — validate first ────────────────────────────────────
    if np.any(S < 0):
        raise ParameterError(
            "Spectral centroid is only defined " "with non-negative energies"
        )

    if freq.ndim == 1:
        # reshape for broadcasting
        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)

    # Column-normalize S
    centroid: np.ndarray = np.sum(
        freq * util.normalize(S, norm=1, axis=-2), axis=-2, keepdims=True
    )
    return centroid



def spectral_bandwidth(
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
    freq: Optional[np.ndarray] = None,
    centroid: Optional[np.ndarray] = None,
    norm: bool = True,
    p: float = 2,
) -> np.ndarray:
    """Compute p'th-order spectral bandwidth.

       The spectral bandwidth [#]_ at frame ``t`` is computed by::

        (sum_k S[k, t] * (freq[k, t] - centroid[t])**p)**(1/p)

    .. [#] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing
        methods for music transcription, chapter 5.
        Springer Science & Business Media.

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
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    freq : None or np.ndarray [shape=(d,) or shape=(..., d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`
    centroid : None or np.ndarray [shape=(..., 1, t)]
        pre-computed centroid frequencies
    norm : bool
        Normalize per-frame spectral energy (sum to one)
    p : float > 0
        Power to raise deviation from spectral centroid.

    Returns
    -------
    bandwidth : np.ndarray [shape=(..., 1, t)]
        frequency bandwidth for each frame

    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    >>> spec_bw
    array([[1273.836, 1228.873, ..., 2952.357, 3013.68 ]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y=y))
    >>> librosa.feature.spectral_bandwidth(S=S)
    array([[1273.836, 1228.873, ..., 2952.357, 3013.68 ]])

    Using variable bin center frequencies

    >>> freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    >>> librosa.feature.spectral_bandwidth(S=np.abs(D), freq=freqs)
    array([[1274.637, 1228.786, ..., 2952.4  , 3013.735]])

    Plot the result

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> times = librosa.times_like(spec_bw)
    >>> centroid = librosa.feature.spectral_centroid(S=S)
    >>> ax[0].semilogy(times, spec_bw[0], label='Spectral bandwidth')
    >>> ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
    >>> ax[0].legend()
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='log Power spectrogram')
    >>> ax[1].fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]),
    ...                 np.minimum(centroid[0] + spec_bw[0], sr/2),
    ...                 alpha=0.5, label='Centroid +- bandwidth')
    >>> ax[1].plot(times, centroid[0], label='Spectral centroid', color='w')
    >>> ax[1].legend(loc='lower right')
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

    if not np.isrealobj(S):
        raise ParameterError(
            "Spectral bandwidth is only defined " "with real-valued input"
        )
    elif np.any(S < 0):
        raise ParameterError(
            "Spectral bandwidth is only defined " "with non-negative energies"
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Keep 2D variable-frequency validation explicit for clearer errors.
    if isinstance(freq, np.ndarray) and S.ndim == 2 and freq.ndim == 2 and freq.shape != S.shape:
        raise ParameterError(
            f"freq.shape mismatch: expected {S.shape}, found {freq.shape}"
        )

    # Rust fused fast path for centroid=None in static 1D frequency bins.
    if (
        centroid is None
        and RUST_AVAILABLE
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
        and isinstance(freq, np.ndarray)
        and freq.ndim == 1
        and freq.dtype == np.float64
    ):
        _bw_auto_name = (
            "spectral_bandwidth_auto_centroid_f32"
            if S.dtype == np.float32
            else "spectral_bandwidth_auto_centroid_f64"
        )
        _bw_auto_kernel = getattr(_rust_ext, _bw_auto_name, None)
        if _bw_auto_kernel is not None:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
            )
            freq_c = np.ascontiguousarray(freq)
            bw_frames = [
                _bw_auto_kernel(channel, freq_c, bool(norm), float(p))
                for channel in s_flat
            ]

            if S.ndim == 2:
                return bw_frames[0]

            return np.stack(bw_frames, axis=0).reshape(*S.shape[:-2], 1, S.shape[-1])

    # If we don't have a centroid provided, compute it using the existing
    # spectrogram S
    if centroid is None:
        centroid = spectral_centroid(
            y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq
        )

    # Rust fast path for static 1D frequency bins.
    if (
        RUST_AVAILABLE
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
        and isinstance(freq, np.ndarray)
        and freq.ndim == 1
        and freq.dtype == np.float64
        and centroid is not None
        and centroid.shape[-2:] == (1, S.shape[-1])
        and centroid.shape[:-2] == S.shape[:-2]
    ):
        _bw_name = "spectral_bandwidth_f32" if S.dtype == np.float32 else "spectral_bandwidth_f64"
        _bw_kernel = getattr(_rust_ext, _bw_name, None)
        if _bw_kernel is not None:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
            )
            c_flat = np.reshape(
                np.ascontiguousarray(centroid), (-1, 1, S.shape[-1])
            )
            freq_c = np.ascontiguousarray(freq)
            bw_frames = [
                _bw_kernel(channel, freq_c, c_channel, bool(norm), float(p))
                for channel, c_channel in zip(s_flat, c_flat)
            ]

            if S.ndim == 2:
                return bw_frames[0]

            return np.stack(bw_frames, axis=0).reshape(*S.shape[:-2], 1, S.shape[-1])

    if freq.ndim == 1:
        deviation = np.abs(
            np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1)
        )
    else:
        deviation = np.abs(freq - centroid)

    # Column-normalize S
    if norm:
        S = util.normalize(S, norm=1, axis=-2)

    bw: np.ndarray = np.sum(S * deviation**p, axis=-2, keepdims=True) ** (1.0 / p)
    return bw


