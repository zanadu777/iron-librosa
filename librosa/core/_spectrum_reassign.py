#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reassigned spectrogram and magphase"""
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

from ._spectrum_stft import stft
def __reassign_frequencies(
    y: np.ndarray,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequencies based on a spectrogram representation.

    The reassignment vector is calculated using equation 5.20 in Flandrin,
    Auger, & Chassande-Mottin 2002::

        omega_reassigned = omega - np.imag(S_dh/S_h)

    where ``S_h`` is the complex STFT calculated using the original window, and
    ``S_dh`` is the complex STFT calculated using the derivative of the original
    window.

    See `reassigned_spectrogram` for references.

    It is recommended to use ``pad_mode="wrap"`` or else ``center=False``, rather
    than the defaults. Frequency reassignment assumes that the energy in each
    FFT bin is associated with exactly one signal component. Reflection padding
    at the edges of the signal may invalidate the reassigned estimates in the
    boundary frames.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to `__reassign_frequencies`

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See ``stft`` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    dtype : numeric type
        Complex numeric type for ``S``. Default is inferred to match
        the numerical precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    freqs : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Instantaneous frequencies:
        ``freqs[f, t]`` is the frequency for bin ``f``, frame ``t``.
    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    Warns
    -----
    RuntimeWarning
        Frequencies with zero support will produce a divide-by-zero warning and
        will be returned as `np.nan`.

    See Also
    --------
    stft : Short-time Fourier Transform
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> frequencies, S = librosa.core.spectrum.__reassign_frequencies(y, sr=sr)
    >>> frequencies
    array([[0.000e+00, 0.000e+00, ..., 0.000e+00, 0.000e+00],
           [3.628e+00, 4.698e+00, ..., 1.239e+01, 1.072e+01],
           ...,
           [1.101e+04, 1.102e+04, ..., 1.105e+04, 1.102e+04],
           [1.102e+04, 1.102e+04, ..., 1.102e+04, 1.102e+04]])
    """
    # retrieve window samples if needed so that the window derivative can be
    # calculated
    if win_length is None:
        win_length = n_fft

    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)

    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)

        S_h = stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    else:
        if dtype is None:
            dtype = S.dtype

        S_h = S

    # cyclic gradient to correctly handle edges of a periodic window
    window_derivative = util.cyclic_gradient(window)

    S_dh = stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_derivative,
        center=center,
        dtype=dtype,
        pad_mode=pad_mode,
    )

    # equation 5.20 of Flandrin, Auger, & Chassande-Mottin 2002
    # the sign of the correction is reversed in some papers - see Plante,
    # Meyer, & Ainsworth 1998 pp. 283-284
    with np.errstate(invalid="ignore"):
        # We can ignore divide-by-zero here because NaN is an acceptable correction value
        correction = -np.imag(S_dh / S_h)

    freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs = util.expand_to(freqs, ndim=correction.ndim, axes=-2) + correction * (
        0.5 * sr / np.pi
    )

    return freqs, S_h


def __reassign_times(
    y: np.ndarray,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Time reassignments based on a spectrogram representation.

    The reassignment vector is calculated using equation 5.23 in Flandrin,
    Auger, & Chassande-Mottin 2002::

        t_reassigned = t + np.real(S_th/S_h)

    where ``S_h`` is the complex STFT calculated using the original window, and
    ``S_th`` is the complex STFT calculated using the original window multiplied
    by the time offset from the window center.

    See `reassigned_spectrogram` for references.

    It is recommended to use ``pad_mode="constant"`` (zero padding) or else
    ``center=False``, rather than the defaults. Time reassignment assumes that
    the energy in each FFT bin is associated with exactly one impulse event.
    Reflection padding at the edges of the signal may invalidate the reassigned
    estimates in the boundary frames.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to `__reassign_times`

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    dtype : numeric type
        Complex numeric type for ``S``. Default is inferred to match
        the precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    times : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Reassigned times:
        ``times[f, t]`` is the time for bin ``f``, frame ``t``.
    S : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    Warns
    -----
    RuntimeWarning
        Time estimates with zero support will produce a divide-by-zero warning
        and will be returned as `np.nan`.

    See Also
    --------
    stft : Short-time Fourier Transform
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> times, S = librosa.core.spectrum.__reassign_times(y, sr=sr)
    >>> times
    array([[ 2.268e-05,  1.144e-02, ...,  5.332e+00,  5.333e+00],
           [ 2.268e-05,  1.451e-02, ...,  5.334e+00,  5.333e+00],
           ...,
           [ 2.268e-05, -6.177e-04, ...,  5.368e+00,  5.327e+00],
           [ 2.268e-05,  1.420e-03, ...,  5.307e+00,  5.328e+00]])
    """
    # retrieve window samples if needed so that the time-weighted window can be
    # calculated
    if win_length is None:
        win_length = n_fft

    window = get_window(window, win_length, fftbins=True)
    window = util.pad_center(window, size=n_fft)

    # retrieve hop length if needed so that the frame times can be calculated
    if hop_length is None:
        hop_length = int(win_length // 4)

    if S is None:
        if dtype is None:
            dtype = util.dtype_r2c(y.dtype)
        S_h = stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    else:
        if dtype is None:
            dtype = S.dtype
        S_h = S

    # calculate window weighted by time
    half_width = n_fft // 2

    window_times: np.ndarray
    if n_fft % 2:
        window_times = np.arange(-half_width, half_width + 1)

    else:
        window_times = np.arange(0.5 - half_width, half_width)

    window_time_weighted = window * window_times

    S_th = stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window_time_weighted,
        center=center,
        dtype=dtype,
        pad_mode=pad_mode,
    )

    # equation 5.23 of Flandrin, Auger, & Chassande-Mottin 2002
    # the sign of the correction is reversed in some papers - see Plante,
    # Meyer, & Ainsworth 1998 pp. 283-284
    with np.errstate(invalid="ignore"):
        # We can ignore divide-by-zero here because NaN is an acceptable correction value
        correction = np.real(S_th / S_h)

    if center:
        pad_length = None

    else:
        pad_length = n_fft

    times = convert.frames_to_time(
        np.arange(S_h.shape[-1]), sr=sr, hop_length=hop_length, n_fft=pad_length
    )

    times = util.expand_to(times, ndim=correction.ndim, axes=-1) + correction / sr

    return times, S_h


def reassigned_spectrogram(
    y: np.ndarray,
    *,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    reassign_frequencies: bool = True,
    reassign_times: bool = True,
    ref_power: Union[float, Callable] = 1e-6,
    fill_nan: bool = False,
    clip: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Time-frequency reassigned spectrogram.

    The reassignment vectors are calculated using equations 5.20 and 5.23 in
    [#]_::

        t_reassigned = t + np.real(S_th/S_h)
        omega_reassigned = omega - np.imag(S_dh/S_h)

    where ``S_h`` is the complex STFT calculated using the original window,
    ``S_dh`` is the complex STFT calculated using the derivative of the original
    window, and ``S_th`` is the complex STFT calculated using the original window
    multiplied by the time offset from the window center. See [#]_ for
    additional algorithms, and [#]_, [#]_, and [#]_ for history and discussion of the
    method.

    .. [#] Flandrin, P., Auger, F., & Chassande-Mottin, E. (2002).
        Time-Frequency reassignment: From principles to algorithms. In
        Applications in Time-Frequency Signal Processing (Vol. 10, pp.
        179-204). CRC Press.

    .. [#] Fulop, S. A., & Fitz, K. (2006). Algorithms for computing the
        time-corrected instantaneous frequency (reassigned) spectrogram, with
        applications. The Journal of the Acoustical Society of America, 119(1),
        360. doi:10.1121/1.2133000

    .. [#] Auger, F., Flandrin, P., Lin, Y.-T., McLaughlin, S., Meignen, S.,
        Oberlin, T., & Wu, H.-T. (2013). Time-Frequency Reassignment and
        Synchrosqueezing: An Overview. IEEE Signal Processing Magazine, 30(6),
        32-41. doi:10.1109/MSP.2013.2265316

    .. [#] Hainsworth, S., Macleod, M. (2003). Time-frequency reassignment: a
        review and analysis. Tech. Rep. CUED/FINFENG/TR.459, Cambridge
        University Engineering Department

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)], real-valued
        audio time series. Multi-channel is supported.

    sr : number > 0 [scalar]
        sampling rate of ``y``

    S : np.ndarray [shape=(..., d, t)] or None
        (optional) complex STFT calculated using the other arguments provided
        to ``reassigned_spectrogram``

    n_fft : int > 0 [scalar]
        FFT window size. Defaults to 2048.

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.

    win_length : int > 0, <= n_fft
        Window length. Defaults to ``n_fft``.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True`` (default), the signal ``y`` is padded so that frame
          ``S[:, t]`` is centered at ``y[t * hop_length]``. See `Notes` for
          recommended usage in this function.
        - If ``False``, then ``S[:, t]`` begins at ``y[t * hop_length]``.

    reassign_frequencies : boolean
        - If ``True`` (default), the returned frequencies will be instantaneous
          frequency estimates.
        - If ``False``, the returned frequencies will be a read-only view of the
          STFT bin frequencies for all frames.

    reassign_times : boolean
        - If ``True`` (default), the returned times will be corrected
          (reassigned) time estimates for each bin.
        - If ``False``, the returned times will be a read-only view of the STFT
          frame times for all bins.

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating time-frequency reassignments.
        Any bin with ``np.abs(S[f, t])**2 < ref_power`` will be returned as
        `np.nan` in both frequency and time, unless ``fill_nan`` is ``True``. If 0
        is provided, then only bins with zero power will be returned as
        `np.nan` (unless ``fill_nan=True``).

    fill_nan : boolean
        - If ``False`` (default), the frequency and time reassignments for bins
          below the power threshold provided in ``ref_power`` will be returned as
          `np.nan`.
        - If ``True``, the frequency and time reassignments for these bins will
          be returned as the bin center frequencies and frame times.

    clip : boolean
        - If ``True`` (default), estimated frequencies outside the range
          `[0, 0.5 * sr]` or times outside the range `[0, len(y) / sr]` will be
          clipped to those ranges.
        - If ``False``, estimated frequencies and times beyond the bounds of the
          spectrogram may be returned.

    dtype : numeric type
        Complex numeric type for STFT calculation. Default is inferred to match
        the precision of the input signal.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    freqs, times, mags : np.ndarray [shape=(..., 1 + n_fft/2, t), dtype=real]
        Instantaneous frequencies:
            ``freqs[..., f, t]`` is the frequency for bin ``f``, frame ``t``.
            If ``reassign_frequencies=False``, this will instead be a read-only array
            of the same shape containing the bin center frequencies for all frames.

        Reassigned times:
            ``times[..., f, t]`` is the time for bin ``f``, frame ``t``.
            If ``reassign_times=False``, this will instead be a read-only array of
            the same shape containing the frame times for all bins.

        Magnitudes from short-time Fourier transform:
            ``mags[..., f, t]`` is the magnitude for bin ``f``, frame ``t``.

    Warns
    -----
    RuntimeWarning
        Frequency or time estimates with zero support will produce a
        divide-by-zero warning, and will be returned as `np.nan` unless
        ``fill_nan=True``.

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    It is recommended to use ``center=False`` with this function rather than the
    librosa default ``True``. Unlike ``stft``, reassigned times are not aligned to
    the left or center of each frame, so padding the signal does not affect the
    meaning of the reassigned times. However, reassignment assumes that the
    energy in each FFT bin is associated with exactly one signal component and
    impulse event.

    If ``reassign_times`` is ``False``, the frame times that are returned will be
    aligned to the left or center of the frame, depending on the value of
    ``center``. In this case, if ``center`` is ``True``, then ``pad_mode="wrap"`` is
    recommended for valid estimation of the instantaneous frequencies in the
    boundary frames.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> amin = 1e-10
    >>> n_fft = 64
    >>> sr = 4000
    >>> y = 1e-3 * librosa.clicks(times=[0.3], sr=sr, click_duration=1.0,
    ...                           click_freq=1200.0, length=8000) +\
    ...     1e-3 * librosa.clicks(times=[1.5], sr=sr, click_duration=0.5,
    ...                           click_freq=400.0, length=8000) +\
    ...     1e-3 * librosa.chirp(fmin=200, fmax=1600, sr=sr, duration=2.0) +\
    ...     1e-6 * np.random.randn(2*sr)
    >>> freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=sr,
    ...                                                     n_fft=n_fft)
    >>> mags_db = librosa.amplitude_to_db(mags, ref=np.max)

    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> imgpow = librosa.display.specshow(mags_db, x_axis="s", y_axis="linear", sr=sr,
    ...                          hop_length=n_fft//4, ax=ax[0])
    >>> ax[0].set(title='Spectrogram', xlabel=None)
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(perceptual_weighting(mags**2, freqs, ref=np.max),
    ...                                 y_axis='cqt_hz', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Perceptually weighted spectrogram')
    >>> fig.colorbar(imgpow, ax=ax[0])
    >>> fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    """
    if not callable(ref_power) and ref_power < 0:
        raise ParameterError("ref_power must be non-negative or callable.")

    if not reassign_frequencies and not reassign_times:
        raise ParameterError("reassign_frequencies or reassign_times must be True.")

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # frequency and time reassignment if requested
    if reassign_frequencies:
        freqs, S = __reassign_frequencies(
            y=y,
            sr=sr,
            S=S,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    if reassign_times:
        times, S = __reassign_times(
            y=y,
            sr=sr,
            S=S,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            dtype=dtype,
            pad_mode=pad_mode,
        )

    assert S is not None

    mags: np.ndarray = np.abs(S)

    # clean up reassignment issues: divide-by-zero, bins with near-zero power,
    # and estimates outside the spectrogram bounds

    # retrieve bin frequencies and frame times to replace missing estimates
    if fill_nan or not reassign_frequencies or not reassign_times:
        if center:
            pad_length = None

        else:
            pad_length = n_fft

        bin_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)

        frame_times = convert.frames_to_time(
            frames=np.arange(S.shape[-1]),
            sr=sr,
            hop_length=hop_length,
            n_fft=pad_length,
        )

    # find bins below the power threshold
    # reassigned bins with zero power will already be NaN
    if callable(ref_power):
        ref_p = ref_power(mags**2)
    else:
        ref_p = ref_power
    mags_low = np.less(mags, ref_p**0.5, where=~np.isnan(mags))

    # for reassigned estimates, optionally set thresholded bins to NaN, return
    # bin frequencies and frame times in place of NaN generated by
    # divide-by-zero and power threshold, and clip to spectrogram bounds
    if reassign_frequencies:
        if ref_p > 0:
            freqs[mags_low] = np.nan

        if fill_nan:
            freqs = np.where(np.isnan(freqs), bin_freqs[:, np.newaxis], freqs)

        if clip:
            np.clip(freqs, 0, sr / 2.0, out=freqs)

    # or if reassignment was not requested, return bin frequencies and frame
    # times for every cell is the spectrogram
    else:
        freqs = np.broadcast_to(bin_freqs[:, np.newaxis], S.shape)

    if reassign_times:
        if ref_p > 0:
            times[mags_low] = np.nan

        if fill_nan:
            times = np.where(np.isnan(times), frame_times[np.newaxis, :], times)

        if clip:
            np.clip(times, 0, y.shape[-1] / float(sr), out=times)

    else:
        times = np.broadcast_to(frame_times[np.newaxis, :], S.shape)

    return freqs, times, mags


def magphase(D: np.ndarray, *, power: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that ``D = S * P``.

    Parameters
    ----------
    D : np.ndarray [shape=(..., d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    D_mag : np.ndarray [shape=(..., d, t), dtype=real]
        magnitude of ``D``, raised to ``power``
    D_phase : np.ndarray [shape=(..., d, t), dtype=complex]
        ``exp(1.j * phi)`` where ``phi`` is the phase of ``D``

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)
    >>> phase
    array([[ 1.   +0.000e+00j,  1.   +0.000e+00j, ...,
            -1.   -8.742e-08j, -1.   -8.742e-08j],
           [-1.   -8.742e-08j, -0.775-6.317e-01j, ...,
            -0.885-4.648e-01j,  0.472-8.815e-01j],
           ...,
           [ 1.   -4.342e-12j,  0.028-9.996e-01j, ...,
            -0.222-9.751e-01j, -0.75 -6.610e-01j],
           [-1.   -8.742e-08j, -1.   -8.742e-08j, ...,
             1.   +0.000e+00j,  1.   +0.000e+00j]], dtype=complex64)

    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[ 0.000e+00,  0.000e+00, ..., -3.142e+00, -3.142e+00],
           [-3.142e+00, -2.458e+00, ..., -2.658e+00, -1.079e+00],
           ...,
           [-4.342e-12, -1.543e+00, ..., -1.794e+00, -2.419e+00],
           [-3.142e+00, -3.142e+00, ...,  0.000e+00,  0.000e+00]],
          dtype=float32)
    """
    mag = np.abs(D)

    # Prevent NaNs and return magnitude 0, phase 1+0j for zero
    zeros_to_ones = mag == 0
    mag_nonzero = mag + zeros_to_ones
    # Compute real and imaginary separately, because complex division can
    # produce NaNs when denormalized numbers are involved (< ~2e-39 for
    # complex64, ~5e-309 for complex128)
    phase = np.empty_like(D, dtype=util.dtype_r2c(D.dtype))
    phase.real = D.real / mag_nonzero + zeros_to_ones
    phase.imag = D.imag / mag_nonzero

    mag **= power

    return mag, phase


