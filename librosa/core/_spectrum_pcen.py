#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PCEN, Griffin-Lim reconstruction, and spectrogram helper"""
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
# Cache ndarray Hann-equivalence checks by object id to avoid repeated O(n_fft) scans.
_RUST_WINDOW_IS_HANN_CACHE: dict[tuple[int, int, str], bool] = {}

from ._spectrum_stft import stft, istft
@overload
def pcen(
    S: np.ndarray,
    *,
    sr: float = ...,
    hop_length: int = ...,
    gain: float = ...,
    bias: float = ...,
    power: float = ...,
    time_constant: float = ...,
    eps: float = ...,
    b: Optional[float] = ...,
    max_size: int = ...,
    ref: Optional[np.ndarray] = ...,
    axis: int = ...,
    max_axis: Optional[int] = ...,
    zi: Optional[np.ndarray] = ...,
    return_zf: Literal[False] = ...,
) -> np.ndarray:
    ...


@overload
def pcen(
    S: np.ndarray,
    *,
    sr: float = ...,
    hop_length: int = ...,
    gain: float = ...,
    bias: float = ...,
    power: float = ...,
    time_constant: float = ...,
    eps: float = ...,
    b: Optional[float] = ...,
    max_size: int = ...,
    ref: Optional[np.ndarray] = ...,
    axis: int = ...,
    max_axis: Optional[int] = ...,
    zi: Optional[np.ndarray] = ...,
    return_zf: Literal[True],
) -> Tuple[np.ndarray, np.ndarray]:
    ...


@overload
def pcen(
    S: np.ndarray,
    *,
    sr: float = ...,
    hop_length: int = ...,
    gain: float = ...,
    bias: float = ...,
    power: float = ...,
    time_constant: float = ...,
    eps: float = ...,
    b: Optional[float] = ...,
    max_size: int = ...,
    ref: Optional[np.ndarray] = ...,
    axis: int = ...,
    max_axis: Optional[int] = ...,
    zi: Optional[np.ndarray] = ...,
    return_zf: bool = ...,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    ...


@cache(level=30)
def pcen(
    S: np.ndarray,
    *,
    sr: float = 22050,
    hop_length: int = 512,
    gain: float = 0.98,
    bias: float = 2,
    power: float = 0.5,
    time_constant: float = 0.400,
    eps: float = 1e-6,
    b: Optional[float] = None,
    max_size: int = 1,
    ref: Optional[np.ndarray] = None,
    axis: int = -1,
    max_axis: Optional[int] = None,
    zi: Optional[np.ndarray] = None,
    return_zf: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Per-channel energy normalization (PCEN)

    This function normalizes a time-frequency representation ``S`` by
    performing automatic gain control, followed by nonlinear compression [#]_ ::

        P[f, t] = (S / (eps + M[f, t])**gain + bias)**power - bias**power

    IMPORTANT: the default values of eps, gain, bias, and power match the
    original publication, in which ``S`` is a 40-band mel-frequency
    spectrogram with 25 ms windowing, 10 ms frame shift, and raw audio values
    in the interval [-2**31; 2**31-1[. If you use these default values, we
    recommend to make sure that the raw audio is properly scaled to this
    interval, and not to [-1, 1[ as is most often the case.

    The matrix ``M`` is the result of applying a low-pass, temporal IIR filter
    to ``S``::

        M[f, t] = (1 - b) * M[f, t - 1] + b * S[f, t]

    If ``b`` is not provided, it is calculated as::

        b = (sqrt(1 + 4* T**2) - 1) / (2 * T**2)

    where ``T = time_constant * sr / hop_length``. [#]_

    This normalization is designed to suppress background noise and
    emphasize foreground signals, and can be used as an alternative to
    decibel scaling (`amplitude_to_db`).

    This implementation also supports smoothing across frequency bins
    by specifying ``max_size > 1``.  If this option is used, the filtered
    spectrogram ``M`` is computed as::

        M[f, t] = (1 - b) * M[f, t - 1] + b * R[f, t]

    where ``R`` has been max-filtered along the frequency axis, similar to
    the SuperFlux algorithm implemented in `onset.onset_strength`::

        R[f, t] = max(S[f - max_size//2: f + max_size//2, t])

    This can be used to perform automatic gain control on signals that cross
    or span multiple frequency bans, which may be desirable for spectrograms
    with high frequency resolution.

    .. [#] Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A.
       (2017, March). Trainable frontend for robust and far-field keyword spotting.
       In Acoustics, Speech and Signal Processing (ICASSP), 2017
       IEEE International Conference on (pp. 5670-5674). IEEE.

    .. [#] Lostanlen, V., Salamon, J., McFee, B., Cartwright, M., Farnsworth, A.,
       Kelling, S., and Bello, J. P. Per-Channel Energy Normalization: Why and How.
       IEEE Signal Processing Letters, 26(1), 39-43.

    Parameters
    ----------
    S : np.ndarray (non-negative)
        The input (magnitude) spectrogram

    sr : number > 0 [scalar]
        The audio sampling rate

    hop_length : int > 0 [scalar]
        The hop length of ``S``; defaults to ``512``

    gain : number >= 0 [scalar]
        The gain factor.  Typical values should be slightly less than 1.

    bias : number >= 0 [scalar]
        The bias point of the nonlinear compression (default: 2)

    power : number >= 0 [scalar]
        The compression exponent.  Typical values should be between 0 and 0.5.
        Smaller values of ``power`` result in stronger compression.
        At the limit ``power=0``, polynomial compression becomes logarithmic.

    time_constant : number > 0 [scalar]
        The time constant for IIR filtering, measured in seconds.

    eps : number > 0 [scalar]
        A small constant used to ensure numerical stability of the filter.

    b : number in [0, 1]  [scalar]
        The filter coefficient for the low-pass filter.
        If not provided, it will be inferred from ``time_constant``.

    max_size : int > 0 [scalar]
        The width of the max filter applied to the frequency axis.
        If left as `1`, no filtering is performed.

    ref : None or np.ndarray (shape=S.shape)
        An optional pre-computed reference spectrum (``R`` in the above).
        If not provided it will be computed from ``S``.

    axis : int [scalar]
        The (time) axis of the input spectrogram.

    max_axis : None or int [scalar]
        The frequency axis of the input spectrogram.
        If `None`, and ``S`` is two-dimensional, it will be inferred
        as the opposite from ``axis``.
        If ``S`` is not two-dimensional, and ``max_size > 1``, an error
        will be raised.

    zi : np.ndarray
        The initial filter delay values.

        This may be the ``zf`` (final delay values) of a previous call to ``pcen``, or
        computed by `scipy.signal.lfilter_zi`.

    return_zf : bool
        If ``True``, return the final filter delay values along with the PCEN output ``P``.
        This is primarily useful in streaming contexts, where the final state of one
        block of processing should be used to initialize the next block.

        If ``False`` (default) only the PCEN values ``P`` are returned.

    Returns
    -------
    P : np.ndarray, non-negative [shape=(n, m)]
        The per-channel energy normalized version of ``S``.
    zf : np.ndarray (optional)
        The final filter delay values.  Only returned if ``return_zf=True``.

    See Also
    --------
    amplitude_to_db
    librosa.onset.onset_strength

    Examples
    --------
    Compare PCEN to log amplitude (dB) scaling on Mel spectra

    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('robin'))

    >>> # We recommend scaling y to the range [-2**31, 2**31[ before applying
    >>> # PCEN's default parameters. Furthermore, we use power=1 to get a
    >>> # magnitude spectrum instead of a power spectrum.
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, power=1)
    >>> log_S = librosa.amplitude_to_db(S, ref=np.max)
    >>> pcen_S = librosa.pcen(S * (2**31))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(log_S, x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='log amplitude (dB)', xlabel=None)
    >>> ax[0].label_outer()
    >>> imgpcen = librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', ax=ax[1])
    >>> ax[1].set(title='Per-channel energy normalization')
    >>> fig.colorbar(img, ax=ax[0], format="%+2.0f dB")
    >>> fig.colorbar(imgpcen, ax=ax[1], format="%+2.0f dB")

    Compare PCEN with and without max-filtering

    >>> pcen_max = librosa.pcen(S * (2**31), max_size=3)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(pcen_S, x_axis='time', y_axis='mel', ax=ax[0])
    >>> ax[0].set(title='Per-channel energy normalization (no max-filter)')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(pcen_max, x_axis='time', y_axis='mel', ax=ax[1])
    >>> ax[1].set(title='Per-channel energy normalization (max_size=3)')
    >>> fig.colorbar(img, ax=ax)
    """
    if power < 0:
        raise ParameterError(f"power={power} must be nonnegative")

    if gain < 0:
        raise ParameterError(f"gain={gain} must be non-negative")

    if bias < 0:
        raise ParameterError(f"bias={bias} must be non-negative")

    if eps <= 0:
        raise ParameterError(f"eps={eps} must be strictly positive")

    if time_constant <= 0:
        raise ParameterError(f"time_constant={time_constant} must be strictly positive")

    if not util.is_positive_int(max_size):
        raise ParameterError(f"max_size={max_size} must be a positive integer")

    if b is None:
        t_frames = time_constant * sr / float(hop_length)
        # By default, this solves the equation for b:
        #   b**2  + (1 - b) / t_frames  - 2 = 0
        # which approximates the full-width half-max of the
        # squared frequency response of the IIR low-pass filter

        b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)

    if not 0 <= b <= 1:
        raise ParameterError(f"b={b} must be between 0 and 1")

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn(
            "pcen was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call pcen(np.abs(D)) instead.",
            stacklevel=2,
        )
        S = np.abs(S)

    if ref is None:
        if max_size == 1:
            ref = S
        elif S.ndim == 1:
            raise ParameterError(
                "Max-filtering cannot be applied to 1-dimensional input"
            )
        else:
            if max_axis is None:
                if S.ndim != 2:
                    raise ParameterError(
                        f"Max-filtering a {S.ndim:d}-dimensional spectrogram "
                        "requires you to specify max_axis"
                    )
                # if axis = 0, max_axis=1
                # if axis = +- 1, max_axis = 0
                max_axis = np.mod(1 - axis, 2)

            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=max_axis)

    if zi is None:
        # Make sure zi matches dimension to input
        shape = tuple([1] * ref.ndim)
        zi = np.empty(shape)
        zi[:] = scipy.signal.lfilter_zi([b], [1, b - 1])[:]

    # Temporal integration
    S_smooth: np.ndarray
    zf: np.ndarray
    S_smooth, zf = scipy.signal.lfilter([b], [1, b - 1], ref, zi=zi, axis=axis)

    # Adaptive gain control
    # Working in log-space gives us some stability, and a slight speedup
    smooth = np.exp(-gain * (np.log(eps) + np.log1p(S_smooth / eps)))

    # Dynamic range compression
    S_out: np.ndarray
    if power == 0:
        S_out = np.log1p(S * smooth)
    elif bias == 0:
        S_out = np.exp(power * (np.log(S) + np.log(smooth)))
    else:
        S_out = (bias**power) * np.expm1(power * np.log1p(S * smooth / bias))

    if return_zf:
        return S_out, zf
    else:
        return S_out


def griffinlim(
    S: np.ndarray,
    *,
    n_iter: int = 32,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    pad_mode: _PadModeSTFT = "constant",
    momentum: float = 0.99,
    init: Optional[str] = "random",
    random_state: Optional[
        Union[int, np.random.RandomState, np.random.Generator]
    ] = None,
) -> np.ndarray:
    """Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm.

    Given a short-time Fourier transform magnitude matrix (``S``), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-STFT
    operations. [#]_

    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that ``S`` contains only the non-negative frequencies (as computed by
    `stft`).

    The "fast" GL method [#]_ uses a momentum parameter to accelerate convergence.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236ΓÇô243, Apr. 1984.

    .. [#] Perraudin, N., Balazs, P., & S├╕ndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    Parameters
    ----------
    S : np.ndarray [shape=(..., n_fft // 2 + 1, t), non-negative]
        An array of short-time Fourier transform magnitudes as produced by
        `stft`.

    n_iter : int > 0
        The number of iterations to run

    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to ``n_fft // 4``

    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal ``n_fft``

    n_fft : None or int > 0
        The number of samples per frame.
        By default (None), this will be inferred from the shape of D as an even number.
        However, if an odd frame length was used, the correct
        frame length can be specified here.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`

    center : boolean
        If ``True``, the STFT is assumed to use centered frames.
        If ``False``, the STFT is assumed to use left-aligned frames.

    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is inferred
        to match the precision of the input spectrogram.

    length : None or int > 0
        If provided, the output ``y`` is zero-padded or clipped to exactly ``length``
        samples.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.

    init : None or 'random' [default]
        If 'random' (the default), then phase values are initialized randomly
        according to ``random_state``.  This is recommended when the input ``S`` is
        a magnitude spectrogram with no initial phase estimates.

        If `None`, then the phase is initialized from ``S``.  This is useful when
        an initial guess for phase can be provided, or when you want to resume
        Griffin-Lim from a previous output.

    random_state : None, int, np.random.RandomState, or np.random.Generator
        If int, random_state is the seed used by the random number generator
        for phase initialization.

        If `np.random.RandomState` or `np.random.Generator` instance, the random number
        generator itself.

        If `None`, defaults to the `np.random.default_rng()` object.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time-domain signal reconstructed from ``S``

    See Also
    --------
    stft
    istft
    magphase
    filters.get_window

    Examples
    --------
    A basic STFT inverse example

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> # Get the magnitude spectrogram
    >>> S = np.abs(librosa.stft(y))
    >>> # Invert using Griffin-Lim
    >>> y_inv = librosa.griffinlim(S)
    >>> # Invert without estimating phase
    >>> y_istft = librosa.istft(S)

    Wave-plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.waveshow(y, sr=sr, color='b', ax=ax[0])
    >>> ax[0].set(title='Original', xlabel=None)
    >>> ax[0].label_outer()
    >>> librosa.display.waveshow(y_inv, sr=sr, color='g', ax=ax[1])
    >>> ax[1].set(title='Griffin-Lim reconstruction', xlabel=None)
    >>> ax[1].label_outer()
    >>> librosa.display.waveshow(y_istft, sr=sr, color='r', ax=ax[2])
    >>> ax[2].set_title('Magnitude-only istft reconstruction')
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)  # type: ignore
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rng = random_state  # type: ignore
    else:
        raise ParameterError(f"Unsupported random_state={random_state!r}")

    if momentum > 1:
        warnings.warn(
            f"Griffin-Lim with momentum={momentum} > 1 can be unstable. "
            "Proceed with caution!",
            stacklevel=2,
        )
    elif momentum < 0:
        raise ParameterError(f"griffinlim() called with momentum={momentum} < 0")

    # Infer n_fft from the spectrogram shape
    if n_fft is None:
        n_fft = 2 * (S.shape[-2] - 1)

    # Infer the dtype from S
    angles = np.empty(S.shape, dtype=util.dtype_r2c(S.dtype))
    eps = util.tiny(angles)

    if init == "random":
        # randomly initialize the phase
        angles[:] = util.phasor((2 * np.pi * rng.random(size=S.shape)))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise ParameterError(f"init={init} must either None or 'random'")

    # Place-holders for temporary data and reconstructed buffer
    rebuilt = None
    tprev = None
    inverse = None

    # Absorb magnitudes into angles
    angles *= S
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = istft(
            angles,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            out=inverse,
        )

        # Rebuild the spectrogram
        rebuilt = stft(
            inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            out=rebuilt,
        )

        # Update our phase estimates
        angles[:] = rebuilt
        if tprev is not None:
            angles -= (momentum / (1 + momentum)) * tprev
        angles /= np.abs(angles) + eps
        angles *= S
        # Store
        rebuilt, tprev = tprev, rebuilt

    # Return the final phase estimates
    return istft(
        angles,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        window=window,
        center=center,
        dtype=dtype,
        length=length,
        out=inverse,
    )


def _spectrogram(
    *,
    y: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    n_fft: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    power: float = 1,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
) -> Tuple[np.ndarray, int]:
    """Retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.

    Parameters
    ----------
    y : None or np.ndarray
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length ``win_length`` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, the STFT is assumed to use left-aligned frames.

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float]
        - If ``S`` is provided as input, then ``S_out == S``
        - Else, ``S_out = |stft(y, ...)|**power``
    n_fft : int > 0
        - If ``S`` is provided, then ``n_fft`` is inferred from ``S``
        - Else, copied from input
    """
    def _is_rust_hann_window(win_spec: _WindowSpec, fft_size: int) -> bool:
        """Check if window is Hann (string or precomputed periodic array)."""
        if isinstance(win_spec, str) and win_spec == "hann":
            return True

        cache_key: Optional[tuple[int, int, str]] = None
        if isinstance(win_spec, np.ndarray):
            cache_key = (id(win_spec), int(fft_size), str(win_spec.dtype))
            cached = _RUST_WINDOW_IS_HANN_CACHE.get(cache_key)
            if cached is not None:
                return cached

        try:
            win_arr = np.asarray(win_spec)
        except Exception:
            return False

        if win_arr.ndim != 1 or win_arr.shape[0] != int(fft_size) or np.iscomplexobj(win_arr):
            return False

        hann_ref = get_window("hann", int(fft_size), fftbins=True)
        is_hann = np.allclose(win_arr, hann_ref, rtol=1e-7, atol=1e-10)

        if cache_key is not None:
            _RUST_WINDOW_IS_HANN_CACHE[cache_key] = is_hann

        return is_hann

    def _extract_window_array(win_spec: _WindowSpec, fft_size: int) -> Optional[np.ndarray]:
        """Extract window array from window spec. Returns None for Hann string or unsupported specs."""
        if isinstance(win_spec, str):
            # String window specs: only pass None to Rust, which falls back to Hann
            return None

        try:
            win_arr = np.asarray(win_spec)
            if win_arr.ndim == 1 and win_arr.shape[0] == int(fft_size) and not np.iscomplexobj(win_arr):
                return win_arr
        except Exception:
            pass

        return None

    if S is not None:
        # Infer n_fft from spectrogram shape, but only if it mismatches
        if n_fft is None or n_fft // 2 + 1 != S.shape[-2]:
            n_fft = 2 * (S.shape[-2] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        if n_fft is None:
            raise ParameterError(f"Unable to compute spectrogram with n_fft={n_fft}")
        if y is None:
            raise ParameterError(
                "Input signal must be provided to compute a spectrogram"
            )
        # Check if we can dispatch to Rust. We support:
        # 1. window="hann" (string)
        # 2. Precomputed window array (any type that can be converted to float32)
        window_for_rust = None
        _rust_window_ok = False

        if isinstance(window, str):
            # String windows: only Hann can dispatch
            _rust_window_ok = (window == "hann")
        else:
            # Precomputed window: extract and validate length
            window_for_rust = _extract_window_array(window, n_fft)
            _rust_window_ok = (window_for_rust is not None)

            # Route periodic Hann arrays through Rust's internal Hann path.
            # This avoids redundant window transfer while preserving behavior.
            if _rust_window_ok and _is_rust_hann_window(window, n_fft):
                window_for_rust = None

        _rust_stft_ok = (
            FORCE_RUST_STFT
            and not FORCE_NUMPY_STFT
            and RUST_AVAILABLE
            and y.ndim == 1
            and y.dtype in (np.float32, np.float64)
            and _rust_window_ok
            and (win_length is None or win_length == n_fft)
            and power == 2.0
            and (
                (y.dtype == np.float32 and hasattr(_rust_ext, "stft_power"))
                or (y.dtype == np.float64 and hasattr(_rust_ext, "stft_power_f64"))
            )
        )

        if _rust_stft_ok:
            # Mirror stft()'s f64-upcast decision for float32 input.
            # stft() uses stft_complex_f64 for float32 y when:
            #   center=True  and n_fft >= 2048, OR
            #   center=False and n_fft <= 1024
            # To ensure _spectrogram(y=y) == np.abs(stft(y))**power we must
            # use the same precision path.  stft_power_f64 on float64 y gives
            # float64 power; cast back to float32 at the end.
            _use_f64_for_f32 = (
                y.dtype == np.float32
                and hasattr(_rust_ext, "stft_power_f64")
                and (
                    (center is False and int(n_fft) <= 1024)
                    or (center is True and int(n_fft) >= 2048)
                )
            )

            if _use_f64_for_f32:
                y_c = np.ascontiguousarray(y, dtype=np.float64)
                win_c = None
                if window_for_rust is not None:
                    win_c = np.ascontiguousarray(window_for_rust, dtype=np.float64)
                S_f64 = _rust_ext.stft_power_f64(
                    y_c,
                    int(n_fft),
                    int(hop_length) if hop_length is not None else n_fft // 4,
                    bool(center),
                    win_c,
                )
                S = S_f64.astype(np.float32)
            else:
                y_c = np.ascontiguousarray(y)

                # Match window dtype to the selected kernel.
                win_c = None
                if window_for_rust is not None:
                    win_dtype = np.float32 if y.dtype == np.float32 else np.float64
                    win_c = np.ascontiguousarray(window_for_rust, dtype=win_dtype)

                rust_stft_power = (
                    _rust_ext.stft_power if y.dtype == np.float32 else _rust_ext.stft_power_f64
                )

                S = rust_stft_power(
                    y_c,
                    int(n_fft),
                    int(hop_length) if hop_length is not None else n_fft // 4,
                    bool(center),
                    win_c,  # None uses Hann in Rust, otherwise uses provided window
                )
        else:
            S = (
                np.abs(
                    stft(
                        y,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        center=center,
                        window=window,
                        pad_mode=pad_mode,
                    )
                )
                ** power
            )

    return S, n_fft
