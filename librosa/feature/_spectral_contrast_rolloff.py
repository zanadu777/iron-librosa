#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral contrast and roll-off features"""

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

# Spectral contrast dispatch policy:
# - auto: use Rust only above tuned workload threshold
# - rust: force Rust when kernel is available
# - python: force Python fallback
_CONTRAST_RUST_MODE = os.getenv("IRON_LIBROSA_CONTRAST_RUST_MODE", "auto").strip().lower()


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Parse non-negative integer env knobs with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        parsed = int(raw.strip())
    except (TypeError, ValueError):
        return default

    if parsed < minimum:
        return default

    return parsed


_CONTRAST_RUST_WORK_THRESHOLD = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_WORK_THRESHOLD", 1_500_000
)
_CONTRAST_RUST_MIN_FRAMES = _env_int("IRON_LIBROSA_CONTRAST_RUST_MIN_FRAMES", 1200)
_CONTRAST_RUST_STEREO_MIN_FRAMES = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_STEREO_MIN_FRAMES", 2000
)
_CONTRAST_RUST_STEREO_WORK_THRESHOLD = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_STEREO_WORK_THRESHOLD", 4_000_000
)
_CONTRAST_RUST_MULTICHANNEL_MIN_FRAMES = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_MULTICHANNEL_MIN_FRAMES", 800
)
_CONTRAST_RUST_HEAVY_CHANNELS = _env_int("IRON_LIBROSA_CONTRAST_RUST_HEAVY_CHANNELS", 4, minimum=1)
_CONTRAST_RUST_HEAVY_WORK_THRESHOLD = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_HEAVY_WORK_THRESHOLD", 1_200_000
)
_CONTRAST_RUST_HEAVY_MIN_FRAMES = _env_int("IRON_LIBROSA_CONTRAST_RUST_HEAVY_MIN_FRAMES", 300)
_CONTRAST_RUST_FUSED_MONO_MIN_FRAMES = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_FUSED_MONO_MIN_FRAMES", 1400
)
_CONTRAST_RUST_FUSED_MONO_WORK_THRESHOLD = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_FUSED_MONO_WORK_THRESHOLD", 1_600_000
)
_CONTRAST_RUST_FUSED_STEREO_MIN_FRAMES = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_FUSED_STEREO_MIN_FRAMES", 2000
)
_CONTRAST_RUST_FUSED_STEREO_WORK_THRESHOLD = _env_int(
    "IRON_LIBROSA_CONTRAST_RUST_FUSED_STEREO_WORK_THRESHOLD", 4_000_000
)


def _contrast_rust_auto_ok(channel_count: int, n_bins: int, n_frames: int) -> bool:
    """Return whether spectral_contrast should use Rust in auto mode.

    Policy tiers are tuned from empirical scans:
    - mono: only large frame counts are profitable
    - stereo: stay conservative in auto mode (recent scans show regressions)
    - 3 channels: profitable from ~800 frames upward
    - 4+ channels: profitable even around 300 frames when total work is high
    """
    work = channel_count * n_bins * n_frames

    if channel_count <= 1:
        return work >= _CONTRAST_RUST_WORK_THRESHOLD and n_frames >= _CONTRAST_RUST_MIN_FRAMES

    if channel_count == 2:
        return (
            work >= _CONTRAST_RUST_STEREO_WORK_THRESHOLD
            and n_frames >= _CONTRAST_RUST_STEREO_MIN_FRAMES
        )

    if channel_count < _CONTRAST_RUST_HEAVY_CHANNELS:
        return (
            work >= _CONTRAST_RUST_WORK_THRESHOLD
            and n_frames >= _CONTRAST_RUST_MULTICHANNEL_MIN_FRAMES
        )

    return (
        work >= _CONTRAST_RUST_HEAVY_WORK_THRESHOLD
        and n_frames >= _CONTRAST_RUST_HEAVY_MIN_FRAMES
    )


def _contrast_rust_fused_ok(channel_count: int, n_bins: int, n_frames: int) -> bool:
    """Return whether fused contrast kernel should be used for this shape."""
    work = channel_count * n_bins * n_frames

    if channel_count <= 1:
        return (
            work >= _CONTRAST_RUST_FUSED_MONO_WORK_THRESHOLD
            and n_frames >= _CONTRAST_RUST_FUSED_MONO_MIN_FRAMES
        )

    if channel_count == 2:
        return (
            work >= _CONTRAST_RUST_FUSED_STEREO_WORK_THRESHOLD
            and n_frames >= _CONTRAST_RUST_FUSED_STEREO_MIN_FRAMES
        )

    if channel_count < _CONTRAST_RUST_HEAVY_CHANNELS:
        return (
            work >= _CONTRAST_RUST_FUSED_STEREO_WORK_THRESHOLD
            or n_frames >= _CONTRAST_RUST_FUSED_STEREO_MIN_FRAMES
        )

    return True

def spectral_contrast(
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
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
    linear: bool = False,
) -> np.ndarray:
    """Compute spectral contrast

    Each frame of a spectrogram ``S`` is divided into sub-bands.
    For each sub-band, the energy contrast is estimated by comparing
    the mean energy in the top quantile (peak energy) to that of the
    bottom quantile (valley energy).  High contrast values generally
    correspond to clear, narrow-band signals, while low contrast values
    correspond to broad-band noise. [#]_

    .. [#] Jiang, Dan-Ning, Lie Lu, Hong-Jiang Zhang, Jian-Hua Tao,
           and Lian-Hong Cai.
           "Music type classification by spectral contrast feature."
           In Multimedia and Expo, 2002. ICME'02. Proceedings.
           2002 IEEE International Conference on, vol. 1, pp. 113-116.
           IEEE, 2002.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number  > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
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
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    freq : None or np.ndarray [shape=(d,)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies.
    fmin : float > 0
        Frequency cutoff for the first bin ``[0, fmin]``
        Subsequent bins will cover ``[fmin, 2*fmin]`, `[2*fmin, 4*fmin]``, etc.
    n_bands : int > 1
        number of frequency bands
    quantile : float in (0, 1)
        quantile for determining peaks and valleys
    linear : bool
        If `True`, return the linear difference of magnitudes:
        ``peaks - valleys``.
        If `False`, return the logarithmic difference:
        ``log(peaks) - log(valleys)``.

    Returns
    -------
    contrast : np.ndarray [shape=(..., n_bands + 1, t)]
        each row of spectral contrast values corresponds to a given
        octave-based frequency

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                  ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
    >>> ax[0].set(title='Power spectrogram')
    >>> ax[0].label_outer()
    >>> img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
    >>> fig.colorbar(img2, ax=[ax[1]])
    >>> ax[1].set(ylabel='Frequency bands', title='Spectral contrast')
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

    freq = np.atleast_1d(freq)

    if freq.ndim != 1 or len(freq) != S.shape[-2]:
        raise ParameterError(f"freq.shape mismatch: expected ({S.shape[-2]:d},)")

    if n_bands < 1 or not isinstance(n_bands, (int, np.integer)):
        raise ParameterError("n_bands must be a positive integer")

    if not 0.0 < quantile < 1.0:
        raise ParameterError("quantile must lie in the range (0, 1)")

    if fmin <= 0:
        raise ParameterError("fmin must be a positive number")

    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))

    if np.any(octa[:-1] >= 0.5 * sr):
        raise ParameterError(
            "Frequency band exceeds Nyquist. " "Reduce either fmin or n_bands."
        )

    # shape of valleys and peaks based on spectrogram
    shape = list(S.shape)
    shape[-2] = n_bands + 1

    valley = np.zeros(shape)
    peak = np.zeros_like(valley)

    # ── Rust fast path for static 1D frequency bins (default case) ──────────
    # Dispatch per-band kernel if available and workload is large enough to
    # amortize per-band flatten/reshape overhead.
    _contrast_channel_count = int(np.prod(S.shape[:-2])) if S.ndim > 2 else 1
    _contrast_auto_ok = _contrast_rust_auto_ok(
        _contrast_channel_count, S.shape[-2], S.shape[-1]
    )
    _use_rust_contrast = (
        RUST_AVAILABLE
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
        and (
            _CONTRAST_RUST_MODE == "rust"
            or (_CONTRAST_RUST_MODE == "auto" and _contrast_auto_ok)
        )
    )
    _contrast_kernel_f32 = None
    _contrast_kernel_f64 = None
    _contrast_fused_kernel = None
    if _use_rust_contrast:
        _name = (
            "spectral_contrast_band_f32"
            if S.dtype == np.float32
            else "spectral_contrast_band_f64"
        )
        _contrast_kernel = getattr(_rust_ext, _name, None)
        if _name == "spectral_contrast_band_f32":
            _contrast_kernel_f32 = _contrast_kernel
        else:
            _contrast_kernel_f64 = _contrast_kernel
        _fused_name = (
            "spectral_contrast_fused_f32"
            if S.dtype == np.float32
            else "spectral_contrast_fused_f64"
        )
        _contrast_fused_kernel = getattr(_rust_ext, _fused_name, None)
        _use_rust_contrast = _contrast_kernel is not None

    _use_fused_contrast = (
        _use_rust_contrast
        and _contrast_fused_kernel is not None
        and _contrast_rust_fused_ok(_contrast_channel_count, S.shape[-2], S.shape[-1])
    )

    _band_meta = []
    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = np.logical_and(freq >= f_low, freq <= f_high)

        idx = np.flatnonzero(current_band)

        if k > 0:
            current_band[idx[0] - 1] = True

        if k == n_bands:
            current_band[idx[-1] + 1 :] = True

        band_idx = np.flatnonzero(current_band)
        band_start = int(band_idx[0])
        band_stop = int(band_idx[-1]) + 1
        if k < n_bands:
            band_stop -= 1

        idx_q = int(np.maximum(np.rint(quantile * np.sum(current_band)), 1))
        _band_meta.append((band_start, band_stop, idx_q))

    if _use_fused_contrast:
        try:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (_contrast_channel_count, S.shape[-2], S.shape[-1])
            )
            band_starts = np.asarray([m[0] for m in _band_meta], dtype=np.int64)
            band_stops = np.asarray([m[1] for m in _band_meta], dtype=np.int64)
            idx_qs = np.asarray([m[2] for m in _band_meta], dtype=np.int64)
            assert _contrast_fused_kernel is not None
            peak_flat, valley_flat = _contrast_fused_kernel(s_flat, band_starts, band_stops, idx_qs)
            peak = np.reshape(peak_flat, shape)
            valley = np.reshape(valley_flat, shape)

            if linear:
                contrast = peak - valley
            else:
                contrast = power_to_db(peak) - power_to_db(valley)
            return contrast
        except Exception:
            pass

    peak_flat = np.reshape(peak, (_contrast_channel_count, n_bands + 1, S.shape[-1]))
    valley_flat = np.reshape(valley, (_contrast_channel_count, n_bands + 1, S.shape[-1]))

    for k, (band_start, band_stop, idx_q) in enumerate(_band_meta):
        sub_band = S[..., band_start:band_stop, :]

        # ── Try Rust kernel for this band ──────────────────────────────────
        if _use_rust_contrast:
            try:
                sub_flat = np.reshape(
                    np.ascontiguousarray(sub_band), (_contrast_channel_count, sub_band.shape[-2], sub_band.shape[-1])
                )

                n_sub = sub_flat.shape[-2]
                # Rust kernel clamps idx <= n_sub-1. Preserve parity by using
                # Rust only when Python's idx_q lies within that range.
                if n_sub < 2 or idx_q > (n_sub - 1):
                    raise RuntimeError("fall back to Python contrast path")

                # Convert quantile -> effective quantile that reproduces idx_q
                # in the Rust kernel's round(quantile * n_sub) logic.
                q_eff = idx_q / float(n_sub)

                for ch_idx in range(_contrast_channel_count):
                    ch = np.ascontiguousarray(sub_flat[ch_idx])
                    if S.dtype == np.float32:
                        assert _contrast_kernel_f32 is not None
                        pk, vl = _contrast_kernel_f32(ch, q_eff)
                    else:
                        assert _contrast_kernel_f64 is not None
                        pk, vl = _contrast_kernel_f64(ch, q_eff)
                    peak_flat[ch_idx, k, :] = pk[0]
                    valley_flat[ch_idx, k, :] = vl[0]
                continue  # Skip to next band
            except Exception:
                pass  # Fall through to Python path

        # ── Python fallback ────────────────────────────────────────────────

        sortedr = np.sort(sub_band, axis=-2)

        valley[..., k, :] = np.mean(sortedr[..., :idx_q, :], axis=-2)
        peak[..., k, :] = np.mean(sortedr[..., -idx_q:, :], axis=-2)

    if linear:
        contrast = peak - valley
    else:
        contrast = power_to_db(peak) - power_to_db(valley)
    return contrast



def spectral_rolloff(
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
    roll_percent: float = 0.85,
) -> np.ndarray:
    """Compute roll-off frequency.

    The roll-off frequency is defined for each frame as the center frequency
    for a spectrogram bin such that at least roll_percent (0.85 by default)
    of the energy of the spectrum in this frame is contained in this bin and
    the bins below. This can be used to, e.g., approximate the maximum (or
    minimum) frequency by setting roll_percent to a value close to 1 (or 0).

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        audio sampling rate of ``y``
    S : np.ndarray [shape=(d, t)] or None
        (optional) spectrogram magnitude
    n_fft : int > 0 [scalar]
        FFT window size
    hop_length : int > 0 [scalar]
        hop length for STFT. See `librosa.stft` for details.
    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
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
        - If `False`, then frame ``t`` begins at ``y[t * hop_length]``
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses zero padding.
    freq : None or np.ndarray [shape=(d,) or shape=(..., d, t)]
        Center frequencies for spectrogram bins.
        If `None`, then FFT bin center frequencies are used.
        Otherwise, it can be a single array of ``d`` center frequencies,
        or a matrix of center frequencies as constructed by
        `librosa.reassigned_spectrogram`
    roll_percent : float [0 < roll_percent < 1]
        Roll-off percentage.

    Returns
    -------
    rolloff : np.ndarray [shape=(..., 1, t)]
        roll-off frequency for each frame

    Examples
    --------
    From time-series input

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> # Approximate maximum frequencies with roll_percent=0.85 (default)
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr)
    array([[2583.984, 3036.182, ..., 9173.145, 9248.511]])
    >>> # Approximate maximum frequencies with roll_percent=0.99
    >>> rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
    >>> rolloff
    array([[ 7192.09 ,  6739.893, ..., 10960.4  , 10992.7  ]])
    >>> # Approximate minimum frequencies with roll_percent=0.01
    >>> rolloff_min = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.01)
    >>> rolloff_min
    array([[516.797, 538.33 , ..., 764.429, 764.429]])

    From spectrogram input

    >>> S, phase = librosa.magphase(librosa.stft(y))
    >>> librosa.feature.spectral_rolloff(S=S, sr=sr)
    array([[2583.984, 3036.182, ..., 9173.145, 9248.511]])

    >>> # With a higher roll percentage:
    >>> librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    array([[ 3919.043,  3994.409, ..., 10443.604, 10594.336]])

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax)
    >>> ax.plot(librosa.times_like(rolloff), rolloff[0], label='Roll-off frequency (0.99)')
    >>> ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
    ...         label='Roll-off frequency (0.01)')
    >>> ax.legend(loc='lower right')
    >>> ax.set(title='log Power spectrogram')
    """
    if not 0.0 < roll_percent < 1.0:
        raise ParameterError("roll_percent must lie in the range (0, 1)")

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
            "Spectral rolloff is only defined " "with real-valued input"
        )
    elif np.any(S < 0):
        raise ParameterError(
            "Spectral rolloff is only defined " "with non-negative energies"
        )

    # Compute the center frequencies of each bin
    if freq is None:
        freq = fft_frequencies(sr=sr, n_fft=n_fft)

    # Keep 2D variable-frequency validation explicit for clearer errors.
    if isinstance(freq, np.ndarray) and S.ndim == 2 and freq.ndim == 2 and freq.shape != S.shape:
        raise ParameterError(
            f"freq.shape mismatch: expected {S.shape}, found {freq.shape}"
        )

    # Rust fast path for static 1D frequency bins.
    if (
        RUST_AVAILABLE
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
        and isinstance(freq, np.ndarray)
        and freq.ndim == 1
        and freq.dtype == np.float64
    ):
        _rolloff_name = "spectral_rolloff_f32" if S.dtype == np.float32 else "spectral_rolloff_f64"
        _rolloff_kernel = getattr(_rust_ext, _rolloff_name, None)
        if _rolloff_kernel is not None:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
            )
            freq_c = np.ascontiguousarray(freq)
            roll_frames = [
                _rolloff_kernel(channel, freq_c, float(roll_percent))
                for channel in s_flat
            ]

            if S.ndim == 2:
                return roll_frames[0]

            return np.stack(roll_frames, axis=0).reshape(*S.shape[:-2], 1, S.shape[-1])

    # Pilot variable-frequency path is disabled until full parity is proven.
    if (
        False
        and
        RUST_AVAILABLE
        and S.dtype in (np.float32, np.float64)
        and S.ndim == 2
        and isinstance(freq, np.ndarray)
        and freq.ndim == 2
        and freq.shape == S.shape
        and freq.dtype == np.float64
    ):
        _var_name = (
            "spectral_rolloff_variable_freq_f32"
            if S.dtype == np.float32
            else "spectral_rolloff_variable_freq_f64"
        )
        _var_kernel = getattr(_rust_ext, _var_name, None)
        if _var_kernel is not None:
            return _var_kernel(
                np.ascontiguousarray(S),
                np.ascontiguousarray(freq),
                float(roll_percent),
            )

    # Make sure that frequency can be broadcast
    if freq.ndim == 1:
        # reshape for broadcasting
        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)

    total_energy = np.cumsum(S, axis=-2)
    # (channels,freq,frames)

    threshold = roll_percent * total_energy[..., -1, :]

    # reshape threshold for broadcasting
    threshold = np.expand_dims(threshold, axis=-2)

    ind = np.where(total_energy < threshold, np.nan, 1)

    rolloff: np.ndarray = np.nanmin(ind * freq, axis=-2, keepdims=True)
    return rolloff


