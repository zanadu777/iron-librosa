#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction"""

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


__all__ = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_rolloff",
    "spectral_flatness",
    "poly_features",
    "rms",
    "zero_crossing_rate",
    "chroma_stft",
    "chroma_cqt",
    "chroma_cens",
    "chroma_vqt",
    "melspectrogram",
    "mfcc",
    "tonnetz",
]


# Heuristic crossover for mel projection backend in 2D path.
# Counts multiply-accumulate ops: n_mels * n_fft_bins * n_frames.
# Auto-calibrated via  python calibrate_mel_threshold.py  (rewrites this line).
# 0 = always use NumPy/BLAS (correct for MKL-backed machines after calibration).
# Set IRON_LIBROSA_MEL_BACKEND=rust env var to force the Rust faer path.
_MEL_RUST_WORK_THRESHOLD = 201_226_955


def _load_external_mel_threshold_registry() -> Dict[str, int]:
    """Load optional per-profile mel thresholds from JSON file."""
    registry_path = os.getenv("IRON_LIBROSA_MEL_THRESHOLD_FILE", "").strip()
    if not registry_path:
        return {}

    try:
        with open(registry_path, "r", encoding="utf-8") as fdesc:
            data = json.load(fdesc)
    except Exception:
        return {}

    # Accept either {"thresholds": {...}} or a flat mapping.
    if isinstance(data, dict) and isinstance(data.get("thresholds"), dict):
        data = data["thresholds"]

    if not isinstance(data, dict):
        return {}

    out: Dict[str, int] = {}
    for key, value in data.items():
        try:
            parsed = int(value)
            if parsed >= 0:
                out[str(key)] = parsed
        except (TypeError, ValueError):
            continue

    return out


def _resolve_mel_work_threshold() -> int:
    """Resolve mel auto-dispatch threshold with cross-CPU profile support.

    Precedence:
      1) IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD (explicit integer override)
      2) IRON_LIBROSA_MEL_PROFILE against external JSON registry
      3) IRON_LIBROSA_MEL_PROFILE against built-in registry
      4) _MEL_RUST_WORK_THRESHOLD fallback constant
    """
    env_override = os.getenv("IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD")
    if env_override is not None:
        try:
            parsed = int(env_override.strip())
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass

    profile = os.getenv("IRON_LIBROSA_MEL_PROFILE", "").strip()
    if profile:
        external = _load_external_mel_threshold_registry()
        if profile in external:
            return external[profile]

        if profile in MEL_WORK_THRESHOLDS:
            return MEL_WORK_THRESHOLDS[profile]

    return _MEL_RUST_WORK_THRESHOLD


# -- Spectral features -- #
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

    # Pilot: variable-frequency fast path for 2-D inputs.
    if (
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
def chroma_stft(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    norm: Optional[float] = np.inf,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    pad_mode: _PadModeSTFT = "constant",
    tuning: Optional[float] = None,
    n_chroma: int = 12,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a chromagram from a waveform or power spectrogram.

    This implementation is derived from ``chromagram_E`` [#]_

    .. [#] Ellis, Daniel P.W.  "Chroma feature analysis and synthesis"
           2007/04/21
           https://www.ee.columbia.edu/~dpwe/resources/matlab/chroma-ansyn/

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        power spectrogram
    norm : float or None
        Column-wise normalization.
        See `librosa.util.normalize` for details.
        If `None`, no normalization is performed.
    n_fft : int  > 0 [scalar]
        FFT window size if provided ``y, sr`` instead of ``S``
    hop_length : int > 0 [scalar]
        hop length if provided ``y, sr`` instead of ``S``
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
    tuning : float [scalar] or None.
        Deviation from A440 tuning in fractional chroma bins.
        If `None`, it is automatically estimated.
    n_chroma : int > 0 [scalar]
        Number of chroma bins to produce (12 by default).
    **kwargs : additional keyword arguments to parameterize chroma filters.
    ctroct : float > 0 [scalar]
    octwidth : float > 0 or None [scalar]
        ``ctroct`` and ``octwidth`` specify a dominance window:
        a Gaussian weighting centered on ``ctroct`` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of ``octwidth``.
        Set ``octwidth`` to `None` to use a flat weighting.
    norm : float > 0 or np.inf
        Normalization factor for each filter
    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    chromagram : np.ndarray [shape=(..., n_chroma, t)]
        Normalized energy for each chroma bin at each frame.

    See Also
    --------
    librosa.filters.chroma : Chroma filter bank construction
    librosa.util.normalize : Vector normalization

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> librosa.feature.chroma_stft(y=y, sr=sr)
    array([[1.   , 0.962, ..., 0.143, 0.278],
           [0.688, 0.745, ..., 0.103, 0.162],
           ...,
           [0.468, 0.598, ..., 0.18 , 0.342],
           [0.681, 0.702, ..., 0.553, 1.   ]], dtype=float32)

    Use an energy (magnitude) spectrum instead of power spectrogram

    >>> S = np.abs(librosa.stft(y))
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[1.   , 0.973, ..., 0.527, 0.569],
           [0.774, 0.81 , ..., 0.518, 0.506],
           ...,
           [0.624, 0.73 , ..., 0.611, 0.644],
           [0.766, 0.822, ..., 0.92 , 1.   ]], dtype=float32)

    Use a pre-computed power spectrogram with a larger frame

    >>> S = np.abs(librosa.stft(y, n_fft=4096))**2
    >>> chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    >>> chroma
    array([[0.994, 0.873, ..., 0.169, 0.227],
           [0.735, 0.64 , ..., 0.141, 0.135],
           ...,
           [0.6  , 0.937, ..., 0.214, 0.257],
           [0.743, 0.937, ..., 0.684, 0.815]], dtype=float32)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax[0])
    >>> fig.colorbar(img, ax=[ax[0]])
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> fig.colorbar(img, ax=[ax[1]])
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

    if tuning is None:
        tuning = estimate_tuning(S=S, sr=sr, bins_per_octave=n_chroma)

    # Get the filter bank
    chromafb = filters.chroma(
        sr=sr, n_fft=n_fft, tuning=tuning, n_chroma=n_chroma, **kwargs
    )

    # Rust fast path for 2-D S with compatible dtype.
    if (
        RUST_AVAILABLE
        and np.isrealobj(S)
        and S.dtype in (np.float32, np.float64)
        and S.ndim >= 2
    ):
        _chroma_name = (
            "chroma_project_f32" if S.dtype == np.float32 else "chroma_project_f64"
        )
        _chroma_kernel = getattr(_rust_ext, _chroma_name, None)

        if _chroma_kernel is not None:
            s_flat = np.reshape(
                np.ascontiguousarray(S), (-1, S.shape[-2], S.shape[-1])
            )
            chromafb_c = np.ascontiguousarray(chromafb.astype(S.dtype))
            chromas = [_chroma_kernel(channel, chromafb_c) for channel in s_flat]

            if S.ndim == 2:
                raw_chroma = chromas[0]
            else:
                raw_chroma = np.stack(chromas, axis=0).reshape(*S.shape[:-2], n_chroma, S.shape[-1])
        else:
            # Fallback to Python einsum
            raw_chroma = np.einsum("cf,...ft->...ct", chromafb, S, optimize=True)
    else:
        # Fallback to Python einsum
        raw_chroma = np.einsum("cf,...ft->...ct", chromafb, S, optimize=True)

    # Compute normalization factor for each frame
    return util.normalize(raw_chroma, norm=norm, axis=-2)


def chroma_cqt(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    C: Optional[np.ndarray] = None,
    hop_length: int = 512,
    fmin: Optional[_FloatLike_co] = None,
    norm: Optional[Union[int, float]] = np.inf,
    threshold: float = 0.0,
    tuning: Optional[float] = None,
    n_chroma: int = 12,
    n_octaves: int = 7,
    window: Optional[np.ndarray] = None,
    bins_per_octave: Optional[int] = 36,
    cqt_mode: str = "full",
) -> np.ndarray:
    r"""Constant-Q chromagram

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time series. Multi-channel is supported.
    sr : number > 0
        sampling rate of ``y``
    C : np.ndarray [shape=(..., d, t)] [Optional]
        a pre-computed constant-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.
    tuning : float [scalar] or None.
        Deviation (in fractions of a CQT bin) from A440 tuning
    n_chroma : int > 0 [scalar]
        Number of chroma bins to produce
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`
    bins_per_octave : int > 0, optional
        Number of bins per octave in the CQT.
        Must be an integer multiple of ``n_chroma``.
        Default: 36 (3 bins per semitone)
        If `None`, it will match ``n_chroma``.
    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    Returns
    -------
    chromagram : np.ndarray [shape=(..., n_chroma, t)]
        The output chromagram

    See Also
    --------
    librosa.util.normalize
    librosa.cqt
    librosa.hybrid_cqt
    chroma_stft

    Examples
    --------
    Compare a long-window STFT chromagram to the CQT chromagram

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr,
    ...                                           n_chroma=12, n_fft=4096)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='chroma_stft')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='chroma_cqt')
    >>> fig.colorbar(img, ax=ax)
    """
    cqt_func = {"full": cqt, "hybrid": hybrid_cqt}

    if bins_per_octave is None:
        bins_per_octave = n_chroma
    elif np.remainder(bins_per_octave, n_chroma) != 0:
        raise ParameterError(
            f"bins_per_octave={bins_per_octave} must be an integer "
            f"multiple of n_chroma={n_chroma}"
        )

    # Build the CQT if we don't have one already
    if C is None:
        if y is None:
            raise ParameterError(
                "At least one of C or y must be provided to compute chroma"
            )
        C = np.abs(
            cqt_func[cqt_mode](
                y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
                tuning=tuning,
            )
        )

    # Map to chroma
    cq_to_chr = filters.cq_to_chroma(
        C.shape[-2],
        bins_per_octave=bins_per_octave,
        n_chroma=n_chroma,
        fmin=fmin,
        window=window,
    )

    chroma = np.einsum("cf,...ft->...ct", cq_to_chr, C, optimize=True)

    if threshold is not None:
        chroma[chroma < threshold] = 0.0

    # Normalize
    chroma = util.normalize(chroma, norm=norm, axis=-2)

    return chroma


def chroma_cens(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    C: Optional[np.ndarray] = None,
    hop_length: int = 512,
    fmin: Optional[_FloatLike_co] = None,
    tuning: Optional[float] = None,
    n_chroma: int = 12,
    n_octaves: int = 7,
    bins_per_octave: int = 36,
    cqt_mode: str = "full",
    window: Optional[np.ndarray] = None,
    norm: Optional[float] = 2,
    win_len_smooth: Optional[int] = 41,
    smoothing_window: _WindowSpec = "hann",
) -> np.ndarray:
    r"""Compute the chroma variant "Chroma Energy Normalized" (CENS)

    To compute CENS features, following steps are taken after obtaining chroma vectors
    using `chroma_cqt`: [#]_.

        1. L-1 normalization of each chroma vector
        2. Quantization of amplitude based on "log-like" amplitude thresholds
        3. (optional) Smoothing with sliding window. Default window length = 41 frames
        4. (not implemented) Downsampling

    CENS features are robust to dynamics, timbre and articulation, thus these are commonly used in audio
    matching and retrieval applications.

    .. [#] Meinard Müller and Sebastian Ewert
           "Chroma Toolbox: MATLAB implementations for extracting variants of chroma-based audio features"
           In Proceedings of the International Conference on Music Information Retrieval (ISMIR), 2011.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time series. Multi-channel is supported.
    sr : number > 0
        sampling rate of ``y``
    C : np.ndarray [shape=(d, t)] [Optional]
        a pre-computed constant-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    tuning : float [scalar] or None.
        Deviation (in fractions of a CQT bin) from A440 tuning
    n_chroma : int > 0
        Number of chroma bins to produce
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`
    bins_per_octave : int > 0
        Number of bins per octave in the CQT.
        Default: 36
    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode
    win_len_smooth : int > 0 or None
        Length of temporal smoothing window. `None` disables temporal smoothing.
        Default: 41
    smoothing_window : str, float or tuple
        Type of window function for temporal smoothing. See `librosa.filters.get_window` for possible inputs.
        Default: 'hann'

    Returns
    -------
    cens : np.ndarray [shape=(..., n_chroma, t)]
        The output cens-chromagram

    See Also
    --------
    chroma_cqt : Compute a chromagram from a constant-Q transform.
    chroma_stft : Compute a chromagram from an STFT spectrogram or waveform.
    librosa.filters.get_window : Compute a window function.

    Examples
    --------
    Compare standard cqt chroma to CENS.

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=15)
    >>> chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='chroma_cq')
    >>> ax[0].label_outer()
    >>> librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='chroma_cens')
    >>> fig.colorbar(img, ax=ax)
    """
    if not (
        (win_len_smooth is None)
        or (isinstance(win_len_smooth, (int, np.integer)) and win_len_smooth > 0)
    ):
        raise ParameterError(
            f"win_len_smooth={win_len_smooth} must be a positive integer or None"
        )

    chroma = chroma_cqt(
        y=y,
        C=C,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        norm=None,
        n_chroma=n_chroma,
        n_octaves=n_octaves,
        cqt_mode=cqt_mode,
        window=window,
    )

    # L1-Normalization
    chroma = util.normalize(chroma, norm=1, axis=-2)

    # Quantize amplitudes
    QUANT_STEPS = [0.4, 0.2, 0.1, 0.05]
    QUANT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

    chroma_quant = np.zeros_like(chroma)

    for cur_quant_step_idx, cur_quant_step in enumerate(QUANT_STEPS):
        chroma_quant += (chroma > cur_quant_step) * QUANT_WEIGHTS[cur_quant_step_idx]

    if win_len_smooth:
        # Apply temporal smoothing
        win = filters.get_window(smoothing_window, win_len_smooth + 2, fftbins=False)
        win /= np.sum(win)

        # reshape for broadcasting
        win = util.expand_to(win, ndim=chroma_quant.ndim, axes=-1)

        cens = scipy.ndimage.convolve(chroma_quant, win, mode="constant")
    else:
        cens = chroma_quant

    # L2-Normalization
    return util.normalize(cens, norm=norm, axis=-2)


def chroma_vqt(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    V: Optional[np.ndarray] = None,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    intervals: Union[str, Collection[float]],
    norm: Optional[float] = np.inf,
    threshold: float = 0.0,
    n_octaves: int = 7,
    bins_per_octave: int = 12,
    gamma: float = 0,
) -> np.ndarray:
    r"""Variable-Q chromagram

    This differs from CQT-based chroma by supporting non-equal temperament
    intervals.

    Note: unlike CQT- and STFT-based chroma, VQT chroma does not aggregate energy
    from neighboring frequency bands.  As a result, the number of chroma
    features produced is equal to the number of intervals used, or equivalently,
    the number of bins per octave in the underlying VQT representation.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time series. Multi-channel is supported.
    sr : number > 0
        sampling rate of ``y``
    V : np.ndarray [shape=(..., d, t)] [Optional]
        a pre-computed variable-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    intervals : str or array of floats in [1, 2)
        Either a string specification for an interval set, e.g.,
        `'equal'`, `'pythagorean'`, `'ji3'`, etc. or an array of
        intervals expressed as numbers between 1 and 2.
        .. see also:: librosa.interval_frequencies
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    bins_per_octave : int > 0, optional
        Number of bins per octave in the VQT.
    gamma : number > 0 [scalar]
        Bandwidth offset for determining filter lengths.
        .. see also:: librosa.vqt

    Returns
    -------
    chromagram : np.ndarray [shape=(..., bins_per_octave, t)]
        The output chromagram

    See Also
    --------
    librosa.util.normalize
    librosa.vqt
    chroma_cqt
    librosa.interval_frequencies

    Examples
    --------
    Compare an equal-temperament CQT chromagram to a 5-limit just intonation
    chromagram.  Both use 36 bins per octave.

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> n_bins = 36
    >>> chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=n_bins)
    >>> chroma_vq = librosa.feature.chroma_vqt(y=y, sr=sr,
    ...                                        intervals='ji5',
    ...                                        bins_per_octave=n_bins)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time',
    ...                          ax=ax[0], bins_per_octave=n_bins)
    >>> ax[0].set(ylabel='chroma_cqt')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(chroma_vq, y_axis='chroma_fjs', x_axis='time',
    ...                                ax=ax[1], bins_per_octave=n_bins,
    ...                                intervals='ji5')
    >>> ax[1].set(ylabel='chroma_vqt')
    >>> fig.colorbar(img, ax=ax)
    """
    # If intervals are provided as an array, override BPO
    if not isinstance(intervals, str):
        bins_per_octave = len(intervals)

    # Build the CQT if we don't have one already
    if V is None:
        if y is None:
            raise ParameterError(
                "At least one of y or V must be provided to compute chroma"
            )
        V = np.abs(
            vqt(
                y=y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                intervals=intervals,
                n_bins=n_octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
                gamma=gamma,
            )
        )

    # Map to chroma
    vq_to_chr = filters.cq_to_chroma(
        V.shape[-2],
        bins_per_octave=bins_per_octave,
        n_chroma=bins_per_octave,
        fmin=fmin,
    )

    chroma = np.einsum("cf,...ft->...ct", vq_to_chr, V, optimize=True)

    if threshold is not None:
        chroma[chroma < threshold] = 0.0

    # Normalize
    chroma = util.normalize(chroma, norm=norm, axis=-2)

    return chroma


def tonnetz(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    chroma: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute the tonal centroid features (tonnetz)

    This representation uses the method of [#]_ to project chroma features
    onto a 6-dimensional basis representing the perfect fifth, minor third,
    and major third each as two-dimensional coordinates.

    .. [#] Harte, C., Sandler, M., & Gasser, M. (2006). "Detecting Harmonic
           Change in Musical Audio." In Proceedings of the 1st ACM Workshop
           on Audio and Music Computing Multimedia (pp. 21-26).
           Santa Barbara, CA, USA: ACM Press. doi:10.1145/1178723.1178727.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        Audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    chroma : np.ndarray [shape=(n_chroma, t)] or None
        Normalized energy for each chroma bin at each frame.
        If `None`, a cqt chromagram is performed.
    **kwargs : Additional keyword arguments to `chroma_cqt`,
        if ``chroma`` is not pre-computed.
    C : np.ndarray [shape=(..., d, t)] [Optional]
        a pre-computed constant-Q spectrogram
    hop_length : int > 0
        number of samples between successive chroma frames
    fmin : float > 0
        minimum frequency to analyze in the CQT.
        Default: `C1 ~= 32.7 Hz`
    norm : int > 0, +-np.inf, or None
        Column-wise normalization of the chromagram.
    threshold : float
        Pre-normalization energy threshold.  Values below the
        threshold are discarded, resulting in a sparse chromagram.
    tuning : float [scalar] or None.
        Deviation (in fractions of a CQT bin) from A440 tuning
    n_chroma : int > 0
        Number of chroma bins to produce
    n_octaves : int > 0
        Number of octaves to analyze above ``fmin``
    window : None or np.ndarray
        Optional window parameter to `filters.cq_to_chroma`
    bins_per_octave : int > 0, optional
        Number of bins per octave in the CQT.
        Must be an integer multiple of ``n_chroma``.
        Default: 36 (3 bins per semitone)
        If `None`, it will match ``n_chroma``.
    cqt_mode : ['full', 'hybrid']
        Constant-Q transform mode

    Returns
    -------
    tonnetz : np.ndarray [shape(..., 6, t)]
        Tonal centroid features for each frame.

        Tonnetz dimensions:
            - 0: Fifth x-axis
            - 1: Fifth y-axis
            - 2: Minor x-axis
            - 3: Minor y-axis
            - 4: Major x-axis
            - 5: Major y-axis

    See Also
    --------
    chroma_cqt : Compute a chromagram from a constant-Q transform.
    chroma_stft : Compute a chromagram from an STFT spectrogram or waveform.

    Examples
    --------
    Compute tonnetz features from the harmonic component of a song

    >>> y, sr = librosa.load(librosa.ex('nutcracker'), duration=10, offset=10)
    >>> y = librosa.effects.harmonic(y)
    >>> tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    >>> tonnetz
    array([[ 0.007, -0.026, ...,  0.055,  0.056],
           [-0.01 , -0.009, ..., -0.012, -0.017],
           ...,
           [ 0.006, -0.021, ..., -0.012, -0.01 ],
           [-0.009,  0.031, ..., -0.05 , -0.037]])

    Compare the tonnetz features to `chroma_cqt`

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img1 = librosa.display.specshow(tonnetz,
    ...                                 y_axis='tonnetz', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Tonal Centroids (Tonnetz)')
    >>> ax[0].label_outer()
    >>> img2 = librosa.display.specshow(librosa.feature.chroma_cqt(y=y, sr=sr),
    ...                                 y_axis='chroma', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Chroma')
    >>> fig.colorbar(img1, ax=[ax[0]])
    >>> fig.colorbar(img2, ax=[ax[1]])
    """
    if y is None and chroma is None:
        raise ParameterError(
            "Either the audio samples or the chromagram must be "
            "passed as an argument."
        )

    if chroma is None:
        chroma = chroma_cqt(y=y, sr=sr, **kwargs)

    # Generate Transformation matrix
    dim_map = np.linspace(0, 12, num=chroma.shape[-2], endpoint=False)

    scale = np.asarray([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])

    V = np.multiply.outer(scale, dim_map)

    # Even rows compute sin()
    V[::2] -= 0.5

    R = np.array([1, 1, 1, 1, 0.5, 0.5])  # Fifths  # Minor  # Major

    phi = R[:, np.newaxis] * np.cos(np.pi * V)

    # Do the transform to tonnetz
    ton: np.ndarray = np.einsum(
        "pc,...ci->...pi", phi, util.normalize(chroma, norm=1, axis=-2), optimize=True
    )
    return ton


# -- Mel spectrogram and MFCCs -- #
def mfcc(
    *,
    y: Optional[np.ndarray] = None,
    sr: float = 22050,
    S: Optional[np.ndarray] = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: Optional[str] = "ortho",
    lifter: float = 0,
    mel_norm: Optional[Union[Literal["slaney"], float]] = "slaney",
    **kwargs: Any,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients (MFCCs)

    .. warning:: If multi-channel audio input ``y`` is provided, the MFCC
        calculation will depend on the peak loudness (in decibels) across
        all channels.  The result may differ from independent MFCC calculation
        of each channel.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)] or None
        audio time series. Multi-channel is supported..
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        log-power Mel spectrogram
    n_mfcc : int > 0 [scalar]
        number of MFCCs to return
    dct_type : {1, 2, 3}
        Discrete cosine transform (DCT) type.
        By default, DCT type-2 is used.
    norm : None or 'ortho'
        If ``dct_type`` is `2 or 3`, setting ``norm='ortho'`` uses an ortho-normal
        DCT basis.
        Normalization is not supported for ``dct_type=1``.
    lifter : number >= 0
        If ``lifter>0``, apply *liftering* (cepstral filtering) to the MFCCs::
            M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter) * lifter / 2)
        Setting ``lifter >= 2 * n_mfcc`` emphasizes the higher-order coefficients.
        As ``lifter`` increases, the coefficient weighting becomes approximately linear.
    mel_norm : `norm` argument to `melspectrogram`
    **kwargs : additional keyword arguments to `melspectrogram`
        if operating on time series input
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
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
    power : float > 0 [scalar]
        Exponent applied to the spectrum before calculating the melspectrogram when the input is a time signal,
        e.g. 1 for magnitude, 2 for power **(default)**, etc.
    **kwargs : additional keyword arguments for Mel filter bank parameters
    n_mels : int > 0 [scalar]
        number of Mel bands to generate
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    M : np.ndarray [shape=(..., n_mfcc, t)]
        MFCC sequence

    See Also
    --------
    melspectrogram
    scipy.fft.dct

    Examples
    --------
    Generate mfccs from a time series

    >>> y, sr = librosa.load(librosa.ex('libri1'))
    >>> librosa.feature.mfcc(y=y, sr=sr)
    array([[-565.919, -564.288, ..., -426.484, -434.668],
           [  10.305,   12.509, ...,   88.43 ,   90.12 ],
           ...,
           [   2.807,    2.068, ...,   -6.725,   -5.159],
           [   2.822,    2.244, ...,   -6.198,   -6.177]], dtype=float32)

    Using a different hop length and HTK-style Mel frequencies

    >>> librosa.feature.mfcc(y=y, sr=sr, hop_length=1024, htk=True)
    array([[-5.471e+02, -5.464e+02, ..., -4.446e+02, -4.200e+02],
           [ 1.361e+01,  1.402e+01, ...,  9.764e+01,  9.869e+01],
           ...,
           [ 4.097e-01, -2.029e+00, ..., -1.051e+01, -1.130e+01],
           [-1.119e-01, -1.688e+00, ..., -3.442e+00, -4.687e+00]],
          dtype=float32)

    Use a pre-computed log-power Mel spectrogram

    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                    fmax=8000)
    >>> librosa.feature.mfcc(S=librosa.power_to_db(S))
    array([[-559.974, -558.449, ..., -411.96 , -420.458],
           [  11.018,   13.046, ...,   76.972,   80.888],
           ...,
           [   2.713,    2.379, ...,    1.464,   -2.835],
           [   2.712,    2.619, ...,    2.209,    0.648]], dtype=float32)

    Get more components

    >>> mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    Visualize the MFCC series

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
    ...                                x_axis='time', y_axis='mel', fmax=8000,
    ...                                ax=ax[0])
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel spectrogram')
    >>> ax.label_outer()
    >>> img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
    >>> fig.colorbar(img, ax=ax[1])
    >>> ax[1].set(title='MFCC')

    Compare different DCT bases

    >>> m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
    >>> m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
    >>> ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
    >>> fig.colorbar(img, ax=[ax[0]])
    >>> img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
    >>> ax[1].set(title='HTK-style (dct_type=3)')
    >>> fig.colorbar(img2, ax=[ax[1]])
    """
    if S is None:
        # multichannel behavior may be different due to relative noise floor differences between channels
        S = power_to_db(melspectrogram(y=y, sr=sr, norm = mel_norm, **kwargs))

    # Rust fast path for the most common MFCC setting.
    if (
        S.ndim == 2
        and S.dtype in (np.float32, np.float64)
        and dct_type == 2
        and norm == "ortho"
        and RUST_AVAILABLE
    ):
        _dct_name = "dct2_ortho_f64" if S.dtype == np.float64 else "dct2_ortho_f32"
        if hasattr(_rust_ext, _dct_name):
            M = getattr(_rust_ext, _dct_name)(np.ascontiguousarray(S), int(n_mfcc))
        else:
            fft = get_fftlib()
            M = fft.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]
    else:
        fft = get_fftlib()
        M = fft.dct(S, axis=-2, type=dct_type, norm=norm)[
            ..., :n_mfcc, :
        ]

    if lifter > 0:
        # shape lifter for broadcasting
        LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
        LI = util.expand_to(LI, ndim=S.ndim, axes=-2)

        M *= 1 + (lifter / 2) * LI
        return M
    elif lifter == 0:
        return M
    else:
        raise ParameterError(f"MFCC lifter={lifter} must be a non-negative number")


def melspectrogram(
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
    power: float = 2.0,
    **kwargs: Any,
) -> np.ndarray:
    """Compute a mel-scaled spectrogram.

    If a spectrogram input ``S`` is provided, then it is mapped directly onto
    the mel basis by ``mel_f.dot(S)``.

    If a time-series input ``y, sr`` is provided, then its magnitude spectrogram
    ``S`` is first computed, and then mapped onto the mel scale by
    ``mel_f.dot(S**power)``.

    By default, ``power=2`` operates on a power spectrum.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)] or None
        audio time-series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, t)] or None
        spectrogram
    n_fft : int > 0 [scalar]
        length of the FFT window
    hop_length : int > 0 [scalar]
        number of samples between successive frames.
        See `librosa.stft`
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
    power : float > 0 [scalar]
        Exponent for the magnitude melspectrogram.
        e.g., 1 for energy, 2 for power **(default)**, etc.
    **kwargs : additional keyword arguments for Mel filter bank parameters
    n_mels : int > 0 [scalar]
        number of Mel bands to generate
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    htk : bool [scalar]
        use HTK formula instead of Slaney
    norm : {None, 'slaney', or number} [scalar]
        If 'slaney', divide the triangular mel weights by the width of
        the mel band (area normalization).
        If numeric, use `librosa.util.normalize` to normalize each filter
        by to unit l_p norm. See `librosa.util.normalize` for a full
        description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0
    dtype : np.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.

    Returns
    -------
    S : np.ndarray [shape=(..., n_mels, t)]
        Mel spectrogram

    See Also
    --------
    librosa.filters.mel : Mel filter bank construction
    librosa.stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> librosa.feature.melspectrogram(y=y, sr=sr)
    array([[3.837e-06, 1.451e-06, ..., 8.352e-14, 1.296e-11],
           [2.213e-05, 7.866e-06, ..., 8.532e-14, 1.329e-11],
           ...,
           [1.115e-05, 5.192e-06, ..., 3.675e-08, 2.470e-08],
           [6.473e-07, 4.402e-07, ..., 1.794e-08, 2.908e-08]],
          dtype=float32)

    Using a pre-computed power spectrogram would give the same result:

    >>> D = np.abs(librosa.stft(y))**2
    >>> S = librosa.feature.melspectrogram(S=D, sr=sr)

    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):

    >>> # Passing through arguments to the Mel filters
    >>> S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
    ...                                     fmax=8000)

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> S_dB = librosa.power_to_db(S, ref=np.max)
    >>> img = librosa.display.specshow(S_dB, x_axis='time',
    ...                          y_axis='mel', sr=sr,
    ...                          fmax=8000, ax=ax)
    >>> fig.colorbar(img, ax=ax, format='%+2.0f dB')
    >>> ax.set(title='Mel-frequency spectrogram')
    """
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Build a Mel filter
    mel_basis = filters.mel(sr=sr, n_fft=n_fft, **kwargs)

    # Fast path for the common 2D case.
    # Backend selection is adaptive by workload size, with env overrides:
    #   IRON_LIBROSA_MEL_BACKEND=numpy|rust|auto (default auto)
    if S.ndim == 2:
        n_mels = mel_basis.shape[0]
        n_fft_bins = mel_basis.shape[1]
        n_frames = S.shape[1]
        work = n_mels * n_fft_bins * n_frames
        mel_threshold = _resolve_mel_work_threshold()

        use_rust = (
            RUST_AVAILABLE
            and not FORCE_NUMPY_MEL
            and (FORCE_RUST_MEL or work <= mel_threshold)
        )

        if use_rust:
            if (
                S.dtype == np.float32
                and mel_basis.dtype == np.float32
                and hasattr(_rust_ext, "mel_project_f32")
            ):
                return _rust_ext.mel_project_f32(
                    np.ascontiguousarray(S),
                    np.ascontiguousarray(mel_basis),
                )

            if hasattr(_rust_ext, "mel_project_f64"):
                out = _rust_ext.mel_project_f64(
                    np.ascontiguousarray(S, dtype=np.float64),
                    np.ascontiguousarray(mel_basis, dtype=np.float64),
                )
                return out.astype(np.result_type(S.dtype, mel_basis.dtype), copy=False)

        # NumPy dot is usually BLAS-backed (MKL/OpenBLAS/Accelerate).
        return mel_basis.dot(S)

    melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
    return melspec
