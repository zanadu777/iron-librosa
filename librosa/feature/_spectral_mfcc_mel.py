#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MFCC and mel-spectrogram features"""

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
