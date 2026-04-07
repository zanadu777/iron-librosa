#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chroma and tonnetz features"""

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
