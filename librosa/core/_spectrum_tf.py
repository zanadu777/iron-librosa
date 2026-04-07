#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase vocoder and IIR filtering"""
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

from ._spectrum_stft import stft, istft
def phase_vocoder(
    D: np.ndarray,
    *,
    rate: float,
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    prefer_rust: bool = True,
) -> np.ndarray:
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of ``rate``

    Based on the implementation provided by [#]_.

    This is a simplified implementation, intended primarily for
    reference and pedagogical purposes.  It makes no attempt to
    handle transients, and is likely to produce many audible
    artifacts.  For a higher quality implementation, we recommend
    the RubberBand library [#]_ and its Python wrapper `pyrubberband`.

    .. [#] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        https://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    .. [#] https://breakfastquay.com/rubberband/

    Examples
    --------
    >>> # Play at double speed
    >>> y, sr   = librosa.load(librosa.ex('trumpet'))
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_fast  = librosa.phase_vocoder(D, rate=2.0, hop_length=512)
    >>> y_fast  = librosa.istft(D_fast, hop_length=512)

    >>> # Or play at 1/3 speed
    >>> y, sr   = librosa.load(librosa.ex('trumpet'))
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_slow  = librosa.phase_vocoder(D, rate=1./3, hop_length=512)
    >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    Parameters
    ----------
    D : np.ndarray [shape=(..., d, t), dtype=complex]
        STFT matrix

    rate : float > 0 [scalar]
        Speed-up factor: ``rate > 1`` is faster, ``rate < 1`` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of ``D``.

        If None, defaults to ``n_fft//4 = (D.shape[0]-1)//2``

    n_fft : int > 0 or None
        The number of samples per frame in D.
        By default (None), this will be inferred from the shape of D.
        However, if D was constructed using an odd-length window, the correct
        frame length can be specified here.

    prefer_rust : bool
        If True (default) and Rust acceleration is available, use the Rust kernel
        for improved performance. If False, always use the pure-Python implementation.
        The Rust kernel produces numerically identical results (within machine
        precision) to the Python reference implementation.

    Returns
    -------
    D_stretched : np.ndarray [shape=(..., d, t / rate), dtype=complex]
        time-stretched STFT

    See Also
    --------
    pyrubberband
    """
    if n_fft is None:
        n_fft = 2 * (D.shape[-2] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    # Create an empty output array
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # Expected phase advance in each bin per frame
    phi_advance = hop_length * convert.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[..., 0])

    # Pad 0 columns to simplify boundary logic
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    # Pre-compute frame indices and interpolation weights once.
    step_int = np.floor(time_steps).astype(int)
    step_alpha = time_steps - step_int

    # Pre-compute phase and magnitude after padding to avoid repeated
    # np.angle/np.abs calls inside the synthesis loop.
    D_phase = np.angle(D)
    D_mag = np.abs(D)

    # ── Rust fast-path: eliminate Python frame loop ────────────────────────────
    # Dispatch to Rust kernel when: Rust is available, input is a real ndarray (not masked),
    # dtype is complex64 or complex128, and prefer_rust=True.
    # Handles mono (ndim==2) and multichannel (ndim>2) by iterating Rust calls per-channel.
    _rust_pv_fn = None
    _pv_float_dtype: DTypeLike = np.float32
    if prefer_rust and RUST_AVAILABLE and isinstance(D, np.ndarray) and D.ndim >= 2:
        if D.dtype == np.complex64 and hasattr(_rust_ext, "phase_vocoder_f32"):
            _rust_pv_fn = _rust_ext.phase_vocoder_f32
            _pv_float_dtype = np.float32
        elif D.dtype == np.complex128 and hasattr(_rust_ext, "phase_vocoder_f64"):
            _rust_pv_fn = _rust_ext.phase_vocoder_f64
            _pv_float_dtype = np.float64

    if _rust_pv_fn is not None:
        _step_int_i64 = np.ascontiguousarray(step_int.astype(np.int64))
        # Keep phase vectors in float64 for complex64 parity; f64 path already uses f64.
        _phi = np.ascontiguousarray(
            phi_advance if D.dtype == np.complex64 else phi_advance.astype(np.float64)
        )
        _step_alpha = np.ascontiguousarray(step_alpha.astype(np.float64))

        if D.ndim == 2:
            # Mono: single Rust call.
            # Transpose to (n_padded_frames, n_bins) for cache-friendly row access.
            _dpt = np.ascontiguousarray(D_phase.astype(_pv_float_dtype).T)
            _dmt = np.ascontiguousarray(D_mag.astype(_pv_float_dtype).T)
            _pa = np.ascontiguousarray(phase_acc.astype(np.float64))
            return _rust_pv_fn(_dpt, _dmt, _phi, _step_int_i64, _step_alpha, _pa)
        else:
            # Multichannel: iterate batch dimensions, fill d_stretch per channel.
            _batch_shape = D.shape[:-2]
            _dph_typed = D_phase.astype(_pv_float_dtype)
            _dmg_typed = D_mag.astype(_pv_float_dtype)
            _pa_typed = phase_acc.astype(np.float64)
            for _ch in np.ndindex(*_batch_shape):
                _dpt = np.ascontiguousarray(_dph_typed[_ch].T)
                _dmt = np.ascontiguousarray(_dmg_typed[_ch].T)
                _pa = np.ascontiguousarray(_pa_typed[_ch])
                d_stretch[_ch] = _rust_pv_fn(
                    _dpt, _dmt, _phi, _step_int_i64, _step_alpha, _pa
                )
            return d_stretch
    # ── end Rust fast-path ─────────────────────────────────────────────────────

    for t, idx in enumerate(step_int):
        # Weighting for linear magnitude interpolation
        alpha = step_alpha[t]
        mag = (1.0 - alpha) * D_mag[..., idx] + alpha * D_mag[..., idx + 1]

        # Store to output array
        d_stretch[..., t] = util.phasor(phase_acc, mag=mag)

        # Compute phase advance
        dphase = D_phase[..., idx + 1] - D_phase[..., idx] - phi_advance

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


@cache(level=20)
def iirt(
    y: np.ndarray,
    *,
    sr: float = 22050,
    win_length: int = 2048,
    hop_length: Optional[int] = None,
    center: bool = True,
    tuning: float = 0.0,
    pad_mode: _PadMode = "constant",
    flayout: str = "sos",
    res_type: str = "soxr_hq",
    **kwargs: Any,
) -> np.ndarray:
    r"""Time-frequency representation using IIR filters

    This function will return a time-frequency representation
    using a multirate filter bank consisting of IIR filters. [#]_

    First, ``y`` is resampled as needed according to the provided ``sample_rates``.

    Then, a filterbank with ``n`` band-pass filters is designed.

    The resampled input signals are processed by the filterbank as a whole.
    (`scipy.signal.filtfilt` resp. `sosfiltfilt` is used to make the phase linear.)
    The output of the filterbank is cut into frames.
    For each band, the short-time mean-square power (STMSP) is calculated by
    summing ``win_length`` subsequent filtered time samples.

    When called with the default set of parameters, it will generate the TF-representation
    (pitch filterbank):

        * 85 filters with MIDI pitches [24, 108] as ``center_freqs``.
        * each filter having a bandwidth of one semitone.

    .. [#] M├╝ller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    win_length : int > 0, <= n_fft
        Window length.
    hop_length : int > 0 [scalar]
        Hop length, number samples between subsequent frames.
        If not supplied, defaults to ``win_length // 4``.
    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``D[..., :, t]`` is centered at ``y[t * hop_length]``.
        - If ``False``, then `D[..., :, t]`` begins at ``y[t * hop_length]``
    tuning : float [scalar]
        Tuning deviation from A440 in fractions of a bin.
    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, this function uses zero padding.
    flayout : string
        - If `sos` (default), a series of second-order filters is used for filtering with `scipy.signal.sosfiltfilt`.
          Minimizes numerical precision errors for high-order filters, but is slower.
        - If `ba`, the standard difference equation is used for filtering with `scipy.signal.filtfilt`.
          Can be unstable for high-order filters.
    res_type : string
        The resampling mode.  See `librosa.resample` for details.
    **kwargs : additional keyword arguments
        Additional arguments for `librosa.filters.semitone_filterbank`
        (e.g., could be used to provide another set of ``center_freqs`` and ``sample_rates``).

    Returns
    -------
    bands_power : np.ndarray [shape=(..., n, t), dtype=dtype]
        Short-time mean-square power for the input signal.

    Raises
    ------
    ParameterError
        If ``flayout`` is not None, `ba`, or `sos`.

    See Also
    --------
    librosa.filters.semitone_filterbank
    librosa.filters.mr_frequencies
    librosa.cqt
    scipy.signal.filtfilt
    scipy.signal.sosfiltfilt

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> D = np.abs(librosa.iirt(y, sr=sr))
    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
    ...                                y_axis='cqt_hz', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Constant-Q transform')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                                y_axis='cqt_hz', x_axis='time', ax=ax[1])
    >>> ax[1].set_title('Semitone spectrogram (iirt)')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    if flayout not in ("ba", "sos"):
        raise ParameterError(f"Unsupported flayout={flayout}")

    # check audio input
    util.valid_audio(y)

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = win_length // 4

    # Pad the time series so that frames are centered
    if center:
        padding = [(0, 0) for _ in y.shape]
        padding[-1] = (win_length // 2, win_length // 2)
        y = np.pad(y, padding, mode=pad_mode)

    # get the semitone filterbank
    filterbank_ct, sample_rates = semitone_filterbank(
        tuning=tuning, flayout=flayout, **kwargs
    )

    # create three downsampled versions of the audio signal
    y_resampled = []

    y_srs = np.unique(sample_rates)

    for cur_sr in y_srs:
        y_resampled.append(resample(y, orig_sr=sr, target_sr=cur_sr, res_type=res_type))

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = int(1 + (y.shape[-1] - win_length) // hop_length)

    # Pre-allocate the output array
    shape = list(y.shape)
    # Time dimension reduces to n_frames
    shape[-1] = n_frames
    # Insert a new axis at position -2 for filter response
    shape.insert(-1, len(filterbank_ct))

    bands_power = np.empty_like(y, shape=shape)

    slices: List[Union[int, slice]] = [slice(None) for _ in bands_power.shape]
    for i, (cur_sr, cur_filter) in enumerate(zip(sample_rates, filterbank_ct)):
        slices[-2] = i

        # filter the signal
        cur_sr_idx = np.flatnonzero(y_srs == cur_sr)[0]

        if flayout == "ba":
            cur_filter_output = scipy.signal.filtfilt(
                cur_filter[0], cur_filter[1], y_resampled[cur_sr_idx], axis=-1
            )
        elif flayout == "sos":
            cur_filter_output = scipy.signal.sosfiltfilt(
                cur_filter, y_resampled[cur_sr_idx], axis=-1
            )

        factor = sr / cur_sr
        hop_length_STMSP = hop_length / factor
        win_length_STMSP_round = int(round(win_length / factor))

        # hop_length_STMSP is used here as a floating-point number.
        # The discretization happens at the end to avoid accumulated rounding errors.
        start_idx = np.arange(
            0, cur_filter_output.shape[-1] - win_length_STMSP_round, hop_length_STMSP
        )
        if len(start_idx) < n_frames:
            min_length = (
                int(np.ceil(n_frames * hop_length_STMSP)) + win_length_STMSP_round
            )
            cur_filter_output = util.fix_length(cur_filter_output, size=min_length)
            start_idx = np.arange(
                0,
                cur_filter_output.shape[-1] - win_length_STMSP_round,
                hop_length_STMSP,
            )
        start_idx = np.round(start_idx).astype(int)[:n_frames]

        idx = np.add.outer(start_idx, np.arange(win_length_STMSP_round))

        bands_power[tuple(slices)] = factor * np.sum(
            cur_filter_output[..., idx] ** 2, axis=-1
        )

    return bands_power


