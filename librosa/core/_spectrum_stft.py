#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STFT and iSTFT core functions"""
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

def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    pad_mode: _PadModeSTFT = "constant",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Short-time Fourier transform (STFT).

    The STFT represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping
    windows.

    This function returns a complex-valued matrix D such that

    - ``np.abs(D[..., f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and

    - ``np.angle(D[..., f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.

    The integers ``t`` and ``f`` can be converted to physical units by means
    of the utility functions `frames_to_samples` and `fft_frequencies`.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)], real-valued
        input signal. Multi-channel is supported.

    n_fft : int > 0 [scalar]
        length of the windowed signal after padding with zeros.
        The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
        The default value, ``n_fft=2048`` samples, corresponds to a physical
        duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
        default sample rate in librosa. This value is well adapted for music
        signals. However, in speech processing, the recommended value is 512,
        corresponding to 23 milliseconds at a sample rate of 22050 Hz.
        In any case, we recommend setting ``n_fft`` to a power of two for
        optimizing the speed of the fast Fourier transform (FFT) algorithm.

    hop_length : int > 0 [scalar]
        number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to ``win_length // 4`` (see below).

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window`` of length ``win_length``
        and then padded with zeros to match ``n_fft``.  Padding is added on
        both the left- and the right-side of the window so that the window
        is centered within the frame.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time)
        at the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal ``y``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        Either:

        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        Defaults to a raised cosine window (`'hann'`), which is adequate for
        most applications in audio signal processing.

        .. see also:: `filters.get_window`

    center : boolean
        If ``True``, the signal ``y`` is padded so that frame
        ``D[:, t]`` is centered at ``y[t * hop_length]``.

        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.

        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of `librosa.frames_to_samples`.
        Note, however, that ``center`` must be set to `False` when analyzing
        signals with `librosa.stream`.

        .. see also:: `librosa.stream`

    dtype : np.dtype, optional
        Complex numeric type for ``D``.  Default is inferred to match the
        precision of the input signal.

    pad_mode : string or function
        If ``center=True``, this argument is passed to `np.pad` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.

        .. note:: Not all padding modes supported by `numpy.pad` are supported here.
            `wrap`, `mean`, `maximum`, `median`, and `minimum` are not supported.

            Other modes that depend at most on input values at the edges of the
            signal (e.g., `constant`, `edge`, `linear_ramp`) are supported.

        If ``center=False``,  this argument is ignored.

        .. see also:: `numpy.pad`

    out : np.ndarray or None
        A pre-allocated, complex-valued array to store the STFT results.
        This must be of compatible shape and dtype for the given input parameters.

        If `out` is larger than necessary for the provided input signal, then only
        a prefix slice of `out` will be used.

        If not provided, a new array is allocated and returned.

    Returns
    -------
    D : np.ndarray [shape=(..., 1 + n_fft/2, n_frames), dtype=dtype]
        Complex-valued matrix of short-term Fourier transform
        coefficients.

        If a pre-allocated `out` array is provided, then `D` will be
        a reference to `out`.

        If `out` is larger than necessary, then `D` will be a sliced
        view: `D = out[..., :n_frames]`.

    See Also
    --------
    istft : Inverse STFT
    reassigned_spectrogram : Time-frequency reassigned spectrogram

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> S = np.abs(librosa.stft(y))
    >>> S
    array([[5.395e-03, 3.332e-03, ..., 9.862e-07, 1.201e-05],
           [3.244e-03, 2.690e-03, ..., 9.536e-07, 1.201e-05],
           ...,
           [7.523e-05, 3.722e-05, ..., 1.188e-04, 1.031e-03],
           [7.640e-05, 3.944e-05, ..., 5.180e-04, 1.346e-03]],
          dtype=float32)

    Use left-aligned frames, instead of centered frames

    >>> S_left = librosa.stft(y, center=False)

    Use a shorter hop length

    >>> D_short = librosa.stft(y, hop_length=64)

    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S,
    ...                                                        ref=np.max),
    ...                                y_axis='log', x_axis='time', ax=ax)
    >>> ax.set_title('Power spectrogram')
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)
    elif not util.is_positive_int(hop_length):
        raise ParameterError(f"hop_length={hop_length} must be a positive integer")

    # Check audio is valid
    util.valid_audio(y)

    fft_window: Optional[np.ndarray] = None

    # Rust fast-path for 1D complex STFT.
    # Keep this conservative to preserve exact behavior for advanced cases.
    # Requires explicit opt-in via IRON_LIBROSA_STFT_BACKEND=rust because
    # rustfft and numpy/scipy FFT have ~1e-5 float32 round-off differences
    # that break the MATLAB-parity tests (test_stft / test___reassign_*).
    _rust_stft_ok = (
        FORCE_RUST_STFT
        and not FORCE_NUMPY_STFT
        and RUST_AVAILABLE
        and y.ndim >= 1
        and y.dtype in (np.float32, np.float64)
        and win_length == n_fft
        and n_fft <= y.shape[-1]
        and out is None
        and (center is False or pad_mode == "constant")
        and (
            (
                y.dtype == np.float32
                and (
                    hasattr(_rust_ext, "stft_complex_f64")
                    or hasattr(_rust_ext, "stft_complex")
                )
            )
            or (y.dtype == np.float64 and hasattr(_rust_ext, "stft_complex_f64"))
        )
    )

    if _rust_stft_ok:
        # Use float64 kernel only for strict center=False small-FFT parity cases;
        # otherwise prefer float32 kernel for throughput.
        use_f64_kernel = (
            y.dtype == np.float32
            and hasattr(_rust_ext, "stft_complex_f64")
            and (
                # Preserve strict parity in small-FFT uncentered paths.
                (center is False and n_fft <= 1024)
                # Improve parity for default display/STFT fixtures.
                or (center is True and n_fft >= 2048)
            )
        )

        y_target_dtype = np.float64 if use_f64_kernel else y.dtype
        y_c = np.ascontiguousarray(y, dtype=y_target_dtype)
        win_dtype = np.float64 if use_f64_kernel or y.dtype == np.float64 else np.float32

        # Default full-frame periodic Hann: use a cached numpy window to avoid
        # repeated Hann construction cost in the Rust extension.
        if isinstance(window, str) and window == "hann" and win_length == n_fft:
            cache_key = (int(n_fft), np.dtype(win_dtype))
            win_cached = _RUST_HANN_WINDOW_CACHE.get(cache_key)
            if win_cached is None:
                win_cached = np.ascontiguousarray(
                    get_window("hann", n_fft, fftbins=True), dtype=win_dtype
                )
                _RUST_HANN_WINDOW_CACHE[cache_key] = win_cached
            win_c = win_cached
        else:
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = util.pad_center(fft_window, size=n_fft)
            win_c = np.ascontiguousarray(fft_window, dtype=win_dtype)

        if use_f64_kernel or y.dtype == np.float64:
            rust_stft_complex = _rust_ext.stft_complex_f64
        else:
            rust_stft_complex = _rust_ext.stft_complex

        if y_c.ndim == 1:
            stft_matrix = rust_stft_complex(
                y_c,
                int(n_fft),
                int(hop_length),
                bool(center),
                win_c,
            )
        else:
            # Batch leading dimensions into channels and use native batched
            # kernels when available; otherwise fall back to per-channel calls.
            lead_shape = y_c.shape[:-1]
            y_batch = y_c.reshape((-1, y_c.shape[-1]))
            if use_f64_kernel or y.dtype == np.float64:
                rust_stft_complex_batch = getattr(_rust_ext, "stft_complex_f64_batch", None)
            else:
                rust_stft_complex_batch = getattr(_rust_ext, "stft_complex_batch", None)

            # For small channel counts (especially stereo), per-channel dispatch can
            # outperform batched kernels due to lower setup/reshape overhead.
            use_batch = rust_stft_complex_batch is not None and y_batch.shape[0] >= 4

            if use_batch:
                assert rust_stft_complex_batch is not None
                stft_batch = rust_stft_complex_batch(
                    y_batch,
                    int(n_fft),
                    int(hop_length),
                    bool(center),
                    win_c,
                )
                stft_matrix = stft_batch.reshape(lead_shape + stft_batch.shape[-2:])
            else:
                stft_list = [
                    rust_stft_complex(
                        y_batch[ch],
                        int(n_fft),
                        int(hop_length),
                        bool(center),
                        win_c,
                    )
                    for ch in range(y_batch.shape[0])
                ]
                stft_matrix = np.stack(stft_list, axis=0).reshape(
                    lead_shape + stft_list[0].shape
                )

        # Normalize FFT sign convention: current Rust kernels emit the
        # conjugate of numpy/scipy rfft output, so map back here.
        stft_matrix = np.conjugate(stft_matrix)

        if y.dtype == np.float32 and stft_matrix.dtype != np.complex64:
            stft_matrix = stft_matrix.astype(np.complex64, copy=False)

        if dtype is not None and np.dtype(dtype) != stft_matrix.dtype:
            stft_matrix = stft_matrix.astype(dtype, copy=False)

        return stft_matrix

    if fft_window is None:
        fft_window = get_window(window, win_length, fftbins=True)
        # Pad the window out to n_fft size
        fft_window = util.pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = util.expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if pad_mode in ("wrap", "maximum", "mean", "median", "minimum"):
            # Note: padding with a user-provided function "works", but
            # use at your own risk.
            # Since we don't pass-through kwargs here, any arguments
            # to a user-provided pad function should be encapsulated
            # by using functools.partial:
            #
            # >>> my_pad_func = functools.partial(pad_func, foo=x, bar=y)
            # >>> librosa.stft(..., pad_mode=my_pad_func)

            raise ParameterError(
                f"pad_mode='{pad_mode}' is not supported by librosa.stft"
            )

        if n_fft > y.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of length={y.shape[-1]}"
            )

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0) for _ in range(y.ndim)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            y = np.pad(y, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            # +1 here is to ensure enough samples to fill the window
            # fixes bug #1567
            y_pre = np.pad(
                y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            y_frames_pre = util.frame(y_pre, frame_length=n_fft, hop_length=hop_length)
            # Trim this down to the exact number of frames we should have
            y_frames_pre = y_frames_pre[..., :start_k]

            # How many extra frames do we have from the head?
            extra = y_frames_pre.shape[-1]

            # Determine if we have any frames that will fit inside the tail pad
            if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
                )
                y_frames_post = util.frame(
                    y_post, frame_length=n_fft, hop_length=hop_length
                )
                # How many extra frames do we have from the tail?
                extra += y_frames_post.shape[-1]
            else:
                # In this event, the first frame that touches tail padding would run off
                # the end of the padded array
                # We'll circumvent this by allocating an empty frame buffer for the tail
                # this keeps the subsequent logic simple
                post_shape = list(y_frames_pre.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)
    else:
        if n_fft > y.shape[-1]:
            raise ParameterError(
                f"n_fft={n_fft} is too large for uncentered analysis of input signal of length={y.shape[-1]}"
            )

        # "Middle" of the signal starts at sample 0
        start = 0
        # We have no extra frames
        extra = 0

    fft = get_fftlib()

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Window the time series.
    y_frames = util.frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra

    if out is None:
        stft_matrix = np.zeros(shape, dtype=dtype, order="F")
    elif not (np.allclose(out.shape[:-1], shape[:-1]) and out.shape[-1] >= shape[-1]):
        raise ParameterError(
            f"Shape mismatch for provided output array out.shape={out.shape} and target shape={shape}"
        )
    elif not np.iscomplexobj(out):
        raise ParameterError(f"output with dtype={out.dtype} is not of complex type")
    else:
        if np.allclose(shape, out.shape):
            stft_matrix = out
        else:
            stft_matrix = out[..., : shape[-1]]

    # Fill in the warm-up
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * y_frames_pre, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = int(
        util.MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)
    )
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix


@cache(level=30)
def istft(
    stft_matrix: np.ndarray,
    *,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window: _WindowSpec = "hann",
    center: bool = True,
    dtype: Optional[DTypeLike] = None,
    length: Optional[int] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``y``
    by minimizing the mean squared error between ``stft_matrix`` and STFT of
    ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified ``stft_matrix``.

    .. [#] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236ΓÇô243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(..., 1 + n_fft//2, t)]
        STFT matrix from ``stft``

    hop_length : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to ``win_length // 4``.

    win_length : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the ``window`` function (see below).

        If unspecified, defaults to ``n_fft``.

    n_fft : int > 0 or None
        The number of samples per frame in the input spectrogram.
        By default, this will be inferred from the shape of ``stft_matrix``.
        However, if an odd frame length was used, you can specify the correct
        length by setting ``n_fft``.

    window : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a user-specified window vector of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, ``D`` is assumed to have centered frames.
        - If ``False``, ``D`` is assumed to have left-aligned frames.

    dtype : numeric type
        Real numeric type for ``y``.  Default is to match the numerical
        precision of the input spectrogram.

    length : int > 0, optional
        If provided, the output ``y`` is zero-padded or clipped to exactly
        ``length`` samples.

    out : np.ndarray or None
        A pre-allocated, complex-valued array to store the reconstructed signal
        ``y``.  This must be of the correct shape for the given input parameters.

        If not provided, a new array is allocated and returned.

    Returns
    -------
    y : np.ndarray [shape=(..., n)]
        time domain signal reconstructed from ``stft_matrix``.
        If ``stft_matrix`` contains more than two axes
        (e.g., from a stereo input signal), then ``y`` will match shape on the leading dimensions.

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([-1.407e-03, -4.461e-04, ...,  5.131e-06, -1.417e-05],
          dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of ``y`` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, size=n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    8.940697e-08
    """
    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add broadcasting axes
    ifft_window = util.pad_center(ifft_window, size=n_fft)
    ifft_window = util.expand_to(ifft_window, ndim=stft_matrix.ndim, axes=-2)

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + 2 * (n_fft // 2)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]

    if dtype is None:
        dtype = util.dtype_c2r(stft_matrix.dtype)

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if length:
        expected_signal_len = length
    elif center:
        expected_signal_len -= 2 * (n_fft // 2)

    shape.append(expected_signal_len)

    if out is None:
        y = np.zeros(shape, dtype=dtype)
    elif not np.allclose(out.shape, shape):
        raise ParameterError(
            f"Shape mismatch for provided output array out.shape={out.shape} != {shape}"
        )
    else:
        y = out
        # Since we'll be doing overlap-add here, this needs to be initialized to zero.
        y.fill(0.0)

    fft = get_fftlib()

    if center:
        # First frame that does not depend on padding
        #  k * hop_length - n_fft//2 >= 0
        # k * hop_length >= n_fft // 2
        # k >= (n_fft//2 / hop_length)

        start_frame = int(np.ceil((n_fft // 2) / hop_length))

        # Do overlap-add on the head block
        ytmp = ifft_window * fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)

        shape[-1] = n_fft + hop_length * (start_frame - 1)
        head_buffer = np.zeros(shape, dtype=dtype)

        __overlap_add(head_buffer, ytmp, hop_length)

        # If y is smaller than the head buffer, take everything
        if y.shape[-1] < shape[-1] - n_fft // 2:
            y[..., :] = head_buffer[..., n_fft // 2 : y.shape[-1] + n_fft // 2]
        else:
            # Trim off the first n_fft//2 samples from the head and copy into target buffer
            y[..., : shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2 :]

        # This offset compensates for any differences between frame alignment
        # and padding truncation
        offset = start_frame * hop_length - n_fft // 2

    else:
        start_frame = 0
        offset = 0

    n_columns = int(
        util.MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    )
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[..., frame * hop_length + offset :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window=window,
        n_frames=n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    if center:
        start = n_fft // 2
    else:
        start = 0

    ifft_window_sum = util.fix_length(ifft_window_sum[..., start:], size=y.shape[-1])

    approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)

    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return y


@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample

        y[..., sample : (sample + N)] += ytmp[..., :N, frame]


