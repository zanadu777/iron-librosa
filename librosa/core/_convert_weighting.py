#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frequency weighting curves and time/sample grid utilities"""
from __future__ import annotations
import numpy as np
from . import notation
from ..util.exceptions import ParameterError
from ..util.decorators import vectorize
from typing import Any, Callable, Dict, Iterable, Optional, Sized, Union, overload
from .._rust_bridge import _rust_ext, RUST_AVAILABLE
from .._typing import (
    _IterableLike,
    _FloatLike_co,
    _SequenceLike,
    _ScalarOrSequence,
    _IntLike_co,
)
from ._convert_time import frames_to_samples, samples_to_time

@overload
def A_weighting(
    frequencies: _FloatLike_co, *, min_db: Optional[float] = ...
) -> np.floating[Any]:  # pylint: disable=invalid-name
    ...


@overload
def A_weighting(
    frequencies: _SequenceLike[_FloatLike_co], *, min_db: Optional[float] = ...
) -> np.ndarray:  # pylint: disable=invalid-name
    ...


@overload
def A_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = ...
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    ...


def A_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = -80.0
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    """Compute the A-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    A_weighting : scalar or np.ndarray [shape=(n,)]
        ``A_weighting[i]`` is the A-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.A_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)',
    ...        ylabel='Weighting (log10)',
    ...        title='A-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    weights: np.ndarray = 2.0 + 20.0 * (
        np.log10(const[0])
        + 2 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
        - 0.5 * np.log10(f_sq + const[3])
    )

    if min_db is None:
        return weights
    else:
        return np.maximum(min_db, weights)


@overload
def B_weighting(
    frequencies: _FloatLike_co, *, min_db: Optional[float] = ...
) -> np.floating[Any]:  # pylint: disable=invalid-name
    ...


@overload
def B_weighting(
    frequencies: _SequenceLike[_FloatLike_co], *, min_db: Optional[float] = ...
) -> np.ndarray:  # pylint: disable=invalid-name
    ...


@overload
def B_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = ...
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    ...


def B_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = -80.0
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    """Compute the B-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    B_weighting : scalar or np.ndarray [shape=(n,)]
        ``B_weighting[i]`` is the B-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the B-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.B_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)',
    ...        ylabel='Weighting (log10)',
    ...        title='B-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([12194.217, 20.598997, 158.48932]) ** 2.0
    weights: np.ndarray = 0.17 + 20.0 * (
        np.log10(const[0])
        + 1.5 * np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
        - 0.5 * np.log10(f_sq + const[2])
    )

    return weights if min_db is None else np.maximum(min_db, weights)


@overload
def C_weighting(
    frequencies: _FloatLike_co, *, min_db: Optional[float] = ...
) -> np.floating[Any]:  # pylint: disable=invalid-name
    ...


@overload
def C_weighting(
    frequencies: _SequenceLike[_FloatLike_co], *, min_db: Optional[float] = ...
) -> np.ndarray:  # pylint: disable=invalid-name
    ...


@overload
def C_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = ...
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    ...


def C_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = -80.0
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    """Compute the C-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    C_weighting : scalar or np.ndarray [shape=(n,)]
        ``C_weighting[i]`` is the C-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    D_weighting

    Examples
    --------
    Get the C-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.C_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='C-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([12194.217, 20.598997]) ** 2.0
    weights: np.ndarray = 0.062 + 20.0 * (
        np.log10(const[0])
        + np.log10(f_sq)
        - np.log10(f_sq + const[0])
        - np.log10(f_sq + const[1])
    )

    return weights if min_db is None else np.maximum(min_db, weights)


@overload
def D_weighting(
    frequencies: _FloatLike_co, *, min_db: Optional[float] = ...
) -> np.floating[Any]:  # pylint: disable=invalid-name
    ...


@overload
def D_weighting(
    frequencies: _SequenceLike[_FloatLike_co], *, min_db: Optional[float] = ...
) -> np.ndarray:  # pylint: disable=invalid-name
    ...


@overload
def D_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = ...
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    ...


def D_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, min_db: Optional[float] = -80.0
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    """Compute the D-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    D_weighting : scalar or np.ndarray [shape=(n,)]
        ``D_weighting[i]`` is the D-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting

    Examples
    --------
    Get the D-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.D_weighting(freqs)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='D-Weighting of CQT frequencies')
    """
    f_sq = np.asanyarray(frequencies) ** 2.0

    const = np.array([8.3046305e-3, 1018.7, 1039.6, 3136.5, 3424, 282.7, 1160]) ** 2.0
    weights: np.ndarray = 20.0 * (
        0.5 * np.log10(f_sq)
        - np.log10(const[0])
        + 0.5
        * (
            +np.log10((const[1] - f_sq) ** 2 + const[2] * f_sq)
            - np.log10((const[3] - f_sq) ** 2 + const[4] * f_sq)
            - np.log10(const[5] + f_sq)
            - np.log10(const[6] + f_sq)
        )
    )

    if min_db is None:
        return weights
    else:
        return np.maximum(min_db, weights)


def Z_weighting(
    frequencies: Sized, *, min_db: Optional[float] = None
) -> np.ndarray:  # pylint: disable=invalid-name
    """Apply no weighting curve (aka Z-weighting).

    This function behaves similarly to `A_weighting`, `B_weighting`, etc.,
    but all frequencies are equally weighted.
    An optional threshold `min_db` can still be used to clip energies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    Z_weighting : scalar or np.ndarray [shape=(n,)]
        ``Z_weighting[i]`` is the Z-weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting
    """
    weights = np.zeros(len(frequencies))
    if min_db is None:
        return weights
    else:
        return np.maximum(min_db, weights)


WEIGHTING_FUNCTIONS: Dict[
    Optional[str], Callable[..., Union[np.floating[Any], np.ndarray]]
] = {
    "A": A_weighting,
    "B": B_weighting,
    "C": C_weighting,
    "D": D_weighting,
    "Z": Z_weighting,
    None: Z_weighting,
}


@overload
def frequency_weighting(
    frequencies: _FloatLike_co, *, kind: str = ..., **kwargs: Any
) -> np.floating[Any]:  # pylint: disable=invalid-name
    ...


@overload
def frequency_weighting(
    frequencies: _SequenceLike[_FloatLike_co], *, kind: str = ..., **kwargs: Any
) -> np.ndarray:  # pylint: disable=invalid-name
    ...


@overload
def frequency_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, kind: str = ..., **kwargs: Any
) -> Union[np.floating[Any], np.ndarray]:  # pylint: disable=invalid-name
    ...


def frequency_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, kind: str = "A", **kwargs: Any
) -> Union[np.floating[Any], np.ndarray]:
    """Compute the weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    kind : str in
        The weighting kind. e.g. `'A'`, `'B'`, `'C'`, `'D'`, `'Z'`
    **kwargs
        Additional keyword arguments to A_weighting, B_weighting, etc.

    Returns
    -------
    weighting : scalar or np.ndarray [shape=(n,)]
        ``weighting[i]`` is the weighting of ``frequencies[i]``

    See Also
    --------
    perceptual_weighting
    multi_frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A-weighting for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weights = librosa.frequency_weighting(freqs, kind='A')
    >>> fig, ax = plt.subplots()
    >>> ax.plot(freqs, weights)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='A-Weighting of CQT frequencies')
    """
    if isinstance(kind, str):
        kind = kind.upper()
    return WEIGHTING_FUNCTIONS[kind](frequencies, **kwargs)


def multi_frequency_weighting(
    frequencies: _ScalarOrSequence[_FloatLike_co],
    *,
    kinds: Iterable[str] = "ZAC",
    **kwargs: Any,
) -> np.ndarray:
    """Compute multiple weightings of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)
    kinds : list or tuple or str
        An iterable of weighting kinds. e.g. `('Z', 'B')`, `'ZAD'`, `'C'`
    **kwargs : keywords to pass to the weighting function.

    Returns
    -------
    weighting : scalar or np.ndarray [shape=(len(kinds), n)]
        ``weighting[i, j]`` is the weighting of ``frequencies[j]``
        using the curve determined by ``kinds[i]``.

    See Also
    --------
    perceptual_weighting
    frequency_weighting
    A_weighting
    B_weighting
    C_weighting
    D_weighting

    Examples
    --------
    Get the A, B, C, D, and Z weightings for CQT frequencies

    >>> import matplotlib.pyplot as plt
    >>> freqs = librosa.cqt_frequencies(n_bins=108, fmin=librosa.note_to_hz('C1'))
    >>> weightings = 'ABCDZ'
    >>> weights = librosa.multi_frequency_weighting(freqs, kinds=weightings)
    >>> fig, ax = plt.subplots()
    >>> for label, w in zip(weightings, weights):
    ...     ax.plot(freqs, w, label=label)
    >>> ax.set(xlabel='Frequency (Hz)', ylabel='Weighting (log10)',
    ...        title='Weightings of CQT frequencies')
    >>> ax.legend()
    """
    return np.stack(
        [frequency_weighting(frequencies, kind=k, **kwargs) for k in kinds], axis=0
    )


def times_like(
    X: Union[np.ndarray, float],
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    axis: int = -1,
) -> np.ndarray:
    """Return an array of time values to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of X.
        By default, the last axis (-1) is taken.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        ndarray of times (in seconds) corresponding to each frame of X.

    See Also
    --------
    samples_like :
        Return an array of sample indices to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> D = librosa.stft(y)
    >>> times = librosa.times_like(D, sr=sr)
    >>> times
    array([0.   , 0.023, ..., 5.294, 5.317])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> times = librosa.times_like(n_frames, sr=sr)
    >>> times
    array([  0.00000000e+00,   2.32199546e-02,   4.64399093e-02, ...,
             6.13935601e+01,   6.14167800e+01,   6.14400000e+01])
    """
    samples = samples_like(X, hop_length=hop_length, n_fft=n_fft, axis=axis)
    time: np.ndarray = samples_to_time(samples, sr=sr)
    return time


def samples_like(
    X: Union[np.ndarray, float],
    *,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    axis: int = -1,
) -> np.ndarray:
    """Return an array of sample indices to match the time axis from a feature matrix.

    Parameters
    ----------
    X : np.ndarray or scalar
        - If ndarray, X is a feature matrix, e.g. STFT, chromagram, or mel spectrogram.
        - If scalar, X represents the number of frames.
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.
    axis : int [scalar]
        The axis representing the time axis of ``X``.
        By default, the last axis (-1) is taken.

    Returns
    -------
    samples : np.ndarray [shape=(n,)]
        ndarray of sample indices corresponding to each frame of ``X``.

    See Also
    --------
    times_like :
        Return an array of time values to match the time axis from a feature matrix.

    Examples
    --------
    Provide a feature matrix input:

    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> X = librosa.stft(y)
    >>> samples = librosa.samples_like(X)
    >>> samples
    array([     0,    512, ..., 116736, 117248])

    Provide a scalar input:

    >>> n_frames = 2647
    >>> samples = librosa.samples_like(n_frames)
    >>> samples
    array([      0,     512,    1024, ..., 1353728, 1354240, 1354752])
    """
    # suppress type checks because mypy does not understand isscalar
    if np.isscalar(X):
        frames = np.arange(X)  # type: ignore
    else:
        frames = np.arange(X.shape[axis])  # type: ignore
    return frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)


