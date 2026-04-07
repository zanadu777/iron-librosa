#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Time, frame, sample and block conversion utilities"""
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

@overload
def frames_to_samples(
    frames: _IntLike_co, *, hop_length: int = 512, n_fft: Optional[int] = None
) -> np.integer[Any]:
    ...


@overload
def frames_to_samples(
    frames: _SequenceLike[_IntLike_co],
    *,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> np.ndarray:
    ...


def frames_to_samples(
    frames: _ScalarOrSequence[_IntLike_co],
    *,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> Union[np.integer[Any], np.ndarray]:
    """Convert frame indices to audio sample indices.

    Parameters
    ----------
    frames : number or np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : number or np.ndarray
        time (in samples) of each given frame number::

            times[i] = frames[i] * hop_length

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_samples = librosa.frames_to_samples(beats, sr=sr)
    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.asanyarray(frames) * hop_length + offset).astype(int)


@overload
def samples_to_frames(
    samples: _IntLike_co, *, hop_length: int = ..., n_fft: Optional[int] = ...
) -> np.integer[Any]:
    ...


@overload
def samples_to_frames(
    samples: _SequenceLike[_IntLike_co],
    *,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> np.ndarray:
    ...


@overload
def samples_to_frames(
    samples: _ScalarOrSequence[_IntLike_co],
    *,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> Union[np.integer[Any], np.ndarray]:
    ...


def samples_to_frames(
    samples: _ScalarOrSequence[_IntLike_co],
    *,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> Union[np.integer[Any], np.ndarray]:
    """Convert sample indices into STFT frames.

    Examples
    --------
    >>> # Get the frame numbers for every 256 samples
    >>> librosa.samples_to_frames(np.arange(0, 22050, 256))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
            7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13,
           14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
           21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
           28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41,
           42, 42, 43])

    Parameters
    ----------
    samples : int or np.ndarray [shape=(n,)]
        sample index or vector of sample indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``- n_fft // 2``
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : int or np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times::

            frames[i] = floor( samples[i] / hop_length )

    See Also
    --------
    samples_to_time : convert sample indices to time values
    frames_to_samples : convert frame indices to sample indices
    """
    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.asanyarray(samples)
    return np.asarray(np.floor((samples - offset) // hop_length), dtype=int)


@overload
def frames_to_time(
    frames: _IntLike_co,
    *,
    sr: float = ...,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> np.floating[Any]:
    ...


@overload
def frames_to_time(
    frames: _SequenceLike[_IntLike_co],
    *,
    sr: float = ...,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> np.ndarray:
    ...


@overload
def frames_to_time(
    frames: _ScalarOrSequence[_IntLike_co],
    *,
    sr: float = ...,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...


def frames_to_time(
    frames: _ScalarOrSequence[_IntLike_co],
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert frame counts to time (seconds).

    Parameters
    ----------
    frames : np.ndarray [shape=(n,)]
        frame index or vector of frame indices
    sr : number > 0 [scalar]
        audio sampling rate
    hop_length : int > 0 [scalar]
        number of samples between successive frames
    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``n_fft // 2``
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number::

            times[i] = frames[i] * hop_length / sr

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('choice'))
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    >>> beat_times = librosa.frames_to_time(beats, sr=sr)
    """
    samples = frames_to_samples(frames, hop_length=hop_length, n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


@overload
def time_to_frames(
    times: _FloatLike_co,
    *,
    sr: float = ...,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> np.integer[Any]:
    ...


@overload
def time_to_frames(
    times: _SequenceLike[_FloatLike_co],
    *,
    sr: float = ...,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> np.ndarray:
    ...


@overload
def time_to_frames(
    times: _ScalarOrSequence[_FloatLike_co],
    *,
    sr: float = ...,
    hop_length: int = ...,
    n_fft: Optional[int] = ...,
) -> Union[np.integer[Any], np.ndarray]:
    ...


def time_to_frames(
    times: _ScalarOrSequence[_FloatLike_co],
    *,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
) -> Union[np.integer[Any], np.ndarray]:
    """Convert time stamps into STFT frames.

    Parameters
    ----------
    times : np.ndarray [shape=(n,)]
        time (in seconds) or vector of time values

    sr : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of ``- n_fft // 2``
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times::

            frames[i] = floor( times[i] * sr / hop_length )

    See Also
    --------
    frames_to_time : convert frame indices to time values
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get the frame numbers for every 100ms

    >>> librosa.time_to_frames(np.arange(0, 1, 0.1),
    ...                         sr=22050, hop_length=512)
    array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])
    """
    samples = time_to_samples(times, sr=sr)

    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


@overload
def time_to_samples(times: _FloatLike_co, *, sr: float = ...) -> np.integer[Any]:
    ...


@overload
def time_to_samples(
    times: _SequenceLike[_FloatLike_co], *, sr: float = ...
) -> np.ndarray:
    ...


@overload
def time_to_samples(
    times: _ScalarOrSequence[_FloatLike_co], *, sr: float = ...
) -> Union[np.integer[Any], np.ndarray]:
    ...


def time_to_samples(
    times: _ScalarOrSequence[_FloatLike_co], *, sr: float = 22050
) -> Union[np.integer[Any], np.ndarray]:
    """Convert timestamps (in seconds) to sample indices.

    Parameters
    ----------
    times : number or np.ndarray
        Time value or array of time values (in seconds)
    sr : number > 0
        Sampling rate

    Returns
    -------
    samples : int or np.ndarray [shape=times.shape, dtype=int]
        Sample indices corresponding to values in ``times``

    See Also
    --------
    time_to_frames : convert time values to frame indices
    samples_to_time : convert sample indices to time values

    Examples
    --------
    >>> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])
    """
    return (np.asanyarray(times) * sr).astype(int)


@overload
def samples_to_time(samples: _IntLike_co, *, sr: float = ...) -> np.floating[Any]:
    ...


@overload
def samples_to_time(
    samples: _SequenceLike[_IntLike_co], *, sr: float = ...
) -> np.ndarray:
    ...


@overload
def samples_to_time(
    samples: _ScalarOrSequence[_IntLike_co], *, sr: float = ...
) -> Union[np.floating[Any], np.ndarray]:
    ...


def samples_to_time(
    samples: _ScalarOrSequence[_IntLike_co], *, sr: float = 22050
) -> Union[np.floating[Any], np.ndarray]:
    """Convert sample indices to time (in seconds).

    Parameters
    ----------
    samples : np.ndarray
        Sample index or array of sample indices
    sr : number > 0
        Sampling rate

    Returns
    -------
    times : np.ndarray [shape=samples.shape]
        Time values corresponding to ``samples`` (in seconds)

    See Also
    --------
    samples_to_frames : convert sample indices to frame indices
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get timestamps corresponding to every 512 samples

    >>> librosa.samples_to_time(np.arange(0, 22050, 512), sr=22050)
    array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,
            0.163,  0.186,  0.209,  0.232,  0.255,  0.279,  0.302,
            0.325,  0.348,  0.372,  0.395,  0.418,  0.441,  0.464,
            0.488,  0.511,  0.534,  0.557,  0.58 ,  0.604,  0.627,
            0.65 ,  0.673,  0.697,  0.72 ,  0.743,  0.766,  0.789,
            0.813,  0.836,  0.859,  0.882,  0.906,  0.929,  0.952,
            0.975,  0.998])
    """
    return np.asanyarray(samples) / float(sr)


@overload
def blocks_to_frames(blocks: _IntLike_co, *, block_length: int) -> np.integer[Any]:
    ...


@overload
def blocks_to_frames(
    blocks: _SequenceLike[_IntLike_co], *, block_length: int
) -> np.ndarray:
    ...


@overload
def blocks_to_frames(
    blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int
) -> Union[np.integer[Any], np.ndarray]:
    ...


def blocks_to_frames(
    blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int
) -> Union[np.integer[Any], np.ndarray]:
    """Convert block indices to frame indices

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block

    Returns
    -------
    frames : np.ndarray [shape=samples.shape, dtype=int]
        The index or indices of frames corresponding to the beginning
        of each provided block.

    See Also
    --------
    blocks_to_samples
    blocks_to_time

    Examples
    --------
    Get frame indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_frame = librosa.blocks_to_frames(n, block_length=16)

    """
    return block_length * np.asanyarray(blocks)


@overload
def blocks_to_samples(
    blocks: _IntLike_co, *, block_length: int, hop_length: int
) -> np.integer[Any]:
    ...


@overload
def blocks_to_samples(
    blocks: _SequenceLike[_IntLike_co], *, block_length: int, hop_length: int
) -> np.ndarray:
    ...


@overload
def blocks_to_samples(
    blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int, hop_length: int
) -> Union[np.integer[Any], np.ndarray]:
    ...


def blocks_to_samples(
    blocks: _ScalarOrSequence[_IntLike_co], *, block_length: int, hop_length: int
) -> Union[np.integer[Any], np.ndarray]:
    """Convert block indices to sample indices

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block
    hop_length : int > 0
        The number of samples to advance between frames

    Returns
    -------
    samples : np.ndarray [shape=samples.shape, dtype=int]
        The index or indices of samples corresponding to the beginning
        of each provided block.

        Note that these correspond to the *first* sample index in
        each block, and are not frame-centered.

    See Also
    --------
    blocks_to_frames
    blocks_to_time

    Examples
    --------
    Get sample indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_sample = librosa.blocks_to_samples(n, block_length=16,
    ...                                          hop_length=512)

    """
    frames = blocks_to_frames(blocks, block_length=block_length)
    return frames_to_samples(frames, hop_length=hop_length)


@overload
def blocks_to_time(
    blocks: _IntLike_co, *, block_length: int, hop_length: int, sr: float
) -> np.floating[Any]:
    ...


@overload
def blocks_to_time(
    blocks: _SequenceLike[_IntLike_co], *, block_length: int, hop_length: int, sr: float
) -> np.ndarray:
    ...


@overload
def blocks_to_time(
    blocks: _ScalarOrSequence[_IntLike_co],
    *,
    block_length: int,
    hop_length: int,
    sr: float,
) -> Union[np.floating[Any], np.ndarray]:
    ...


def blocks_to_time(
    blocks: _ScalarOrSequence[_IntLike_co],
    *,
    block_length: int,
    hop_length: int,
    sr: float,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert block indices to time (in seconds)

    Parameters
    ----------
    blocks : np.ndarray
        Block index or array of block indices
    block_length : int > 0
        The number of frames per block
    hop_length : int > 0
        The number of samples to advance between frames
    sr : int > 0
        The sampling rate (samples per second)

    Returns
    -------
    times : np.ndarray [shape=samples.shape]
        The time index or indices (in seconds) corresponding to the
        beginning of each provided block.

        Note that these correspond to the time of the *first* sample
        in each block, and are not frame-centered.

    See Also
    --------
    blocks_to_frames
    blocks_to_samples

    Examples
    --------
    Get time indices for each block in a stream

    >>> filename = librosa.ex('brahms')
    >>> sr = librosa.get_samplerate(filename)
    >>> stream = librosa.stream(filename, block_length=16,
    ...                         frame_length=2048, hop_length=512)
    >>> for n, y in enumerate(stream):
    ...     n_time = librosa.blocks_to_time(n, block_length=16,
    ...                                     hop_length=512, sr=sr)

    """
    samples = blocks_to_samples(
        blocks, block_length=block_length, hop_length=hop_length
    )
    return samples_to_time(samples, sr=sr)


