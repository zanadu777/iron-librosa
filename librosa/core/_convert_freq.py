#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frequency scale and spectrogram frequency utilities"""
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
def hz_to_mel(frequencies: _FloatLike_co, *, htk: bool = ...) -> np.floating[Any]:
    ...


@overload
def hz_to_mel(
    frequencies: _SequenceLike[_FloatLike_co], *, htk: bool = ...
) -> np.ndarray:
    ...


@overload
def hz_to_mel(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, htk: bool = ...
) -> Union[np.floating[Any], np.ndarray]:
    ...


def hz_to_mel(
    frequencies: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating[Any], np.ndarray]:
    """Convert Hz to Mels

    Examples
    --------
    >>> librosa.hz_to_mel(60)
    0.9
    >>> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies : number or np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels : number or np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """
    frequencies = np.asanyarray(frequencies)

    # --- iron-librosa: Rust acceleration ---
    if (
        RUST_AVAILABLE
        and hasattr(_rust_ext, "hz_to_mel")
        and frequencies.ndim == 1
    ):
        # Coerce htk from numpy scalar/array to native Python bool for Rust dispatch
        htk = bool(np.asarray(htk).item())
        freq_rust = np.ascontiguousarray(frequencies, dtype=np.float64)
        return _rust_ext.hz_to_mel(freq_rust, htk=htk)
    # --- end Rust acceleration ---

    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


@overload
def mel_to_hz(mels: _FloatLike_co, *, htk: bool = ...) -> np.floating[Any]:
    ...


@overload
def mel_to_hz(mels: _SequenceLike[_FloatLike_co], *, htk: bool = ...) -> np.ndarray:
    ...


@overload
def mel_to_hz(
    mels: _ScalarOrSequence[_FloatLike_co], *, htk: bool = ...
) -> Union[np.floating[Any], np.ndarray]:
    ...


def mel_to_hz(
    mels: _ScalarOrSequence[_FloatLike_co], *, htk: bool = False
) -> Union[np.floating[Any], np.ndarray]:
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >>> librosa.mel_to_hz(3)
    200.

    >>> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """
    mels = np.asanyarray(mels)

    # --- iron-librosa: Rust acceleration ---
    if RUST_AVAILABLE and hasattr(_rust_ext, "mel_to_hz") and mels.ndim == 1:
        # Coerce htk from numpy scalar/array to native Python bool for Rust dispatch
        htk = bool(np.asarray(htk).item())
        mels_rust = np.ascontiguousarray(mels, dtype=np.float64)
        return _rust_ext.mel_to_hz(mels_rust, htk=htk)
    # --- end Rust acceleration ---

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


@overload
def hz_to_octs(
    frequencies: _FloatLike_co, *, tuning: float = ..., bins_per_octave: int = ...
) -> np.floating[Any]:
    ...


@overload
def hz_to_octs(
    frequencies: _SequenceLike[_FloatLike_co],
    *,
    tuning: float = ...,
    bins_per_octave: int = ...,
) -> np.ndarray:
    ...


@overload
def hz_to_octs(
    frequencies: _ScalarOrSequence[_FloatLike_co],
    *,
    tuning: float = ...,
    bins_per_octave: int = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...


def hz_to_octs(
    frequencies: _ScalarOrSequence[_FloatLike_co],
    *,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert frequencies (Hz) to (fractional) octave numbers.

    Examples
    --------
    >>> librosa.hz_to_octs(440.0)
    4.
    >>> librosa.hz_to_octs([32, 64, 128, 256])
    array([ 0.219,  1.219,  2.219,  3.219])

    Parameters
    ----------
    frequencies : number >0 or np.ndarray [shape=(n,)] or float
        scalar or vector of frequencies
    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    octaves : number or np.ndarray [shape=(n,)]
        octave number for each frequency

    See Also
    --------
    octs_to_hz
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

    octs: np.ndarray = np.log2(np.asanyarray(frequencies) / (float(A440) / 16))
    return octs


@overload
def octs_to_hz(
    octs: _FloatLike_co, *, tuning: float = ..., bins_per_octave: int = ...
) -> np.floating[Any]:
    ...


@overload
def octs_to_hz(
    octs: _SequenceLike[_FloatLike_co],
    *,
    tuning: float = ...,
    bins_per_octave: int = ...,
) -> np.ndarray:
    ...


@overload
def octs_to_hz(
    octs: _ScalarOrSequence[_FloatLike_co],
    *,
    tuning: float = ...,
    bins_per_octave: int = ...,
) -> Union[np.floating[Any], np.ndarray]:
    ...


def octs_to_hz(
    octs: _ScalarOrSequence[_FloatLike_co],
    *,
    tuning: float = 0.0,
    bins_per_octave: int = 12,
) -> Union[np.floating[Any], np.ndarray]:
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    Examples
    --------
    >>> librosa.octs_to_hz(1)
    55.
    >>> librosa.octs_to_hz([-2, -1, 0, 1, 2])
    array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

    Parameters
    ----------
    octs : np.ndarray [shape=(n,)] or float
        octave number for each frequency
    tuning : float
        Tuning deviation from A440 in (fractional) bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    frequencies : number or np.ndarray [shape=(n,)]
        scalar or vector of frequencies

    See Also
    --------
    hz_to_octs
    """
    A440 = 440.0 * 2.0 ** (tuning / bins_per_octave)

    return (float(A440) / 16) * (2.0 ** np.asanyarray(octs))


@overload
def A4_to_tuning(A4: _FloatLike_co, *, bins_per_octave: int = ...) -> np.floating[Any]:
    ...


@overload
def A4_to_tuning(
    A4: _SequenceLike[_FloatLike_co], *, bins_per_octave: int = ...
) -> np.ndarray:
    ...


@overload
def A4_to_tuning(
    A4: _ScalarOrSequence[_FloatLike_co], *, bins_per_octave: int = ...
) -> Union[np.floating[Any], np.ndarray]:
    ...


def A4_to_tuning(
    A4: _ScalarOrSequence[_FloatLike_co], *, bins_per_octave: int = 12
) -> Union[np.floating[Any], np.ndarray]:
    """Convert a reference pitch frequency (e.g., ``A4=435``) to a tuning
    estimation, in fractions of a bin per octave.

    This is useful for determining the tuning deviation relative to
    A440 of a given frequency, assuming equal temperament. By default,
    12 bins per octave are used.

    This method is the inverse of `tuning_to_A4`.

    Examples
    --------
    The base case of this method in which A440 yields 0 tuning offset
    from itself.

    >>> librosa.A4_to_tuning(440.0)
    0.

    Convert a non-A440 frequency to a tuning offset relative
    to A440 using the default of 12 bins per octave.

    >>> librosa.A4_to_tuning(432.0)
    -0.318

    Convert two reference pitch frequencies to corresponding
    tuning estimations at once, but using 24 bins per octave.

    >>> librosa.A4_to_tuning([440.0, 444.0], bins_per_octave=24)
    array([   0.,   0.313   ])

    Parameters
    ----------
    A4 : float or np.ndarray [shape=(n,), dtype=float]
        Reference frequency(s) corresponding to A4.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    tuning : float or np.ndarray [shape=(n,), dtype=float]
        Tuning deviation from A440 in (fractional) bins per octave.

    See Also
    --------
    tuning_to_A4
    """
    tuning: np.ndarray = bins_per_octave * (np.log2(np.asanyarray(A4)) - np.log2(440.0))
    return tuning


@overload
def tuning_to_A4(
    tuning: _FloatLike_co, *, bins_per_octave: int = ...
) -> np.floating[Any]:
    ...


@overload
def tuning_to_A4(
    tuning: _SequenceLike[_FloatLike_co], *, bins_per_octave: int = ...
) -> np.ndarray:
    ...


@overload
def tuning_to_A4(
    tuning: _ScalarOrSequence[_FloatLike_co], *, bins_per_octave: int = ...
) -> Union[np.floating[Any], np.ndarray]:
    ...


def tuning_to_A4(
    tuning: _ScalarOrSequence[_FloatLike_co], *, bins_per_octave: int = 12
) -> Union[np.floating[Any], np.ndarray]:
    """Convert a tuning deviation (from 0) in fractions of a bin per
    octave (e.g., ``tuning=-0.1``) to a reference pitch frequency
    relative to A440.

    This is useful if you are working in a non-A440 tuning system
    to determine the reference pitch frequency given a tuning
    offset and assuming equal temperament. By default, 12 bins per
    octave are used.

    This method is the inverse of  `A4_to_tuning`.

    Examples
    --------
    The base case of this method in which a tuning deviation of 0
    gets us to our A440 reference pitch.

    >>> librosa.tuning_to_A4(0.0)
    440.

    Convert a nonzero tuning offset to its reference pitch frequency.

    >>> librosa.tuning_to_A4(-0.318)
    431.992

    Convert 3 tuning deviations at once to respective reference
    pitch frequencies, using 36 bins per octave.

    >>> librosa.tuning_to_A4([0.1, 0.2, -0.1], bins_per_octave=36)
    array([   440.848,    441.698   439.154])

    Parameters
    ----------
    tuning : float or np.ndarray [shape=(n,), dtype=float]
        Tuning deviation from A440 in fractional bins per octave.
    bins_per_octave : int > 0
        Number of bins per octave.

    Returns
    -------
    A4 : float or np.ndarray [shape=(n,), dtype=float]
        Reference frequency corresponding to A4.

    See Also
    --------
    A4_to_tuning
    """
    return 440.0 * 2.0 ** (np.asanyarray(tuning) / bins_per_octave)


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    """Alternative interface for `np.fft.rfftfreq`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size

    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``

    Examples
    --------
    >>> librosa.fft_frequencies(sr=22050, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])
    """
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def cqt_frequencies(
    n_bins: int, *, fmin: float, bins_per_octave: int = 12, tuning: float = 0.0
) -> np.ndarray:
    """Compute the center frequencies of Constant-Q bins.

    Examples
    --------
    >>> # Get the CQT frequencies for 24 notes, starting at C2
    >>> librosa.cqt_frequencies(24, fmin=librosa.note_to_hz('C2'))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
             92.499,   97.999,  103.826,  110.   ,  116.541,  123.471,
            130.813,  138.591,  146.832,  155.563,  164.814,  174.614,
            184.997,  195.998,  207.652,  220.   ,  233.082,  246.942])

    Parameters
    ----------
    n_bins : int > 0 [scalar]
        Number of constant-Q bins
    fmin : float > 0 [scalar]
        Minimum frequency
    bins_per_octave : int > 0 [scalar]
        Number of bins per octave
    tuning : float
        Deviation from A440 tuning in fractional bins

    Returns
    -------
    frequencies : np.ndarray [shape=(n_bins,)]
        Center frequency for each CQT bin
    """
    correction: float = 2.0 ** (float(tuning) / bins_per_octave)
    frequencies: np.ndarray = 2.0 ** (
        np.arange(0, n_bins, dtype=float) / bins_per_octave
    )

    return correction * fmin * frequencies


def mel_frequencies(
    n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False
) -> np.ndarray:
    """Compute an array of acoustic frequencies tuned to the mel scale.

    The mel scale is a quasi-logarithmic function of acoustic frequency
    designed such that perceptually similar pitch intervals (e.g. octaves)
    appear equal in width over the full hearing range.

    Because the definition of the mel scale is conditioned by a finite number
    of subjective psychoacoustical experiments, several implementations coexist
    in the audio signal processing literature [#]_. By default, librosa replicates
    the behavior of the well-established MATLAB Auditory Toolbox of Slaney [#]_.
    According to this default implementation,  the conversion from Hertz to mel is
    linear below 1 kHz and logarithmic above 1 kHz. Another available implementation
    replicates the Hidden Markov Toolkit [#]_ (HTK) according to the following formula::

        mel = 2595.0 * np.log10(1.0 + f / 700.0).

    The choice of implementation is determined by the ``htk`` keyword argument: setting
    ``htk=False`` leads to the Auditory toolbox implementation, whereas setting it ``htk=True``
    leads to the HTK implementation.

    .. [#] Umesh, S., Cohen, L., & Nelson, D. Fitting the mel scale.
        In Proc. International Conference on Acoustics, Speech, and Signal Processing
        (ICASSP), vol. 1, pp. 217-220, 1998.

    .. [#] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory
        Modeling Work. Technical Report, version 2, Interval Research Corporation, 1998.

    .. [#] Young, S., Evermann, G., Gales, M., Hain, T., Kershaw, D., Liu, X.,
        Moore, G., Odell, J., Ollason, D., Povey, D., Valtchev, V., & Woodland, P.
        The HTK book, version 3.4. Cambridge University, March 2009.

    See Also
    --------
    hz_to_mel
    mel_to_hz
    librosa.feature.melspectrogram
    librosa.feature.mfcc

    Parameters
    ----------
    n_mels : int > 0 [scalar]
        Number of mel bins.
    fmin : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax : float >= 0 [scalar]
        Maximum frequency (Hz).
    htk : bool
        If True, use HTK formula to convert Hz to mel.
        Otherwise (False), use Slaney's Auditory Toolbox.

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >>> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz


def tempo_frequencies(
    n_bins: int, *, hop_length: int = 512, sr: float = 22050
) -> np.ndarray:
    """Compute the frequencies (in beats per minute) corresponding
    to an onset auto-correlation or tempogram matrix.

    Parameters
    ----------
    n_bins : int > 0
        The number of lag bins
    hop_length : int > 0
        The number of samples between each bin
    sr : number > 0
        The audio sampling rate

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_bins,)]
        vector of bin frequencies measured in BPM.

        .. note:: ``bin_frequencies[0] = +np.inf`` corresponds to 0-lag

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.tempo_frequencies(384, sr=22050)
    array([      inf,  2583.984,  1291.992, ...,     6.782,
               6.764,     6.747])
    """
    bin_frequencies = np.zeros(int(n_bins), dtype=np.float64)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies


def fourier_tempo_frequencies(
    *, sr: float = 22050, win_length: int = 384, hop_length: int = 512
) -> np.ndarray:
    """Compute the frequencies (in beats per minute) corresponding
    to a Fourier tempogram matrix.

    Parameters
    ----------
    sr : number > 0
        The audio sampling rate
    win_length : int > 0
        The number of frames per analysis window
    hop_length : int > 0
        The number of samples between each bin

    Returns
    -------
    bin_frequencies : ndarray [shape=(win_length // 2 + 1 ,)]
        vector of bin frequencies measured in BPM.

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >>> librosa.fourier_tempo_frequencies(win_length=384, sr=22050)
    array([ 0.   ,  0.117,  0.234, ..., 22.266, 22.383, 22.5  ])
    """
    # sr / hop_length gets the frame rate
    # multiplying by 60 turns frames / sec into frames / minute
    return fft_frequencies(sr=sr * 60 / float(hop_length), n_fft=win_length)


# A-weighting should be capitalized: suppress the naming warning
