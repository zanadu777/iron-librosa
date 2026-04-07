#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indian music (svara) and FJS pitch notation utilities"""
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
from ._convert_pitch import hz_to_midi, note_to_midi, hz_to_note

@overload
def midi_to_svara_h(
    midi: _FloatLike_co,
    *,
    Sa: _FloatLike_co,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def midi_to_svara_h(
    midi: np.ndarray,
    *,
    Sa: _FloatLike_co,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def midi_to_svara_h(
    midi: Union[_FloatLike_co, np.ndarray],
    *,
    Sa: _FloatLike_co,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


@vectorize(excluded=["Sa", "abbr", "octave", "unicode"])
def midi_to_svara_h(
    midi: Union[_FloatLike_co, np.ndarray],
    *,
    Sa: _FloatLike_co,
    abbr: bool = True,
    octave: bool = True,
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert MIDI numbers to Hindustani svara

    Parameters
    ----------
    midi : numeric or np.ndarray
        The MIDI number or numbers to convert

    Sa : number > 0
        MIDI number of the reference Sa.

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'r', 'R', 'g', 'G', ...)

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given MIDI number(s)

    See Also
    --------
    hz_to_svara_h
    note_to_svara_h
    midi_to_svara_c
    midi_to_note

    Examples
    --------
    Convert a single midi number:

    >>> librosa.midi_to_svara_h(65, Sa=60)
    'm'

    The first three svara with Sa at midi number 60:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=60)
    array(['S', 'r', 'R'], dtype='<U1')

    With Sa=67, midi 60-62 are in the octave below:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=67)
    array(['ṃ', 'Ṃ', 'P̣'], dtype='<U2')

    Or without unicode decoration:

    >>> librosa.midi_to_svara_h([60, 61, 62], Sa=67, unicode=False)
    array(['m,', 'M,', 'P,'], dtype='<U2')

    Or going up an octave, with Sa=60, and using unabbreviated notes

    >>> librosa.midi_to_svara_h([72, 73, 74], Sa=60, abbr=False)
    array(['Ṡa', 'ṙe', 'Ṙe'], dtype='<U3')
    """
    SVARA_MAP = [
        "Sa",
        "re",
        "Re",
        "ga",
        "Ga",
        "ma",
        "Ma",
        "Pa",
        "dha",
        "Dha",
        "ni",
        "Ni",
    ]

    SVARA_MAP_SHORT = list(s[0] for s in SVARA_MAP)

    # mypy does not understand vectorization
    svara_num = int(np.round(midi - Sa))  # type: ignore

    if abbr:
        svara = SVARA_MAP_SHORT[svara_num % 12]
    else:
        svara = SVARA_MAP[svara_num % 12]

    if octave:
        if 24 > svara_num >= 12:
            if unicode:
                svara = svara[0] + "\u0307" + svara[1:]
            else:
                svara += "'"
        elif -12 <= svara_num < 0:
            if unicode:
                svara = svara[0] + "\u0323" + svara[1:]
            else:
                svara += ","

    return svara


@overload
def hz_to_svara_h(
    frequencies: _FloatLike_co,
    *,
    Sa: _FloatLike_co,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def hz_to_svara_h(
    frequencies: _SequenceLike[_FloatLike_co],
    *,
    Sa: _FloatLike_co,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def hz_to_svara_h(
    frequencies: _ScalarOrSequence[_FloatLike_co],
    *,
    Sa: _FloatLike_co,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


def hz_to_svara_h(
    frequencies: _ScalarOrSequence[_FloatLike_co],
    *,
    Sa: _FloatLike_co,
    abbr: bool = True,
    octave: bool = True,
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert frequencies (in Hz) to Hindustani svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    frequencies : positive number or np.ndarray
        The frequencies (in Hz) to convert

    Sa : positive number
        Frequency (in Hz) of the reference Sa.

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'r', 'R', 'g', 'G', ...)

        If `False`, return long-form names ('Sa', 're', 'Re', 'ga', 'Ga', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given frequency/frequencies

    See Also
    --------
    midi_to_svara_h
    note_to_svara_h
    hz_to_svara_c
    hz_to_note

    Examples
    --------
    Convert Sa in three octaves:

    >>> librosa.hz_to_svara_h([261/2, 261, 261*2], Sa=261)
    ['Ṣ', 'S', 'Ṡ']

    Convert one octave worth of frequencies with full names:

    >>> freqs = librosa.cqt_frequencies(n_bins=12, fmin=261)
    >>> librosa.hz_to_svara_h(freqs, Sa=freqs[0], abbr=False)
    ['Sa', 're', 'Re', 'ga', 'Ga', 'ma', 'Ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
    """
    midis = hz_to_midi(frequencies)
    return midi_to_svara_h(
        midis, Sa=hz_to_midi(Sa), abbr=abbr, octave=octave, unicode=unicode
    )


@overload
def note_to_svara_h(
    notes: str,
    *,
    Sa: str,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def note_to_svara_h(
    notes: _IterableLike[str],
    *,
    Sa: str,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def note_to_svara_h(
    notes: Union[str, _IterableLike[str]],
    *,
    Sa: str,
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


def note_to_svara_h(
    notes: Union[str, _IterableLike[str]],
    *,
    Sa: str,
    abbr: bool = True,
    octave: bool = True,
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert western notes to Hindustani svara.

    Note that this conversion assumes 12-tone equal temperament.
    """
    midis = note_to_midi(notes, round_midi=False)

    return midi_to_svara_h(
        midis, Sa=note_to_midi(Sa), abbr=abbr, octave=octave, unicode=unicode
    )


@overload
def midi_to_svara_c(
    midi: _FloatLike_co,
    *,
    Sa: _FloatLike_co,
    mela: Union[int, str],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def midi_to_svara_c(
    midi: np.ndarray,
    *,
    Sa: _FloatLike_co,
    mela: Union[int, str],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def midi_to_svara_c(
    midi: Union[float, np.ndarray],
    *,
    Sa: _FloatLike_co,
    mela: Union[int, str],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


@vectorize(excluded=["Sa", "mela", "abbr", "octave", "unicode"])  # type: ignore
def midi_to_svara_c(
    midi: Union[float, np.ndarray],
    *,
    Sa: _FloatLike_co,
    mela: Union[int, str],
    abbr: bool = True,
    octave: bool = True,
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert MIDI numbers to Carnatic svara within a given melakarta raga

    Parameters
    ----------
    midi : numeric
        The MIDI numbers to convert

    Sa : number > 0
        MIDI number of the reference Sa.

        Default: 60 (261.6 Hz, `C4`)

    mela : int or str
        The name or index of the melakarta raga

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information and subscript
        numbers.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given MIDI number(s)

    See Also
    --------
    hz_to_svara_c
    note_to_svara_c
    mela_to_degrees
    mela_to_svara
    list_mela
    """
    svara_num = int(np.round(midi - Sa))

    svara_map = notation.mela_to_svara(mela, abbr=abbr, unicode=unicode)

    svara = svara_map[svara_num % 12]

    if octave:
        if 24 > svara_num >= 12:
            if unicode:
                svara = svara[0] + "\u0307" + svara[1:]
            else:
                svara += "'"
        elif -12 <= svara_num < 0:
            if unicode:
                svara = svara[0] + "\u0323" + svara[1:]
            else:
                svara += ","

    return svara


@overload
def hz_to_svara_c(
    frequencies: float,
    *,
    Sa: float,
    mela: Union[int, str],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def hz_to_svara_c(
    frequencies: np.ndarray,
    *,
    Sa: float,
    mela: Union[int, str],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def hz_to_svara_c(
    frequencies: Union[float, np.ndarray],
    *,
    Sa: float,
    mela: Union[int, str],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


def hz_to_svara_c(
    frequencies: Union[float, np.ndarray],
    *,
    Sa: float,
    mela: Union[int, str],
    abbr: bool = True,
    octave: bool = True,
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert frequencies (in Hz) to Carnatic svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    frequencies : positive number or np.ndarray
        The frequencies (in Hz) to convert

    Sa : positive number
        Frequency (in Hz) of the reference Sa.

    mela : str or int [1, 72]
        Melakarta raga name or index

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given frequency/frequencies

    See Also
    --------
    note_to_svara_c
    midi_to_svara_c
    hz_to_svara_h
    hz_to_note
    list_mela

    Examples
    --------
    Convert Sa in three octaves:

    >>> librosa.hz_to_svara_c([261/2, 261, 261*2], Sa=261, mela='kanakangi')
    ['Ṣ', 'S', 'Ṡ']

    Convert one octave worth of frequencies using melakarta #36:

    >>> freqs = librosa.cqt_frequencies(n_bins=12, fmin=261)
    >>> librosa.hz_to_svara_c(freqs, Sa=freqs[0], mela=36)
    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']
    """
    midis = hz_to_midi(frequencies)
    return midi_to_svara_c(
        midis, Sa=hz_to_midi(Sa), mela=mela, abbr=abbr, octave=octave, unicode=unicode
    )


@overload
def note_to_svara_c(
    notes: str,
    *,
    Sa: str,
    mela: Union[str, int],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def note_to_svara_c(
    notes: _IterableLike[str],
    *,
    Sa: str,
    mela: Union[str, int],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def note_to_svara_c(
    notes: Union[str, _IterableLike[str]],
    *,
    Sa: str,
    mela: Union[str, int],
    abbr: bool = ...,
    octave: bool = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


def note_to_svara_c(
    notes: Union[str, _IterableLike[str]],
    *,
    Sa: str,
    mela: Union[str, int],
    abbr: bool = True,
    octave: bool = True,
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert western notes to Carnatic svara

    Note that this conversion assumes 12-tone equal temperament.

    Parameters
    ----------
    notes : str or iterable of str
        Notes to convert (e.g., `'C#'` or `['C4', 'Db4', 'D4']`

    Sa : str
        Note corresponding to Sa (e.g., `'C'` or `'C5'`).

        If no octave information is provided, it will default to octave 0
        (``C0`` ~= 16 Hz)

    mela : str or int [1, 72]
        Melakarta raga to use.

    abbr : bool
        If `True` (default) return abbreviated names ('S', 'R1', 'R2', 'G1', 'G2', ...)

        If `False`, return long-form names ('Sa', 'Ri1', 'Ri2', 'Ga1', 'Ga2', ...)

    octave : bool
        If `True`, decorate svara in neighboring octaves with over- or under-dots.

        If `False`, ignore octave height information.

    unicode : bool
        If `True`, use unicode symbols to decorate octave information.

        If `False`, use low-order ASCII (' and ,) for octave decorations.

        This only takes effect if `octave=True`.

    Returns
    -------
    svara : str or np.ndarray of str
        The svara corresponding to the given notes

    See Also
    --------
    midi_to_svara_c
    hz_to_svara_c
    note_to_svara_h
    note_to_midi
    note_to_hz
    list_mela

    Examples
    --------
    >>> librosa.note_to_svara_h(['C4', 'G4', 'C5', 'D5', 'G5'], Sa='C5', mela=1)
    ['Ṣ', 'P̣', 'S', 'G₁', 'P']
    """
    midis = note_to_midi(notes, round_midi=False)

    return midi_to_svara_c(
        midis, Sa=note_to_midi(Sa), mela=mela, abbr=abbr, octave=octave, unicode=unicode
    )


@overload
def hz_to_fjs(
    frequencies: _FloatLike_co,
    *,
    fmin: Optional[float] = ...,
    unison: Optional[str] = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def hz_to_fjs(
    frequencies: _SequenceLike[_FloatLike_co],
    *,
    fmin: Optional[float] = ...,
    unison: Optional[str] = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


def hz_to_fjs(
    frequencies: _ScalarOrSequence[_FloatLike_co],
    *,
    fmin: Optional[float] = None,
    unison: Optional[str] = None,
    unicode: bool = False,
) -> Union[str, np.ndarray]:
    """Convert one or more frequencies (in Hz) from a just intonation
    scale to notes in FJS notation.

    Parameters
    ----------
    frequencies : float or iterable of float
        Input frequencies, specified in Hz
    fmin : float (optional)
        The minimum frequency, corresponding to a unison note.
        If not provided, it will be inferred as `min(frequencies)`
    unison : str (optional)
        The name of the unison note.
        If not provided, it will be inferred as the scientific pitch
        notation name of `fmin`, that is, `hz_to_note(fmin)`
    unicode : bool
        If `True`, then unicode symbols are used for accidentals.
        If `False`, then low-order ASCII symbols are used for accidentals.

    Returns
    -------
    notes : str or np.ndarray(dtype=str)
        ``notes[i]`` is the closest note name to ``frequency[i]``
        (or ``frequency`` if the input is scalar)

    See Also
    --------
    hz_to_note
    interval_to_fjs

    Examples
    --------
    Get a single note name for a frequency, relative to A=55 Hz

    >>> librosa.hz_to_fjs(66, fmin=55, unicode=True)
    'C₅'

    Get notation for a 5-limit frequency set starting at A=55

    >>> freqs = librosa.interval_frequencies(24, intervals="ji5", fmin=55)
    >>> freqs
    array([ 55.   ,  58.667,  61.875,  66.   ,  68.75 ,  73.333,  77.344,
        82.5  ,  88.   ,  91.667,  99.   , 103.125, 110.   , 117.333,
       123.75 , 132.   , 137.5  , 146.667, 154.687, 165.   , 176.   ,
       183.333, 198.   , 206.25 ])
    >>> librosa.hz_to_fjs(freqs, unicode=True)
    array(['A', 'B♭₅', 'B', 'C₅', 'C♯⁵', 'D', 'D♯⁵', 'E', 'F₅', 'F♯⁵', 'G₅',
       'G♯⁵', 'A', 'B♭₅', 'B', 'C₅', 'C♯⁵', 'D', 'D♯⁵', 'E', 'F₅', 'F♯⁵',
       'G₅', 'G♯⁵'], dtype='<U3')

    """
    if fmin is None:
        # mypy doesn't know that min can handle scalars
        fmin = np.min(frequencies)  # type: ignore
    if unison is None:
        unison = hz_to_note(fmin, octave=False, unicode=False)

    if np.isscalar(frequencies):
        # suppress type check - mypy does not understand scalar checks
        intervals = frequencies / fmin  # type: ignore
    else:
        intervals = np.asarray(frequencies) / fmin

    # mypy does not understand vectorization
    return notation.interval_to_fjs(intervals, unison=unison, unicode=unicode)  # type: ignore
