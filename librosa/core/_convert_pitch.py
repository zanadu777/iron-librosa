#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pitch and note conversion utilities"""
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
def note_to_hz(note: str, **kwargs: Any) -> np.floating[Any]:
    ...


@overload
def note_to_hz(note: _IterableLike[str], **kwargs: Any) -> np.ndarray:
    ...


@overload
def note_to_hz(
    note: Union[str, _IterableLike[str], Iterable[str]], **kwargs: Any
) -> Union[np.floating[Any], np.ndarray]:
    ...


def note_to_hz(
    note: Union[str, _IterableLike[str], Iterable[str]], **kwargs: Any
) -> Union[np.floating[Any], np.ndarray]:
    """Convert one or more note names to frequency (Hz)

    Examples
    --------
    >>> # Get the frequency of a note
    >>> librosa.note_to_hz('C')
    array([ 16.352])
    >>> # Or multiple notes
    >>> librosa.note_to_hz(['A3', 'A4', 'A5'])
    array([ 220.,  440.,  880.])
    >>> # Or notes with tuning deviations
    >>> librosa.note_to_hz('C2-32', round_midi=False)
    array([ 64.209])

    Parameters
    ----------
    note : str or iterable of str
        One or more note names to convert
    **kwargs : additional keyword arguments
        Additional parameters to `note_to_midi`

    Returns
    -------
    frequencies : number or np.ndarray [shape=(len(note),)]
        Array of frequencies (in Hz) corresponding to ``note``

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    """
    return midi_to_hz(note_to_midi(note, **kwargs))


@overload
def note_to_midi(note: str, *, round_midi: bool = ...) -> Union[float, int]:
    ...


@overload
def note_to_midi(note: _IterableLike[str], *, round_midi: bool = ...) -> np.ndarray:
    ...


@overload
def note_to_midi(
    note: Union[str, _IterableLike[str], Iterable[str]], *, round_midi: bool = ...
) -> Union[float, int, np.ndarray]:
    ...


def note_to_midi(
    note: Union[str, _IterableLike[str], Iterable[str]], *, round_midi: bool = True
) -> Union[float, np.ndarray]:
    """Convert one or more spelled notes to MIDI number(s).

    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with ``#``, flats may be indicated with ``!`` or ``b``.

    Parameters
    ----------
    note : str or iterable of str
        One or more note names.
    round_midi : bool
        - If ``True``, midi numbers are rounded to the nearest integer.
        - If ``False``, allow fractional midi numbers.

    Returns
    -------
    midi : float or np.array
        Midi note numbers corresponding to inputs.

    Raises
    ------
    ParameterError
        If the input is not in valid note format

    See Also
    --------
    midi_to_note
    note_to_hz

    Examples
    --------
    >>> librosa.note_to_midi('C')
    12
    >>> librosa.note_to_midi('C#3')
    49
    >>> librosa.note_to_midi('C♯3')  # Using Unicode sharp
    49
    >>> librosa.note_to_midi('C♭3')  # Using Unicode flat
    47
    >>> librosa.note_to_midi('f4')
    65
    >>> librosa.note_to_midi('Bb-1')
    10
    >>> librosa.note_to_midi('A!8')
    116
    >>> librosa.note_to_midi('G𝄪6')  # Double-sharp
    93
    >>> librosa.note_to_midi('B𝄫6')  # Double-flat
    93
    >>> librosa.note_to_midi('C♭𝄫5')  # Triple-flats also work
    69
    >>> # Lists of notes also work
    >>> librosa.note_to_midi(['C', 'E', 'G'])
    array([12, 16, 19])
    """
    if not isinstance(note, str):
        return np.array([note_to_midi(n, round_midi=round_midi) for n in note])

    pitch_map: Dict[str, int] = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }
    acc_map: Dict[str, int] = {
        "#": 1,
        "": 0,
        "b": -1,
        "!": -1,
        "♯": 1,
        "𝄪": 2,
        "♭": -1,
        "𝄫": -2,
        "♮": 0,
    }

    match = notation.NOTE_RE.match(note)

    if not match:
        raise ParameterError(f"Improper note format: {note:s}")

    pitch = match.group("note").upper()
    offset = np.sum([acc_map[o] for o in match.group("accidental")])
    octave = match.group("octave")
    cents = match.group("cents")

    if not octave:
        octave = 0
    else:
        octave = int(octave)

    if not cents:
        cents = 0
    else:
        cents = int(cents) * 1e-2

    note_value: float = 12 * (octave + 1) + pitch_map[pitch] + offset + cents

    if round_midi:
        return int(np.round(note_value))
    else:
        return note_value


@overload
def midi_to_note(
    midi: _FloatLike_co,
    *,
    octave: bool = ...,
    cents: bool = ...,
    key: str = ...,
    unicode: bool = ...,
) -> str:
    ...


@overload
def midi_to_note(
    midi: _SequenceLike[_FloatLike_co],
    *,
    octave: bool = ...,
    cents: bool = ...,
    key: str = ...,
    unicode: bool = ...,
) -> np.ndarray:
    ...


@overload
def midi_to_note(
    midi: _ScalarOrSequence[_FloatLike_co],
    *,
    octave: bool = ...,
    cents: bool = ...,
    key: str = ...,
    unicode: bool = ...,
) -> Union[str, np.ndarray]:
    ...


@vectorize(excluded=["octave", "cents", "key", "unicode"])
def midi_to_note(
    midi: _ScalarOrSequence[_FloatLike_co],
    *,
    octave: bool = True,
    cents: bool = False,
    key: str = "C:maj",
    unicode: bool = True,
) -> Union[str, np.ndarray]:
    """Convert one or more MIDI numbers to note strings.

    MIDI numbers will be rounded to the nearest integer.

    Notes will be of the format 'C0', 'C♯0', 'D0', ...

    Examples
    --------
    >>> librosa.midi_to_note(0)
    'C-1'

    >>> librosa.midi_to_note(37)
    'C♯2'

    >>> librosa.midi_to_note(37, unicode=False)
    'C#2'

    >>> librosa.midi_to_note(-2)
    'A♯-2'

    >>> librosa.midi_to_note(104.7)
    'A7'

    >>> librosa.midi_to_note(104.7, cents=True)
    'A7-30'

    >>> librosa.midi_to_note(np.arange(12, 24)))
    array(['C0', 'C♯0', 'D0', 'D♯0', 'E0', 'F0', 'F♯0', 'G0', 'G♯0', 'A0',
           'A♯0', 'B0'], dtype='<U3')

    Use a key signature to resolve enharmonic equivalences

    >>> librosa.midi_to_note(range(12, 24), key='F:min')
    array(['C0', 'D♭0', 'D0', 'E♭0', 'E0', 'F0', 'G♭0', 'G0', 'A♭0', 'A0',
           'B♭0', 'B0'], dtype='<U3')

    Parameters
    ----------
    midi : int or iterable of int
        Midi numbers to convert.

    octave : bool
        If True, include the octave number

    cents : bool
        If true, cent markers will be appended for fractional notes.
        Eg, ``midi_to_note(69.3, cents=True) == 'A4+03'``

    key : str
        A key signature to use when resolving enharmonic equivalences.

    unicode : bool
        If ``True`` (default), accidentals will use Unicode notation: ♭ or ♯

        If ``False``, accidentals will use ASCII-compatible notation: b or #

    Returns
    -------
    notes : str or np.ndarray of str
        Strings describing each midi note.

    Raises
    ------
    ParameterError
        if ``cents`` is True and ``octave`` is False

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    key_to_notes
    """
    if cents and not octave:
        raise ParameterError("Cannot encode cents without octave information.")

    note_map = notation.key_to_notes(key=key, unicode=unicode)

    # mypy does not understand vectorization, suppress type checks
    note_num = int(np.round(midi))  # type: ignore
    note_cents = int(100 * np.around(midi - note_num, 2))  # type: ignore

    note = note_map[note_num % 12]

    if octave:
        note = "{:s}{:0d}".format(note, int(note_num / 12) - 1)
    if cents:
        note = f"{note:s}{note_cents:+02d}"

    return note


@overload
def midi_to_hz(notes: _FloatLike_co) -> np.floating[Any]:
    ...


@overload
def midi_to_hz(notes: _SequenceLike[_FloatLike_co]) -> np.ndarray:
    ...


@overload
def midi_to_hz(
    notes: _ScalarOrSequence[_FloatLike_co],
) -> Union[np.ndarray, np.floating[Any]]:
    ...


def midi_to_hz(
    notes: _ScalarOrSequence[_FloatLike_co],
) -> Union[np.ndarray, np.floating[Any]]:
    """Get the frequency (Hz) of MIDI note(s)

    Examples
    --------
    >>> librosa.midi_to_hz(36)
    65.406

    >>> librosa.midi_to_hz(np.arange(36, 48))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,
             87.307,   92.499,   97.999,  103.826,  110.   ,
            116.541,  123.471])

    Parameters
    ----------
    notes : int or np.ndarray [shape=(n,), dtype=int]
        midi number(s) of the note(s)

    Returns
    -------
    frequency : number or np.ndarray [shape=(n,), dtype=float]
        frequency (frequencies) of ``notes`` in Hz

    See Also
    --------
    hz_to_midi
    note_to_hz
    """
    return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))


@overload
def hz_to_midi(frequencies: _FloatLike_co) -> np.floating[Any]:
    ...


@overload
def hz_to_midi(frequencies: _SequenceLike[_FloatLike_co]) -> np.ndarray:
    ...


@overload
def hz_to_midi(
    frequencies: _ScalarOrSequence[_FloatLike_co],
) -> Union[np.ndarray, np.floating[Any]]:
    ...


def hz_to_midi(
    frequencies: _ScalarOrSequence[_FloatLike_co],
) -> Union[np.ndarray, np.floating[Any]]:
    """Get MIDI note number(s) for given frequencies

    Examples
    --------
    >>> librosa.hz_to_midi(60)
    34.506
    >>> librosa.hz_to_midi([110, 220, 440])
    array([ 45.,  57.,  69.])

    Parameters
    ----------
    frequencies : float or np.ndarray [shape=(n,), dtype=float]
        frequencies to convert

    Returns
    -------
    note_nums : number or np.ndarray [shape=(n,), dtype=float]
        MIDI notes to ``frequencies``

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    """
    midi: np.ndarray = 12 * (np.log2(np.asanyarray(frequencies)) - np.log2(440.0)) + 69
    return midi


@overload
def hz_to_note(frequencies: _FloatLike_co, **kwargs: Any) -> str:
    ...


@overload
def hz_to_note(frequencies: _SequenceLike[_FloatLike_co], **kwargs: Any) -> np.ndarray:
    ...


@overload
def hz_to_note(
    frequencies: _ScalarOrSequence[_FloatLike_co], **kwargs: Any
) -> Union[str, np.ndarray]:
    ...


def hz_to_note(
    frequencies: _ScalarOrSequence[_FloatLike_co], **kwargs: Any
) -> Union[str, np.ndarray]:
    """Convert one or more frequencies (in Hz) to the nearest note names.

    Parameters
    ----------
    frequencies : float or iterable of float
        Input frequencies, specified in Hz
    **kwargs : additional keyword arguments
        Arguments passed through to `midi_to_note`

    Returns
    -------
    notes : str or np.ndarray of str
        ``notes[i]`` is the closest note name to ``frequency[i]``
        (or ``frequency`` if the input is scalar)

    See Also
    --------
    hz_to_midi
    midi_to_note
    note_to_hz

    Examples
    --------
    Get a single note name for a frequency

    >>> librosa.hz_to_note(440.0)
    'A5'

    Get multiple notes with cent deviation

    >>> librosa.hz_to_note([32, 64], cents=True)
    ['C1-38', 'C2-38']

    Get multiple notes, but suppress octave labels

    >>> librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 12)),
    ...                    octave=False)
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#', 'A']

    """
    return midi_to_note(hz_to_midi(frequencies), **kwargs)


