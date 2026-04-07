#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split librosa/core/convert.py into focused submodule files.

Fixed to correctly handle @overload decorated functions.

Each output file stays under ~900 lines.  The original convert.py is
replaced with a thin re-export wrapper.

Run from the repo root:
    python scripts/split_convert.py
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERT_PATH = os.path.join(ROOT, "librosa", "core", "convert.py")
CORE_DIR = os.path.join(ROOT, "librosa", "core")

with open(CONVERT_PATH, encoding="utf-8-sig") as fh:
    lines = fh.readlines()

total = len(lines)
print(f"convert.py: {total} lines")

IMPORTS = """\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

"""


def extract(start_1based, end_1based_inclusive):
    return "".join(lines[start_1based - 1 : end_1based_inclusive])


# Known first-line (1-based) of each public function:
FUNC_STARTS = {
    "frames_to_samples":       66,
    "samples_to_frames":      127,
    "frames_to_time":         209,
    "time_to_frames":         287,
    "time_to_samples":        372,
    "samples_to_time":        422,
    "blocks_to_frames":       479,
    "blocks_to_samples":      536,
    "blocks_to_time":         602,
    "note_to_hz":             680,
    "note_to_midi":           735,
    "midi_to_note":           864,
    "midi_to_hz":            1001,
    "hz_to_midi":            1051,
    "hz_to_note":            1100,
    "hz_to_mel":             1163,
    "mel_to_hz":             1251,
    "hz_to_octs":            1331,
    "octs_to_hz":            1397,
    "A4_to_tuning":          1464,
    "tuning_to_A4":          1535,
    "fft_frequencies":       1607,
    "cqt_frequencies":       1631,
    "mel_frequencies":       1669,
    "tempo_frequencies":     1753,
    "fourier_tempo_frequencies": 1791,
    "A_weighting":           1825,
    "B_weighting":           1904,
    "C_weighting":           1979,
    "D_weighting":           2052,
    "Z_weighting":           2132,
    "frequency_weighting":   2184,
    "multi_frequency_weighting": 2249,
    "times_like":            2300,
    "samples_like":          2360,
    "midi_to_svara_h":       2420,
    "hz_to_svara_h":         2571,
    "note_to_svara_h":       2675,
    "midi_to_svara_c":       2730,
    "hz_to_svara_c":         2844,
    "note_to_svara_c":       2956,
    "hz_to_fjs":             3065,
}

# Compute end lines: start of next function - 1, or total for the last
func_order = sorted(FUNC_STARTS.items(), key=lambda x: x[1])
FUNC_ENDS = {}
for i, (name, start) in enumerate(func_order):
    if i + 1 < len(func_order):
        FUNC_ENDS[name] = func_order[i + 1][1] - 1
    else:
        FUNC_ENDS[name] = total


def write_submodule(path, docstring, func_names):
    parts = [f'#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n"""{docstring}"""\n']
    parts.append(IMPORTS.replace("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n", ""))
    for name in func_names:
        start = FUNC_STARTS[name]
        end = FUNC_ENDS[name]
        parts.append(extract(start, end))
    content = "".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    lc = content.count("\n")
    print(f"  wrote {os.path.basename(path)}: {lc} lines")
    return lc


# --- File 1: _convert_time.py ---
write_submodule(
    os.path.join(CORE_DIR, "_convert_time.py"),
    "Time, frame, sample and block conversion utilities",
    [
        "frames_to_samples", "samples_to_frames",
        "frames_to_time", "time_to_frames",
        "time_to_samples", "samples_to_time",
        "blocks_to_frames", "blocks_to_samples", "blocks_to_time",
    ],
)

# --- File 2: _convert_pitch.py ---
write_submodule(
    os.path.join(CORE_DIR, "_convert_pitch.py"),
    "Pitch and note conversion utilities",
    [
        "note_to_hz", "note_to_midi", "midi_to_note",
        "midi_to_hz", "hz_to_midi", "hz_to_note",
    ],
)

# --- File 3: _convert_freq.py ---
write_submodule(
    os.path.join(CORE_DIR, "_convert_freq.py"),
    "Frequency scale and spectrogram frequency utilities",
    [
        "hz_to_mel", "mel_to_hz",
        "hz_to_octs", "octs_to_hz",
        "A4_to_tuning", "tuning_to_A4",
        "fft_frequencies", "cqt_frequencies", "mel_frequencies",
        "tempo_frequencies", "fourier_tempo_frequencies",
    ],
)

# --- File 4: _convert_weighting.py ---
write_submodule(
    os.path.join(CORE_DIR, "_convert_weighting.py"),
    "Frequency weighting curves and time/sample grid utilities",
    [
        "A_weighting", "B_weighting", "C_weighting",
        "D_weighting", "Z_weighting",
        "frequency_weighting", "multi_frequency_weighting",
        "times_like", "samples_like",
    ],
)

# --- File 5: _convert_svara.py ---
write_submodule(
    os.path.join(CORE_DIR, "_convert_svara.py"),
    "Indian music (svara) and FJS pitch notation utilities",
    [
        "midi_to_svara_h", "hz_to_svara_h", "note_to_svara_h",
        "midi_to_svara_c", "hz_to_svara_c", "note_to_svara_c",
        "hz_to_fjs",
    ],
)

# --- Thin re-export convert.py ---
ALL_FUNCS = [name for name, _ in func_order]

THIN = '''\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit conversion utilities.

All functions are re-exported from focused submodules for maintainability.
Public import paths are unchanged:

    librosa.frames_to_time(...)
    from librosa.core.convert import hz_to_mel
"""
from __future__ import annotations

from ._convert_time import (
    frames_to_samples,
    samples_to_frames,
    frames_to_time,
    time_to_frames,
    time_to_samples,
    samples_to_time,
    blocks_to_frames,
    blocks_to_samples,
    blocks_to_time,
)
from ._convert_pitch import (
    note_to_hz,
    note_to_midi,
    midi_to_note,
    midi_to_hz,
    hz_to_midi,
    hz_to_note,
)
from ._convert_freq import (
    hz_to_mel,
    mel_to_hz,
    hz_to_octs,
    octs_to_hz,
    A4_to_tuning,
    tuning_to_A4,
    fft_frequencies,
    cqt_frequencies,
    mel_frequencies,
    tempo_frequencies,
    fourier_tempo_frequencies,
)
from ._convert_weighting import (
    A_weighting,
    B_weighting,
    C_weighting,
    D_weighting,
    Z_weighting,
    frequency_weighting,
    multi_frequency_weighting,
    times_like,
    samples_like,
)
from ._convert_svara import (
    midi_to_svara_h,
    hz_to_svara_h,
    note_to_svara_h,
    midi_to_svara_c,
    hz_to_svara_c,
    note_to_svara_c,
    hz_to_fjs,
)

__all__ = [
    "frames_to_samples",
    "frames_to_time",
    "samples_to_frames",
    "samples_to_time",
    "time_to_samples",
    "time_to_frames",
    "blocks_to_samples",
    "blocks_to_frames",
    "blocks_to_time",
    "note_to_hz",
    "note_to_midi",
    "midi_to_hz",
    "midi_to_note",
    "hz_to_note",
    "hz_to_midi",
    "hz_to_mel",
    "hz_to_octs",
    "hz_to_fjs",
    "mel_to_hz",
    "octs_to_hz",
    "A4_to_tuning",
    "tuning_to_A4",
    "fft_frequencies",
    "cqt_frequencies",
    "mel_frequencies",
    "tempo_frequencies",
    "fourier_tempo_frequencies",
    "A_weighting",
    "B_weighting",
    "C_weighting",
    "D_weighting",
    "Z_weighting",
    "frequency_weighting",
    "multi_frequency_weighting",
    "samples_like",
    "times_like",
    "midi_to_svara_h",
    "midi_to_svara_c",
    "note_to_svara_h",
    "note_to_svara_c",
    "hz_to_svara_h",
    "hz_to_svara_c",
]
'''

convert_out = os.path.join(CORE_DIR, "convert.py")
with open(convert_out, "w", encoding="utf-8") as fh:
    fh.write(THIN)
print(f"  wrote convert.py: {THIN.count(chr(10))} lines")

print("Done.")


