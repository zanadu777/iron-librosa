#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split librosa/core/convert.py into focused submodule files (v2).

Correctly handles @overload decorated functions by looking backwards
from the known first-def line to find decorators.

Run from the repo root:
    python scripts/split_convert_v2.py
"""

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(ROOT, "librosa", "core")

# Read original convert.py from git (before our thin-wrapper change)
raw = subprocess.run(
    ["git", "-C", ROOT, "show", "HEAD:librosa/core/convert.py"],
    capture_output=True
).stdout.decode("utf-8-sig")

lines = raw.splitlines(keepends=True)
total = len(lines)
print(f"convert.py (from git): {total} lines")

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

# First 'def funcname' line numbers (1-based), from ast.parse scan
# These are the FIRST overloaded def for each function
FIRST_DEF = {
    "frames_to_samples":      66,
    "samples_to_frames":     127,
    "frames_to_time":        209,
    "time_to_frames":        287,
    "time_to_samples":       372,
    "samples_to_time":       422,
    "blocks_to_frames":      479,
    "blocks_to_samples":     536,
    "blocks_to_time":        602,
    "note_to_hz":            680,
    "note_to_midi":          735,
    "midi_to_note":          864,
    "midi_to_hz":           1001,
    "hz_to_midi":           1051,
    "hz_to_note":           1100,
    "hz_to_mel":            1163,
    "mel_to_hz":            1251,
    "hz_to_octs":           1331,
    "octs_to_hz":           1397,
    "A4_to_tuning":         1464,
    "tuning_to_A4":         1535,
    "fft_frequencies":      1607,
    "cqt_frequencies":      1631,
    "mel_frequencies":      1669,
    "tempo_frequencies":    1753,
    "fourier_tempo_frequencies": 1791,
    "A_weighting":          1825,
    "B_weighting":          1904,
    "C_weighting":          1979,
    "D_weighting":          2052,
    "Z_weighting":          2132,
    "frequency_weighting":  2184,
    "multi_frequency_weighting": 2249,
    "times_like":           2300,
    "samples_like":         2360,
    "midi_to_svara_h":      2420,
    "hz_to_svara_h":        2571,
    "note_to_svara_h":      2675,
    "midi_to_svara_c":      2730,
    "hz_to_svara_c":        2844,
    "note_to_svara_c":      2956,
    "hz_to_fjs":            3065,
}

func_order = sorted(FIRST_DEF.items(), key=lambda x: x[1])


def group_start_1based(func_name, first_def_1based):
    """Find the 1-based start line of a function group including decorators.

    Walks backward from first_def_1based - 1 to find @overload decorators
    and blank lines between overloads that are part of this group.
    """
    gs_0 = first_def_1based - 1  # 0-indexed default (the def line itself)
    i = first_def_1based - 2     # 0-indexed line just before the def

    while i >= 0:
        stripped = lines[i].strip()
        if stripped == "@overload":
            gs_0 = i
            i -= 1
        elif stripped == "":
            # blank line — might be between two @overload blocks
            # look further back to see if there's another @overload
            j = i - 1
            while j >= 0 and lines[j].strip() == "":
                j -= 1
            if j >= 0 and lines[j].strip() == "@overload":
                gs_0 = i
                i -= 1
            else:
                break
        else:
            break

    return gs_0 + 1  # back to 1-based


GROUP_STARTS = {
    name: group_start_1based(name, fd) for name, fd in FIRST_DEF.items()
}

GROUP_ENDS = {}
for i, (name, _) in enumerate(func_order):
    if i + 1 < len(func_order):
        next_name = func_order[i + 1][0]
        GROUP_ENDS[name] = GROUP_STARTS[next_name] - 1
    else:
        GROUP_ENDS[name] = total

# Report adjustments
print("Group start adjustments (decorator prefix detection):")
for name, fd in FIRST_DEF.items():
    gs = GROUP_STARTS[name]
    if gs != fd:
        print(f"  {name}: first_def={fd} -> group_start={gs}")


def extract(start_1, end_1):
    """Return lines[start_1-1 : end_1] as string."""
    return "".join(lines[start_1 - 1: end_1])


def write_submodule(path, docstring, func_names_list):
    header = f'#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n"""{docstring}"""\n'
    body = IMPORTS.replace("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n", "")
    parts = [header, body]
    for name in func_names_list:
        parts.append(extract(GROUP_STARTS[name], GROUP_ENDS[name]))
    content = "".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    lc = content.count("\n")
    print(f"  wrote {os.path.basename(path)}: {lc} lines")
    return lc


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

write_submodule(
    os.path.join(CORE_DIR, "_convert_pitch.py"),
    "Pitch and note conversion utilities",
    ["note_to_hz", "note_to_midi", "midi_to_note", "midi_to_hz", "hz_to_midi", "hz_to_note"],
)

write_submodule(
    os.path.join(CORE_DIR, "_convert_freq.py"),
    "Frequency scale and spectrogram frequency utilities",
    [
        "hz_to_mel", "mel_to_hz", "hz_to_octs", "octs_to_hz",
        "A4_to_tuning", "tuning_to_A4",
        "fft_frequencies", "cqt_frequencies", "mel_frequencies",
        "tempo_frequencies", "fourier_tempo_frequencies",
    ],
)

write_submodule(
    os.path.join(CORE_DIR, "_convert_weighting.py"),
    "Frequency weighting curves and time/sample grid utilities",
    [
        "A_weighting", "B_weighting", "C_weighting", "D_weighting", "Z_weighting",
        "frequency_weighting", "multi_frequency_weighting",
        "times_like", "samples_like",
    ],
)

write_submodule(
    os.path.join(CORE_DIR, "_convert_svara.py"),
    "Indian music (svara) and FJS pitch notation utilities",
    [
        "midi_to_svara_h", "hz_to_svara_h", "note_to_svara_h",
        "midi_to_svara_c", "hz_to_svara_c", "note_to_svara_c",
        "hz_to_fjs",
    ],
)

# ---------------------------------------------------------------------------
# Thin re-export convert.py
# ---------------------------------------------------------------------------
THIN = """\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"Unit conversion utilities.

All functions are re-exported from focused submodules for maintainability.
Public import paths are unchanged.
\"\"\"
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
"""

convert_out = os.path.join(CORE_DIR, "convert.py")
with open(convert_out, "w", encoding="utf-8") as fh:
    fh.write(THIN)
print(f"  wrote convert.py: {THIN.count(chr(10))} lines")

print("Done.")

