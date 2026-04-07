#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split librosa/core/spectrum.py into focused submodule files.

Run from the repo root:
    python scripts/split_spectrum.py
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(ROOT, "librosa", "core")
SPECTRUM_PATH = os.path.join(CORE_DIR, "spectrum.py")

with open(SPECTRUM_PATH, encoding="utf-8-sig") as fh:
    lines = fh.readlines()

total = len(lines)
print(f"spectrum.py (working tree): {total} lines")

# Common imports block (same for all submodules)
IMPORTS = """\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

"""

# Function start lines (1-based) in the CURRENT spectrum.py.
# The current file has a modified _spectrogram function (~30 extra lines).
# All other functions are at the same position as git HEAD.
FIRST_DEF = {
    "stft":                    61,
    "istft":                  527,
    "__overlap_add":           763,
    "__reassign_frequencies":  779,
    "__reassign_times":        940,
    "reassigned_spectrogram": 1119,
    "magphase":               1425,
    "phase_vocoder":          1494,
    "iirt":                   1665,
    "power_to_db":            1856,   # first @overload
    "db_to_power":            2041,   # first @overload
    "amplitude_to_db":        2111,   # first @overload
    "db_to_amplitude":        2211,   # first @overload
    "perceptual_weighting":   2262,
    "fmt":                    2335,
    "pcen":                   2536,   # first @overload
    "griffinlim":             2870,
    "_spectrogram":           3100,
}

func_order = sorted(FIRST_DEF.items(), key=lambda x: x[1])


def group_start_1based(func_name, first_def_1based):
    """Find actual group start including @overload decorators."""
    gs_0 = first_def_1based - 1
    i = first_def_1based - 2
    while i >= 0:
        stripped = lines[i].strip()
        if stripped == "@overload":
            gs_0 = i
            i -= 1
        elif stripped == "":
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
    return gs_0 + 1


GROUP_STARTS = {n: group_start_1based(n, fd) for n, fd in FIRST_DEF.items()}
GROUP_ENDS = {}
for i, (name, _) in enumerate(func_order):
    if i + 1 < len(func_order):
        GROUP_ENDS[name] = GROUP_STARTS[func_order[i + 1][0]] - 1
    else:
        GROUP_ENDS[name] = total


def extract(s1, e1):
    return "".join(lines[s1 - 1: e1])


def write_submodule(path, docstring, func_names_list, extra_imports=""):
    header = f'#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n"""{docstring}"""\n'
    body = IMPORTS.replace("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n", "")
    parts = [header, body]
    if extra_imports:
        parts.append(extra_imports + "\n")
    for name in func_names_list:
        parts.append(extract(GROUP_STARTS[name], GROUP_ENDS[name]))
    content = "".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    lc = content.count("\n")
    print(f"  wrote {os.path.basename(path)}: {lc} lines")
    return lc


# File 1: _spectrum_stft.py – stft, istft, __overlap_add
write_submodule(
    os.path.join(CORE_DIR, "_spectrum_stft.py"),
    "STFT and iSTFT core functions",
    ["stft", "istft", "__overlap_add"],
)

# File 2: _spectrum_reassign.py – reassigned_spectrogram helpers + magphase
write_submodule(
    os.path.join(CORE_DIR, "_spectrum_reassign.py"),
    "Reassigned spectrogram and magphase",
    ["__reassign_frequencies", "__reassign_times", "reassigned_spectrogram", "magphase"],
    extra_imports="from ._spectrum_stft import stft",
)

# File 3: _spectrum_tf.py – phase_vocoder + iirt
write_submodule(
    os.path.join(CORE_DIR, "_spectrum_tf.py"),
    "Phase vocoder and IIR filtering",
    ["phase_vocoder", "iirt"],
    extra_imports="from ._spectrum_stft import stft, istft",
)

# File 4: _spectrum_db.py – DB/power conversions + perceptual + fmt
write_submodule(
    os.path.join(CORE_DIR, "_spectrum_db.py"),
    "Decibel and power scale conversion utilities",
    [
        "power_to_db", "db_to_power",
        "amplitude_to_db", "db_to_amplitude",
        "perceptual_weighting", "fmt",
    ],
)

# File 5: _spectrum_pcen.py – pcen + griffinlim + _spectrogram
write_submodule(
    os.path.join(CORE_DIR, "_spectrum_pcen.py"),
    "PCEN, Griffin-Lim reconstruction, and spectrogram helper",
    ["pcen", "griffinlim", "_spectrogram"],
    extra_imports="from ._spectrum_stft import stft, istft",
)

# ---------------------------------------------------------------------------
# Thin re-export spectrum.py
# ---------------------------------------------------------------------------
THIN = """\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"Utilities for spectral processing.

All public functions are re-exported from focused submodules.
Import paths are unchanged for backward compatibility.
\"\"\"
from __future__ import annotations

from ._spectrum_stft import stft, istft
from ._spectrum_reassign import (
    reassigned_spectrogram,
    magphase,
)
from ._spectrum_tf import phase_vocoder, iirt
from ._spectrum_db import (
    power_to_db,
    db_to_power,
    amplitude_to_db,
    db_to_amplitude,
    perceptual_weighting,
    fmt,
)
from ._spectrum_pcen import pcen, griffinlim, _spectrogram

__all__ = [
    "stft",
    "istft",
    "magphase",
    "iirt",
    "reassigned_spectrogram",
    "phase_vocoder",
    "perceptual_weighting",
    "power_to_db",
    "db_to_power",
    "amplitude_to_db",
    "db_to_amplitude",
    "fmt",
    "pcen",
    "griffinlim",
]
"""

spectrum_out = os.path.join(CORE_DIR, "spectrum.py")
with open(spectrum_out, "w", encoding="utf-8") as fh:
    fh.write(THIN)
print(f"  wrote spectrum.py: {THIN.count(chr(10))} lines")

print("Done.")

