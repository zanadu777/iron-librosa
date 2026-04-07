#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split librosa/feature/spectral.py into focused submodule files.

Each output file stays under 1200 lines.  The original spectral.py is
replaced with a thin re-export wrapper so the public interface is
unchanged.

Run from the repo root:
    python scripts/split_spectral.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECTRAL_PATH = os.path.join(ROOT, "librosa", "feature", "spectral.py")
FEATURE_DIR = os.path.join(ROOT, "librosa", "feature")

with open(SPECTRAL_PATH, encoding="utf-8") as fh:
    lines = fh.readlines()

total = len(lines)
print(f"spectral.py: {total} lines")

# ---------------------------------------------------------------------------
# Common import block (lines 1-35, 0-indexed 0-34)
# ---------------------------------------------------------------------------
IMPORTS = """\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import scipy
import scipy.signal

from .. import util
from .. import filters
from ..util.exceptions import ParameterError

from ..core.fft import get_fftlib
from ..core.convert import fft_frequencies
from ..core.audio import zero_crossings
from ..core.spectrum import power_to_db, _spectrogram
from ..core.constantq import cqt, hybrid_cqt, vqt
from ..core.pitch import estimate_tuning
from .._rust_bridge import (
    _rust_ext,
    RUST_AVAILABLE,
    FORCE_RUST_MEL,
    FORCE_NUMPY_MEL,
)
from typing import Any, Optional, Union, Collection, Dict
from typing_extensions import Literal
from numpy.typing import DTypeLike
from .._typing import _FloatLike_co, _WindowSpec, _PadMode, _PadModeSTFT

try:
    from ._mel_threshold_registry import MEL_WORK_THRESHOLDS
except Exception:
    MEL_WORK_THRESHOLDS = {}

"""


def extract(start_1based, end_1based_inclusive):
    """Return lines[start-1 : end] as a single string (1-based line numbers)."""
    return "".join(lines[start_1based - 1 : end_1based_inclusive])


# ---------------------------------------------------------------------------
# Identify key line numbers by scanning for def/class boundaries
# ---------------------------------------------------------------------------
# We'll use the known function start lines (verified by inspection):
#   spectral_centroid:  246
#   spectral_bandwidth: 452
#   spectral_contrast:  681
#   spectral_rolloff:   966
#   spectral_flatness: 1172
#   rms:               1317
#   poly_features:     1486
#   zero_crossing_rate:1630
#   chroma_stft:       1704
#   chroma_cqt:        1887
#   chroma_cens:       2017
#   chroma_vqt:        2165
#   tonnetz:           2298
#   mfcc:              2432
#   melspectrogram:    2626

# Boundary: last line is total lines
BOUNDS = {
    # (first_line, last_line)  – 1-based, inclusive
    "spectral_centroid":   (246,  451),
    "spectral_bandwidth":  (452,  680),
    "spectral_contrast":   (681,  965),
    "spectral_rolloff":    (966, 1171),
    "spectral_flatness":  (1172, 1316),
    "rms":                (1317, 1485),
    "poly_features":      (1486, 1629),
    "zero_crossing_rate": (1630, 1703),
    "chroma_stft":        (1704, 1886),
    "chroma_cqt":         (1887, 2016),
    "chroma_cens":        (2017, 2164),
    "chroma_vqt":         (2165, 2297),
    "tonnetz":            (2298, 2431),
    "mfcc":               (2432, 2625),
    "melspectrogram":     (2626, total),
}

# Private helpers specific to each group
# contrast helpers: lines 38-155 in original (env constants + auto_ok/fused_ok)
CONTRAST_HELPERS = extract(38, 155)

# mel threshold helpers: lines 176-243
MEL_HELPERS = extract(176, 243)

# ---------------------------------------------------------------------------
# Build files
# ---------------------------------------------------------------------------

def write_submodule(path, docstring, extra_module_code, func_names):
    """Write a submodule file."""
    parts = [IMPORTS]
    if docstring:
        parts.insert(0, f'#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n"""{docstring}"""\n\n')
        parts[1] = IMPORTS.replace("#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n", "")
    else:
        parts[0] = IMPORTS
    if extra_module_code:
        parts.append("\n" + extra_module_code + "\n")
    for name in func_names:
        start, end = BOUNDS[name]
        parts.append("\n" + extract(start, end))
    content = "".join(parts)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    lc = content.count("\n")
    print(f"  wrote {os.path.basename(path)}: {lc} lines")
    return lc


# File 1: _spectral_centroid_bw.py
write_submodule(
    os.path.join(FEATURE_DIR, "_spectral_centroid_bw.py"),
    "Spectral centroid and bandwidth features",
    "",
    ["spectral_centroid", "spectral_bandwidth"],
)

# File 2: _spectral_contrast_rolloff.py  (contrast needs helpers)
write_submodule(
    os.path.join(FEATURE_DIR, "_spectral_contrast_rolloff.py"),
    "Spectral contrast and roll-off features",
    CONTRAST_HELPERS.rstrip(),
    ["spectral_contrast", "spectral_rolloff"],
)

# File 3: _spectral_misc.py
_ENABLE_RUST_RMS_TIME_BLOCK = """\
_ENABLE_RUST_RMS_TIME = os.getenv("IRON_LIBROSA_ENABLE_RUST_RMS_TIME", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
"""
write_submodule(
    os.path.join(FEATURE_DIR, "_spectral_misc.py"),
    "Spectral flatness, RMS, polynomial features, zero-crossing rate",
    _ENABLE_RUST_RMS_TIME_BLOCK,
    ["spectral_flatness", "rms", "poly_features", "zero_crossing_rate"],
)

# File 4: _spectral_chroma.py
write_submodule(
    os.path.join(FEATURE_DIR, "_spectral_chroma.py"),
    "Chroma and tonnetz features",
    "",
    ["chroma_stft", "chroma_cqt", "chroma_cens", "chroma_vqt", "tonnetz"],
)

# File 5: _spectral_mfcc_mel.py  (includes mel threshold helpers)
write_submodule(
    os.path.join(FEATURE_DIR, "_spectral_mfcc_mel.py"),
    "MFCC and mel-spectrogram features",
    MEL_HELPERS.rstrip(),
    ["mfcc", "melspectrogram"],
)

# ---------------------------------------------------------------------------
# Thin re-export spectral.py
# ---------------------------------------------------------------------------
THIN_SPECTRAL = '''\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction.

This module re-exports all public spectral features from focused submodules.
Import paths are preserved for backward compatibility:

    librosa.feature.spectral_centroid(...)
    from librosa.feature.spectral import spectral_centroid
    import librosa.feature.spectral as spectral; spectral.mfcc(...)
"""

from ._spectral_centroid_bw import spectral_centroid, spectral_bandwidth
from ._spectral_contrast_rolloff import spectral_contrast, spectral_rolloff
from ._spectral_misc import spectral_flatness, rms, poly_features, zero_crossing_rate
from ._spectral_chroma import chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, tonnetz
from ._spectral_mfcc_mel import mfcc, melspectrogram

__all__ = [
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_rolloff",
    "spectral_flatness",
    "poly_features",
    "rms",
    "zero_crossing_rate",
    "chroma_stft",
    "chroma_cqt",
    "chroma_cens",
    "chroma_vqt",
    "melspectrogram",
    "mfcc",
    "tonnetz",
]
'''

spectral_out = os.path.join(FEATURE_DIR, "spectral.py")
with open(spectral_out, "w", encoding="utf-8") as fh:
    fh.write(THIN_SPECTRAL)
print(f"  wrote spectral.py: {THIN_SPECTRAL.count(chr(10))} lines")

print("Done.")

