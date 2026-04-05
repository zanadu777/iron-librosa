"""
iron-librosa Rust bridge
========================

This module is the single import point for the compiled Rust extension
``librosa._rust``.  All Python submodules that want to dispatch to Rust
should import from here::

    from .._rust_bridge import _rust_ext, RUST_AVAILABLE

Design:
  - ``_rust_ext``      : the compiled extension module, or ``None`` when
                         the Rust build is not available (e.g., a source-
                         only install, CI without Rust, etc.).
  - ``RUST_AVAILABLE`` : convenience boolean for ``_rust_ext is not None``.

Adding new Rust accelerations
------------------------------
1. Implement the function in ``src/<module>.rs`` and register it in
   ``src/lib.rs`` (``m.add_function(...)``).
2. In the corresponding Python file, add a guard at the top of the
   function body::

       if RUST_AVAILABLE and hasattr(_rust_ext, "my_function"):
           return _rust_ext.my_function(...)
       # ... original Python implementation follows ...

   The ``hasattr`` check lets you deploy incrementally: the Python
   fallback remains active until the Rust version is wired up.
"""

from __future__ import annotations

import os

__all__ = [
    "_rust_ext",
    "RUST_AVAILABLE",
    "RUST_EXTENSION_AVAILABLE",
    "FORCE_NUMPY_MEL",
    "FORCE_RUST_MEL",
    "FORCE_NUMPY_CQT_VQT",
    "FORCE_RUST_CQT_VQT",
    "FORCE_NUMPY_BEAT",
    "FORCE_RUST_BEAT",
]

# Mel backend policy override for librosa.feature.melspectrogram 2D path.
# Accepted values: "auto" (default), "numpy", "rust".
_mel_backend = os.getenv("IRON_LIBROSA_MEL_BACKEND", "auto").strip().lower()
if _mel_backend not in {"auto", "numpy", "rust"}:
    _mel_backend = "auto"

FORCE_NUMPY_MEL: bool = _mel_backend == "numpy"
FORCE_RUST_MEL: bool = _mel_backend == "rust"

# Phase 13 CQT/VQT backend policy. This seam remains opt-in until benchmark
# evidence is strong enough to justify default promotion.
# Accepted values: "auto" (default, keep NumPy path), "numpy", "rust".
_cqt_vqt_backend = os.getenv("IRON_LIBROSA_CQT_VQT_BACKEND", "auto").strip().lower()
if _cqt_vqt_backend not in {"auto", "numpy", "rust"}:
    _cqt_vqt_backend = "auto"

FORCE_NUMPY_CQT_VQT: bool = _cqt_vqt_backend == "numpy"
FORCE_RUST_CQT_VQT: bool = _cqt_vqt_backend == "rust"

# Phase 14 beat backend policy. Keep default on NumPy/Numba path until
# benchmark and parity gates support promotion.
# Accepted values: "auto" (default), "numpy", "rust".
_beat_backend = os.getenv("IRON_LIBROSA_BEAT_BACKEND", "auto").strip().lower()
if _beat_backend not in {"auto", "numpy", "rust"}:
    _beat_backend = "auto"

FORCE_NUMPY_BEAT: bool = _beat_backend == "numpy"
FORCE_RUST_BEAT: bool = _beat_backend == "rust"

try:
    from librosa import _rust as _rust_ext  # type: ignore[attr-defined]
except ImportError:
    _rust_ext = None

# Keep extension availability separate from dispatch policy.
RUST_EXTENSION_AVAILABLE: bool = _rust_ext is not None


# Global dispatch gate: default to Rust accelerated dispatch paths unless explicitly disabled.
# Set IRON_LIBROSA_RUST_DISPATCH=0 to force legacy NumPy/SciPy parity mode.
_rust_dispatch = os.getenv("IRON_LIBROSA_RUST_DISPATCH", "1").strip().lower()
RUST_AVAILABLE: bool = RUST_EXTENSION_AVAILABLE and _rust_dispatch not in {"0", "false", "no", "off"}
