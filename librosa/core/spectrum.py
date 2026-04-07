#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for spectral processing.

All public functions are re-exported from focused submodules.
Import paths are unchanged for backward compatibility.
"""
from __future__ import annotations

from functools import wraps

from .._rust_bridge import (
    _rust_ext as _RUST_EXT_DEFAULT,
    RUST_AVAILABLE as _RUST_AVAILABLE_DEFAULT,
    FORCE_NUMPY_STFT as _FORCE_NUMPY_STFT_DEFAULT,
    FORCE_RUST_STFT as _FORCE_RUST_STFT_DEFAULT,
)

from . import _spectrum_stft
from . import _spectrum_reassign
from . import _spectrum_tf
from . import _spectrum_db
from . import _spectrum_pcen

# Keep these module globals mutable for backward-compatible monkeypatching.
_rust_ext = _RUST_EXT_DEFAULT
RUST_AVAILABLE = _RUST_AVAILABLE_DEFAULT
FORCE_NUMPY_STFT = _FORCE_NUMPY_STFT_DEFAULT
FORCE_RUST_STFT = _FORCE_RUST_STFT_DEFAULT


def _sync_rust_dispatch_state() -> None:
    _spectrum_stft._rust_ext = _rust_ext
    _spectrum_stft.RUST_AVAILABLE = RUST_AVAILABLE
    _spectrum_stft.FORCE_NUMPY_STFT = FORCE_NUMPY_STFT
    _spectrum_stft.FORCE_RUST_STFT = FORCE_RUST_STFT
    _spectrum_tf._rust_ext = _rust_ext
    _spectrum_tf.RUST_AVAILABLE = RUST_AVAILABLE
    _spectrum_tf.FORCE_NUMPY_STFT = FORCE_NUMPY_STFT
    _spectrum_tf.FORCE_RUST_STFT = FORCE_RUST_STFT
    _spectrum_pcen._rust_ext = _rust_ext
    _spectrum_pcen.RUST_AVAILABLE = RUST_AVAILABLE
    _spectrum_pcen.FORCE_NUMPY_STFT = FORCE_NUMPY_STFT
    _spectrum_pcen.FORCE_RUST_STFT = FORCE_RUST_STFT
    _spectrum_reassign._rust_ext = _rust_ext
    _spectrum_reassign.RUST_AVAILABLE = RUST_AVAILABLE
    _spectrum_reassign.FORCE_NUMPY_STFT = FORCE_NUMPY_STFT
    _spectrum_reassign.FORCE_RUST_STFT = FORCE_RUST_STFT


@wraps(_spectrum_stft.stft)
def stft(*args, **kwargs):
    _sync_rust_dispatch_state()
    return _spectrum_stft.stft(*args, **kwargs)


@wraps(_spectrum_stft.istft)  # type: ignore[has-type]
def istft(*args, **kwargs):
    return _spectrum_stft.istft(*args, **kwargs)


@wraps(_spectrum_reassign.reassigned_spectrogram)
def reassigned_spectrogram(*args, **kwargs):
    _sync_rust_dispatch_state()
    _spectrum_reassign.__reassign_frequencies = __reassign_frequencies
    _spectrum_reassign.__reassign_times = __reassign_times
    return _spectrum_reassign.reassigned_spectrogram(*args, **kwargs)


magphase = _spectrum_reassign.magphase


__reassign_frequencies = _spectrum_reassign.__reassign_frequencies
__reassign_times = _spectrum_reassign.__reassign_times


@wraps(_spectrum_tf.phase_vocoder)
def phase_vocoder(*args, **kwargs):
    _sync_rust_dispatch_state()
    return _spectrum_tf.phase_vocoder(*args, **kwargs)


iirt = _spectrum_tf.iirt  # type: ignore[has-type]

power_to_db = _spectrum_db.power_to_db
db_to_power = _spectrum_db.db_to_power
amplitude_to_db = _spectrum_db.amplitude_to_db
db_to_amplitude = _spectrum_db.db_to_amplitude
perceptual_weighting = _spectrum_db.perceptual_weighting  # type: ignore[has-type]
fmt = _spectrum_db.fmt  # type: ignore[has-type]

pcen = _spectrum_pcen.pcen


@wraps(_spectrum_pcen.griffinlim)
def griffinlim(*args, **kwargs):
    _sync_rust_dispatch_state()
    return _spectrum_pcen.griffinlim(*args, **kwargs)


@wraps(_spectrum_pcen._spectrogram)
def _spectrogram(*args, **kwargs):
    _sync_rust_dispatch_state()
    return _spectrum_pcen._spectrogram(*args, **kwargs)

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
