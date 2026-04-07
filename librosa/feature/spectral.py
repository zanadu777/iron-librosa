#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Spectral feature extraction compatibility wrapper."""

from __future__ import annotations

import importlib

from .._rust_bridge import _rust_ext as _RUST_EXT_DEFAULT, RUST_AVAILABLE as _RUST_AVAILABLE_DEFAULT

from . import _spectral_centroid_bw
from . import _spectral_contrast_rolloff
from . import _spectral_misc
from . import _spectral_mfcc_mel
from ._spectral_chroma import chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, tonnetz
from ._spectral_mfcc_mel import mfcc, melspectrogram, _MEL_RUST_WORK_THRESHOLD
from ..core.spectrum import _spectrogram

# Reload contrast policy module on import so env-based thresholds update when this module is reloaded.
_spectral_contrast_rolloff = importlib.reload(_spectral_contrast_rolloff)

# Mutable compatibility globals used by tests and monkeypatch-based dispatch checks.
_rust_ext = _RUST_EXT_DEFAULT
RUST_AVAILABLE = _RUST_AVAILABLE_DEFAULT
_ENABLE_RUST_RMS_TIME = _spectral_misc._ENABLE_RUST_RMS_TIME

# Contrast policy internals preserved for backward compatibility.
_CONTRAST_RUST_MODE = _spectral_contrast_rolloff._CONTRAST_RUST_MODE
_CONTRAST_RUST_WORK_THRESHOLD = _spectral_contrast_rolloff._CONTRAST_RUST_WORK_THRESHOLD
_CONTRAST_RUST_MIN_FRAMES = _spectral_contrast_rolloff._CONTRAST_RUST_MIN_FRAMES
_CONTRAST_RUST_HEAVY_CHANNELS = _spectral_contrast_rolloff._CONTRAST_RUST_HEAVY_CHANNELS
_contrast_rust_auto_ok = _spectral_contrast_rolloff._contrast_rust_auto_ok
_contrast_rust_fused_ok = _spectral_contrast_rolloff._contrast_rust_fused_ok

MEL_WORK_THRESHOLDS = _spectral_mfcc_mel.MEL_WORK_THRESHOLDS


def _sync_feature_dispatch_state() -> None:
    _spectral_misc._rust_ext = _rust_ext
    _spectral_misc.RUST_AVAILABLE = RUST_AVAILABLE
    _spectral_misc._ENABLE_RUST_RMS_TIME = _ENABLE_RUST_RMS_TIME

    _spectral_centroid_bw._rust_ext = _rust_ext
    _spectral_centroid_bw.RUST_AVAILABLE = RUST_AVAILABLE

    _spectral_contrast_rolloff._rust_ext = _rust_ext
    _spectral_contrast_rolloff.RUST_AVAILABLE = RUST_AVAILABLE
    _spectral_contrast_rolloff._CONTRAST_RUST_MODE = _CONTRAST_RUST_MODE
    _spectral_contrast_rolloff._CONTRAST_RUST_WORK_THRESHOLD = _CONTRAST_RUST_WORK_THRESHOLD
    _spectral_contrast_rolloff._CONTRAST_RUST_MIN_FRAMES = _CONTRAST_RUST_MIN_FRAMES
    _spectral_contrast_rolloff._CONTRAST_RUST_HEAVY_CHANNELS = _CONTRAST_RUST_HEAVY_CHANNELS


def spectral_centroid(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_centroid_bw.spectral_centroid(*args, **kwargs)


def spectral_bandwidth(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_centroid_bw.spectral_bandwidth(*args, **kwargs)


def spectral_contrast(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_contrast_rolloff.spectral_contrast(*args, **kwargs)


def spectral_rolloff(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_contrast_rolloff.spectral_rolloff(*args, **kwargs)


def spectral_flatness(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_misc.spectral_flatness(*args, **kwargs)


def rms(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_misc.rms(*args, **kwargs)


def poly_features(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_misc.poly_features(*args, **kwargs)


def zero_crossing_rate(*args, **kwargs):
    _sync_feature_dispatch_state()
    return _spectral_misc.zero_crossing_rate(*args, **kwargs)


def _resolve_mel_work_threshold() -> int:
    _spectral_mfcc_mel.MEL_WORK_THRESHOLDS = MEL_WORK_THRESHOLDS
    return _spectral_mfcc_mel._resolve_mel_work_threshold()

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
