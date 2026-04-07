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
from ..core.spectrum import _spectrogram

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
