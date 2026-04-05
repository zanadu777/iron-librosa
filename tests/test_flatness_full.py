#!/usr/bin/env python
"""Tests for spectral_flatness Rust kernels and API dispatch."""

import numpy as np
import pytest

import librosa
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")


def _sample_s() -> np.ndarray:
    rng = np.random.default_rng(6001)
    return np.abs(rng.standard_normal((513, 20))).astype(np.float32)


def test_spectral_flatness_symbols_present() -> None:
    assert hasattr(_rust_ext, "spectral_flatness_f32")
    assert hasattr(_rust_ext, "spectral_flatness_f64")


def test_spectral_flatness_f32_kernel() -> None:
    s = _sample_s()
    out = _rust_ext.spectral_flatness_f32(s, 1e-10, 2.0)

    assert out.shape == (1, 20)
    assert out.dtype == np.float32
    assert np.all(out > 0)
    assert np.all(out <= 1.0)


def test_spectral_flatness_f64_kernel() -> None:
    s64 = _sample_s().astype(np.float64)
    out = _rust_ext.spectral_flatness_f64(s64, 1e-10, 2.0)

    assert out.shape == (1, 20)
    assert out.dtype == np.float64


def test_spectral_flatness_api_dispatch() -> None:
    s = _sample_s()
    out = librosa.feature.spectral_flatness(S=s)
    assert out.shape == (1, 20)
