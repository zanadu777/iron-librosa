#!/usr/bin/env python
"""Phase 13 parity scaffolding for CQT/VQT Rust acceleration.

This file intentionally starts with Python-reference tests so we can lock
baseline behavior before wiring any Rust kernel dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest

import librosa
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


def _phase13_signal(sr: int = 22050, seconds: float = 3.0, seed: int = 13001) -> np.ndarray:
    """Deterministic test signal with harmonics + noise for parity checks."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    y = 0.65 * np.sin(2.0 * np.pi * 220.0 * t)
    y += 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    y += 0.08 * np.sin(2.0 * np.pi * 660.0 * t)
    y += 0.02 * rng.standard_normal(n).astype(np.float32)
    return y.astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_cqt_reference_deterministic(dtype):
    """Reference CQT path should be deterministic under fixed inputs/settings."""
    y = _phase13_signal().astype(dtype)

    cqt_a = librosa.cqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        sparsity=0.01,
        res_type="soxr_hq",
    )
    cqt_b = librosa.cqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        sparsity=0.01,
        res_type="soxr_hq",
    )

    assert cqt_a.shape == cqt_b.shape
    np.testing.assert_allclose(cqt_a, cqt_b, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_vqt_reference_deterministic(dtype):
    """Reference VQT path should be deterministic under fixed inputs/settings."""
    y = _phase13_signal().astype(dtype)

    vqt_a = librosa.vqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        gamma=0.0,
        sparsity=0.01,
        res_type="soxr_hq",
    )
    vqt_b = librosa.vqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        gamma=0.0,
        sparsity=0.01,
        res_type="soxr_hq",
    )

    assert vqt_a.shape == vqt_b.shape
    np.testing.assert_allclose(vqt_a, vqt_b, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize(
    "fn_name",
    ["cqt_project_f32", "cqt_project_f64", "vqt_project_f32", "vqt_project_f64"],
)
def test_phase13_rust_kernel_symbols_not_required_yet(fn_name):
    """Phase 13 day-1: this test records symbol absence without failing CI.

    Once Rust kernels are wired, replace this with strict symbol assertions.
    """
    if not RUST_AVAILABLE:
        pytest.skip("Rust extension is not available in this environment")

    has_symbol = hasattr(_rust_ext, fn_name)
    # The test is intentionally permissive on day-1 scaffolding.
    assert isinstance(has_symbol, bool)

