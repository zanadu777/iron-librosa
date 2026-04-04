"""Phase 9 pilot tests: spectral_rolloff variable-frequency Rust path."""

import importlib
from contextlib import contextmanager

import numpy as np
import pytest

import librosa
from librosa.util.exceptions import ParameterError
from librosa._rust_bridge import RUST_AVAILABLE


@contextmanager
def _force_python_spectral_fallback(enabled: bool):
    mod = importlib.import_module("librosa.feature.spectral")
    prev_available = mod.RUST_AVAILABLE
    prev_ext = mod._rust_ext
    try:
        if enabled:
            mod.RUST_AVAILABLE = False
            mod._rust_ext = None
        yield
    finally:
        mod.RUST_AVAILABLE = prev_available
        mod._rust_ext = prev_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestVariableFreqRolloff:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_variable_freq_matches_python_fallback(self, dtype):
        rng = np.random.default_rng(9120)
        n_bins, n_frames = 1025, 120
        S = np.abs(rng.standard_normal((n_bins, n_frames))).astype(dtype)

        # Monotonic-ish per-frame frequency grids to emulate reassigned bins.
        base = np.linspace(0.0, 11025.0, n_bins, dtype=np.float64)[:, None]
        jitter = rng.normal(0.0, 3.0, size=(n_bins, n_frames))
        freq = np.maximum(0.0, base + jitter).astype(np.float64)

        roll_percent = 0.93
        with _force_python_spectral_fallback(True):
            expected = librosa.feature.spectral_rolloff(
                S=S, freq=freq, roll_percent=roll_percent
            )

        actual = librosa.feature.spectral_rolloff(S=S, freq=freq, roll_percent=roll_percent)

        assert actual.shape == expected.shape == (1, n_frames)
        rtol = 1e-5 if dtype == np.float32 else 1e-10
        atol = 1e-6 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def test_invalid_variable_freq_shape_uses_python_validation(self):
        S = np.abs(np.random.randn(513, 20)).astype(np.float32)
        bad_freq = np.abs(np.random.randn(513, 19)).astype(np.float64)
        with pytest.raises(ParameterError, match="freq.shape mismatch"):
            librosa.feature.spectral_rolloff(S=S, freq=bad_freq)

    @pytest.mark.parametrize("roll_percent", [0.0, 1.0, -0.1, 1.1])
    def test_invalid_roll_percent_raises(self, roll_percent):
        S = np.abs(np.random.randn(257, 16)).astype(np.float32)
        freq = np.abs(np.random.randn(257, 16)).astype(np.float64)
        with pytest.raises(Exception):
            librosa.feature.spectral_rolloff(S=S, freq=freq, roll_percent=roll_percent)

