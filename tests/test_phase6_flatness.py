"""Phase 6 tests: spectral_flatness Rust kernel parity + API dispatch."""

import importlib

import numpy as np
import pytest

import librosa
import librosa.feature.spectral as spectral_mod
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatness_numpy(S, amin=1e-10, power=2.0):
    """Reference: exactly what librosa computes (pure-Python path)."""
    S_thresh = np.maximum(amin, S ** power)
    gmean = np.exp(np.mean(np.log(S_thresh), axis=-2, keepdims=True))
    amean = np.mean(S_thresh, axis=-2, keepdims=True)
    return gmean / amean


def _force_python_fallback(enabled: bool):
    """Context manager: disable Rust dispatch when enabled=True."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        prev_available = spectral_mod.RUST_AVAILABLE
        prev_ext = spectral_mod._rust_ext
        try:
            if enabled:
                spectral_mod.RUST_AVAILABLE = False
                spectral_mod._rust_ext = None
            yield
        finally:
            spectral_mod.RUST_AVAILABLE = prev_available
            spectral_mod._rust_ext = prev_ext

    return _ctx()


# ─────────────────────────────────────────────────────────────────────────────
# Raw kernel tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSpectralFlatnessKernels:
    """Parity tests for raw spectral_flatness_f32 / f64 vs NumPy reference."""

    @pytest.mark.parametrize("n_bins,n_frames", [(513, 20), (1025, 50), (2049, 10)])
    def test_flatness_f32_default_power(self, n_bins, n_frames):
        rng = np.random.default_rng(6001)
        S = np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float32)

        expected = _flatness_numpy(S, amin=1e-10, power=2.0).astype(np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)

        assert actual.shape == (1, n_frames)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("n_bins,n_frames", [(513, 20), (1025, 50)])
    def test_flatness_f64_default_power(self, n_bins, n_frames):
        rng = np.random.default_rng(6002)
        S = np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float64)

        expected = _flatness_numpy(S, amin=1e-10, power=2.0)
        actual = _rust_ext.spectral_flatness_f64(S, 1e-10, 2.0)

        assert actual.shape == (1, n_frames)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_flatness_f32_power_one(self):
        """power=1.0 fast path (no squaring)."""
        rng = np.random.default_rng(6003)
        S = np.abs(rng.standard_normal((1025, 30))).astype(np.float32)
        amin = 1e-8

        expected = _flatness_numpy(S, amin=amin, power=1.0).astype(np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, amin, 1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_flatness_f32_power_nonstandard(self):
        """Non-power-of-2 value (uses powf path)."""
        rng = np.random.default_rng(6004)
        S = np.abs(rng.standard_normal((513, 25))).astype(np.float32)

        expected = _flatness_numpy(S, amin=1e-10, power=1.5).astype(np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, 1e-10, 1.5)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

    def test_flatness_f32_silent_frame(self):
        """All-zero frame should produce flatness ≈ 1 (all values clamped to amin)."""
        S = np.zeros((1025, 5), dtype=np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)
        # geometric_mean(amin) / arithmetic_mean(amin) = 1.0; allow for minor float32 rounding
        np.testing.assert_allclose(actual, np.ones((1, 5), dtype=np.float32), rtol=1e-3, atol=1e-3)

    def test_flatness_f32_single_frame(self):
        """Edge case: n_frames=1."""
        rng = np.random.default_rng(6005)
        S = np.abs(rng.standard_normal((513, 1))).astype(np.float32)
        expected = _flatness_numpy(S, amin=1e-10, power=2.0).astype(np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_flatness_output_range(self):
        """Flatness values should lie in (0, 1]."""
        rng = np.random.default_rng(6006)
        S = np.abs(rng.standard_normal((1025, 100))).astype(np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)
        assert np.all(actual > 0)
        assert np.all(actual <= 1.0 + 1e-5)  # small tolerance for float rounding

    def test_flatness_bad_amin(self):
        S = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(Exception):
            _rust_ext.spectral_flatness_f32(S, 0.0, 2.0)

    def test_flatness_large_array_f32(self):
        """Ensure the parallel path (>PAR_THRESHOLD) gives same result as serial."""
        rng = np.random.default_rng(6007)
        # 1025 * 300 = 307_500 > PAR_THRESHOLD(200_000)
        S = np.abs(rng.standard_normal((1025, 300))).astype(np.float32)
        expected = _flatness_numpy(S, amin=1e-10, power=2.0).astype(np.float32)
        actual = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# API-level dispatch tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSpectralFlatnessAPI:
    """Verify the public API dispatches to Rust and matches the Python fallback."""

    @pytest.mark.parametrize("n_fft,n_frames", [(2048, 50), (1024, 100)])
    def test_api_f32_matches_python_fallback(self, n_fft, n_frames):
        n_bins = n_fft // 2 + 1
        rng = np.random.default_rng(6010)
        S = np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float32)

        with _force_python_fallback(True):
            expected = librosa.feature.spectral_flatness(S=S)

        actual = librosa.feature.spectral_flatness(S=S)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_api_f64_matches_python_fallback(self):
        rng = np.random.default_rng(6011)
        S = np.abs(rng.standard_normal((1025, 40))).astype(np.float64)

        with _force_python_fallback(True):
            expected = librosa.feature.spectral_flatness(S=S)

        actual = librosa.feature.spectral_flatness(S=S)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_api_power_one(self):
        rng = np.random.default_rng(6012)
        S = np.abs(rng.standard_normal((1025, 40))).astype(np.float32)

        with _force_python_fallback(True):
            expected = librosa.feature.spectral_flatness(S=S, power=1.0)

        actual = librosa.feature.spectral_flatness(S=S, power=1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_api_custom_amin(self):
        rng = np.random.default_rng(6013)
        S = np.abs(rng.standard_normal((513, 30))).astype(np.float32)

        with _force_python_fallback(True):
            expected = librosa.feature.spectral_flatness(S=S, amin=1e-6)

        actual = librosa.feature.spectral_flatness(S=S, amin=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_api_multichannel(self):
        """3-D input (2-channel): shape (..., n_bins, n_frames)."""
        rng = np.random.default_rng(6014)
        S = np.abs(rng.standard_normal((2, 1025, 40))).astype(np.float32)

        with _force_python_fallback(True):
            expected = librosa.feature.spectral_flatness(S=S)

        actual = librosa.feature.spectral_flatness(S=S)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)

    def test_api_output_dtype_f32(self):
        rng = np.random.default_rng(6015)
        S = np.abs(rng.standard_normal((1025, 30))).astype(np.float32)
        out = librosa.feature.spectral_flatness(S=S)
        assert out.dtype == np.float32

    def test_api_output_dtype_f64(self):
        rng = np.random.default_rng(6016)
        S = np.abs(rng.standard_normal((1025, 30))).astype(np.float64)
        out = librosa.feature.spectral_flatness(S=S)
        assert out.dtype == np.float64

    def test_api_negative_S_raises(self):
        """Negative spectrogram values must still raise ParameterError."""
        S = np.full((1025, 10), -1.0, dtype=np.float32)
        with pytest.raises(Exception):
            librosa.feature.spectral_flatness(S=S)

    def test_api_bad_amin_raises(self):
        S = np.ones((1025, 10), dtype=np.float32)
        with pytest.raises(Exception):
            librosa.feature.spectral_flatness(S=S, amin=0.0)

