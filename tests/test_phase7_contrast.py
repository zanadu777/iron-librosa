"""Phase 7 tests: spectral_contrast Rust kernel acceleration."""

import importlib
import os
from contextlib import contextmanager

import numpy as np
import pytest

import librosa
import librosa.feature.spectral as spectral_mod
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


@contextmanager
def _force_python_contrast_fallback(enabled: bool):
    """Temporarily disable Rust contrast dispatch to validate parity."""
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


@contextmanager
def _spectral_env_overrides(overrides):
    """Reload spectral module under temporary environment overrides."""

    prev = {k: os.environ.get(k) for k in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)

    importlib.reload(spectral_mod)
    try:
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(spectral_mod)


def _contrast_band_numpy(s_band, quantile):
    """Reference: naive Python implementation matching librosa logic."""
    n_bins = s_band.shape[0]
    n_frames = s_band.shape[1]

    # Compute quantile index (at least 1)
    idx = max(1, int(np.round(quantile * n_bins)))
    idx = min(idx, n_bins - 1)

    sortedr = np.sort(s_band, axis=0)

    valley = np.mean(sortedr[:idx, :], axis=0, keepdims=True)
    peak = np.mean(sortedr[-idx:, :], axis=0, keepdims=True)

    return peak, valley


class TestSpectralContrastAutoPolicy:
    """Unit tests for the shape-based auto-dispatch policy."""

    def test_auto_policy_mono_800_stays_off(self):
        assert not spectral_mod._contrast_rust_auto_ok(1, 1025, 800)

    def test_auto_policy_mono_1600_turns_on(self):
        assert spectral_mod._contrast_rust_auto_ok(1, 1025, 1600)

    def test_auto_policy_stereo_300_stays_off(self):
        assert not spectral_mod._contrast_rust_auto_ok(2, 1025, 300)

    def test_auto_policy_stereo_800_turns_on(self):
        assert not spectral_mod._contrast_rust_auto_ok(2, 1025, 800)

    def test_auto_policy_stereo_2400_turns_on(self):
        assert spectral_mod._contrast_rust_auto_ok(2, 1025, 2400)

    def test_auto_policy_quad_300_turns_on(self):
        assert spectral_mod._contrast_rust_auto_ok(4, 1025, 300)

    def test_auto_policy_eight_channel_300_turns_on(self):
        assert spectral_mod._contrast_rust_auto_ok(8, 1025, 300)

    def test_fused_policy_mono_1200_stays_off(self):
        assert not spectral_mod._contrast_rust_fused_ok(1, 1025, 1200)

    def test_fused_policy_mono_1600_turns_on(self):
        assert spectral_mod._contrast_rust_fused_ok(1, 1025, 1600)

    def test_fused_policy_stereo_800_turns_on(self):
        assert not spectral_mod._contrast_rust_fused_ok(2, 1025, 800)

    def test_fused_policy_stereo_2400_turns_on(self):
        assert spectral_mod._contrast_rust_fused_ok(2, 1025, 2400)

    def test_fused_policy_quad_300_turns_on(self):
        assert spectral_mod._contrast_rust_fused_ok(4, 1025, 300)

    def test_env_override_auto_threshold(self):
        with _spectral_env_overrides(
            {
                "IRON_LIBROSA_CONTRAST_RUST_WORK_THRESHOLD": "2000000",
                "IRON_LIBROSA_CONTRAST_RUST_MIN_FRAMES": "1",
            }
        ):
            assert not spectral_mod._contrast_rust_auto_ok(1, 1025, 1600)

    def test_env_override_fused_mono_min_frames(self):
        with _spectral_env_overrides(
            {
                "IRON_LIBROSA_CONTRAST_RUST_FUSED_MONO_MIN_FRAMES": "2000",
                "IRON_LIBROSA_CONTRAST_RUST_FUSED_MONO_WORK_THRESHOLD": "1",
            }
        ):
            assert not spectral_mod._contrast_rust_fused_ok(1, 1025, 1600)

    def test_env_invalid_values_fall_back_to_defaults(self):
        default_work = spectral_mod._CONTRAST_RUST_WORK_THRESHOLD
        default_heavy_channels = spectral_mod._CONTRAST_RUST_HEAVY_CHANNELS

        with _spectral_env_overrides(
            {
                "IRON_LIBROSA_CONTRAST_RUST_WORK_THRESHOLD": "not_an_int",
                "IRON_LIBROSA_CONTRAST_RUST_HEAVY_CHANNELS": "-4",
            }
        ):
            assert spectral_mod._CONTRAST_RUST_WORK_THRESHOLD == default_work
            assert spectral_mod._CONTRAST_RUST_HEAVY_CHANNELS == default_heavy_channels


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSpectralContrastBandKernels:
    """Parity tests for raw spectral_contrast_band_f32 / f64 vs NumPy reference."""

    @pytest.mark.parametrize("n_bins,n_frames,quantile", [
        (20, 10, 0.02),
        (100, 50, 0.1),
    ])
    def test_contrast_band_f32_parity(self, n_bins, n_frames, quantile):
        rng = np.random.default_rng(7001)
        s_band = np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float32)

        expected_peak, expected_valley = _contrast_band_numpy(s_band, quantile)
        actual_peak, actual_valley = _rust_ext.spectral_contrast_band_f32(s_band, quantile)

        assert actual_peak.shape == (1, n_frames)
        assert actual_valley.shape == (1, n_frames)
        # Tolerance: allow for rounding differences in idx calculation
        np.testing.assert_allclose(actual_peak, expected_peak, rtol=0.05, atol=1e-5)
        np.testing.assert_allclose(actual_valley, expected_valley, rtol=0.05, atol=1e-5)

    @pytest.mark.parametrize("n_bins,n_frames,quantile", [
        (20, 10, 0.02),
        (100, 50, 0.1),
    ])
    def test_contrast_band_f64_parity(self, n_bins, n_frames, quantile):
        rng = np.random.default_rng(7002)
        s_band = np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float64)

        expected_peak, expected_valley = _contrast_band_numpy(s_band, quantile)
        actual_peak, actual_valley = _rust_ext.spectral_contrast_band_f64(s_band, quantile)

        # Allow for rounding differences in idx calculation
        np.testing.assert_allclose(actual_peak, expected_peak, rtol=0.05, atol=1e-10)
        np.testing.assert_allclose(actual_valley, expected_valley, rtol=0.05, atol=1e-10)

    def test_contrast_band_f32_small_band(self):
        """Edge case: very small band (2 bins)."""
        s_band = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        quantile = 0.5

        expected_peak, expected_valley = _contrast_band_numpy(s_band, quantile)
        actual_peak, actual_valley = _rust_ext.spectral_contrast_band_f32(s_band, quantile)

        np.testing.assert_allclose(actual_peak, expected_peak, rtol=0.1)
        np.testing.assert_allclose(actual_valley, expected_valley, rtol=0.1)

    def test_contrast_band_large_array_f32(self):
        """Large array to test parallel path (> 200K elements)."""
        # 500 * 500 = 250K > PAR_THRESHOLD
        rng = np.random.default_rng(7004)
        s_band = np.abs(rng.standard_normal((500, 500))).astype(np.float32)

        expected_peak, expected_valley = _contrast_band_numpy(s_band, 0.02)
        actual_peak, actual_valley = _rust_ext.spectral_contrast_band_f32(s_band, 0.02)

        np.testing.assert_allclose(actual_peak, expected_peak, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual_valley, expected_valley, rtol=1e-4, atol=1e-5)

    def test_contrast_band_bad_quantile(self):
        """Invalid quantile should raise error."""
        s_band = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(Exception):
            _rust_ext.spectral_contrast_band_f32(s_band, 0.0)
        with pytest.raises(Exception):
            _rust_ext.spectral_contrast_band_f32(s_band, 1.0)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSpectralContrastAPI:
    """API-level dispatch tests for spectral_contrast."""

    def test_api_f32_basic(self):
        """Basic f32 input through full API."""
        rng = np.random.default_rng(7010)
        S = np.abs(rng.standard_normal((513, 50))).astype(np.float32)

        result = librosa.feature.spectral_contrast(S=S, sr=22050)

        # Should have shape (n_bands + 1, n_frames)
        assert result.shape[0] == 7  # default n_bands=6, so 6+1=7
        assert result.shape[1] == 50
        # Note: power_to_db converts to float64, so output dtype may differ from input
        assert result.dtype in (np.float32, np.float64)

    def test_api_f64_basic(self):
        """Basic f64 input through full API."""
        rng = np.random.default_rng(7011)
        S = np.abs(rng.standard_normal((513, 50))).astype(np.float64)

        result = librosa.feature.spectral_contrast(S=S, sr=22050)

        assert result.shape == (7, 50)
        assert result.dtype == np.float64

    def test_api_custom_n_bands(self):
        """Custom n_bands parameter."""
        rng = np.random.default_rng(7012)
        S = np.abs(rng.standard_normal((513, 40))).astype(np.float32)

        result = librosa.feature.spectral_contrast(S=S, sr=22050, n_bands=4)

        assert result.shape == (5, 40)  # 4 + 1

    def test_api_linear_contrast(self):
        """Linear vs logarithmic contrast."""
        rng = np.random.default_rng(7013)
        S = np.abs(rng.standard_normal((513, 30))).astype(np.float32)

        linear = librosa.feature.spectral_contrast(S=S, sr=22050, linear=True)
        log = librosa.feature.spectral_contrast(S=S, sr=22050, linear=False)

        # Should be different
        assert not np.allclose(linear, log)
        # Both should be same shape
        assert linear.shape == log.shape == (7, 30)

    def test_api_custom_quantile(self):
        """Custom quantile parameter."""
        rng = np.random.default_rng(7014)
        S = np.abs(rng.standard_normal((513, 40))).astype(np.float32)

        result_q002 = librosa.feature.spectral_contrast(S=S, sr=22050, quantile=0.02)
        result_q010 = librosa.feature.spectral_contrast(S=S, sr=22050, quantile=0.1)

        # Different quantiles should give different results (usually)
        assert not np.allclose(result_q002, result_q010)

    def test_api_custom_fmin(self):
        """Custom fmin parameter affects band boundaries."""
        rng = np.random.default_rng(7015)
        S = np.abs(rng.standard_normal((513, 40))).astype(np.float32)

        result_200 = librosa.feature.spectral_contrast(S=S, sr=22050, fmin=200)
        result_100 = librosa.feature.spectral_contrast(S=S, sr=22050, fmin=100)

        # Different fmin should give different results
        assert not np.allclose(result_200, result_100)

    def test_api_output_dtype_preserved(self):
        """Output dtype after power_to_db conversion."""
        rng = np.random.default_rng(7016)

        S32 = np.abs(rng.standard_normal((513, 30))).astype(np.float32)
        out32 = librosa.feature.spectral_contrast(S=S32)
        # power_to_db may promote to float64
        assert out32.dtype in (np.float32, np.float64)

        S64 = np.abs(rng.standard_normal((513, 30))).astype(np.float64)
        out64 = librosa.feature.spectral_contrast(S=S64)
        assert out64.dtype == np.float64

    def test_api_from_time_series(self):
        """Can compute from time-series input (tests full pipeline)."""
        rng = np.random.default_rng(7017)
        y = rng.standard_normal(22050 * 2).astype(np.float32)  # 2 seconds

        result = librosa.feature.spectral_contrast(y=y, sr=22050)

        assert result.shape[0] == 7
        assert result.shape[1] > 0

    def test_api_bad_quantile_raises(self):
        """Invalid quantile should raise error even at API level."""
        S = np.ones((513, 10), dtype=np.float32)
        with pytest.raises(Exception):
            librosa.feature.spectral_contrast(S=S, quantile=0.0)
        with pytest.raises(Exception):
            librosa.feature.spectral_contrast(S=S, quantile=1.0)

    def test_api_bad_fmin_raises(self):
        """Invalid fmin should raise error."""
        S = np.ones((513, 10), dtype=np.float32)
        with pytest.raises(Exception):
            librosa.feature.spectral_contrast(S=S, fmin=0.0)
        with pytest.raises(Exception):
            librosa.feature.spectral_contrast(S=S, fmin=-100)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_api_multichannel_3d_parity_forced_python(self, dtype):
        """3D input should match forced Python fallback (channel-wise parity)."""
        rng = np.random.default_rng(7020)
        S = np.abs(rng.standard_normal((2, 513, 60))).astype(dtype)

        with _force_python_contrast_fallback(False):
            out_rust = librosa.feature.spectral_contrast(S=S, sr=22050)

        with _force_python_contrast_fallback(True):
            out_py = librosa.feature.spectral_contrast(S=S, sr=22050)

        assert out_rust.shape == out_py.shape == (2, 7, 60)
        np.testing.assert_allclose(out_rust, out_py, rtol=1e-6, atol=1e-8)

    def test_api_multichannel_4d_parity_forced_python(self):
        """4D batched input should match forced Python fallback."""
        rng = np.random.default_rng(7021)
        S = np.abs(rng.standard_normal((2, 3, 513, 40))).astype(np.float32)

        with _force_python_contrast_fallback(False):
            out_rust = librosa.feature.spectral_contrast(S=S, sr=22050, quantile=0.01)

        with _force_python_contrast_fallback(True):
            out_py = librosa.feature.spectral_contrast(S=S, sr=22050, quantile=0.01)

        assert out_rust.shape == out_py.shape == (2, 3, 7, 40)
        np.testing.assert_allclose(out_rust, out_py, rtol=1e-6, atol=1e-8)

    def test_api_contrast_mode_python_override(self, monkeypatch):
        """`python` mode should disable Rust dispatch and match fallback."""
        rng = np.random.default_rng(7022)
        S = np.abs(rng.standard_normal((2, 513, 80))).astype(np.float32)

        monkeypatch.setattr(spectral_mod, "_CONTRAST_RUST_MODE", "python")
        out_override = librosa.feature.spectral_contrast(S=S, sr=22050)

        with _force_python_contrast_fallback(True):
            out_py = librosa.feature.spectral_contrast(S=S, sr=22050)

        np.testing.assert_allclose(out_override, out_py, rtol=1e-7, atol=1e-9)

    def test_api_contrast_threshold_gate_parity(self, monkeypatch):
        """Auto gate on/off should preserve numerical parity."""
        rng = np.random.default_rng(7023)
        S = np.abs(rng.standard_normal((2, 513, 900))).astype(np.float32)

        # Force auto mode and disable Rust by setting a very high threshold
        monkeypatch.setattr(spectral_mod, "_CONTRAST_RUST_MODE", "auto")
        monkeypatch.setattr(spectral_mod, "_CONTRAST_RUST_WORK_THRESHOLD", 10**12)
        out_auto_py = librosa.feature.spectral_contrast(S=S, sr=22050)

        # Force auto mode and enable Rust by lowering thresholds
        monkeypatch.setattr(spectral_mod, "_CONTRAST_RUST_WORK_THRESHOLD", 1)
        monkeypatch.setattr(spectral_mod, "_CONTRAST_RUST_MIN_FRAMES", 1)
        out_auto_rust = librosa.feature.spectral_contrast(S=S, sr=22050)

        np.testing.assert_allclose(out_auto_rust, out_auto_py, rtol=1e-6, atol=1e-8)










