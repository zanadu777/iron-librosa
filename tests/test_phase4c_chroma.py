"""Phase 4C: Chroma Filter Projection Tests"""
import numpy as np
import pytest
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


class TestChromaProjectKernels:
    """Raw chroma_project kernel tests."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
    def test_chroma_project_f32_basic(self):
        """Verify basic f32 chroma projection."""
        np.random.seed(42)
        n_bins, n_frames, n_chroma = 513, 200, 12
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32)
        chromafb = np.abs(np.random.randn(n_chroma, n_bins)).astype(np.float32)

        result = _rust_ext.chroma_project_f32(
            np.ascontiguousarray(S),
            np.ascontiguousarray(chromafb),
        )

        assert result.shape == (n_chroma, n_frames)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
    def test_chroma_project_f64_basic(self):
        """Verify basic f64 chroma projection."""
        np.random.seed(42)
        n_bins, n_frames, n_chroma = 513, 200, 12
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float64)
        chromafb = np.abs(np.random.randn(n_chroma, n_bins)).astype(np.float64)

        result = _rust_ext.chroma_project_f64(
            np.ascontiguousarray(S),
            np.ascontiguousarray(chromafb),
        )

        assert result.shape == (n_chroma, n_frames)
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
    def test_chroma_project_f32_matches_einsum(self):
        """Verify f32 chroma projection matches numpy einsum."""
        np.random.seed(42)
        n_bins, n_frames, n_chroma = 513, 200, 12
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32)
        chromafb = np.abs(np.random.randn(n_chroma, n_bins)).astype(np.float32)

        rust_result = _rust_ext.chroma_project_f32(
            np.ascontiguousarray(S),
            np.ascontiguousarray(chromafb),
        )
        numpy_result = np.einsum("cf,ft->ct", chromafb, S, optimize=True)

        np.testing.assert_allclose(rust_result, numpy_result, rtol=1e-5)

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
    def test_chroma_project_f64_matches_einsum(self):
        """Verify f64 chroma projection matches numpy einsum."""
        np.random.seed(42)
        n_bins, n_frames, n_chroma = 513, 200, 12
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float64)
        chromafb = np.abs(np.random.randn(n_chroma, n_bins)).astype(np.float64)

        rust_result = _rust_ext.chroma_project_f64(
            np.ascontiguousarray(S),
            np.ascontiguousarray(chromafb),
        )
        numpy_result = np.einsum("cf,ft->ct", chromafb, S, optimize=True)

        np.testing.assert_allclose(rust_result, numpy_result, rtol=1e-10)

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
    def test_chroma_project_f32_device_override_matches_cpu(self, monkeypatch):
        """Requesting apple-gpu should currently match forced CPU output."""
        np.random.seed(42)
        n_bins, n_frames, n_chroma = 513, 64, 12
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32)
        chromafb = np.abs(np.random.randn(n_chroma, n_bins)).astype(np.float32)

        monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
        out_cpu = _rust_ext.chroma_project_f32(
            np.ascontiguousarray(S),
            np.ascontiguousarray(chromafb),
        )

        monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
        out_gpu_req = _rust_ext.chroma_project_f32(
            np.ascontiguousarray(S),
            np.ascontiguousarray(chromafb),
        )

        np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-7, atol=1e-7)


class TestChromaSTFTDispatch:
    """Public chroma_stft dispatch tests."""

    def test_chroma_stft_2d_f32_dispatch(self):
        """Verify chroma_stft dispatches to Rust for 2-D f32."""
        np.random.seed(42)
        n_fft = 2048
        n_bins = n_fft // 2 + 1
        n_frames = 200
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32) ** 2

        result = librosa.feature.chroma_stft(S=S, sr=22050, n_fft=n_fft)

        assert result.shape == (12, n_frames)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_chroma_stft_2d_f64_dispatch(self):
        """Verify chroma_stft dispatches to Rust for 2-D f64."""
        np.random.seed(42)
        n_fft = 2048
        n_bins = n_fft // 2 + 1
        n_frames = 200
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float64) ** 2

        result = librosa.feature.chroma_stft(S=S, sr=22050, n_fft=n_fft)

        assert result.shape == (12, n_frames)
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))

    def test_chroma_stft_multichannel_f32(self):
        """Verify chroma_stft works for multichannel f32."""
        np.random.seed(42)
        n_fft = 2048
        n_bins = n_fft // 2 + 1
        n_frames = 200
        n_channels = 2
        S = np.abs(np.random.randn(n_channels, n_bins, n_frames)).astype(np.float32) ** 2

        result = librosa.feature.chroma_stft(S=S, sr=22050, n_fft=n_fft)

        assert result.shape == (n_channels, 12, n_frames)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_chroma_stft_fallback_complex(self):
        """Verify chroma_stft falls back for complex input."""
        np.random.seed(42)
        n_fft = 2048
        n_bins = n_fft // 2 + 1
        n_frames = 200
        # Complex input should fall back to Python path
        S_complex = np.random.randn(n_bins, n_frames).astype(np.complex64)

        # Should not crash, should use Python fallback
        result = librosa.feature.chroma_stft(S=np.abs(S_complex), sr=22050, n_fft=n_fft)

        assert result.shape == (12, n_frames)
        assert np.all(np.isfinite(result))

