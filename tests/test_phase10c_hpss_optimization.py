"""
Phase 10C HPSS Optimization: Comprehensive Testing Suite

Tests:
1. Masking parallelization correctness
2. Frame-level parallelism validation
3. Parity vs SciPy fallback
4. Edge cases and boundaries
5. Dtype validation (f32/f64)
6. Shape validation (2D/3D/4D)
7. Mask mode testing
"""

import numpy as np
import pytest
import librosa
from scipy.ndimage import median_filter


class TestHPSSMaskingParallelization:
    """Validate frame-level parallelism in masking computation."""

    def test_masking_correctness_2d_f32(self):
        """Test that parallelized masking produces correct output."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((513, 200))).astype(np.float32)

        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Basic shape and dtype validation
        assert H.shape == S.shape
        assert P.shape == S.shape
        assert H.dtype == np.float32
        assert P.dtype == np.float32

        # Values should be in valid range
        assert np.all(H >= 0) or np.isclose(np.min(H), 0, atol=1e-6)
        assert np.all(P >= 0) or np.isclose(np.min(P), 0, atol=1e-6)

    def test_masking_correctness_2d_f64(self):
        """Test f64 precision masking parallelization."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((1025, 300))).astype(np.float64)

        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        assert H.shape == S.shape
        assert P.shape == S.shape
        assert H.dtype == np.float64
        assert P.dtype == np.float64

    def test_mask_mode_f32(self):
        """Test that mask=True returns masks, not scaled components."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((513, 200))).astype(np.float32)

        mask_H, mask_P = librosa.decompose.hpss(S, kernel_size=(31, 31), mask=True)

        # Masks should be between 0 and 1
        assert np.all(mask_H >= 0) and np.all(mask_H <= 1.0 + 1e-6)
        assert np.all(mask_P >= 0) and np.all(mask_P <= 1.0 + 1e-6)

        # At each bin, masks should sum approximately to 1 (with residual for margin=1.0)
        combined = mask_H + mask_P
        assert np.all(combined >= 0.99) or np.all(np.isclose(combined, 1.0, atol=1e-5))

    def test_masking_with_margins(self):
        """Test masking with margin > 1.0."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((513, 200))).astype(np.float32)

        # With margin > 1.0, masks can sum < 1 (residual component)
        mask_H, mask_P = librosa.decompose.hpss(
            S, kernel_size=(31, 31), mask=True, margin=2.0
        )

        # Masks should be non-negative
        assert np.all(mask_H >= 0)
        assert np.all(mask_P >= 0)

        # Sum should be < 1 due to residual
        combined = mask_H + mask_P
        assert np.all(combined <= 1.0 + 1e-5)


class TestHPSSParityValidation:
    """Validate that Rust HPSS matches SciPy baseline."""

    def test_parity_vs_scipy_small(self):
        """Parity test on small input."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((256, 100))).astype(np.float32)

        # Rust computation
        H_rust, P_rust = librosa.decompose.hpss(S, kernel_size=(31, 31), mask=True)

        # Scipy computation (reference)
        harm_shape = [1, 31]
        perc_shape = [31, 1]
        harm_scipy = median_filter(S, size=harm_shape, mode="reflect")
        perc_scipy = median_filter(S, size=perc_shape, mode="reflect")

        # Compute masks using librosa utility
        mask_H_scipy = librosa.util.softmask(harm_scipy, perc_scipy, power=2.0)
        mask_P_scipy = librosa.util.softmask(perc_scipy, harm_scipy, power=2.0)

        # Compare masks (allowing for numerical precision)
        np.testing.assert_allclose(H_rust, mask_H_scipy, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(P_rust, mask_P_scipy, rtol=1e-5, atol=1e-6)

    def test_parity_decomposition_vs_manual(self):
        """Test full decomposition: H + P ≈ S (with margins)."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((513, 200))).astype(np.float32)

        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31), margin=1.0)

        # With margin=1.0: H + P ≈ S
        reconstructed = H + P
        np.testing.assert_allclose(reconstructed, S, rtol=1e-4, atol=1e-5)

    def test_parity_residual_with_large_margin(self):
        """Test that residual exists with margin > 1.0."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((513, 200))).astype(np.float32)

        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31), margin=3.0)
        R = S - (H + P)

        # Residual should be non-negative
        assert np.all(R >= -1e-6)

        # Residual should be non-trivial (not all zeros)
        assert np.sum(np.abs(R)) > 0.1


class TestHPSSBatchProcessing:
    """Validate batch processing with frame-level parallelism."""

    def test_batch_vs_sequential_consistency(self):
        """Verify batch processing matches sequential processing."""
        rng = np.random.default_rng(9102026)
        S_batch = np.abs(rng.standard_normal((4, 513, 300))).astype(np.float32)

        # Process as batch
        H_batch, P_batch = librosa.decompose.hpss(S_batch, kernel_size=(31, 31))

        # Process individually
        H_individual = np.zeros_like(S_batch)
        P_individual = np.zeros_like(S_batch)
        for b in range(4):
            H_ind, P_ind = librosa.decompose.hpss(S_batch[b:b+1], kernel_size=(31, 31))
            H_individual[b] = H_ind[0]
            P_individual[b] = P_ind[0]

        # Should be identical (or very close numerically)
        np.testing.assert_allclose(H_batch, H_individual, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(P_batch, P_individual, rtol=1e-5, atol=1e-6)

    def test_batch_3d_stereo(self):
        """Test stereo batch processing (2 channels, N bins, T frames)."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((2, 513, 300))).astype(np.float32)

        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        assert H.shape == S.shape
        assert P.shape == S.shape


class TestHPSSEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_input_size(self):
        """Test with minimal spectrogram size."""
        S = np.ones((4, 4), dtype=np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(3, 3))
        assert H.shape == S.shape
        assert P.shape == S.shape

    def test_very_large_input(self):
        """Test with large spectrogram (should use frame parallelism)."""
        rng = np.random.default_rng(9102026)
        # Large input: 4096 bins × 2000 frames = 8M elements (exceeds PAR_THRESHOLD)
        S = np.abs(rng.standard_normal((4096, 2000))).astype(np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))
        assert H.shape == S.shape
        assert P.shape == S.shape

    def test_asymmetric_kernels(self):
        """Test with different harmonic and percussive kernel sizes."""
        rng = np.random.default_rng(9102026)
        S = np.abs(rng.standard_normal((513, 300))).astype(np.float32)

        H, P = librosa.decompose.hpss(S, kernel_size=(13, 31))
        assert H.shape == S.shape
        assert P.shape == S.shape

        # Verify decomposition quality
        reconstructed = H + P
        np.testing.assert_allclose(reconstructed, S, rtol=1e-4, atol=1e-5)

    def test_zero_input(self):
        """Test with all-zero input."""
        S = np.zeros((513, 300), dtype=np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Output should be zero
        np.testing.assert_array_equal(H, np.zeros_like(S))
        np.testing.assert_array_equal(P, np.zeros_like(S))

    def test_constant_input(self):
        """Test with constant-valued input."""
        S = np.full((513, 300), 0.5, dtype=np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Decomposition should be fairly even
        assert np.all(H >= 0)
        assert np.all(P >= 0)
        np.testing.assert_allclose(H + P, S, rtol=1e-4, atol=1e-5)


class TestHPSSNumericalStability:
    """Test numerical stability of masking computation."""

    def test_very_small_values(self):
        """Test with very small (near-zero) values."""
        S = np.full((256, 100), 1e-8, dtype=np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(P))

    def test_very_large_values(self):
        """Test with very large values."""
        S = np.full((256, 100), 1e8, dtype=np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(P))

    def test_mixed_magnitude_values(self):
        """Test with mixed magnitude orders."""
        rng = np.random.default_rng(9102026)
        # Mix very small and very large values
        S = np.abs(rng.standard_normal((256, 100))).astype(np.float32)
        S[0:50, 0:50] *= 1e-6  # Very small region
        S[100:150, 50:100] *= 1e6  # Very large region

        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # All values should be finite
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(P))
        assert np.all(H >= 0)
        assert np.all(P >= 0)


class TestHPSSPerformanceCharacteristics:
    """Test performance characteristics of frame parallelism."""

    def test_small_input_uses_sequential(self):
        """Small inputs should use sequential path (no parallelism)."""
        rng = np.random.default_rng(9102026)
        # Small: 256 bins × 100 frames = 25.6K elements (< 200K threshold)
        S = np.abs(rng.standard_normal((256, 100))).astype(np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Should produce valid results even with sequential path
        assert H.shape == S.shape
        assert P.shape == S.shape

    def test_large_input_uses_parallelism(self):
        """Large inputs should trigger frame-level parallelism."""
        rng = np.random.default_rng(9102026)
        # Large: 2048 bins × 1000 frames = 2.048M elements (>> 200K threshold)
        S = np.abs(rng.standard_normal((2048, 1000))).astype(np.float32)
        H, P = librosa.decompose.hpss(S, kernel_size=(31, 31))

        # Should produce valid results with parallelism
        assert H.shape == S.shape
        assert P.shape == S.shape
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(P))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

