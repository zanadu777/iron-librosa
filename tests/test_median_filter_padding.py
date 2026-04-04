"""Direct unit tests for Rust median filter padding verification."""

import numpy as np
import pytest
from scipy.ndimage import median_filter

try:
    import librosa._rust as _rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend not available")
class TestMedianFilterPadding:
    """Verify Rust median filter matches scipy output on various inputs."""

    def test_harmonic_filter_f32_simple(self):
        """Test harmonic (horizontal/time) median filter on simple array."""
        # Simple 3x3 array
        S = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float32)

        # scipy reference: median with size=[1, 3] (horizontal / time)
        expected = median_filter(S, size=[1, 3], mode='reflect')

        # Rust harmonic with kernel_size=3
        actual = _rust.median_filter_harmonic_f32(S, kernel_size=3)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7,
                                   err_msg="Harmonic filter mismatch")

    def test_percussive_filter_f32_simple(self):
        """Test percussive (vertical/frequency) median filter on simple array."""
        S = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]], dtype=np.float32)

        # scipy reference: median with size=[3, 1] (vertical / frequency)
        expected = median_filter(S, size=[3, 1], mode='reflect')

        # Rust percussive with kernel_size=3
        actual = _rust.median_filter_percussive_f32(S, kernel_size=3)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7,
                                   err_msg="Percussive filter mismatch")

    def test_harmonic_filter_f32_random(self):
        """Test harmonic filter on random data."""
        rng = np.random.default_rng(42)
        S = np.abs(rng.standard_normal((257, 100))).astype(np.float32)

        expected = median_filter(S, size=[1, 31], mode='reflect')
        actual = _rust.median_filter_harmonic_f32(S, kernel_size=31)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6,
                                   err_msg="Harmonic filter random data mismatch")

    def test_percussive_filter_f32_random(self):
        """Test percussive filter on random data."""
        rng = np.random.default_rng(42)
        S = np.abs(rng.standard_normal((257, 100))).astype(np.float32)

        expected = median_filter(S, size=[31, 1], mode='reflect')
        actual = _rust.median_filter_percussive_f32(S, kernel_size=31)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6,
                                   err_msg="Percussive filter random data mismatch")

    def test_harmonic_filter_f64_random(self):
        """Test harmonic filter with f64."""
        rng = np.random.default_rng(42)
        S = np.abs(rng.standard_normal((257, 100))).astype(np.float64)

        expected = median_filter(S, size=[1, 31], mode='reflect')
        actual = _rust.median_filter_harmonic_f64(S, kernel_size=31)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                                   err_msg="Harmonic filter f64 mismatch")

    def test_percussive_filter_f64_random(self):
        """Test percussive filter with f64."""
        rng = np.random.default_rng(42)
        S = np.abs(rng.standard_normal((257, 100))).astype(np.float64)

        expected = median_filter(S, size=[31, 1], mode='reflect')
        actual = _rust.median_filter_percussive_f64(S, kernel_size=31)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12,
                                   err_msg="Percussive filter f64 mismatch")

    def test_edge_cases(self):
        """Test edge cases: single element, all same value, etc."""
        # Single value array
        S_single = np.array([[5.0]], dtype=np.float32)
        expected_single = median_filter(S_single, size=[1, 3], mode='reflect')
        actual_single = _rust.median_filter_harmonic_f32(S_single, kernel_size=3)
        np.testing.assert_allclose(actual_single, expected_single, rtol=1e-6)

        # All same values
        S_same = np.full((5, 5), 7.0, dtype=np.float32)
        expected_same = median_filter(S_same, size=[1, 3], mode='reflect')
        actual_same = _rust.median_filter_harmonic_f32(S_same, kernel_size=3)
        np.testing.assert_allclose(actual_same, expected_same, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

