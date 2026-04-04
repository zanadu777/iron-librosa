"""Quick test to verify median filter behavior."""

import numpy as np
from scipy.ndimage import median_filter
import librosa

# Simple test array
S = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]], dtype=np.float32)

print("Original:")
print(S)
print()

# scipy vertical median filter (size = [3, 1])
print("scipy harmonic (size=[3,1]):")
harm_scipy = median_filter(S, size=[3, 1], mode="reflect")
print(harm_scipy)
print()

# Try Rust version
if hasattr(librosa._rust, 'median_filter_harmonic_f32'):
    print("Rust harmonic:")
    harm_rust = librosa._rust.median_filter_harmonic_f32(S, kernel_size=3)
    print(harm_rust)
else:
    print("Rust backend not available")

