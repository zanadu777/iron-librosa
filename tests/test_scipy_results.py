"""Compare Rust vs scipy median_filter on known cases."""

import numpy as np
from scipy.ndimage import median_filter

# Simple test case
S = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0]], dtype=np.float32)

print("Original:")
print(S)
print()

# scipy harmonic (vertical, size=[3,1])
print("scipy harmonic (size=[3,1]):")
harm_scipy = median_filter(S, size=[3, 1], mode="reflect")
print(harm_scipy)
print()

# scipy percussive (horizontal, size=[1,3])
print("scipy percussive (size=[1,3]):")
perc_scipy = median_filter(S, size=[1, 3], mode="reflect")
print(perc_scipy)
print()

# Now test larger array to see the pattern
S_large = np.arange(1, 16, dtype=np.float32).reshape(3, 5)
print("Larger array:")
print(S_large)
print()

print("scipy vertical median (size=[3,1]):")
harm_large = median_filter(S_large, size=[3, 1], mode="reflect")
print(harm_large)
print()

print("scipy horizontal median (size=[1,3]):")
perc_large = median_filter(S_large, size=[1, 3], mode="reflect")
print(perc_large)

