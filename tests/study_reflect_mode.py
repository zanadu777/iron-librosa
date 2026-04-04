"""Study scipy.ndimage.median_filter reflect mode behavior."""

import numpy as np
from scipy.ndimage import median_filter

# Test reflect padding behavior on a simple 1D array
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print("Original array:", arr)
print()

# Apply median filter with kernel size 3 to understand padding
# scipy reflect mode: mirrors about the edge pixel (excludes the edge pixel in reflection)
result = median_filter(arr, size=3, mode='reflect')
print("Median filter with size=3:", result)
print()

# Let's manually trace what reflect padding should do:
# For reflect mode with kernel size k and array [a, b, c, d, e]:
# - Pad width = k // 2 = 1
# - Left padding: mirror [a, b, c, d, e] excluding rightmost, so [b, a, b, c, d, e]
# - Actually: reflect([1,2,3,4,5], pad=1) → [2, 1, 2, 3, 4, 5, 4]
# (mirror excludes the point being reflected, so: 2 is the mirror of 1 with exclusion)

# Let's test with different sizes to understand the pattern
for kernel_size in [3, 5, 7]:
    pad_width = kernel_size // 2
    result = median_filter(arr, size=kernel_size, mode='reflect')
    print(f"Kernel size {kernel_size} (pad={pad_width}): {result}")

print()

# Test 2D case
arr_2d = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])
print("Original 2D array:")
print(arr_2d)
print()

# Vertical median filter (size=[3, 1])
print("Vertical median filter (size=[3, 1]):")
result_v = median_filter(arr_2d, size=[3, 1], mode='reflect')
print(result_v)
print()

# What we should get for column [1, 4, 7] with reflect padding:
# Original: [1, 4, 7]
# With reflect pad=1: [4, 1, 4, 7, 4]  (no! let me think...)
# Actually reflect mode mirrors the axis:
# [1, 4, 7] → pad left by reflecting: what's the reflection of 1?
# In scipy reflect, we exclude the edge, so left padding of [1,4,7] is [4] (the next element)
# Right padding is [4] (the element before last)
# So: [4, 1, 4, 7, 4]
# Then sliding window [3]:
#  - [4, 1, 4] → median = 4
#  - [1, 4, 7] → median = 4
#  - [4, 7, 4] → median = 4

# Let me verify with the actual scipy result
print("For first column [1, 4, 7], median_filter result:", result_v[:, 0])
# Expected from reflect: [4, 4, 4] but let's see

