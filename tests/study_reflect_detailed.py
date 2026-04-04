"""Detailed reflect padding study."""

import numpy as np
from scipy.ndimage import median_filter

# Test simple case - array of length 5
arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
print("Original:", arr)

# Check how scipy pads with reflect mode
# According to scipy: mode='reflect' mirrors about the edge EXCLUDING the edge pixel
# For [10, 20, 30, 40, 50] with pad=1:
# - left: reflect [20] (mirror excluding edge 10)
# - right: reflect [40] (mirror excluding edge 50)
# Result: [20, 10, 20, 30, 40, 50, 40]

# Let's verify using a filter that shows the padding
# Median of [20, 10, 20] = 20
# Median of [10, 20, 30] = 20
# etc.

result = median_filter(arr, size=3, mode='reflect')
print("Median(size=3, reflect):", result)

# Now 2D test - more explicit
arr_2d = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]])

print("\nOriginal 2D:")
print(arr_2d)

# Vertical median (along axis 0, kernel size 3)
result_v = median_filter(arr_2d, size=(3, 1), mode='reflect')
print("\nVertical median(3, 1, reflect):")
print(result_v)

# For first column [1, 4, 7] with pad=1:
# reflect mode pads to [4, 1, 4, 7, 4]
# Medians:
#   [4, 1, 4] → 4
#   [1, 4, 7] → 4
#   [4, 7, 4] → 4
# But scipy gives [1, 4, 7]... so maybe it's not doing what I think?

# Let me test with an array where the effect is clearer
arr_test = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
print("\n\nTest array:", arr_test)
result_test = median_filter(arr_test, size=3, mode='reflect')
print("Median(size=3, reflect):", result_test)

# Expected with reflect [10, 1, 10, 100, 1000, 10000, 1000]:
#  [10, 1, 10] → 10
#  [1, 10, 100] → 10
#  [10, 100, 1000] → 100
#  [100, 1000, 10000] → 1000
#  [1000, 10000, 1000] → 1000

# Another test: clear difference array
arr_test2 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
result_test2 = median_filter(arr_test2, size=3, mode='reflect')
print("\n\nTest array 2:", arr_test2)
print("Median(size=3, reflect):", result_test2)

# With reflect [1, 0, 1, 2, 3, 4, 3]:
#  [1, 0, 1] → 1
#  [0, 1, 2] → 1
#  [1, 2, 3] → 2
#  [2, 3, 4] → 3
#  [3, 4, 3] → 3
# Expected: [1, 1, 2, 3, 3] but let's see what scipy gives

