#!/usr/bin/env python
"""
Validation script for Rust dispatch safety patch + dimensional guard.
This confirms that hz_to_mel and mel_to_hz work with both 1D and 2D arrays.
"""
import sys
import numpy as np
import librosa
from librosa._rust_bridge import RUST_AVAILABLE, RUST_EXTENSION_AVAILABLE

print("=" * 70)
print("RUST DISPATCH SAFETY VALIDATION")
print("=" * 70)
print(f"RUST_EXTENSION_AVAILABLE: {RUST_EXTENSION_AVAILABLE}")
print(f"RUST_AVAILABLE (dispatch enabled): {RUST_AVAILABLE}")
print()

test_results = []

# Test 1: hz_to_mel with 1D array
print("Test 1: hz_to_mel(1D array)")
try:
    input_arr = np.array([110, 220, 440])
    result = librosa.hz_to_mel(input_arr)
    expected = np.array([1.65, 3.3, 6.6])
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} != {expected.shape}"
    assert np.allclose(result, expected, rtol=0.01), f"Values mismatch: {result} != {expected}"
    print(f"  ✓ PASSED: {result}")
    test_results.append(("hz_to_mel(1D)", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    test_results.append(("hz_to_mel(1D)", False))

# Test 2: hz_to_mel with scalar
print("\nTest 2: hz_to_mel(scalar)")
try:
    result = librosa.hz_to_mel(440.0)
    assert isinstance(result, (np.floating, float)), f"Expected scalar, got {type(result)}"
    print(f"  ✓ PASSED: {result}")
    test_results.append(("hz_to_mel(scalar)", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    test_results.append(("hz_to_mel(scalar)", False))

# Test 3: mel_to_hz with 1D array
print("\nTest 3: mel_to_hz(1D array)")
try:
    input_arr = np.array([1.65, 3.3, 6.6])
    result = librosa.mel_to_hz(input_arr)
    expected = np.array([110, 220, 440])
    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} != {expected.shape}"
    assert np.allclose(result, expected, rtol=0.01), f"Values mismatch: {result} != {expected}"
    print(f"  ✓ PASSED: {result}")
    test_results.append(("mel_to_hz(1D)", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    test_results.append(("mel_to_hz(1D)", False))

# Test 4: mel_to_hz with scalar
print("\nTest 4: mel_to_hz(scalar)")
try:
    result = librosa.mel_to_hz(3.3)
    assert isinstance(result, (np.floating, float)), f"Expected scalar, got {type(result)}"
    print(f"  [PASS] mel_to_hz(scalar): {result}")
    test_results.append(("mel_to_hz(scalar)", True))
except Exception as e:
    print(f"  [FAIL] mel_to_hz(scalar): {e}")
    test_results.append(("mel_to_hz(scalar)", False))

# Test 5: mel_to_hz with 2D array (multi-channel safety check)
print("\nTest 5: mel_to_hz(2D array) - should use NumPy fallback")
try:
    mels_2d = np.random.randn(2, 128)
    result = librosa.mel_to_hz(mels_2d)
    assert result.shape == mels_2d.shape, f"Shape mismatch: {result.shape} != {mels_2d.shape}"
    print(f"  ✓ PASSED: shape {result.shape}")
    test_results.append(("mel_to_hz(2D)", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    test_results.append(("mel_to_hz(2D)", False))

# Test 6: hz_to_mel with 2D array (multichannel safety check)
print("\nTest 6: hz_to_mel(2D array) - should use NumPy fallback")
try:
    hz_2d = np.random.randn(2, 100) * 500 + 100  # frequencies
    result = librosa.hz_to_mel(hz_2d)
    assert result.shape == hz_2d.shape, f"Shape mismatch: {result.shape} != {hz_2d.shape}"
    print(f"  ✓ PASSED: shape {result.shape}")
    test_results.append(("hz_to_mel(2D)", True))
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    test_results.append(("hz_to_mel(2D)", False))

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
passed = sum(1 for _, result in test_results if result)
total = len(test_results)
print(f"Passed: {passed}/{total}")
for test_name, result in test_results:
    status = "[PASS]" if result else "[FAIL]"
    print(f"  {status} {test_name}")

print("=" * 70)
sys.exit(0 if passed == total else 1)

