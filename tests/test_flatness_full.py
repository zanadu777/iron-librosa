#!/usr/bin/env python
"""Quick test of spectral_flatness implementation."""

import sys
import os

# Try to build and install first
print("Building Rust extension...")
import subprocess
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "."],
    cwd="D:\\Dev\\Programming 2026\\Rust\\iron-librosa",
    capture_output=True,
    timeout=120
)

if result.returncode != 0:
    print("Build STDOUT:")
    print(result.stdout.decode('utf-8', errors='replace')[-2000:])
    print("Build STDERR:")
    print(result.stderr.decode('utf-8', errors='replace')[-2000:])
    sys.exit(1)

print("Build succeeded. Testing kernels...")

# Now test
import numpy as np
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE

if not RUST_AVAILABLE:
    print("ERROR: Rust extension not available")
    sys.exit(1)

# Check symbols
if not hasattr(_rust_ext, 'spectral_flatness_f32'):
    print("ERROR: spectral_flatness_f32 not found")
    sys.exit(1)

if not hasattr(_rust_ext, 'spectral_flatness_f64'):
    print("ERROR: spectral_flatness_f64 not found")
    sys.exit(1)

print("✓ Both kernel symbols are present")

# Test f32 kernel
rng = np.random.default_rng(6001)
S = np.abs(rng.standard_normal((513, 20))).astype(np.float32)

try:
    out = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)
    print(f"✓ spectral_flatness_f32 works: output shape {out.shape}, dtype {out.dtype}")
    assert out.shape == (1, 20), f"wrong shape: {out.shape}"
    assert out.dtype == np.float32, f"wrong dtype: {out.dtype}"
    assert np.all(out > 0) and np.all(out <= 1.0), f"values out of range: [{out.min()}, {out.max()}]"
except Exception as e:
    print(f"✗ spectral_flatness_f32 failed: {e}")
    sys.exit(1)

# Test f64 kernel
S64 = S.astype(np.float64)
try:
    out64 = _rust_ext.spectral_flatness_f64(S64, 1e-10, 2.0)
    print(f"✓ spectral_flatness_f64 works: output shape {out64.shape}, dtype {out64.dtype}")
    assert out64.shape == (1, 20), f"wrong shape: {out64.shape}"
    assert out64.dtype == np.float64, f"wrong dtype: {out64.dtype}"
except Exception as e:
    print(f"✗ spectral_flatness_f64 failed: {e}")
    sys.exit(1)

# Test API dispatch
import librosa
try:
    result = librosa.feature.spectral_flatness(S=S)
    print(f"✓ API dispatch works: librosa.feature.spectral_flatness(S=S) -> shape {result.shape}")
except Exception as e:
    print(f"✗ API dispatch failed: {e}")
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")

