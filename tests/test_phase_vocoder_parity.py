#!/usr/bin/env python
"""
Standalone test to verify Rust phase-vocoder parity after ties-to-even rounding fix.
Run: python test_phase_vocoder_parity.py
"""
import numpy as np
import sys
import os

# Ensure we load librosa from the repo
sys.path.insert(0, os.path.dirname(__file__))

import librosa
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext

def phase_vocoder_reference_loop(D, rate, hop_length, n_fft):
    """Reference Python implementation from librosa."""
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(D[..., 0])

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D_padded = np.pad(D, padding, mode="constant")

    step_int = np.floor(time_steps).astype(int)
    step_alpha = time_steps - step_int

    D_phase = np.angle(D_padded)
    D_mag = np.abs(D_padded)

    for t, idx in enumerate(step_int):
        alpha = step_alpha[t]
        mag = (1.0 - alpha) * D_mag[..., idx] + alpha * D_mag[..., idx + 1]
        d_stretch[..., t] = librosa.util.phasor(phase_acc, mag=mag)

        dphase = D_phase[..., idx + 1] - D_phase[..., idx] - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase

    return d_stretch


def test_parity_f32():
    """Test complex64 (f32) parity."""
    if not (RUST_AVAILABLE and hasattr(_rust_ext, "phase_vocoder_f32")):
        print("⚠ Rust phase_vocoder_f32 not available, skipping")
        return False

    print("\n=== Testing complex64 (f32) ===")

    rng = np.random.default_rng(2041)
    D = (
        rng.standard_normal((257, 24)).astype(np.float64)
        + 1j * rng.standard_normal((257, 24)).astype(np.float64)
    ).astype(np.complex64)

    rate = 1.25
    hop_length = 128
    n_fft = 512

    # Reference
    ref = phase_vocoder_reference_loop(D, rate=rate, hop_length=hop_length, n_fft=n_fft)

    # Prepare inputs for Rust kernel
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    step_int = np.floor(time_steps).astype(np.int64)
    step_alpha = (time_steps - step_int).astype(np.float64)

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D_padded = np.pad(D, padding, mode="constant")

    D_phase_t = np.ascontiguousarray(np.angle(D_padded).astype(np.float32).T)
    D_mag_t = np.ascontiguousarray(np.abs(D_padded).astype(np.float32).T)
    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(D[..., 0]).astype(np.float64)

    # Call Rust
    rust_out = _rust_ext.phase_vocoder_f32(
        D_phase_t,
        D_mag_t,
        phi_advance.astype(np.float64),
        step_int,
        step_alpha,
        phase_acc,
    )

    # Compare
    max_diff = np.max(np.abs(rust_out - ref))
    mean_diff = np.mean(np.abs(rust_out - ref))

    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Stricter f32 tolerance
    if max_diff < 1e-5:
        print("✓ PASS: f32 parity within tolerance (1e-5)")
        return True
    else:
        print(f"✗ FAIL: f32 max diff {max_diff:.2e} >= 1e-5")
        # Find first divergence
        for b in range(ref.shape[0]):
            for t in range(ref.shape[1]):
                if np.abs(rust_out[b, t] - ref[b, t]) > 1e-5:
                    print(f"  First divergence at (bin={b}, frame={t})")
                    print(f"    Python: {ref[b, t]}")
                    print(f"    Rust:   {rust_out[b, t]}")
                    return False
    return False


def test_parity_f64():
    """Test complex128 (f64) parity."""
    if not (RUST_AVAILABLE and hasattr(_rust_ext, "phase_vocoder_f64")):
        print("⚠ Rust phase_vocoder_f64 not available, skipping")
        return False

    print("\n=== Testing complex128 (f64) ===")

    rng = np.random.default_rng(2042)
    D = (
        rng.standard_normal((257, 24)).astype(np.float64)
        + 1j * rng.standard_normal((257, 24)).astype(np.float64)
    ).astype(np.complex128)

    rate = 1.25
    hop_length = 128
    n_fft = 512

    # Reference
    ref = phase_vocoder_reference_loop(D, rate=rate, hop_length=hop_length, n_fft=n_fft)

    # Prepare inputs for Rust kernel
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    step_int = np.floor(time_steps).astype(np.int64)
    step_alpha = (time_steps - step_int).astype(np.float64)

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D_padded = np.pad(D, padding, mode="constant")

    D_phase_t = np.ascontiguousarray(np.angle(D_padded).astype(np.float64).T)
    D_mag_t = np.ascontiguousarray(np.abs(D_padded).astype(np.float64).T)
    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(D[..., 0]).astype(np.float64)

    # Call Rust
    rust_out = _rust_ext.phase_vocoder_f64(
        D_phase_t,
        D_mag_t,
        phi_advance.astype(np.float64),
        step_int,
        step_alpha,
        phase_acc,
    )

    # Compare
    max_diff = np.max(np.abs(rust_out - ref))
    mean_diff = np.mean(np.abs(rust_out - ref))

    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Very strict f64 tolerance
    if max_diff < 1e-11:
        print("✓ PASS: f64 parity within tolerance (1e-11)")
        return True
    else:
        print(f"✗ FAIL: f64 max diff {max_diff:.2e} >= 1e-11")
        # Find first divergence
        for b in range(ref.shape[0]):
            for t in range(ref.shape[1]):
                if np.abs(rust_out[b, t] - ref[b, t]) > 1e-11:
                    print(f"  First divergence at (bin={b}, frame={t})")
                    print(f"    Python: {ref[b, t]}")
                    print(f"    Rust:   {rust_out[b, t]}")
                    return False
    return False


if __name__ == "__main__":
    if not RUST_AVAILABLE:
        print("Rust extension not available!")
        sys.exit(1)

    print("Testing Rust phase-vocoder parity after ties-to-even rounding fix...")

    f32_pass = test_parity_f32()
    f64_pass = test_parity_f64()

    print("\n" + "="*50)
    if f32_pass and f64_pass:
        print("✓ All parity tests PASSED!")
        sys.exit(0)
    else:
        print("✗ Some parity tests FAILED")
        sys.exit(1)

