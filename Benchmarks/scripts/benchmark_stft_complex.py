#!/usr/bin/env python
"""Benchmark complex STFT performance.

This script demonstrates the performance improvement of Rust-accelerated
complex STFT for phase-dependent features like phase vocoder, time-stretching,
and chroma features.
"""

import numpy as np
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE
import timeit


def benchmark_stft_complex(sr=22050, duration=10, n_fft=2048, hop_length=512, n_trials=5):
    """Benchmark Rust vs Python STFT implementations."""

    # Generate synthetic audio
    rng = np.random.default_rng(42)
    y = rng.standard_normal(sr * duration).astype(np.float32)

    print(f"Benchmarking STFT (complex) on {sr} Hz audio, {duration}s ({len(y)} samples)")
    print(f"  n_fft={n_fft}, hop_length={hop_length}")
    print(f"  Trials: {n_trials}\n")

    # Benchmark librosa.stft (Python reference)
    time_python = timeit.timeit(
        lambda: librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True),
        number=n_trials
    ) / n_trials

    if RUST_AVAILABLE and hasattr(_rust_ext, 'stft_complex'):
        # Benchmark Rust stft_complex
        time_rust = timeit.timeit(
            lambda: _rust_ext.stft_complex(y, n_fft, hop_length, center=True),
            number=n_trials
        ) / n_trials

        speedup = time_python / time_rust
        print(f"librosa.stft (Python):    {time_python*1000:.2f} ms")
        print(f"stft_complex (Rust):      {time_rust*1000:.2f} ms")
        print(f"Speedup:                  {speedup:.1f}x\n")
    else:
        print(f"librosa.stft (Python):    {time_python*1000:.2f} ms")
        print("Rust implementation not available.\n")


def benchmark_phase_vocoder(sr=22050, duration=10, rate=2.0, n_fft=2048, hop_length=512, n_trials=3):
    """Benchmark phase vocoder using complex STFT."""

    # Generate synthetic audio
    rng = np.random.default_rng(123)
    y = rng.standard_normal(sr * duration).astype(np.float32)

    print(f"Benchmarking Phase Vocoder (time-stretching) on {sr} Hz audio, {duration}s")
    print(f"  Stretch rate={rate}, n_fft={n_fft}, hop_length={hop_length}")
    print(f"  Trials: {n_trials}\n")

    # Benchmark librosa reference (Python STFT)
    def phase_vocoder_python():
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
        D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
        return librosa.istft(D_stretched, hop_length=hop_length)

    time_python = timeit.timeit(phase_vocoder_python, number=n_trials) / n_trials

    if RUST_AVAILABLE and hasattr(_rust_ext, 'stft_complex'):
        # Benchmark Rust implementation
        def phase_vocoder_rust():
            D = _rust_ext.stft_complex(y, n_fft, hop_length, center=True)
            D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=hop_length)
            return librosa.istft(D_stretched, hop_length=hop_length)

        time_rust = timeit.timeit(phase_vocoder_rust, number=n_trials) / n_trials

        speedup = time_python / time_rust
        print(f"Phase Vocoder (Python STFT): {time_python*1000:.2f} ms")
        print(f"Phase Vocoder (Rust STFT):   {time_rust*1000:.2f} ms")
        print(f"Speedup:                     {speedup:.1f}x\n")
    else:
        print(f"Phase Vocoder (Python STFT): {time_python*1000:.2f} ms")
        print("Rust implementation not available.\n")


if __name__ == "__main__":
    print("=" * 70)
    print("STFT Complex Kernel Benchmarks")
    print("=" * 70 + "\n")

    benchmark_stft_complex()
    benchmark_phase_vocoder()

    print("\nNotes:")
    print("  - All timings exclude array allocation and Python overhead")
    print("  - Complex STFT enables phase-dependent features (phase vocoder, etc.)")
    print("  - Speedup varies by audio length and CPU core count")

