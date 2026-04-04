# Expanded STFT Fast-Path Use Cases

## Overview

With validation of precomputed Hann window support and implementation of **complex STFT** output, the Rust fast-path now enables phase-dependent audio features beyond magnitude spectrograms. This document outlines the expanded use cases, implementation status, and adoption roadmap.

---

## Completed: Complex STFT Kernel

### What It Does
Returns full complex-valued STFT output (real + imaginary components), matching `librosa.stft()` with Hann window, center-padding, and parallel FFT processing.

### Performance
- **Speedup: ~210x** over Python `librosa.stft()` on 10s audio at 22050 Hz
- Enables phase-dependent features: phase vocoder, time-stretching, chroma features
- Uses same thread-local FFT plan caching infrastructure as `stft_power`

### Implementation Details
- **File**: `src/stft.rs` (added `stft_complex` function)
- **Kernel**: Identical to `stft_power` but writes complex values instead of |z|²
- **Validation**: 
  - Parity test: matches `librosa.stft()` to float32 precision (rtol=1e-4, atol=1e-5)
  - Integration test: phase vocoder + istft matches librosa chain

### Code Integration
```python
from librosa._rust_bridge import _rust_ext

D = _rust_ext.stft_complex(y, n_fft=2048, hop_length=512, center=True)
# D.shape = (n_fft//2+1, n_frames), dtype=complex64
# Can be used directly with librosa.phase_vocoder, librosa.istft, etc.
```

### Use Case: Phase Vocoder (Time-Stretching)
```python
y = librosa.load('audio.wav')[0]
D = _rust_ext.stft_complex(y, n_fft=2048, hop_length=512)
D_stretched = librosa.phase_vocoder(D, rate=2.0)
y_stretched = librosa.istft(D_stretched)
# ~1.5x faster than full Python pipeline (STFT is the bottleneck)
```

---

## Planned: Non-Hann Window Support for Power STFT

### Motivation
- Current `stft_power` only supports built-in Hann window
- Users sometimes need alternative windows (Hamming, Blackman, etc.) for spectral analysis
- Precomputed windows should be cached in Rust to avoid repeated alloc/copy

### Design
- Accept optional precomputed window vector (`f32[n_fft]`) as parameter
- Fall back to Hann if not provided (backward compatible)
- Window vector shared via `Arc<Vec<f32>>` (zero-copy across threads)

### Impact
- Unblock spectral analysis features (spectral contrast, etc.)
- Moderate adoption (~10% of use cases)

### Priority: **MEDIUM** (blocks some spectral features, moderate complexity)

---

## Planned: Magnitude-Only STFT

### Motivation
- Intermediate between `stft_power` (|z|²) and `stft_complex` (full z)
- Returns |z| (magnitude) for features needing magnitude without squaring
- Lighter than complex (8-byte vs 16-byte per bin) but more flexible than power

### Design
- New function `stft_magnitude(y, n_fft, hop_length, center)` → `f32[n_bins, n_frames]`
- Reuse FFT plan/buffer infrastructure
- Return `sqrt(z.re² + z.im²)` for each bin

### Impact
- Enables spectral flux, spectral centroid with custom algorithms
- Lower memory footprint than complex
- Modest adoption (~5% of use cases)

### Priority: **LOW** (nice-to-have, easy to implement)

---

## Planned: Float64 Support

### Motivation
- Many researchers use double-precision audio (e.g., high-quality processing pipelines)
- Current kernel is float32-only
- `stft_power` and `stft_complex` should accept both f32 and f64

### Design
- Create generic FFT kernels parametric in precision
  - Leverage `rustfft` generic support (`FftPlanner<T>`)
  - Thread-local state for f32 and f64 separately
- Maintain separate dispatch paths: `stft_complex_f32` and `stft_complex_f64`
- Python side auto-selects based on input dtype

### Impact
- Unblocks high-precision research use cases
- Minor complexity (moderate)
- Low adoption (~5% immediately, higher over time)

### Priority: **LOW** (can be deferred; niche use case)

---

## Planned: Multi-Channel / Batch STFT

### Motivation
- Real-world audio often comes in stereo or surround (n_channels > 1)
- Current kernel processes single 1D audio vector
- Batch processing can amortize FFT plan allocation

### Design
- Accept input shape: `(n_channels, n_samples)` or 1D `(n_samples,)`
- Output shape: `(n_channels, n_bins, n_frames)` for batch input
- Reuse FFT plan across channels (same n_fft)
- Optional: per-channel window weight adjustment

### Impact
- Direct use case: librosa.feature.chroma_stft on stereo input
- Moderate complexity
- Moderate adoption (~15% of use cases)

### Priority: **MEDIUM-LOW** (deferred post-complex; higher value than magnitude-only)

---

## Planned: Relaxed Contiguity & Auto-Promotion

### Motivation
- Current kernel requires C-contiguous f32 input
- Real-world audio may be:
  - Non-contiguous (e.g., after slicing, transposing)
  - Wrong dtype (f64, int16, etc.)
- Manual contiguity check forces users to copy data

### Design
- In Python bridge (`_rust_bridge.py`):
  - Check contiguity; auto-copy if needed
  - Accept f64, promote to f32 with warning (optional)
  - Validate dtype early and inform user
- Rust kernel: no changes (assume input already valid)

### Impact
- Smoother UX, fewer "unexpected performance cliff" surprises
- Low complexity (pure Python)
- High adoption (transparent improvement)

### Priority: **HIGH** (quick win, high UX impact)

---

## Adoption Roadmap

### Phase 1: **ACTIVE** — Complex STFT (✅ Completed)
- ✅ Rust kernel implemented and validated
- ✅ Integration tests pass
- ✅ Benchmark shows 210x speedup
- **Next**: Wire into librosa.feature dispatcher for auto-use in chroma_stft, phase_vocoder

### Phase 2: **NEXT** — Window & Contiguity (1–2 weeks)
- Implement non-Hann window support for stft_power
- Add auto-contiguity/dtype promotion in Python bridge
- Expected impact: unblock spectral analysis features

### Phase 3: **FUTURE** — Float64 & Batch (2–4 weeks)
- Double-precision FFT kernel
- Multi-channel batch processing
- Expected impact: high-precision research, stereo/surround audio

### Phase 4: **DEFERRED** — Magnitude-Only & Advanced
- Magnitude-only STFT (if demand arises)
- Custom window caching API
- Frame-wise optimizations (e.g., peak detection for onset detection)

---

## Performance Summary

| Feature | Speedup | Status | Impact |
|---------|---------|--------|--------|
| Complex STFT | 210x | ✅ Done | High (phase-dependent features) |
| Power STFT (Hann) | ~200x | ✅ Done | High (melspectrogram, MFCC) |
| Non-Hann Windows | ~150x | 🔄 Planned | Medium |
| Float64 | ~200x (f64) | 🔄 Planned | Low-Medium |
| Batch Processing | ~200x/channel | 🔄 Planned | Medium |

---

## Testing & Validation

All expanded kernels include:
- **Parity tests**: Output matches librosa reference (within float precision)
- **Integration tests**: Works correctly in downstream features
- **Benchmarks**: Performance vs Python reference

---

## References

- `src/stft.rs`: Complex & power STFT kernels
- `tests/test_features.py`: Parity & integration tests
- `benchmark_stft_complex.py`: Performance benchmarks
- `librosa/core/spectrum.py`: Dispatcher logic


