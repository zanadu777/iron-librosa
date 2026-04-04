# Phase 5 Kickoff: Spectral Rolloff + Bandwidth

## Status: ✅ COMPLETE (Current Safe Scope)

This kickoff implements the first Phase 5 priority from `FEATURES_REMAINING.md`:
- spectral rolloff acceleration
- spectral bandwidth acceleration
- chroma filter-bank acceleration (`filters.chroma`)

## Implemented

### Rust kernels (`src/spectrum_utils.rs`)
- `spectral_rolloff_f32(s, freq, roll_percent) -> (1, t)`
- `spectral_rolloff_f64(s, freq, roll_percent) -> (1, t)`
- `spectral_bandwidth_f32(s, freq, centroid, norm, p) -> (1, t)`
- `spectral_bandwidth_f64(s, freq, centroid, norm, p) -> (1, t)`
- `spectral_bandwidth_auto_centroid_f32(s, freq, norm, p) -> (1, t)`
- `spectral_bandwidth_auto_centroid_f64(s, freq, norm, p) -> (1, t)`

### Rust kernels (`src/chroma.rs`)
- `chroma_filter_f32(sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c)`
- `chroma_filter_f64(sr, n_fft, n_chroma, tuning, ctroct, octwidth, base_c)`
- `chroma_project_f32(...)`
- `chroma_project_f64(...)`

### Python dispatch (`librosa/feature/spectral.py`)
- Added conservative fast paths for:
  - `spectral_rolloff(...)`
  - `spectral_bandwidth(...)`
- Guards require:
  - real-valued `S`
  - `S.dtype in {float32, float64}`
  - static 1D `freq` (`float64`)
- `spectral_bandwidth(..., centroid=None)` now uses fused auto-centroid Rust kernel when guards match.
- All unsupported/complex cases still fall back to existing Python behavior.

### Python dispatch (`librosa/filters.py`)
- Added conservative Rust fast path for `filters.chroma(...)`
- Guards currently require:
  - `dtype in {float32, float64}`
  - `norm == 2`
  - positive `sr`, valid `n_fft`, positive `n_chroma`
- Non-L2 norm and other unsupported cases fall back to the original Python implementation.

### Exports (`src/lib.rs`)
- Registered rolloff / bandwidth kernels and new chroma filter-bank kernels in the `_rust` module.

### Tests (`tests/test_phase4_features.py`)
Added 8 tests:
- raw rolloff kernel parity
- raw bandwidth kernel parity
- raw fused auto-centroid bandwidth parity
- rolloff public dispatch (multichannel)
- bandwidth public dispatch (multichannel)
- bandwidth auto-centroid dispatch selection
- rolloff fallback on variable `freq`
- bandwidth fallback on variable `freq`

### Tests (`tests/test_phase5_chroma_filters.py`)
Added 5 tests:
- raw `chroma_filter_f32` parity
- raw `chroma_filter_f64` parity
- `filters.chroma` dispatch selection
- `filters.chroma` fallback for unsupported norm
- `filters.chroma` public output parity

## Validation

### Test runs
- `python -m pytest tests/test_phase4_features.py -q`
  - **17 passed**
- `python -m pytest tests/test_features.py -k "spectral_rolloff or spectral_bandwidth" -q`
  - **29 passed, 6 xfailed**
- `python -m pytest tests/test_phase4c_chroma.py -q`
  - **8 passed**
- `python -m pytest tests/test_phase5_chroma_filters.py -q`
  - **5 passed**
- `python -m pytest tests/test_phase4c_chroma.py tests/test_phase5_chroma_filters.py -q`
  - **13 passed**

### Benchmark
- New harness: `benchmark_phase5_spectral.py`
- Raw kernel results:
  - rolloff f32: **~4.5x to ~62x** vs NumPy reference (size-dependent)
  - bandwidth f32 (manual centroid): **~13x to ~176x** vs NumPy reference (size-dependent)
  - bandwidth f32 (fused auto-centroid): **~14x to ~174x** vs NumPy reference (size-dependent)

- Public API A/B (Rust enabled vs forced Python fallback in the same runtime):
  - `spectral_rolloff`: **~5.0x**, **~13.7x**, **~17.9x**
    - (`n_fft=1024`, `2048`, `4096`)
  - `spectral_bandwidth` (auto centroid): **~13.3x**, **~45.7x**, **~55.2x**
  - `spectral_bandwidth` (provided centroid): **~9.7x**, **~25.1x**, **~32.9x**

- `filters.chroma` direct A/B (Rust enabled vs forced Python fallback):
  - `n_fft=2048, n_chroma=12`: **~2.67x**
  - `n_fft=4096, n_chroma=12`: **~3.86x**
  - `n_fft=4096, n_chroma=24`: **~5.74x**

- `chroma_stft(S=..., tuning=0.0)` end-to-end A/B with tuning estimation bypassed:
  - `n_fft=2048, n_chroma=12`: **~1.73x**
  - `n_fft=4096, n_chroma=12`: **~2.23x**
  - `n_fft=4096, n_chroma=24`: **~4.05x**

## Notes

- Direct `librosa` vs `iron_librosa` comparisons can look near-parity in this workspace because both imports may hit the same patched dispatch path.
- Forced-fallback A/B numbers above are the reliable end-to-end speed indicator for this phase.
- Fused auto-centroid narrows bandwidth overhead and keeps behavior-safe fallbacks for unsupported inputs.
- `filters.chroma` is now accelerated for the common default L2-normalized path, which materially improves fixed-tuning `chroma_stft` workloads.
- This kickoff keeps behavior-safe constraints; broader support (e.g. variable frequency grids) remains for later Phase 5 steps.

## Follow-up After Phase 5
1. Decide whether `estimate_tuning` should remain experimental / threshold-gated or receive deeper upstream optimization.
2. Optionally expand support for broader edge cases (e.g. variable frequency grids, non-default norms).
3. Advance to the remaining post-Phase-5 backlog.

