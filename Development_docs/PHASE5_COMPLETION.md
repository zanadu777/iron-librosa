# Phase 5: Spectral Feature Completion & Chroma Bottleneck Reduction

## Status: ✅ COMPLETE (Current Safe Scope)

Phase 5 completed the next tranche of high-impact feature acceleration beyond Phase 4:

- `spectral_rolloff()`
- `spectral_bandwidth()`
- `filters.chroma()`
- `estimate_tuning()` post-processing helper (implemented, but kept experimental / threshold-gated)

This phase focused on **common safe paths first**, then hardened the behavior with regression coverage and benchmark validation.

---

## What Was Implemented

### 1. Spectral rolloff acceleration

Files:
- `src/spectrum_utils.rs`
- `librosa/feature/spectral.py`

Added:
- `spectral_rolloff_f32(...) -> (1, t)`
- `spectral_rolloff_f64(...) -> (1, t)`

Behavior:
- static 1D frequency-bin fast path
- conservative dispatch for real-valued `float32` / `float64` spectrograms
- unsupported cases fall back to Python

### 2. Spectral bandwidth acceleration

Files:
- `src/spectrum_utils.rs`
- `librosa/feature/spectral.py`

Added:
- `spectral_bandwidth_f32(...) -> (1, t)`
- `spectral_bandwidth_f64(...) -> (1, t)`
- `spectral_bandwidth_auto_centroid_f32(...) -> (1, t)`
- `spectral_bandwidth_auto_centroid_f64(...) -> (1, t)`

Behavior:
- static 1D frequency-bin fast path
- fused auto-centroid path when `centroid=None`
- p-specialized math for common `p=1` / `p=2`
- unsupported cases fall back to Python

### 3. Chroma filter-bank acceleration

Files:
- `src/chroma.rs`
- `librosa/filters.py`
- `src/lib.rs`

Added:
- `chroma_filter_f32(...) -> (n_chroma, 1 + n_fft//2)`
- `chroma_filter_f64(...) -> (n_chroma, 1 + n_fft//2)`

Behavior:
- conservative fast path in `filters.chroma(...)`
- currently enabled for common `norm == 2` and `dtype in {float32, float64}` cases
- preserves Python fallback for non-L2 and unsupported cases

### 4. Tuning estimation helper

Files:
- `src/tuning.rs`
- `librosa/core/pitch.py`
- `src/lib.rs`

Added:
- `estimate_tuning_from_piptrack_f32(...) -> float`
- `estimate_tuning_from_piptrack_f64(...) -> float`

Behavior:
- accelerates the **post-piptrack** tuning stage (median threshold + histogram vote)
- currently protected by:
  - `IRON_LIBROSA_ENABLE_RUST_TUNING`
  - `_RUST_TUNING_MIN_WORK`
- left experimental because full `estimate_tuning()` speedups are modest on smaller workloads

---

## Hardening Pass Summary

### Focused Phase 5/Phase 4 validation
- `tests/test_phase4_features.py`
- `tests/test_phase4c_chroma.py`
- `tests/test_phase5_chroma_filters.py`
- `tests/test_phase5_tuning.py`

Result:
- **35 passed**

### Public API feature regressions
- `tests/test_features.py -k "spectral_rolloff or spectral_bandwidth or chroma_stft"`

Result:
- **29 passed, 6 xfailed**

### Core pitch/CQT regressions
- `tests/test_core.py -k "estimate_tuning or cqt"`

Result:
- **32 passed**

### Chroma filter regressions
- `tests/test_filters.py -k chroma`

Result:
- **1009 passed, 1 failed, 1 skipped, 322 deselected**

Notes:
- the single failing `test_chroma_issue1295[261.63]` reproduces even when forcing Python fallback only
- comparison of Rust and Python `filters.chroma()` outputs for the tested case showed **maxabs 0.0**
- therefore this failure is treated as **pre-existing / unrelated to the new Rust chroma filter path**

---

## Benchmark Summary

### Phase 5 spectral benchmark (`benchmark_phase5_spectral.py`)

#### Raw kernel speedups
- `spectral_rolloff_f32`: **~4.5x to ~62x**
- `spectral_bandwidth_f32` (manual centroid): **~13x to ~176x**
- `spectral_bandwidth_auto_centroid_f32`: **~14x to ~174x**

#### Public API A/B (Rust vs forced Python fallback)
- `spectral_rolloff`: **~5.0x / ~13.7x / ~17.9x**
- `spectral_bandwidth` auto-centroid: **~13.3x / ~45.7x / ~55.2x**
- `spectral_bandwidth` provided centroid: **~9.7x / ~25.1x / ~32.9x**

### Phase 5 chroma benchmark (`benchmark_phase5_chroma.py`)

#### Direct `filters.chroma`
- `n_fft=2048, n_chroma=12`: **~2.67x**
- `n_fft=4096, n_chroma=12`: **~3.86x**
- `n_fft=4096, n_chroma=24`: **~5.74x**

#### End-to-end `chroma_stft(S=..., tuning=0.0)`
- `n_fft=2048, n_chroma=12`: **~1.73x**
- `n_fft=4096, n_chroma=12`: **~2.23x**
- `n_fft=4096, n_chroma=24`: **~4.05x**

### Phase 5 tuning benchmark (`benchmark_phase5_tuning.py`)

#### Full `estimate_tuning()`
- small: **~1.03x**
- medium: **~1.17x**
- large: **~1.19x**

#### Post-piptrack helper only
- small: **~1.13x**
- medium: **~1.89x**
- large: **~3.18x**

Interpretation:
- the Rust tuning helper is useful
- but full `estimate_tuning()` remains dominated by upstream `piptrack`
- keeping the Rust path experimental / threshold-gated is the correct default choice for now

---

## Safety / Default-On Policy

### Default-on in Phase 5
- `spectral_rolloff()` fast path
- `spectral_bandwidth()` fast path
- `filters.chroma()` fast path for common L2-normalized cases

### Experimental / guarded in Phase 5
- `estimate_tuning()` Rust post-processing path
  - requires opt-in environment flag
  - requires sufficiently large workload

This preserves behavior safety while avoiding performance regressions on small inputs.

---

## Files Added / Updated

### New source files
- `src/tuning.rs`

### Modified Rust files
- `src/spectrum_utils.rs`
- `src/chroma.rs`
- `src/lib.rs`

### Modified Python files
- `librosa/feature/spectral.py`
- `librosa/filters.py`
- `librosa/core/pitch.py`

### New tests
- `tests/test_phase5_tuning.py`
- `tests/test_phase5_chroma_filters.py`

### New benchmarks
- `benchmark_phase5_spectral.py`
- `benchmark_phase5_tuning.py`
- `benchmark_phase5_chroma.py`

### New docs
- `PHASE5_PERF_SNAPSHOT.md`
- `PHASE5_COMPLETION.md`

---

## Remaining Work After Phase 5

Phase 5 core scope is complete for the current safe scope.

Remaining backlog is now primarily:
- variable-frequency-grid spectral feature acceleration
- broader `filters.chroma()` norm support beyond current common path
- deeper upstream pitch tracking / `piptrack` acceleration
- time-domain `rms(y=...)`
- lower-priority spectral features (`flatness`, `contrast`)
- advanced transforms / decomposition (`CQT`, phase vocoder, HPSS)

---

## Conclusion

Phase 5 successfully finished the next major feature layer:

- high-value spectral reductions are accelerated
- chroma filter-bank generation is accelerated
- fixed-tuning `chroma_stft` now sees meaningful end-to-end gains
- tuning estimation has a working Rust helper, but remains responsibly guarded

For the current safe scope, **Phase 5 is complete**.

