# Phase 12 CQT/VQT Acceleration Plan

Date: 2026-04-04

## Scope

This document closes the CQT/VQT checklist requirement for:
- targeted acceleration plan, and
- parity/performance harness.

## Baseline Harness

- `benchmark_phase12_cqt_vqt.py`
  - runs `librosa.cqt(...)` and `librosa.vqt(...)` at 10s and 30s durations
  - reports output shapes and avg/min runtime

## Hotspot Candidates (CPU)

1. Recursive filterbank convolution loops in `librosa/core/constantq.py`.
2. Sparse basis projection/multiplication paths for repeated frames.
3. Repeated basis construction in repeated calls with same parameter sets.

## Proposed Mitigation Order

1. Add memoized basis reuse metrics and benchmark variants (low risk).
2. Add focused profile script for `cqt` and `vqt` internal stages.
3. Evaluate Rust candidate for dense/sparse projection micro-kernel only after profiling confirms payoff.

## Progress Update (2026-04-04)

- First low-risk pass completed in `librosa/core/constantq.py`:
  - `__cqt_response` now uses a batched dense projection path (`np.tensordot`) for ndarray bases.
  - Sparse/custom basis objects keep the prior per-channel fallback path.
- Benchmark harness improved in `benchmark_phase12_cqt_vqt.py`:
  - added warm-up calls
  - added mono + stereo workload coverage for 10s/30s durations
- Validation:
  - `python -m pytest tests/test_constantq.py -q` -> 631 passed, 10 xfailed
  - benchmark output captured in `benchmarks/results/tmp_phase12_bench_cqt_vqt_after_cqt_opt.txt`

## Stage Profiling Snapshot (2026-04-04)

- Added internal stage profiler to `benchmark_phase12_cqt_vqt.py` (runtime monkeypatch wrappers, no library behavior changes).
- Profile output captured in `benchmarks/results/tmp_phase12_bench_cqt_vqt_profiled.txt`.
- Reference workload (`mono-30s`) hotspots:
  - `__vqt_filter_fft` + `filters.wavelet`: ~58-60% combined
  - `__cqt_response` + `stft`: ~25-28% combined
  - `audio.resample`: ~13%
- Next target: reduce repeated filter-basis build overhead (basis reuse/caching metrics pass) before deeper kernel rewrites.

## Basis Reuse Cache Pass (2026-04-04)

- Implemented in `librosa/core/constantq.py`:
  - added in-memory VQT FFT basis cache with deterministic keys
  - added cache counters and helper APIs (`_vqt_filter_fft_cache_info`, `_vqt_filter_fft_cache_clear`)
  - routed `pseudo_cqt`, `icqt`, and `vqt` through cached wrapper
- Benchmark instrumentation:
  - `benchmark_phase12_cqt_vqt.py` now reports VQT cache hit/miss stats in the stage profile section
- Validation:
  - `python -m pytest tests/test_constantq.py -q` -> 631 passed, 10 xfailed
  - benchmark output: `benchmarks/results/tmp_phase12_bench_cqt_vqt_after_cache.txt`
- Sample gains (min runtime, after cache pass):
  - `cqt mono-30s`: `33.625 ms` -> `16.078 ms`
  - `vqt mono-30s`: `37.145 ms` -> `14.590 ms`
  - `cqt stereo-30s`: `49.806 ms` -> `33.065 ms`
  - `vqt stereo-30s`: `50.700 ms` -> `29.730 ms`

## Resample Guard Pass (2026-04-04)

- Implemented in `librosa/core/constantq.py` (`vqt` loop):
  - downsample now occurs only when another octave remains (`i < n_octaves - 1`)
  - avoids one unnecessary final `audio.resample` call per transform
- Regression test added:
  - `tests/test_constantq.py::test_vqt_skips_final_unneeded_downsample`
- Validation:
  - `python -m pytest tests/test_constantq.py -q` -> 632 passed, 10 xfailed
  - benchmark output: `benchmarks/results/tmp_phase12_bench_cqt_vqt_after_resample_guard.txt`
- Sample gains (min runtime, after resample guard):
  - `cqt mono-30s`: `16.078 ms` -> `15.703 ms`
  - `vqt mono-30s`: `14.590 ms` -> `14.377 ms`
  - `cqt stereo-30s`: `33.065 ms` -> `32.592 ms`
  - `vqt stereo-30s`: `29.730 ms` -> `28.060 ms`

## Exit Criteria Mapping

- [x] Plan documented.
- [x] Benchmark harness added.
- [ ] Rust acceleration implementation (future, profile-gated).

