# Phase 12 CPU Remaining Plan

Date: 2026-04-04

## Target Items

1. Mel cross-CPU threshold strategy
2. phase_vocoder benchmark/parity harness
3. cqt/vqt acceleration plan + baseline harness
4. tonnetz fixture-safe parity policy
5. beat_track hotspot benchmark baseline

## Status Update (2026-04-04)

- `beat_track` hotspot item: completed
  - Mitigation: beat-specific tempo autocorrelation window in `librosa/beat.py` (`_BEAT_TRACK_TEMPO_AC_SIZE = 4.0`)
  - Validation: `tests/test_beat.py` passes; `benchmark_phase12_beat_track.py` shows improved end-to-end timing on noisy 30s/120s workloads

- `phase_vocoder` kernel parity fix & promotion: **COMPLETE**
  - **Root cause identified:** Rounding semantics mismatch (Rust `round()` vs NumPy `round()` ties-to-even)
  - **Fix applied:** Changed both f32 and f64 paths to use `round_ties_even()` in `src/phase_vocoder.rs`
  - **Tests updated:** Removed `xfail` marker; dispatch tests updated for default behavior
  - **Promotion:** Rust dispatch now enabled by default with `prefer_rust=True` parameter for backward compat
  - **Verification:** All parity & dispatch tests passing; see `PHASE_VOCODER_PROMOTION_COMPLETE.md`
  - **Documentation:** `PHASE_VOCODER_FIX.md`, `PHASE_VOCODER_PARITY_CHECKLIST.md`, promotion guide included
  - **Status:** Ready for production release

- `Phase 13 CQT/VQT kickoff`: **STARTED**
  - Added spike tracker: `Development_docs/PHASE13_CQT_VQT_SPIKE.md`
  - Added parity scaffolding tests: `tests/test_phase13_cqt_vqt_parity.py`
  - Upgraded benchmark harness: `Benchmarks/scripts/benchmark_phase12_cqt_vqt.py` (`--durations`, `--repeats`, `--json-out`)
  - Captured smoke baseline artifact: `Benchmarks/results/phase13_cqt_vqt_baseline_smoke.json`

## Deliverables

- Cross-CPU mel resolver + tests + calibration profile support
- `benchmarks/scripts/benchmark_phase12_phase_vocoder.py`
- `benchmarks/scripts/benchmark_phase12_cqt_vqt.py`
- `benchmarks/scripts/benchmark_phase12_beat_track.py`
- `PHASE12_CQT_VQT_ACCELERATION_PLAN.md`
- `PHASE12_TONNETZ_DECISION.md`

## Validation Commands

- `python -m pytest tests/test_mel_threshold_policy.py -q`
- `python -m pytest tests/test_features.py -q -k "phase_vocoder or tonnetz or melspectrogram"`
- `python -m pytest tests/test_constantq.py -q`
- `python -m pytest tests/test_beat.py -q`
- `python benchmarks/scripts/benchmark_phase12_phase_vocoder.py`
- `python benchmarks/scripts/benchmark_phase12_cqt_vqt.py`
- `python benchmarks/scripts/benchmark_phase12_beat_track.py`
