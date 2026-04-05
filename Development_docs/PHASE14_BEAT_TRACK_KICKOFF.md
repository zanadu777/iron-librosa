# Phase 14 Beat-Track Kickoff

Date: 2026-04-04

## Goal

Start Phase 14 by defining and validating the first Rust acceleration seam for beat tracking and tempo estimation with strict parity gates.

## Inputs from Phase 13

- CQT/VQT seam is complete and parity-validated.
- CQT/VQT remains opt-in pending stronger benchmark evidence.
- We carry forward the same promotion rule: no default Rust dispatch without clear, reproducible gains.

Reference:
- `Development_docs/PHASE13_CQT_VQT_COMPLETION_REPORT.md`
- `Development_docs/PHASE14_BEAT_TRACK_SEAM_CONTRACT.md`

## Scope (Phase 14)

### In scope
1. Baseline performance profiling for beat-track and tempo paths.
2. Candidate seam isolation in `librosa.beat` (`__beat_track_dp` first).
3. Rust prototype with guarded Python fallback.
4. Parity tests for beat frames and tempo outputs.
5. Benchmark comparisons and promotion decision.

### Out of scope
- Full rewrite of all `librosa.segment` algorithms.
- Behavioral changes to beat tracking defaults.

## First Tasks

1. Baseline and profile
   - Targets:
     - `librosa.beat.beat_track`
     - tempo path (`librosa.feature.tempo` as called by beat pipeline)
   - Record mono/stereo timings and top hotspots.

2. Define seam contract
   - Candidate: dynamic programming inner loop (`librosa.beat.__beat_track_dp`).
   - Inputs/outputs and dtype rules documented before Rust wiring.

3. Add test gates
   - New/updated tests in `tests/test_beat.py`:
     - deterministic beat frame parity
     - mono/stereo parity
     - tempo parity thresholds

4. Add benchmark artifact
   - Use or extend benchmark script for beat workloads.
   - Save results under `Benchmarks/results/`.

5. Implement guarded dispatch
   - Rust dispatch must be behind availability and symbol guards.
   - Python fallback remains default until promotion criteria are met.

## Validation Commands (initial)

```powershell
python -m pytest -q -o addopts="" tests/test_beat.py
python -m pytest -q -o addopts="" tests/test_tempo.py
```

Benchmark baseline command:

```powershell
python Benchmarks/scripts/benchmark_phase14_beat_track.py --backend compare --json-out Benchmarks/results/phase14_beat_track_baseline.json
```

## Promotion Criteria

- No parity regressions in beat/tempo tests.
- Measurable speedup on medium workloads (target 1.5-2.0x).
- Stable behavior on mono and stereo inputs.
- Clear fallback behavior when Rust extension is unavailable.

## Current Status

- [x] Kickoff scope documented
- [ ] Baseline profile captured
- [ ] First Rust seam implemented
- [ ] Parity suite green with Rust path
- [ ] Benchmark delta captured
- [ ] Promotion decision documented

