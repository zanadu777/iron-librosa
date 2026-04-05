# Phase 16 Onset/Tempo Promotion Kickoff

Date: 2026-04-05

## Goal

Promote the new upstream beat-track accelerations from Phase 15 toward default-safe deployment by hardening cross-host validation and CI gating.

## Inputs

- `Development_docs/PHASE15_BEAT_UPSTREAM_COMPLETION_REPORT.md`
- `Benchmarks/results/phase15_bench_numpy_baseline.json`
- `Benchmarks/results/phase15_bench_phase15_rust.json`

## Scope

### In scope
1. Cross-host benchmark reproducibility for beat-track workloads.
2. CI-ready schema validation for benchmark artifacts.
3. Auto-review enforcement for speedups below `1.5x`.
4. Promotion decision update for `IRON_LIBROSA_RUST_DISPATCH` default policy.

### Out of scope
- New DSP kernels.
- API behavior changes in `librosa.beat`, `librosa.onset`, or `librosa.feature.tempo`.

## Immediate Tasks

1. Add CI step to run benchmark payload schema validation:
   - `Benchmarks/scripts/validate_benchmark_payloads.py`
2. Capture at least one additional host benchmark pair (`baseline` + `rust`) using Phase 15 harness.
3. Confirm `auto_review_cases` behavior in all benchmark JSON writers (Phase 5/12/14/15). [x]
4. Produce a promotion decision note with evidence table.

## Validation Commands

```powershell
python -m pytest -q -o addopts="" tests/test_benchmark_payload_schema.py tests/test_benchmark_guard_policy.py
python Benchmarks/scripts/benchmark_phase15_beat_upstream.py --json-out Benchmarks/results/phase16_host_baseline.json
$env:IRON_LIBROSA_RUST_DISPATCH='1'
python Benchmarks/scripts/benchmark_phase15_beat_upstream.py --json-out Benchmarks/results/phase16_host_rust.json
$env:IRON_LIBROSA_RUST_DISPATCH=$null
python Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/phase16_host_*.json --require-files
```

## Promotion Criteria

- No parity regressions.
- Benchmark artifacts pass schema validation.
- No unexplained sub-`1.5x` results without documented review.
- Reproducible speedups on at least one additional host profile.

## Status

- [x] Kickoff doc created
- [x] Cross-host benchmark pair captured
- [x] CI schema check wired
- [x] Promotion decision recorded (see: Development_docs/PHASE16_PROMOTION_DECISION.md)

Rust acceleration is now enabled by default. To disable (opt-out), set:

```powershell
$env:IRON_LIBROSA_RUST_DISPATCH='0'
```
