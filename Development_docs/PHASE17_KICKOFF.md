# Phase 17 Kickoff

Date: 2026-04-05

## Goal

Establish a Phase 17 baseline with strict parity/performance/CI artifact gates and start with a low-risk benchmark tooling hardening step.

## Inputs

- `Development_docs/PHASE16_ONSET_TEMPO_KICKOFF.md`
- `Development_docs/PHASE16_PROMOTION_DECISION.md`
- `Development_docs/PHASE16_PUSH_MONITORING_REPORT.md`
- `Development_docs/PHASE_COMPLETION_PROCESS.md`
- `Benchmarks/results/phase16_host_baseline.json`
- `Benchmarks/results/phase16_host_rust.json`

## Scope

### In scope
1. Preserve Phase 16 promotion safety while beginning Phase 17 work.
2. Standardize benchmark artifact generation for repeatable cross-host captures.
3. Define Phase 17 acceptance gates before kernel work begins.

### Out of scope
- New API surface changes.
- Large seam rewrites before baseline captures are refreshed.

## Acceptance Gates

1. **Parity gate**
   - No new test failures in the pre-push gate (`run_full_tests.py`).
2. **Performance gate**
   - Benchmarks captured with review threshold `1.5x` and review flags documented.
3. **Artifact gate**
   - Benchmark JSON payloads pass schema validation.
4. **Promotion gate**
   - Decision recorded as Promote / Opt-in / Defer with evidence.

## Task Order

1. Harden benchmark harness output controls (`--json-out`) for deterministic artifact naming.
2. Capture fresh baseline/rust benchmark pair for candidate Phase 17 workloads.
3. Validate artifacts with `Benchmarks/scripts/validate_benchmark_payloads.py`.
4. Execute parity tests and targeted phase tests.
5. Record promotion decision and completion report.

## First Implementation (Completed)

- Added `--json-out` handling to `Benchmarks/scripts/benchmark_phase15_beat_upstream.py`.
- Default output behavior remains unchanged when `--json-out` is not provided.
- Schema enforcement remains active via `assert_benchmark_payload_schema(...)`.

## Validation Commands

```powershell
python -m pytest -q -o addopts="" tests/test_benchmark_payload_schema.py tests/test_benchmark_guard_policy.py
python Benchmarks/scripts/benchmark_phase15_beat_upstream.py --json-out Benchmarks/results/phase17_host_baseline.json
$env:IRON_LIBROSA_RUST_DISPATCH='1'
python Benchmarks/scripts/benchmark_phase15_beat_upstream.py --json-out Benchmarks/results/phase17_host_rust.json
$env:IRON_LIBROSA_RUST_DISPATCH=$null
python Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/phase17_host_*.json --require-files
```

## Status

- [x] Harness hardening (`--json-out`) completed.
- [x] Baseline artifact captured: `Benchmarks/results/phase17_host_baseline.json`.
- [x] Rust artifact captured: `Benchmarks/results/phase17_host_rust.json`.
- [x] Artifact schema validation passed.

### Initial host results

- `noisy_30s`: 29.1 ms -> 12.2 ms (`2.39x`)
- `noisy_120s`: 157.3 ms -> 53.8 ms (`2.93x`)

