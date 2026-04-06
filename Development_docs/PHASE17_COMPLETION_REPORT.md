# Phase 17 Beat Hardening and Artifact Gate Completion Report

Date: 2026-04-05

## Scope Completed

Phase 17 delivered two items:

1. Beat hardening for non-finite onset envelopes in `librosa.beat.beat_track`.
2. Deterministic benchmark artifact output control (`--json-out`) and schema-gated host artifacts.

No new Rust kernels were added in this phase.

## Code Changes

### 1) Beat hardening

- File: `librosa/beat.py`
- Change: `beat_track` now treats non-finite onset envelopes (`NaN`, `Inf`, `-Inf`) as empty detections.
- Behavior details:
  - If `bpm` is not provided: returns zero tempo and empty beats.
  - If `bpm` is provided: preserves provided tempo and returns empty beats.
- Rationale: Align with existing non-finite handling policy used in onset detection (`librosa.onset.onset_detect`).

### 2) Regression coverage

- File: `tests/test_beat.py`
- Added tests:
  - `test_beat_nonfinite_onsets_returns_empty`
  - `test_beat_nonfinite_onsets_dispatch_invariant`
- Coverage:
  - sparse and dense return modes
  - with and without user-provided `bpm`
  - forced NumPy and forced Rust beat dispatch flags

### 3) Artifact flow hardening

- File: `Benchmarks/scripts/benchmark_phase15_beat_upstream.py`
- Added optional `--json-out` support for explicit artifact naming.
- Existing default output behavior preserved.
- Existing schema assertion preserved.

## Validation

### Targeted parity and policy tests

Command:

```powershell
python -m pytest -q -o addopts="" tests/test_beat.py -k "nonfinite_onsets or test_beat_no_onsets" tests/test_phase14_beat_parity.py tests/test_phase15_beat_upstream.py tests/test_benchmark_payload_schema.py tests/test_benchmark_guard_policy.py
```

Result:

- 15 passed

### Artifact schema validation

Command:

```powershell
python Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/phase17_host_*.json --require-files
```

Result:

- PASS: all payloads match schema

## Performance Evidence

From Phase 17 host artifacts:

- `noisy_30s`: 29.1 ms -> 12.2 ms (2.39x)
- `noisy_120s`: 157.3 ms -> 53.8 ms (2.93x)

Both cases exceed the 1.5x review threshold.

## Promotion Outcome

See `Development_docs/PHASE17_PROMOTION_DECISION.md`.

Decision: Promote.

## Checklist

- [x] Code complete for planned Phase 17 hardening scope
- [x] Parity tests green for the affected seam
- [x] Benchmark artifacts schema-valid
- [x] Performance review threshold satisfied on host artifacts
- [x] Promotion decision documented

