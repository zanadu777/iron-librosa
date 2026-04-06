# Phase 17 Promotion Decision

Date: 2026-04-05

## Summary

Phase 17 completes beat-track non-finite onset hardening and validates benchmark artifact flow for cross-host reporting.

All Phase 17 promotion criteria are met:

- No new parity regressions in targeted beat/onset parity suites.
- Phase 17 benchmark artifacts pass schema validation.
- Measured host speedups remain above the 1.5x review threshold.
- Behavior is backend-invariant for forced NumPy and forced Rust beat dispatch flags.

## Evidence Table

| Case | Baseline (ms) | Rust (ms) | Speedup | Review Required |
|---|---:|---:|---:|---|
| noisy_30s | 29.1 | 12.2 | 2.39x | No |
| noisy_120s | 157.3 | 53.8 | 2.93x | No |

- Review threshold: 1.5x
- Auto-review cases: none

## Validation Evidence

- `python -m pytest -q -o addopts="" tests/test_beat.py -k "nonfinite_onsets or test_beat_no_onsets" tests/test_phase14_beat_parity.py tests/test_phase15_beat_upstream.py tests/test_benchmark_payload_schema.py tests/test_benchmark_guard_policy.py`
  - Result: 15 passed
- `python Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/phase17_host_*.json --require-files`
  - Result: pass

## Artifacts

- `Benchmarks/results/phase17_host_baseline.json`
- `Benchmarks/results/phase17_host_rust.json`
- `Development_docs/PHASE17_KICKOFF.md`

## Decision

Promote Phase 17 changes.

Rust acceleration remains enabled by default. Phase 17 hardening introduces safer behavior for non-finite onset envelopes without changing valid-input outputs.

