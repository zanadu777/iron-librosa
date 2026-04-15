# Phase 21 CUDA Drop-In Task Board

Goal: deliver CUDA acceleration as a true drop-in replacement (no required user environment variables), while preserving correctness and safe fallback behavior.

## Working Principles

- Auto-first runtime behavior; no user knobs required for normal use.
- CPU fallback remains guaranteed on any CUDA runtime/init/exec failure.
- Small PRs with explicit acceptance criteria and rollback notes.
- Performance changes are only promoted after benchmark gate verification.

## Task Board

| Task ID | Scope | Files | Estimate | Owner | Status | Acceptance Criteria | Risks | Rollback |
|---|---|---|---:|---|---|---|---|---|
| CUD-001 | Make Auto dispatch drop-in (no required env vars) | `src/backend.rs`, `src/stft.rs`, `src/istft.rs` | 0.5-1d | Runtime | Done | CUDA-capable host uses GPU in auto mode when profitable; CPU-only host unchanged | Unexpected GPU use on marginal workloads | Revert to prior heuristic constants in one commit |
| CUD-002 | Remove env-var dependency for normal CUDA dispatch path | `src/stft.rs`, `src/istft.rs`, `Benchmarks/scripts/benchmark_phase21_cuda_baseline.py` | 0.5d | Runtime/Bench | Done | Bench script validates auto-mode path without force env vars | Benchmark mismatch from mixed modes | Keep compatibility path during transition |
| CUD-003 | Integrate production workspace path (persistent allocations, pinned transfer path) | `src/cuda_fft.rs`, `src/cuda_fft_production.rs` | 1-1.5d | GPU Core | Done | Core wrappers use persistent workspace with existing fallback guarantees | Stream/memory lifecycle bugs | Keep old path callable and fallback on error |
| CUD-004 | Standardize dispatch diagnostics for auto mode | `src/backend.rs`, `src/cuda_fft.rs`, `src/lib.rs` | 0.5d | GPU Core | Done | Diagnostics report selected device and reason | Diagnostics drift from runtime behavior | Keep existing keys; additive only |
| CUD-005 | Add auto-dispatch fallback/parity tests | `tests/test_rust_backend_info.py`, `tests/test_cuda_auto_dispatch.py` | 1d | QA | Done | Tests prove CUDA-unavailable and CUDA-error fallback correctness | CI GPU availability gaps | Split GPU-required vs fallback-only tests |
| CUD-006 | Retune workload heuristic for small-workload safety | `src/stft.rs`, `src/istft.rs`, `src/cuda_fft.rs` | 1d | Performance | In Progress | No small-workload regression trend; medium/large improve | Overfit to one GPU | Conservative defaults with measured thresholds |
| CUD-007 | Benchmark gate update for auto mode promotion | `Benchmarks/scripts/benchmark_phase21_cuda_baseline.py`, `tests/test_phase21_cuda_benchmark_gate.py` | 0.5d | Bench/QA | In Progress | Promotion computed from auto mode runs with large-workload guard | Gate calibration instability | Versioned threshold metadata |
| CUD-008 | Packaging/release readiness for drop-in behavior | `Cargo.toml`, `pyproject.toml`, release docs | 1d | Build/Release | Todo | Build/release strategy documented for CUDA-capable + CPU-safe behavior | Artifact complexity | Keep CPU-only lane unchanged |
| CUD-009 | Documentation cleanup for drop-in policy | `README.md`, `Development_docs/PHASE21_IMPLEMENTATION_STATUS.md` | 0.5d | Docs | Todo | Docs show auto-first behavior and debug-only env vars | stale docs confusion | Changelog and deprecation note |

## Today Sprint (Start Here)

1. CUD-001 - Auto dispatch drop-in policy.
2. CUD-003 - Production workspace integration.
3. CUD-005 - Auto fallback/parity tests.

## Daily Done Definition

- Code merged for task scope.
- Benchmark sample attached (or test run attached for non-perf tasks).
- Rollback path confirmed.
- Docs/task board status updated.

