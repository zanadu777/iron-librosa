# Phase 20 macOS Closeout Checklist (2026-04-12)

## Goal

Close Phase 20 with artifact-backed confidence, then hand off to CUDA work without reopening macOS optimization unexpectedly.

## Must-Pass Checklist

### A) Regression and Parity

- [x] Full regression run complete with zero new failures
- [x] STFT/iSTFT focused tests green
- [x] Metal FFT dispatch/fallback tests green

Commands:

```bash
cd /Users/kenjohnson/Dev/Rust/iron-librosa
source .venv-mac/bin/activate
pytest -q tests/ --tb=short | tee artifacts/run_logs/phase20_pytest_full_2026-04-12.log
pytest -xvs tests/ -k "stft or istft" --tb=short | tee artifacts/run_logs/phase20_stft_istft_2026-04-12.log
pytest -q tests/test_metal_fft_dispatch.py --tb=short | tee artifacts/run_logs/phase20_metal_dispatch_2026-04-12.log
```

### B) Benchmark Artifacts

- [x] Phase 20 benchmark JSON generated
- [x] Phase 20 benchmark markdown generated
- [x] Artifact paths recorded in decision doc

Commands:

```bash
cd /Users/kenjohnson/Dev/Rust/iron-librosa
source .venv-mac/bin/activate
python -u Benchmarks/scripts/benchmark_phase19_chunk_ab.py \
  --chunks default \
  --min-frames 200 \
  --rounds 3 --repeats 4 --warmup 1 --cpu-outer-runs 3 \
  --json-out Benchmarks/results/phase20_chunk_ab_2026-04-12.json \
  --md-out Benchmarks/results/phase20_chunk_ab_2026-04-12.md
```

### C) Decision and Promotion State

- [x] `Development_docs/PHASE20_PROMOTION_DECISION_2026-04-12.md` completed
- [x] Decision selected: Promote / Opt-in / Defer
- [x] Residual regressions documented with workload shape

### D) CUDA Handoff Readiness

- [x] `Development_docs/PHASE21_CUDA_KICKOFF_2026-04-12.md` completed
- [x] CUDA scope and fallback invariants signed off
- [x] Next-phase owner and first 3 tasks assigned

## Same-Day Timeline

- 09:00-10:00: scope freeze + env prep
- 10:00-12:00: test gates and logs
- 13:00-15:00: benchmarks and artifacts
- 15:00-16:00: decision write-up
- 16:00-17:30: CUDA kickoff packet and handoff

## Exit Criteria

- [x] All required artifacts exist in `Benchmarks/results/` and `artifacts/run_logs/`
- [x] Decision doc complete and linked from phase report/index
- [x] CUDA kickoff doc approved for implementation start

## Status: ✅ COMPLETE — Phase 20 macOS closed. Phase 21 PC/CUDA ready to start.

