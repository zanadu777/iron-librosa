# Phase Completion Process

Last updated: 2026-04-05

---

## Mandatory Gate: Performance Review Before Every Commit

> **For every phase (Phase 14 onward, and retroactively enforced on any
> reopened earlier phase): code must pass a performance review before any
> commit lands.  No exceptions.**

A phase is never considered "commit-ready" from code-complete alone.
The sequence is:

```
Code Complete  →  Parity Tests Green  →  Perf Review  →  Commit / Promotion Decision
```

---

## Step-by-Step Checklist

The following must all be checked before committing the phase:

### 1. Code Complete
- [ ] All planned Rust kernels implemented and building without warnings.
- [ ] Python dispatch seam wired with availability guard and fallback.
- [ ] No `unwrap()` / `expect()` on untrusted inputs; all error paths return to Python safely.

### 2. Parity Tests Green
- [ ] Existing test suite passes with no new failures:
  ```powershell
  python -m pytest -q -o addopts="" tests/
  ```
- [ ] Phase-specific parity tests green (exact match / tolerance as per seam contract).
- [ ] Fallback path tested explicitly (force-numpy environment variable).

### 3. Performance Review  ← **Required gate before commit**
- [ ] Benchmark script executed with `--backend compare` (or equivalent).
- [ ] Baseline (NumPy) vs Rust timings captured and saved to `Benchmarks/results/`.
- [ ] Any case with speedup `< 1.5x` is automatically flagged for review and documented.
- [ ] Results reviewed against the promotion criteria defined in the phase kickoff doc.
- [ ] One of the following decisions is explicitly documented:
  - **Promote:** Rust path enabled by default (speedup meets threshold, no regressions).
  - **Opt-in:** Rust available but not default (gains insufficient or not reproducible).
  - **Revert / Defer:** Implementation dropped or deferred with rationale recorded.
- [ ] Any notable regressions (operations that got *slower*) are called out with explanation.

Auto-review policy command example:

```powershell
python Benchmarks/scripts/benchmark_phase14_beat_track.py --backend compare --review-threshold 1.5
```

### 4. Commit
- [ ] Commit message references the phase and includes the promotion decision.
- [ ] Benchmark JSON artifact committed under `Benchmarks/results/`.
- [ ] Phase completion report written under `Development_docs/PHASE<N>_*_COMPLETION_REPORT.md`.

---

## Performance Review Minimum Bar

| Scenario | Minimum acceptable result to promote |
|---|---|
| Single-channel, medium input | ≥ 1.5× speedup vs NumPy fallback |
| Multichannel / batch | ≥ 1.2× speedup (dispatch overhead larger) |
| Parity tolerance (f32) | `rtol=1e-6`, `atol=1e-6` |
| Parity tolerance (f64) | `rtol=1e-9`, `atol=1e-9` |
| Test suite delta | **Zero new failures** |

Opt-in (not default) is acceptable if gains are real but below the promotion
threshold, or are workload-specific.  Document the rationale in the completion
report.

---

## Perf Review Artifacts

Each phase's benchmark results must be saved as:

```
Benchmarks/results/phase<N>_<feature>_baseline.json
Benchmarks/results/phase<N>_<feature>_rust.json   (or combined comparison JSON)
```

The completion report must reference these files and include a human-readable
summary table (operation, Python time, Rust time, speedup).

---

## Applies To

This process applies to **all phases**, including:
- Phase 14 (Beat Track DP) — in progress
- All future phases
- Any phase that is reopened for additional work

For historical phases (1–13) the requirement is considered satisfied by the
benchmark artifacts already committed.  If a phase-N kernel is modified
post-commit, the perf review gate re-applies to that kernel.

---

## Related Documents

- `Development_docs/PHASE14_BEAT_TRACK_KICKOFF.md` — current phase in progress
- `Development_docs/PHASE14_BEAT_TRACK_SEAM_CONTRACT.md` — seam contract and parity rules
- `CONTRIBUTING.md` — general contribution guidelines

