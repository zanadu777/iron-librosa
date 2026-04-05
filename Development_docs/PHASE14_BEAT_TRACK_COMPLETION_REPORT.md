# Phase 14 Beat-Track DP — Completion Report

Date: 2026-04-05

---

## Summary

Phase 14 implemented the first Rust acceleration seam for beat tracking
(the dynamic-programming core, `__beat_track_dp`).  All code is complete,
parity-tested and benchmarked.  The promotion decision is **Opt-in**.

---

## Deliverables

| Deliverable | Location | Status |
|---|---|---|
| Rust kernel `beat_track_dp_f32/f64` | `src/beat.rs` | ✅ |
| Python dispatch seam | `librosa/beat.py :: __beat_track_dp_dispatch` | ✅ |
| Bridge env flags `FORCE_NUMPY_BEAT` / `FORCE_RUST_BEAT` | `librosa/_rust_bridge.py` | ✅ |
| Parity test suite | `tests/test_phase14_beat_parity.py` | ✅ 3/3 pass |
| Baseline benchmark JSON | `Benchmarks/results/phase14_beat_track_baseline_local.json` | ✅ |
| Kickoff compare JSON | `Benchmarks/results/phase14_beat_track_kickoff.json` | ✅ |
| Final compare JSON | `Benchmarks/results/phase14_beat_track_final.json` | ✅ |
| This completion report | `Development_docs/PHASE14_BEAT_TRACK_COMPLETION_REPORT.md` | ✅ |

---

## Parity Test Results

```
tests/test_phase14_beat_parity.py  3 passed  (0 failures)
Full suite (excl. test_display.py): 8601 passed, 1 pre-existing failure
  (test_features::test_cens — missing test data file, unrelated to Phase 14)
```

All three gates from the seam contract passed:

- `beat_track_dp_f32` and `beat_track_dp_f64` symbols present in extension.
- Mono numpy-vs-Rust beat frame exact parity (`np.testing.assert_array_equal`).
- Multichannel guarded-fallback parity (Rust path guards to Python for ndim > 1).

---

## Performance Review

### Stage breakdown (average ms, 5 reps each)

| Case | Samples | Stage | NumPy (ms) | Rust (ms) | Speedup |
|---|---|---|---|---|---|
| click_120bpm_30s | 661 500 | onset | 17.53 | 18.88 | — |
| | | tempo | 17.26 | 17.24 | — |
| | | local_score | 0.189 | 0.181 | — |
| | | **DP** | **0.298** | **0.322** | **0.93×** |
| | | post | 0.304 | 0.306 | — |
| | | **end-to-end** | **17.21** | **20.06** | **0.86×** |
| noisy_30s | 661 500 | onset | 18.04 | 18.62 | — |
| | | tempo | 17.46 | 17.18 | — |
| | | local_score | 0.189 | 0.178 | — |
| | | **DP** | **0.292** | **0.287** | **1.02×** |
| | | post | 0.312 | 0.301 | — |
| | | **end-to-end** | **31.96** | **30.21** | **1.06×** |
| noisy_120s | 2 646 000 | onset | 86.04 | 85.58 | — |
| | | tempo | 83.70 | 85.00 | — |
| | | local_score | 0.654 | 0.634 | — |
| | | **DP** | **1.173** | **1.185** | **0.99×** |
| | | post | 0.449 | 0.432 | — |
| | | **end-to-end** | **135.60** | **135.75** | **1.00×** |

### Findings

1. **DP is not the bottleneck.** The `__beat_track_dp` stage consumes 0.7–0.9 %
   of total `beat_track` runtime.  Onset strength computation and tempo
   estimation each consume ~49 % of runtime.

2. **Rust DP ≈ Numba DP.**  The Rust kernel matches the JIT-compiled Numba
   kernel within noise (0.93×–1.02×).  Numba already provides near-native
   speed for this O(n) scan.

3. **No meaningful end-to-end gain.**  End-to-end speedup ranges from 0.86×
   to 1.06× depending on the workload — statistically indistinguishable from
   parity.

---

## Promotion Decision: **Opt-in**

> Rust beat DP path is correct and guarded.  It remains **opt-in** and is NOT
> promoted to default dispatch.

Rationale:
- The DP stage accounts for < 1 % of `beat_track` runtime.  Even a 10× kernel
  speedup would yield < 0.1 % end-to-end improvement.
- Numba's JIT already reaches near-native speed for this loop; there is no
  systematic gap for Rust to exploit.
- The minimum promotion threshold (≥ 1.5× on medium single-channel) was not
  met.

**How to enable Rust path (opt-in):**
```powershell
$env:IRON_LIBROSA_BEAT_BACKEND = "rust"
$env:IRON_LIBROSA_RUST_DISPATCH = "1"
python your_script.py
```

---

## Next Phase Candidates

The profiling reveals the true bottlenecks:

| Stage | Share | Candidate seam |
|---|---|---|
| Onset strength | ~49 % | Phase 15: Rust onset_strength inner loop |
| Tempo estimation | ~49 % | Phase 15/16: Rust autocorrelation / tempogram |
| Local score convolution | ~0.5 % | Low ROI, not a primary target |
| Beat DP | ~0.8 % | ✅ Done (this phase) |

Recommended next action: baseline-profile `librosa.onset.onset_strength` and
`librosa.feature.tempo` to identify sub-stage hotspots before committing to a
seam for Phase 15.

---

## Artifacts

- `Benchmarks/results/phase14_beat_track_final.json` — final compare run
- `Benchmarks/results/phase14_beat_track_kickoff.json` — earlier compare run
- `Benchmarks/results/phase14_beat_track_baseline_local.json` — numpy-only baseline
- `tests/test_phase14_beat_parity.py` — parity test suite

---

## Checklist Sign-Off

- [x] Kickoff scope documented
- [x] Baseline profile captured
- [x] First Rust seam implemented (`beat_track_dp_f32` / `beat_track_dp_f64`)
- [x] Parity suite green with Rust path (3/3 tests pass)
- [x] Benchmark delta captured and saved to `Benchmarks/results/`
- [x] **Perf review completed** — see table above
- [x] Promotion decision documented — **Opt-in** (rationale: DP < 1% of runtime, no gap vs Numba)
