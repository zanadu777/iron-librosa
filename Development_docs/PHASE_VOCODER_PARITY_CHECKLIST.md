# Phase-Vocoder Parity Fix Checklist

## What was the mismatch?
Rust and Python phase-vocoder kernels diverged because of **rounding direction asymmetry** in phase wrapping:
- Rust's default `round()` = ties-away-from-zero
- NumPy's `round()` = ties-to-even
- Small wrap differences → accumulated phase divergence over frames

## What was fixed?
Changed phase wrapping in `src/phase_vocoder.rs`:
```rust
// Both f32 and f64 paths now use:
dp -= two_pi * (dp / two_pi).round_ties_even();
```

## How to verify the fix works?

### Quick test (post-rebuild):
```bash
python test_phase_vocoder_parity.py
```
Expected output:
```
✓ PASS: f32 parity within tolerance (1e-5)
✓ PASS: f64 parity within tolerance (1e-11)
✓ All parity tests PASSED!
```

### Full pytest suite:
```bash
pytest tests/test_features.py -k "phase_vocoder" -v
```
All should **pass** (not xfail):
- `test_phase_vocoder_dispatch_default_stays_python`
- `test_phase_vocoder_dispatch_opt_in_calls_rust`
- `test_phase_vocoder_dispatch_opt_in_calls_rust_per_channel`
- `test_phase_vocoder_rust_kernel_matches_reference_loop[complex64-...]`
- `test_phase_vocoder_rust_kernel_matches_reference_loop[complex128-...]`

## What changed in code?

### `src/phase_vocoder.rs`
- Line 128: f32 path uses `round_ties_even()` in wrap
- Line 213: f64 path uses `round_ties_even()` in wrap

### `tests/test_features.py`
- Added: `_phase_vocoder_trace_divergence()` helper (detailed mismatch reporting)
- Removed: `@pytest.mark.xfail` from parity test (now expects pass)
- Tightened tolerances: f32 `1e-5`, f64 `1e-11`

## When can Rust dispatch be enabled by default?

**Current state:** Opt-in only via `IRON_LIBROSA_ENABLE_RUST_PHASE_VOCODER=1`

**Promotion criteria (ALL must pass):**
1. ✓ `test_phase_vocoder_rust_kernel_matches_reference_loop` passes for f32 and f64
2. ✓ `test_phase_vocoder_dispatch_*` tests all pass
3. ✓ `test_stft_complex_phase_vocoder_parity` passes (end-to-end audio)
4. ✓ Benchmark shows ≥1.1× speedup on medium workloads
5. ✓ No other tests fail due to dispatch changes

**Promotion action:** 
- Remove env var gate in `librosa/core/spectrum.py` (lines ~1543–1545)
- Change default to Python, add explicit `prefer_rust=True` parameter to `phase_vocoder()`
- Update release notes

## Files touched
- `src/phase_vocoder.rs` (rounding fix)
- `tests/test_features.py` (parity tests + tracer)
- `test_phase_vocoder_parity.py` (verification harness, new)
- `PHASE_VOCODER_FIX.md` (detailed analysis, new)
- `Development_docs/PHASE12_CPU_REMAINING_PLAN.md` (status update)

## Questions?
See `PHASE_VOCODER_FIX.md` for:
- Root cause deep dive
- Numeric precision policy per dtype
- Detailed divergence trace example
- Promotion strategy & timeline

