# Phase 14 Beat-Track Seam Contract

Date: 2026-04-05

## Scope

This contract defines the first Rust acceleration seam for beat tracking:

- Python call site: `librosa.beat.__beat_track_dp_dispatch`
- Candidate Rust symbols:
  - `beat_track_dp_f32`
  - `beat_track_dp_f64`

This seam targets the dynamic-programming (DP) stage only. Public behavior of
`librosa.beat.beat_track` must remain unchanged.

## Python/Rust Input Contract

Inputs provided by the dispatch seam:

1. `localscore`: contiguous 1D numeric array, shape `(n_frames,)`
2. `frames_per_beat`: scalar-like array or scalar-expanded value
3. `tightness`: positive finite `float`

Constraints:

- `localscore` must be `float32` or `float64` for Rust path.
- Non-float dtypes or unsupported shapes must use Python fallback.
- Empty arrays and degenerate frames must return valid empty/default outputs
  without panic.

## Output Contract

Rust DP kernel must return tuple:

1. `backlink`: signed integer array, shape `(n_frames,)`
2. `cumscore`: float array matching dtype family and shape `(n_frames,)`

Parity requirements against `librosa.beat.__beat_track_dp`:

- `backlink` exact equality
- `cumscore` numeric parity with strict tolerance
  - `float32`: `rtol=1e-6`, `atol=1e-6`
  - `float64`: `rtol=1e-9`, `atol=1e-9`

## Dispatch Rules

Order of precedence:

1. `FORCE_NUMPY_BEAT=True` -> always Python fallback
2. `FORCE_RUST_BEAT=True` and Rust symbol available -> force Rust path
3. Default auto mode -> Rust allowed only when contract preconditions are met

If Rust symbol is missing or input is unsupported, dispatch must silently and
safely fall back to Python.

## Non-goals for this seam

- No changes to onset-strength computation
- No changes to tempo estimation strategy
- No changes to beat trimming policy
- No changes to public API return structure

## Validation Gates

Required checks before promotion decision:

- Unit parity tests for dispatch behavior and stub-backed Rust calls
- End-to-end beat/tempo parity checks on deterministic fixtures
- Benchmark report comparing `numpy` vs `rust` dispatch modes

Commands:

```powershell
python -m pytest -q -o addopts="" tests/test_beat.py
python Benchmarks/scripts/benchmark_phase14_beat_track.py --backend compare
```

