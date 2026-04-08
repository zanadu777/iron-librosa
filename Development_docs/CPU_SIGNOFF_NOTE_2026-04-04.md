# CPU Signoff Note (2026-04-04)

## Decision (updated 2026-04-08)

- Mel-threshold blocker status: `CLOSED`
- CPU-complete global signoff: `GO`

## Why

- `librosa.beat.beat_track` gate is now closed (`PASS`) with parity and benchmark evidence.
- Mel threshold policy now has merged multi-host calibration evidence (Linux/macOS) from the calibration workflow.

## Evidence

- Checklist status: `CPU_COMPLETE_CHECKLIST.md`
- Dated run log: `cpu_complete_eval_20260404.txt`
- Mel policy tests: `tests/test_mel_threshold_policy.py` (`6 passed`)
- Mel parity sample: `tests/test_features.py -k "melspectrogram"` (`3 passed`)
- Registry artifacts: `scripts/mel_threshold_registry.json`, `librosa/feature/_mel_threshold_registry.py`
- Collection workflow: `.github/workflows/mel-threshold-calibration.yml` (`workflow_dispatch`, run `24115024508`)
- Merge evidence: PR `#1` (`Update mel threshold registry from multi-host calibration`), merge commit `29d89d95`

## Remaining Blocker

- None. Mel and non-mel checklist blockers are closed.

## Exit Criteria (to flip to GO)

1. Keep the calibration workflow runnable for future host/profile refreshes.
2. Keep policy resolution behavior covered by `tests/test_mel_threshold_policy.py`.
3. Keep `cpu_complete_eval_20260404.txt` updated when new sub-1.5x benchmark reviews are added.

