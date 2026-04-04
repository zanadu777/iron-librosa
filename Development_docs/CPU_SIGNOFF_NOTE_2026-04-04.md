# CPU Signoff Note (2026-04-04)

## Decision

- Global CPU-complete signoff: `NO-GO` (temporary)
- Windows-host validation stream: `GO`

## Why

- `librosa.beat.beat_track` gate is now closed (`PASS`) with parity and benchmark evidence.
- Mel threshold policy is validated locally, but promotion remains `OPEN` pending measured multi-host profile thresholds.

## Evidence

- Checklist status: `CPU_COMPLETE_CHECKLIST.md`
- Dated run log: `cpu_complete_eval_20260404.txt`
- Mel policy tests: `tests/test_mel_threshold_policy.py` (`6 passed`)
- Mel parity sample: `tests/test_features.py -k "melspectrogram"` (`3 passed`)
- Registry artifacts: `mel_threshold_registry.json`, `librosa/feature/_mel_threshold_registry.py`

## Remaining Blocker

- Multi-host measured mel threshold entries are not yet populated for representative non-local profiles.

## Exit Criteria (to flip to GO)

1. Add measured thresholds for representative additional profiles (at minimum one Linux and one macOS profile).
2. Confirm policy resolution behavior for those profiles (env override, external registry, built-in fallback precedence).
3. Re-run `benchmark_melspectrogram.py` on those hosts and append dated evidence to `cpu_complete_eval_20260404.txt`.
4. Update `CPU_COMPLETE_CHECKLIST.md` mel gate from `OPEN` to `PASS`.

