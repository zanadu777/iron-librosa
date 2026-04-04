# CPU-Complete Checklist

Date: 2026-04-04

This checklist is the release gate for CPU readiness before Linux/macOS expansion.

Quick-cycle snapshot (2026-04-04): based on `cpu_complete_eval_20260404.txt`.

## 1) CPU Baseline (Default Settings Only)

Pass only if all items below are true using default env settings.

- [x] `stft` / `istft` CPU paths are default-on and parity-stable. (`PASS`)
  - Evidence: `tests/test_core.py` (`stft`, `istft` coverage)
  - Pass criteria: no new regressions in focused STFT/ISTFT tests.
- [x] Spectral core (`rms(S=...)`, `spectral_centroid`, `spectral_rolloff`, `spectral_bandwidth`, `spectral_flatness`) remains default-on and parity-stable. (`PASS`)
  - Evidence: `tests/test_phase4_features.py`, `benchmark_phase5_spectral.py`
  - Pass criteria: parity tests pass; no benchmark collapse on medium+ workloads.
- [x] `spectral_contrast` auto policy is parity-stable and performant for stereo/quad+. (`PASS` for medium+)
  - Evidence: `tests/test_phase7_contrast.py`, `benchmark_phase11_contrast_multichannel.py`
  - Pass criteria: parity tests pass; auto mode does not regress medium+ multichannel.
- [x] Chroma/MFCC pipeline remains CPU-stable (`chroma_stft`, `mfcc`). (`PASS`)
  - Evidence: `tests/test_phase4c_chroma.py`, `tests/test_features.py -k "chroma or mfcc or melspectrogram"`, `benchmark_phase5_chroma.py`, `benchmark_mfcc.py`
  - Pass criteria: no dtype/shape regressions; no major benchmark regression.
- [x] HPSS fused paths are stable for CPU multichannel workloads. (`PASS`)
  - Evidence: `tests/test_phase10a_hpss.py`, `tests/test_phase10c_hpss_optimization.py`, `tests/test_decompose.py -k "hpss"`, `benchmark_phase10a_hpss.py`, `benchmark_phase10b_batch_parallel.py`, `benchmark_phase10c_hpss_optimization.py`
  - Pass criteria: no correctness drift and no severe perf regression.

## 2) Gated Paths To Promote (Default-On Candidates)

Promotion means removing or relaxing gate conservatively after evidence.

- [x] `estimate_tuning` Rust helper (`IRON_LIBROSA_ENABLE_RUST_TUNING`). (`PASS` for medium+ speedup criterion)
  - Evidence: `tests/test_phase5_tuning.py`, `benchmark_phase5_tuning.py`
  - Promote when: parity holds and end-to-end min speedup >= 1.1x on medium+ workloads.
- [x] `piptrack` Rust path (`IRON_LIBROSA_PIPTRACK_RUST_MODE`, `...MIN_WORK`). (`PASS` for medium+ non-regression criterion)
  - Evidence: `tests/test_phase5_tuning.py`, `tests/test_core.py::test_piptrack_properties`
  - Promote when: parity holds and medium+ workloads are non-regressive.
- [x] Time-domain `rms(y=...)` (`IRON_LIBROSA_ENABLE_RUST_RMS_TIME`). (`PASS`)
  - Evidence: `tests/test_phase4_features.py -k "rms"`, `benchmark_phase5_spectral.py` section for `rms(y=...)`
  - Promote when: no small-workload collapse and medium+ workloads are >= parity. (met on 2026-04-04 run)
- [ ] Mel backend policy threshold (`_MEL_RUST_WORK_THRESHOLD`). (`OPEN` - cross-CPU strategy in place)
  - Evidence: `calibrate_mel_threshold.py` (profile-aware registry support), `tests/test_mel_threshold_policy.py` (`6 passed`), `benchmark_melspectrogram.py`, `mel_threshold_registry.json`, `librosa/feature/_mel_threshold_registry.py`
  - Status note: representative profile keys are populated with conservative defaults; measured multi-host thresholds are still pending.
  - Promote when: profile registry has measured entries for representative hosts and threshold behavior is validated across those hosts.

## 3) Remaining Niche / Unaccelerated Public APIs

These do not block CPU-complete baseline, but must be tracked.

- [x] `librosa.phase_vocoder`
  - Current: Python path (parity tests pass: `tests/test_multichannel.py -k "phase_vocoder"`; benchmark harness: `benchmark_phase12_phase_vocoder.py`)
  - Exit criterion: parity tests + medium-workload benchmark harness. (`PASS`)
- [x] `librosa.core.constantq.cqt` / `vqt`
  - Current: Python-heavy sparse pipeline (baseline tests pass: `tests/test_constantq.py`; plan+harness: `PHASE12_CQT_VQT_ACCELERATION_PLAN.md`, `benchmark_phase12_cqt_vqt.py`)
  - Exit criterion: targeted acceleration plan and parity/perf harness. (`PASS`)
- [x] `librosa.feature.tonnetz`
  - Current: mostly Python path (fixture-safe parity policy documented: `PHASE12_TONNETZ_DECISION.md`; targeted tests run with skip when optional fixtures are absent)
  - Exit criterion: parity + profile-driven optimization decision. (`PASS`)
- [x] `librosa.beat.beat_track` full pipeline (`PASS`)
  - Current: hotspot-driven mitigation landed in `librosa/beat.py` (beat-specific tempo window reduction via `_BEAT_TRACK_TEMPO_AC_SIZE = 4.0`, plus prior local-score JIT/cache + static convolution path)
  - Evidence: `tests/test_beat.py` (`861 passed, 1 skipped, 19 xfailed`), `benchmark_phase12_beat_track.py` before/after rerun shows end-to-end improvement on noisy workloads (`30.693 -> 26.713 ms` for 30s, `148.751 -> 99.487 ms` for 120s)
  - Exit criterion: end-to-end profiling + targeted hotspot mitigation. (`PASS`)

## 4) Validation Artifacts Required Before CPU-Complete Signoff

- [x] Focused parity tests for modified areas pass. (`PASS`)
  - Minimum: `tests/test_phase4_features.py`, `tests/test_phase5_tuning.py`, `tests/test_phase7_contrast.py`
- [x] Focused core tests for pitch/time-frequency pass. (`PASS`)
  - Minimum: `tests/test_core.py -k "piptrack or estimate_tuning or yin or pyin"`
- [x] Benchmarks captured for medium and long workloads. (`PASS`)
  - Minimum scripts:
    - `benchmark_phase5_spectral.py`
    - `benchmark_phase5_tuning.py`
    - `benchmark_phase11_contrast_multichannel.py`
- [x] Results archived to a dated text file in repo root. (`PASS`)
  - Captured: `cpu_complete_eval_20260404.txt`

## 5) Signoff Rule

CPU-complete is achieved when:

1. Section 1 is fully checked.
2. Section 2 has either:
   - promoted default-on, or
   - documented justification for leaving gated with threshold evidence.
3. Section 4 artifacts are present and reproducible.

After signoff, proceed to Linux port, then macOS, and re-run this checklist per platform.

## 6) Current Decision (2026-04-04)

- Decision: `NO-GO` for full CPU-complete signoff (global), `GO` for continued Windows-host validation work.
- Reason: all gates are closed except mel threshold promotion in Section 2.
- Open blocker: measured multi-host mel threshold entries are still pending for representative non-local profiles.
- Evidence anchors: `cpu_complete_eval_20260404.txt`, `tests/test_mel_threshold_policy.py` (`6 passed`), `mel_threshold_registry.json`, `librosa/feature/_mel_threshold_registry.py`.
- Exit criteria to flip to `GO`:
  1. Record measured mel thresholds for representative additional hosts (beyond local `windows-amd64-openblas`).
  2. Validate threshold resolution behavior for those profiles using existing policy tests.
  3. Re-run `benchmark_melspectrogram.py` and attach dated results to `cpu_complete_eval_20260404.txt`.

For review-friendly release notes, see `CPU_SIGNOFF_NOTE_2026-04-04.md`.

