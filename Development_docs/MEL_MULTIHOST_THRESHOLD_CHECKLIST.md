# Mel Multi-Host Threshold Checklist

Date: 2026-04-04

Use this checklist to collect measured mel thresholds on representative hosts and close the remaining mel gate.

## 1) Required Host Profiles

Collect measured values for at least:

- `linux-x86_64-openblas`
- `darwin-arm64-accelerate`
- local Windows profile already measured: `windows-amd64-openblas`

Optional if available:

- `windows-amd64-mkl`

## 2) Per-Host Run Commands

Run from repo root on each host.

Create output directory once:

```bash
mkdir -p benchmarks/results
```

### Linux (bash)

```bash
cd /path/to/iron-librosa
python -m pytest tests/test_mel_threshold_policy.py -q > benchmarks/results/tmp_mel_policy_linux.txt 2>&1
python -m pytest tests/test_features.py -q -k "melspectrogram" > benchmarks/results/tmp_mel_features_linux.txt 2>&1
python calibrate_mel_threshold.py --dry-run --profile linux-x86_64-openblas --skip-registry > benchmarks/results/tmp_mel_calibrate_linux.txt 2>&1
python benchmarks/scripts/benchmark_melspectrogram.py > benchmarks/results/tmp_mel_bench_linux.txt 2>&1
```

### macOS (bash)

```bash
cd /path/to/iron-librosa
python -m pytest tests/test_mel_threshold_policy.py -q > benchmarks/results/tmp_mel_policy_macos.txt 2>&1
python -m pytest tests/test_features.py -q -k "melspectrogram" > benchmarks/results/tmp_mel_features_macos.txt 2>&1
python calibrate_mel_threshold.py --dry-run --profile darwin-arm64-accelerate --skip-registry > benchmarks/results/tmp_mel_calibrate_macos.txt 2>&1
python benchmarks/scripts/benchmark_melspectrogram.py > benchmarks/results/tmp_mel_bench_macos.txt 2>&1
```

### Windows (PowerShell)

```powershell
Set-Location "<path-to-iron-librosa-repo>"
New-Item -ItemType Directory -Path ".\benchmarks\results" -Force | Out-Null
python -m pytest tests/test_mel_threshold_policy.py -q 2>&1 | Out-File -FilePath ".\benchmarks\results\tmp_mel_policy_windows.txt" -Encoding utf8
python -m pytest tests/test_features.py -q -k "melspectrogram" 2>&1 | Out-File -FilePath ".\benchmarks\results\tmp_mel_features_windows.txt" -Encoding utf8
python calibrate_mel_threshold.py --dry-run --profile windows-amd64-openblas --skip-registry 2>&1 | Out-File -FilePath ".\benchmarks\results\tmp_mel_calibrate_windows_openblas.txt" -Encoding utf8
python benchmarks/scripts/benchmark_melspectrogram.py 2>&1 | Out-File -FilePath ".\benchmarks\results\tmp_mel_bench_windows.txt" -Encoding utf8
```

## 3) Update Registry

After collecting each host's measured threshold from the corresponding `benchmarks/results/tmp_mel_calibrate_*.txt` file:

1. Update `mel_threshold_registry.json` `thresholds` with measured integer values.
2. Mirror the same values in `librosa/feature/_mel_threshold_registry.py` (`MEL_WORK_THRESHOLDS`).
3. Keep key format `<os>-<arch>-<blas>`.

## 4) Append Evidence Log

Append a dated addendum in `cpu_complete_eval_20260404.txt` including:

- Test results from mel policy and melspectrogram parity tests.
- Calibrated threshold line per host/profile.
- `benchmark_melspectrogram.py` summary per host.
- Final registry snapshot lines for updated profiles.

## 5) Mel Gate Closure Criteria

Mel gate can be promoted from `OPEN` to `PASS` when:

1. Representative additional host thresholds are measured (at least Linux + macOS in addition to local Windows).
2. Threshold resolution behavior remains validated by `tests/test_mel_threshold_policy.py`.
3. `tests/test_features.py -k "melspectrogram"` remains green.
4. Dated benchmark evidence is recorded in `cpu_complete_eval_20260404.txt`.
5. `CPU_COMPLETE_CHECKLIST.md` mel item is updated to `PASS`.

