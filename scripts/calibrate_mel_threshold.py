"""Calibrate _MEL_RUST_WORK_THRESHOLD for this machine.

Sweeps (n_mels, n_fft_bins, n_frames) combinations, measures the minimum
wall-clock time for:
  - numpy:  mel_basis.dot(S)          (NumPy / BLAS)
  - rust:   _rust_ext.mel_project_f32 (faer Rayon GEMM)

Then finds the FLOP count (n_mels * n_bins * n_frames) below which Rust
is consistently faster, writes that value back into spectral.py.

Usage:
    python calibrate_mel_threshold.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np


def _detect_blas_vendor() -> str:
    """Best-effort BLAS vendor detection from numpy config output."""

    try:
        import io
        from contextlib import redirect_stdout

        stream = io.StringIO()
        with redirect_stdout(stream):
            np.__config__.show()
        text = stream.getvalue().lower()
    except Exception:
        text = ""

    if "mkl" in text:
        return "mkl"
    if "openblas" in text:
        return "openblas"
    if "accelerate" in text or "veclib" in text:
        return "accelerate"
    if "blis" in text:
        return "blis"
    return "unknown"


def _default_profile_key() -> str:
    system = platform.system().lower() or "unknown-os"
    machine = platform.machine().lower() or "unknown-arch"
    blas = _detect_blas_vendor()
    return f"{system}-{machine}-{blas}"


def _update_registry(registry_path: Path, profile_key: str, threshold: int, dry_run: bool) -> None:
    """Update or create profile threshold registry JSON file."""

    existing = {}
    if registry_path.exists():
        try:
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    if not isinstance(existing, dict):
        existing = {}

    thresholds = existing.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    thresholds[profile_key] = int(threshold)
    payload = {
        "version": 1,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": thresholds,
    }

    if dry_run:
        print(f"[dry-run] would write profile '{profile_key}' -> {threshold:_} to {registry_path}")
        return

    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] Wrote profile '{profile_key}' -> {threshold:_} to {registry_path}")

# ── bootstrap ────────────────────────────────────────────────────────────────
try:
    from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE
except ImportError:
    print("ERROR: iron-librosa not installed / Rust extension not built.")
    sys.exit(1)

if not RUST_AVAILABLE or not hasattr(_rust_ext, "mel_project_f32"):
    print("ERROR: mel_project_f32 not available in the Rust extension.")
    sys.exit(1)

# ── sweep grid ────────────────────────────────────────────────────────────────
# (n_mels, n_fft_bins, n_frames)
#   n_fft_bins  = n_fft // 2 + 1   (typical librosa values)
#   n_frames    = audio_duration * sr / hop_length
SWEEP = []
for n_mels in [32, 64, 80, 128, 160, 256]:
    for n_bins in [257, 513, 1025, 2049, 4097]:       # n_fft: 512,1k,2k,4k,8k
        for n_frames in [100, 300, 500, 800, 1200]:
            SWEEP.append((n_mels, n_bins, n_frames))

N_REPS = 12          # timing reps (excluding 2 warmups)
N_WARMUP = 2


def _min_ms(fn) -> float:
    for _ in range(N_WARMUP):
        fn()
    ts = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def bench_case(n_mels: int, n_bins: int, n_frames: int):
    rng = np.random.default_rng(42)
    S = rng.random((n_bins, n_frames), dtype=np.float32)
    mb = rng.random((n_mels, n_bins), dtype=np.float32)
    S_c = np.ascontiguousarray(S)
    mb_c = np.ascontiguousarray(mb)

    t_np = _min_ms(lambda: mb_c.dot(S_c))
    t_rs = _min_ms(lambda: _rust_ext.mel_project_f32(S_c, mb_c))
    return t_np, t_rs


# ── run sweep ─────────────────────────────────────────────────────────────────
print(f"{'n_mels':>7} {'n_bins':>7} {'n_frames':>8} {'work(M)':>9}"
      f" {'numpy(us)':>10} {'rust(us)':>9} {'winner':>7}")
print("-" * 75)

results: list[tuple[int, float, float, float]] = []   # (work, t_np, t_rs, ratio)

for n_mels, n_bins, n_frames in SWEEP:
    work = n_mels * n_bins * n_frames
    t_np, t_rs = bench_case(n_mels, n_bins, n_frames)
    ratio = t_np / t_rs           # > 1 → Rust wins
    winner = "RUST" if ratio > 1.0 else "numpy"
    print(f"{n_mels:>7} {n_bins:>7} {n_frames:>8} {work/1e6:>9.1f}"
          f" {t_np*1e3:>10.1f} {t_rs*1e3:>9.1f} {winner:>7}  {ratio:.2f}x")
    results.append((work, t_np, t_rs, ratio))

# ── find crossover via log-bucket median ──────────────────────────────────────
# Bucket work values into ~16 log-scale bins.  For each bucket compute the
# *median* speedup ratio.  The threshold is the upper edge of the last bucket
# where the median ratio is still >= 1.0 (Rust is not slower on average).
# This avoids one-off wins at anomalous shapes skewing the threshold.
import math

results.sort(key=lambda r: r[0])

work_vals = [r[0] for r in results]
w_min, w_max = min(work_vals), max(work_vals)
N_BUCKETS = 16
edges = [w_min * (w_max / w_min) ** (i / N_BUCKETS) for i in range(N_BUCKETS + 1)]

bucket_ratios: list[list[float]] = [[] for _ in range(N_BUCKETS)]
for work, _, _, ratio in results:
    # find which bucket
    b = min(N_BUCKETS - 1, int(math.log(work / w_min + 1e-9) /
                                  math.log(w_max / w_min + 1e-9) * N_BUCKETS))
    bucket_ratios[b].append(ratio)

print("\nLog-bucket summary (median speedup; >1.0 = Rust faster):")
print(f"  {'work range (M)':>22}  {'n':>4}  {'median ratio':>13}  winner")
last_rust_bucket_edge = 0.0
for b, ratios in enumerate(bucket_ratios):
    if not ratios:
        continue
    ratios.sort()
    med = ratios[len(ratios) // 2]
    lo = edges[b] / 1e6
    hi = edges[b + 1] / 1e6
    win = "RUST" if med >= 1.0 else "numpy"
    print(f"  {lo:>10.1f} – {hi:>8.1f}  {len(ratios):>4}  {med:>13.3f}x  {win}")
    if med >= 1.0:
        last_rust_bucket_edge = edges[b + 1]

if last_rust_bucket_edge == 0.0:
    threshold = 0
    summary = "Rust never wins on this machine — defaulting to NumPy (threshold=0)"
else:
    threshold = int(last_rust_bucket_edge)
    summary = (
        f"Rust median-wins for work ≤ {last_rust_bucket_edge/1e6:.1f}M  "
        f"→ threshold set to {threshold:,}  "
        f"(≈ {threshold/1e6:.1f}M macs)"
    )

print()
print("=" * 75)
print(f"  {summary}")
print("=" * 75)

# ── patch spectral.py ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--profile",
    default=None,
    help="Profile key for registry update (default: auto-detected <os>-<arch>-<blas>)",
)
parser.add_argument(
    "--registry",
    default=str(Path(__file__).parent / "mel_threshold_registry.json"),
    help="Path to JSON registry file used for cross-CPU thresholds",
)
parser.add_argument(
    "--skip-registry",
    action="store_true",
    help="Skip writing the JSON threshold registry",
)
args = parser.parse_args()

profile_key = args.profile or _default_profile_key()
print(f"Profile key: {profile_key}")

spectral_path = Path(__file__).parent / "librosa" / "feature" / "spectral.py"
if not spectral_path.exists():
    print(f"WARN: spectral.py not found at {spectral_path}; skipping patch.")
    sys.exit(0)

src = spectral_path.read_text(encoding="utf-8")
pattern = re.compile(
    r"(_MEL_RUST_WORK_THRESHOLD\s*=\s*)(\d[\d_]*)"
)
new_src, n_subs = pattern.subn(rf"\g<1>{threshold:_}", src)

if n_subs == 0:
    print("WARN: _MEL_RUST_WORK_THRESHOLD not found in spectral.py; skipping patch.")
elif args.dry_run:
    print(f"\n[dry-run] would write threshold={threshold:_} to {spectral_path}")
else:
    spectral_path.write_text(new_src, encoding="utf-8")
    print(f"\n[OK] Wrote _MEL_RUST_WORK_THRESHOLD = {threshold:_} -> {spectral_path}")

if not args.skip_registry:
    _update_registry(
        registry_path=Path(args.registry),
        profile_key=profile_key,
        threshold=threshold,
        dry_run=args.dry_run,
    )




