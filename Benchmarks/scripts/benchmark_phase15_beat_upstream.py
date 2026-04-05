"""Quick benchmark to compare numpy baseline vs Phase 15 Rust."""
import argparse
import json
import os
import platform
import time

# Determine dispatch mode BEFORE importing librosa
# If not explicitly set, default to numpy baseline for fair testing
rust_active = os.environ.get('IRON_LIBROSA_RUST_DISPATCH', '0') == '1'
if not rust_active:
    # Ensure numpy baseline by disabling Rust dispatch
    os.environ['IRON_LIBROSA_RUST_DISPATCH'] = '0'
else:
    # Explicitly enable Rust dispatch
    os.environ['IRON_LIBROSA_RUST_DISPATCH'] = '1'

import librosa
import numpy as np

from benchmark_guard import (
    REVIEW_SPEEDUP_THRESHOLD,
    assert_benchmark_payload_schema,
    evaluate_speedup,
)

SR, HOP = 22050, 512


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 15 beat upstream benchmark")
    parser.add_argument(
        "--compare-with",
        type=str,
        default="Benchmarks/results/phase15_bench_numpy_baseline.json",
        help="Optional baseline JSON to compare against for speedup/review flags.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=REVIEW_SPEEDUP_THRESHOLD,
        help="Speedup threshold below which a case requires manual review.",
    )
    return parser.parse_args()

def mk(sec, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, sec, sec*SR, dtype=np.float32)
    y = 0.5*np.sin(2*np.pi*110*t) + 0.3*np.sin(2*np.pi*220*t)
    return y + 0.2*rng.standard_normal(sec*SR).astype(np.float32)

def bench(y, n=5):
    for _ in range(2):
        librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
        times.append((time.perf_counter()-t0)*1e3)
    return float(np.mean(times)), float(np.min(times))

args = _parse_args()

from librosa._rust_bridge import _rust_ext
print(f"RUST_AVAILABLE = {librosa.onset.RUST_AVAILABLE}")
print(f"onset_flux_median_ref_f32 = {hasattr(_rust_ext, 'onset_flux_median_ref_f32')}")
print(f"tempogram_ac_f32          = {hasattr(_rust_ext, 'tempogram_ac_f32')}")
print()

rows = []
for name, sec, seed in [('noisy_30s', 30, 8801), ('noisy_120s', 120, 8802)]:
    y = mk(sec, seed)
    avg, mn = bench(y)
    rows.append({'case': name, 'avg_ms': avg, 'min_ms': mn})
    print(f"  {name}: avg={avg:.1f}ms  min={mn:.1f}ms")

backend = 'phase15_rust' if rust_active else 'numpy_baseline'

baseline_rows = {}
if rust_active and os.path.exists(args.compare_with):
    try:
        with open(args.compare_with, "r", encoding="utf-8") as fdesc:
            base_payload = json.load(fdesc)
            for row in base_payload.get("rows", []):
                baseline_rows[row["case"]] = float(row["avg_ms"])
    except Exception:
        baseline_rows = {}

auto_review_cases = []
for row in rows:
    base = baseline_rows.get(row["case"])
    if base is None:
        row["speedup_vs_baseline"] = None
        row["review_required"] = None
        continue
    speedup = base / row["avg_ms"] if row["avg_ms"] > 0 else 0.0
    review = evaluate_speedup(speedup, args.review_threshold)
    row["speedup_vs_baseline"] = review["speedup"]
    row["review_required"] = review["review_required"]
    if review["review_required"]:
        auto_review_cases.append(f"{row['case']} ({speedup:.2f}x)")

if auto_review_cases:
    print("auto-review required (< threshold):")
    for case in auto_review_cases:
        print(f"  - {case}")

payload = {
    'meta': {
        'backend': backend,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'review_threshold': args.review_threshold,
    },
    'auto_review_cases': auto_review_cases,
    'rows': rows,
}
assert_benchmark_payload_schema(payload, "phase15_beat_upstream")
fname = f'Benchmarks/results/phase15_bench_{backend}.json'
with open(fname, 'w') as f:
    json.dump(payload, f, indent=2)
print(f"saved {fname}")

