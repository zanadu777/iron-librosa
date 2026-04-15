#!/usr/bin/env python
"""Phase 21 CUDA baseline benchmark.

Captures CPU timing on the standard Phase 19/20 workload matrix so that
CUDA results can be compared against a stable, reproducible baseline once
the cuFFT FFI implementation is active on PC.

This script is intentionally CPU-only. Run it on the target PC before enabling
CUDA to establish the comparison baseline stored in Benchmarks/results/.

Usage (PC, before CUDA is wired):
    python -u Benchmarks/scripts/benchmark_phase21_cuda_baseline.py \\
        --rounds 5 --repeats 5 --warmup 2 \\
        --json-out Benchmarks/results/phase21_cuda_cpu_baseline_$(date +%F).json \\
        --md-out  Benchmarks/results/phase21_cuda_cpu_baseline_$(date +%F).md

Usage (compare after CUDA active, drop-in auto mode):
    python -u Benchmarks/scripts/benchmark_phase21_cuda_baseline.py \\
        --rounds 5 --repeats 5 --warmup 2 --device auto \
        --json-out Benchmarks/results/phase21_cuda_gpu_$(date +%F).json \\
        --baseline-json Benchmarks/results/phase21_cuda_cpu_baseline_<date>.json

Score weights mirror Phase 19/20 (iSTFT 0.6, STFT 0.4, regression penalty 0.05).
Promotion gate: composite score >= 0.887, no workload <1.0x on two consecutive runs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import timeit
from pathlib import Path

W_STFT = 0.4
W_ISTFT = 0.6
REGRESSION_PENALTY = 0.05
LARGE_WORKLOAD_SPEEDUP_TARGET = 1.0
SPEEDUP_NOISE_TOLERANCE = 0.01
PROMOTION_SCORE_TARGET = 0.887
OPT_IN_SCORE_TARGET = 0.82
GATE_POLICY_VERSION = "phase21-cud007-v1"

# Same workload matrix used in Phase 19/20 benchmarks.
# Each entry: (label, sr, duration_sec, n_fft, hop_length)
WORKLOADS = [
    ("short_512",   22050, 1.0,  512,  128),
    ("short_1024",  22050, 1.0, 1024,  256),
    ("medium_512",  22050, 5.0,  512,  128),
    ("medium_1024", 22050, 5.0, 1024,  256),
    ("long_1024",   22050, 20.0, 1024, 256),
]

LARGE_WORKLOADS = {"medium_512", "medium_1024", "long_1024"}


def _time_workload(
    label: str,
    sr: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    """Returns (stft_ms, istft_ms) median over `repeats` timed runs."""
    import numpy as np
    try:
        from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
    except ImportError:
        raise SystemExit("librosa._rust_bridge not importable — activate env first.")

    if not RUST_AVAILABLE or _rust_ext is None:
        raise SystemExit("Rust extension unavailable — build with maturin develop first.")

    n_samples = int(sr * duration)
    rng = np.random.default_rng(seed=42)
    y = rng.standard_normal(n_samples).astype(np.float32)

    # Pre-compute STFT for iSTFT timing (don't count STFT time in iSTFT measurement).
    S = _rust_ext.stft_complex(y, n_fft, hop_length, True, None)

    def _stft():
        _rust_ext.stft_complex(y, n_fft, hop_length, True, None)

    def _istft():
        _rust_ext.istft_f32(S, n_fft, hop_length, None, None)

    # Warmup
    for _ in range(warmup):
        _stft()
        _istft()

    stft_times = []
    istft_times = []
    for _ in range(repeats):
        t0 = timeit.default_timer()
        _stft()
        stft_times.append((timeit.default_timer() - t0) * 1000.0)

        t0 = timeit.default_timer()
        _istft()
        istft_times.append((timeit.default_timer() - t0) * 1000.0)

    return statistics.median(stft_times), statistics.median(istft_times)


def _run_all(
    *,
    rounds: int,
    repeats: int,
    warmup: int,
) -> dict[str, tuple[float, float]]:
    """Run all workloads over `rounds` passes and return median-of-medians."""
    samples: dict[str, list[tuple[float, float]]] = {label: [] for label, *_ in WORKLOADS}
    for rnd in range(rounds):
        print(f"  Round {rnd + 1}/{rounds} ...", flush=True)
        for label, sr, dur, n_fft, hop in WORKLOADS:
            stft_ms, istft_ms = _time_workload(
                label, sr, dur, n_fft, hop, repeats=repeats, warmup=warmup
            )
            samples[label].append((stft_ms, istft_ms))
            print(
                f"    {label:15s}  stft={stft_ms:.2f}ms  istft={istft_ms:.2f}ms",
                flush=True,
            )

    return {
        label: (
            statistics.median(s for s, _ in times),
            statistics.median(i for _, i in times),
        )
        for label, times in samples.items()
    }


def _compare_with_baseline(
    current: dict[str, tuple[float, float]],
    baseline: dict[str, tuple[float, float]],
) -> dict:
    stft_speedups, istft_speedups = [], []
    stft_reg, istft_reg = 0, 0
    stft_near, istft_near = 0, 0
    regression_floor = 1.0 - SPEEDUP_NOISE_TOLERANCE
    rows = []
    for label, *_ in WORKLOADS:
        cpu_stft, cpu_istft = baseline[label]
        cur_stft, cur_istft = current[label]
        sp_stft = cpu_stft / cur_stft
        sp_istft = cpu_istft / cur_istft
        stft_speedups.append(sp_stft)
        istft_speedups.append(sp_istft)
        if sp_stft < regression_floor:
            stft_reg += 1
        elif sp_stft < 1.0:
            stft_near += 1
        if sp_istft < regression_floor:
            istft_reg += 1
        elif sp_istft < 1.0:
            istft_near += 1
        rows.append({
            "workload": label,
            "baseline_stft_ms": cpu_stft,
            "baseline_istft_ms": cpu_istft,
            "current_stft_ms": cur_stft,
            "current_istft_ms": cur_istft,
            "stft_speedup": sp_stft,
            "istft_speedup": sp_istft,
        })

    mean_stft = statistics.mean(stft_speedups)
    mean_istft = statistics.mean(istft_speedups)
    large_rows = [row for row in rows if row["workload"] in LARGE_WORKLOADS]
    mean_large_stft = statistics.mean(r["stft_speedup"] for r in large_rows)
    mean_large_istft = statistics.mean(r["istft_speedup"] for r in large_rows)
    large_gate_floor = LARGE_WORKLOAD_SPEEDUP_TARGET - SPEEDUP_NOISE_TOLERANCE
    large_gate = (
        mean_large_stft >= large_gate_floor
        and mean_large_istft >= large_gate_floor
    )
    regressions = stft_reg + istft_reg
    score = W_STFT * mean_stft + W_ISTFT * mean_istft - REGRESSION_PENALTY * regressions
    score_pass = score >= PROMOTION_SCORE_TARGET
    regression_gate = regressions == 0

    return {
        "workloads": rows,
        "summary": {
            "mean_stft_speedup": mean_stft,
            "mean_istft_speedup": mean_istft,
            "mean_large_stft_speedup": mean_large_stft,
            "mean_large_istft_speedup": mean_large_istft,
            "stft_regressions": stft_reg,
            "istft_regressions": istft_reg,
            "stft_near_parity": stft_near,
            "istft_near_parity": istft_near,
            "regressions_total": regressions,
            "score": score,
        },
        "promotion_gate": {
            "policy_version": GATE_POLICY_VERSION,
            "score_target": PROMOTION_SCORE_TARGET,
            "opt_in_score_target": OPT_IN_SCORE_TARGET,
            "large_workload_speedup_target": LARGE_WORKLOAD_SPEEDUP_TARGET,
            "speedup_noise_tolerance": SPEEDUP_NOISE_TOLERANCE,
            "regression_floor": regression_floor,
            "large_workload_floor": large_gate_floor,
            "score_pass": score_pass,
            "regression_gate": regression_gate,
            "large_workload_gate": large_gate,
            "decision": "PROMOTE" if score_pass and regression_gate and large_gate
                        else "OPT-IN" if score >= OPT_IN_SCORE_TARGET
                        else "DEFER",
        },
    }


def _write_md(path: Path, results: dict, *, device: str, compared: bool) -> None:
    lines = [
        "# Phase 21 CUDA Baseline Benchmark",
        "",
        f"Device: `{device}`",
        f"Mode: {'CPU vs CUDA comparison' if compared else 'CPU baseline capture'}",
        "",
        "## Workload Timings",
        "",
        "| Workload | STFT ms | iSTFT ms |",
        "|---|---:|---:|",
    ]
    for label, (stft_ms, istft_ms) in results["timings"].items():
        lines.append(f"| {label} | {stft_ms:.2f} | {istft_ms:.2f} |")

    if compared and "comparison" in results:
        cmp = results["comparison"]
        lines += [
            "",
            "## Comparison vs CPU Baseline",
            "",
            "| Workload | CPU STFT | CPU iSTFT | Current STFT | Current iSTFT | STFT speedup | iSTFT speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in cmp["workloads"]:
            lines.append(
                f"| {row['workload']} | {row['baseline_stft_ms']:.2f} | {row['baseline_istft_ms']:.2f} | "
                f"{row['current_stft_ms']:.2f} | {row['current_istft_ms']:.2f} | "
                f"{row['stft_speedup']:.3f}x | {row['istft_speedup']:.3f}x |"
            )
        s = cmp["summary"]
        p = cmp["promotion_gate"]
        lines += [
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|---|---:|",
            f"| Mean STFT speedup | {s['mean_stft_speedup']:.3f}x |",
            f"| Mean iSTFT speedup | {s['mean_istft_speedup']:.3f}x |",
            f"| Mean LARGE STFT speedup | {s['mean_large_stft_speedup']:.3f}x |",
            f"| Mean LARGE iSTFT speedup | {s['mean_large_istft_speedup']:.3f}x |",
            f"| STFT regressions | {s['stft_regressions']} |",
            f"| iSTFT regressions | {s['istft_regressions']} |",
            f"| STFT near-parity (<1.0x, >= floor) | {s['stft_near_parity']} |",
            f"| iSTFT near-parity (<1.0x, >= floor) | {s['istft_near_parity']} |",
            f"| Composite score | {s['score']:.3f} |",
            f"| Score target | {p['score_target']} |",
            f"| Opt-in score target | {p['opt_in_score_target']} |",
            f"| Large-workload speedup target | {p['large_workload_speedup_target']:.3f}x |",
            f"| Speedup noise tolerance | {p['speedup_noise_tolerance']:.3f} |",
            f"| Regression floor | {p['regression_floor']:.3f}x |",
            f"| Large-workload floor | {p['large_workload_floor']:.3f}x |",
            f"| Large-workload gate | {p['large_workload_gate']} |",
            f"| **Decision** | **{p['decision']}** |",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 21 CUDA baseline / comparison benchmark")
    ap.add_argument("--rounds", type=int, default=5, help="Repeated measurement rounds")
    ap.add_argument("--repeats", type=int, default=5, help="Timed iterations per round")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing")
    ap.add_argument(
        "--device", default="cpu",
        help="Device to benchmark: cpu | auto | cuda-gpu. Default: cpu.",
    )
    ap.add_argument(
        "--cuda-experimental", default="",
        help="Debug-only: set IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL (e.g. force-on)",
    )
    ap.add_argument("--json-out", default=None, help="Write results JSON here")
    ap.add_argument("--md-out", default=None, help="Write results markdown here")
    ap.add_argument(
        "--baseline-json", default=None,
        help="Existing CPU baseline JSON to compare against (optional)",
    )
    args = ap.parse_args()

    if args.device != "cpu":
        os.environ["IRON_LIBROSA_RUST_DEVICE"] = args.device
    if args.cuda_experimental:
        os.environ["IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL"] = args.cuda_experimental

    print(f"Phase 21 CUDA Benchmark — device={args.device}", flush=True)
    print(f"Rounds={args.rounds}  Repeats={args.repeats}  Warmup={args.warmup}", flush=True)

    timings = _run_all(rounds=args.rounds, repeats=args.repeats, warmup=args.warmup)

    backend_info = None
    try:
        from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext

        if RUST_AVAILABLE and _rust_ext is not None:
            backend_info = _rust_ext.rust_backend_info()
    except Exception:
        backend_info = None

    payload: dict = {
        "meta": {
            "phase": 21,
            "device": args.device,
            "cuda_experimental": args.cuda_experimental or None,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "workloads": [label for label, *_ in WORKLOADS],
            "score_weights": {"stft": W_STFT, "istft": W_ISTFT, "regression_penalty": REGRESSION_PENALTY},
            "gate_policy": {
                "version": GATE_POLICY_VERSION,
                "score_target": PROMOTION_SCORE_TARGET,
                "opt_in_score_target": OPT_IN_SCORE_TARGET,
                "large_workload_speedup_target": LARGE_WORKLOAD_SPEEDUP_TARGET,
                "speedup_noise_tolerance": SPEEDUP_NOISE_TOLERANCE,
            },
        },
        "timings": {label: list(vals) for label, vals in timings.items()},
    }
    if backend_info is not None:
        payload["backend_info"] = backend_info

    compared = False
    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if not baseline_path.exists():
            print(f"WARNING: baseline JSON not found: {baseline_path}", flush=True)
        else:
            baseline_data = json.loads(baseline_path.read_text())
            baseline_timings = {
                label: tuple(vals)
                for label, vals in baseline_data["timings"].items()
            }
            comparison = _compare_with_baseline(timings, baseline_timings)
            payload["comparison"] = comparison
            compared = True

            s = comparison["summary"]
            p = comparison["promotion_gate"]
            print("\n=== Comparison vs CPU Baseline ===", flush=True)
            print(
                f"  Mean STFT speedup:  {s['mean_stft_speedup']:.3f}x\n"
                f"  Mean iSTFT speedup: {s['mean_istft_speedup']:.3f}x\n"
                f"  Mean LARGE STFT speedup:  {s['mean_large_stft_speedup']:.3f}x\n"
                f"  Mean LARGE iSTFT speedup: {s['mean_large_istft_speedup']:.3f}x\n"
                f"  Composite score:    {s['score']:.3f}  (target >= {p['score_target']})\n"
                f"  Large gate:         {p['large_workload_gate']}  (target >= {p['large_workload_speedup_target']:.3f}x)\n"
                f"  Decision:           {p['decision']}",
                flush=True,
            )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON written: {out}", flush=True)

    if args.md_out:
        out = Path(args.md_out)
        _write_md(out, payload, device=args.device, compared=compared)
        print(f"Markdown written: {out}", flush=True)


if __name__ == "__main__":
    main()

