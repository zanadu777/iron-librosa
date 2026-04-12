#!/usr/bin/env python
"""A/B chunk-size benchmark harness for Phase 19 GPU FFT dispatch.

This script holds a single CPU baseline and alternates GPU chunk variants
across rounds to reduce drift when comparing chunk settings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

WORKLOADS = [
    "short_512",
    "short_1024",
    "medium_512",
    "medium_1024",
    "long_1024",
]

# Score weights for selecting a preferred chunk profile.
# End-to-end quality is currently more sensitive to iSTFT behavior,
# so iSTFT gets a higher weight than STFT.
W_STFT = 0.4
W_ISTFT = 0.6
REGRESSION_PENALTY = 0.05


def _run_worker(
    script_path: Path,
    *,
    device: str,
    threshold: int | None,
    metal: str,
    repeats: int,
    warmup: int,
    outer_runs: int,
    chunk_size: int | None,
    min_frames: int | None = None,
) -> dict[str, tuple[float, float]]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--device",
        device,
        "--metal",
        metal,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--outer-runs",
        str(outer_runs),
    ]
    if threshold is not None:
        cmd.extend(["--threshold", str(threshold)])

    env = dict(**__import__("os").environ)
    if chunk_size is None:
        env.pop("IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE", None)
    else:
        env["IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE"] = str(chunk_size)
    if min_frames is None:
        env.pop("IRON_LIBROSA_FFT_GPU_MIN_FRAMES", None)
    else:
        env["IRON_LIBROSA_FFT_GPU_MIN_FRAMES"] = str(min_frames)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed (exit={proc.returncode})\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    payload = json.loads(proc.stdout)
    return {
        label: (float(v["stft_ms"]), float(v["istft_ms"]))
        for label, v in payload["workloads"].items()
    }


def _chunk_label(chunk: int | None) -> str:
    return "gpu_default" if chunk is None else f"gpu_chunk_{chunk}"


def _case_metrics(
    *,
    cpu: dict[str, tuple[float, float]],
    by_workload: dict[str, tuple[float, float]],
) -> dict[str, float | int]:
    stft_sp: list[float] = []
    istft_sp: list[float] = []
    stft_reg = 0
    istft_reg = 0
    for w in WORKLOADS:
        cpu_stft, cpu_istft = cpu[w]
        stft_ms, istft_ms = by_workload[w]
        s_stft = cpu_stft / stft_ms
        s_istft = cpu_istft / istft_ms
        stft_sp.append(s_stft)
        istft_sp.append(s_istft)
        if s_stft < 1.0:
            stft_reg += 1
        if s_istft < 1.0:
            istft_reg += 1

    mean_stft = statistics.mean(stft_sp)
    mean_istft = statistics.mean(istft_sp)
    regressions = stft_reg + istft_reg
    score = (W_STFT * mean_stft) + (W_ISTFT * mean_istft) - (REGRESSION_PENALTY * regressions)

    return {
        "mean_stft_speedup": mean_stft,
        "mean_istft_speedup": mean_istft,
        "stft_regressions": stft_reg,
        "istft_regressions": istft_reg,
        "regressions_total": regressions,
        "score": score,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 19 chunk A/B harness")
    ap.add_argument("--chunks", default="default,256", help="Comma list: default or integer chunk sizes")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cpu-outer-runs", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--min-frames", type=int, default=None, help="Value of IRON_LIBROSA_FFT_GPU_MIN_FRAMES for GPU cases")
    args = ap.parse_args()

    if args.rounds <= 0 or args.repeats <= 0 or args.warmup < 0 or args.cpu_outer_runs <= 0:
        raise SystemExit("ERROR: rounds>0, repeats>0, warmup>=0, cpu-outer-runs>0 required")

    chunk_values: list[int | None] = []
    for token in [x.strip().lower() for x in args.chunks.split(",") if x.strip()]:
        if token in {"default", "unset", "none"}:
            chunk_values.append(None)
        else:
            chunk_values.append(int(token))
    if not chunk_values:
        raise SystemExit("ERROR: --chunks must include at least one value")

    script_path = Path(__file__).with_name("benchmark_phase19_auto_thresholds.py")

    cpu = _run_worker(
        script_path,
        device="cpu",
        threshold=None,
        metal="",
        repeats=args.repeats,
        warmup=args.warmup,
        outer_runs=args.cpu_outer_runs,
        chunk_size=None,
        min_frames=None,
    )

    samples: dict[str, dict[str, dict[str, list[float]]]] = {
        _chunk_label(c): {
            w: {"stft": [], "istft": []} for w in WORKLOADS
        }
        for c in chunk_values
    }

    for _ in range(args.rounds):
        for chunk in chunk_values:
            run = _run_worker(
                script_path,
                device="apple-gpu",
                threshold=args.threshold,
                metal="force-on",
                repeats=args.repeats,
                warmup=args.warmup,
                outer_runs=1,
                chunk_size=chunk,
                min_frames=args.min_frames,
            )
            label = _chunk_label(chunk)
            for w in WORKLOADS:
                stft_ms, istft_ms = run[w]
                samples[label][w]["stft"].append(stft_ms)
                samples[label][w]["istft"].append(istft_ms)

    gpu: dict[str, dict[str, tuple[float, float]]] = {}
    for label, by_workload in samples.items():
        gpu[label] = {}
        for w in WORKLOADS:
            stft_med = statistics.median(by_workload[w]["stft"])
            istft_med = statistics.median(by_workload[w]["istft"])
            gpu[label][w] = (stft_med, istft_med)

    summary: dict[str, dict[str, float | int]] = {}
    best_case = None
    best_score = float("-inf")
    for label, by_workload in gpu.items():
        metrics = _case_metrics(cpu=cpu, by_workload=by_workload)
        summary[label] = metrics
        score = float(metrics["score"])
        if score > best_score:
            best_score = score
            best_case = label

    print("case,workload,stft_ms,istft_ms,stft_speedup_vs_cpu,istft_speedup_vs_cpu")
    for w in WORKLOADS:
        cpu_stft, cpu_istft = cpu[w]
        print(f"cpu,{w},{cpu_stft:.3f},{cpu_istft:.3f},1.000,1.000")
    for label in gpu:
        for w in WORKLOADS:
            stft_ms, istft_ms = gpu[label][w]
            cpu_stft, cpu_istft = cpu[w]
            print(
                f"{label},{w},{stft_ms:.3f},{istft_ms:.3f},"
                f"{cpu_stft / stft_ms:.3f},{cpu_istft / istft_ms:.3f}"
            )

    payload = {
        "meta": {
            "rounds": args.rounds,
            "cpu_outer_runs": args.cpu_outer_runs,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "threshold": args.threshold,
            "chunks": ["default" if c is None else c for c in chunk_values],
        },
        "cpu": cpu,
        "gpu": gpu,
        "summary": summary,
        "selection": {
            "weights": {
                "stft": W_STFT,
                "istft": W_ISTFT,
                "regression_penalty": REGRESSION_PENALTY,
            },
            "best_case": best_case,
        },
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    if args.md_out:
        out = Path(args.md_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Phase 19 Chunk A/B Report",
            "",
            f"Rounds: {args.rounds}; CPU outer-runs: {args.cpu_outer_runs}; repeats: {args.repeats}; warmup: {args.warmup}",
            "",
            "## Median GPU vs Fixed CPU",
            "",
            "| Case | Mean STFT speedup | Mean iSTFT speedup | STFT<1.0x | iSTFT<1.0x | Score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for label, by_workload in gpu.items():
            metrics = summary[label]
            lines.append(
                f"| `{label}` | {float(metrics['mean_stft_speedup']):.3f}x | {float(metrics['mean_istft_speedup']):.3f}x | "
                f"{int(metrics['stft_regressions'])} | {int(metrics['istft_regressions'])} | {float(metrics['score']):.3f} |"
            )

        lines.extend(
            [
                "",
                "## Auto Selection",
                "",
                f"Selected case: `{best_case}`",
                "",
                f"Scoring: `{W_STFT:.1f} * mean_stft + {W_ISTFT:.1f} * mean_istft - {REGRESSION_PENALTY:.2f} * regressions_total`",
            ]
        )
        out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

