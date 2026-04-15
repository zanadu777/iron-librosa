#!/usr/bin/env python
"""Comprehensive three-level benchmark report.

Compares:
1. Original Python librosa
2. Rust CPU implementation
3. Rust + CUDA implementation

Usage:
    python -u Benchmarks/scripts/benchmark_comprehensive_three_level.py \\
        --rounds 3 --repeats 3 --warmup 1 \\
        --json-out Benchmarks/results/three_level_benchmark_$(date +%F).json \\
        --html-out Benchmarks/results/three_level_benchmark_$(date +%F).html
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import timeit
from pathlib import Path
from typing import Optional

W_STFT = 0.4
W_ISTFT = 0.6

# Workload matrix
WORKLOADS = [
    ("short_512", 22050, 1.0, 512, 128),
    ("short_1024", 22050, 1.0, 1024, 256),
    ("medium_512", 22050, 5.0, 512, 128),
    ("medium_1024", 22050, 5.0, 1024, 256),
    ("long_1024", 22050, 20.0, 1024, 256),
]


def _time_python_librosa(
    label: str,
    sr: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    """Time original Python librosa."""
    import numpy as np
    import librosa

    n_samples = int(sr * duration)
    rng = np.random.default_rng(seed=42)
    y = rng.standard_normal(n_samples).astype(np.float32)

    # Pre-compute STFT for iSTFT timing
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    def _stft():
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    def _istft():
        librosa.istft(S, hop_length=hop_length)

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


def _time_rust_cpu(
    label: str,
    sr: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    """Time Rust CPU implementation."""
    import numpy as np
    try:
        from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
    except ImportError:
        raise SystemExit("librosa._rust_bridge not importable")

    if not RUST_AVAILABLE or _rust_ext is None:
        raise SystemExit("Rust extension unavailable")

    n_samples = int(sr * duration)
    rng = np.random.default_rng(seed=42)
    y = rng.standard_normal(n_samples).astype(np.float32)

    # Pre-compute STFT for iSTFT timing
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


def _time_rust_cuda(
    label: str,
    sr: int,
    duration: float,
    n_fft: int,
    hop_length: int,
    *,
    repeats: int,
    warmup: int,
) -> tuple[float, float]:
    """Time Rust + CUDA implementation."""
    import numpy as np
    try:
        from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
    except ImportError:
        raise SystemExit("librosa._rust_bridge not importable")

    if not RUST_AVAILABLE or _rust_ext is None:
        raise SystemExit("Rust extension unavailable")

    n_samples = int(sr * duration)
    rng = np.random.default_rng(seed=42)
    y = rng.standard_normal(n_samples).astype(np.float32)

    # Pre-compute STFT for iSTFT timing
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
    implementation: str,
    rounds: int,
    repeats: int,
    warmup: int,
) -> dict[str, tuple[float, float]]:
    """Run all workloads for specified implementation."""
    if implementation == "python":
        time_fn = _time_python_librosa
    elif implementation == "rust-cpu":
        time_fn = _time_rust_cpu
    elif implementation == "rust-cuda":
        time_fn = _time_rust_cuda
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    samples: dict[str, list[tuple[float, float]]] = {label: [] for label, *_ in WORKLOADS}
    for rnd in range(rounds):
        print(f"  Round {rnd + 1}/{rounds} ...", flush=True)
        for label, sr, dur, n_fft, hop in WORKLOADS:
            try:
                stft_ms, istft_ms = time_fn(
                    label, sr, dur, n_fft, hop, repeats=repeats, warmup=warmup
                )
                samples[label].append((stft_ms, istft_ms))
                print(
                    f"    {label:15s}  stft={stft_ms:.2f}ms  istft={istft_ms:.2f}ms",
                    flush=True,
                )
            except Exception as e:
                print(f"    {label:15s}  ERROR: {e}", flush=True)
                # Use previous value if available, otherwise mark as failed
                if samples[label]:
                    samples[label].append(samples[label][-1])
                else:
                    raise

    return {
        label: (
            statistics.median(s for s, _ in times) if times else 0.0,
            statistics.median(i for _, i in times) if times else 0.0,
        )
        for label, times in samples.items()
    }


def _compute_speedups(
    python_times: dict[str, tuple[float, float]],
    rust_cpu_times: dict[str, tuple[float, float]],
    rust_cuda_times: Optional[dict[str, tuple[float, float]]] = None,
) -> dict:
    """Compute speedup metrics for all implementations."""
    results = {
        "workloads": [],
        "summary": {},
    }

    stft_cpu_speedups = []
    istft_cpu_speedups = []
    stft_cuda_speedups = []
    istft_cuda_speedups = []

    for label, *_ in WORKLOADS:
        py_stft, py_istft = python_times[label]
        rc_stft, rc_istft = rust_cpu_times[label]

        # Speedup: original / new (higher = better)
        sp_stft_cpu = py_stft / rc_stft if rc_stft > 0 else 0.0
        sp_istft_cpu = py_istft / rc_istft if rc_istft > 0 else 0.0

        stft_cpu_speedups.append(sp_stft_cpu)
        istft_cpu_speedups.append(sp_istft_cpu)

        row = {
            "workload": label,
            "python_stft_ms": py_stft,
            "python_istft_ms": py_istft,
            "rust_cpu_stft_ms": rc_stft,
            "rust_cpu_istft_ms": rc_istft,
            "rust_cpu_stft_speedup": sp_stft_cpu,
            "rust_cpu_istft_speedup": sp_istft_cpu,
        }

        if rust_cuda_times:
            ru_stft, ru_istft = rust_cuda_times[label]
            sp_stft_cuda = py_stft / ru_stft if ru_stft > 0 else 0.0
            sp_istft_cuda = py_istft / ru_istft if ru_istft > 0 else 0.0

            stft_cuda_speedups.append(sp_stft_cuda)
            istft_cuda_speedups.append(sp_istft_cuda)

            row.update({
                "rust_cuda_stft_ms": ru_stft,
                "rust_cuda_istft_ms": ru_istft,
                "rust_cuda_stft_speedup": sp_stft_cuda,
                "rust_cuda_istft_speedup": sp_istft_cuda,
                "cuda_vs_cpu_stft_speedup": rc_stft / ru_stft if ru_stft > 0 else 0.0,
                "cuda_vs_cpu_istft_speedup": rc_istft / ru_istft if ru_istft > 0 else 0.0,
            })

        results["workloads"].append(row)

    results["summary"] = {
        "mean_rust_cpu_stft_speedup": statistics.mean(stft_cpu_speedups) if stft_cpu_speedups else 0.0,
        "mean_rust_cpu_istft_speedup": statistics.mean(istft_cpu_speedups) if istft_cpu_speedups else 0.0,
        "min_rust_cpu_stft_speedup": min(stft_cpu_speedups) if stft_cpu_speedups else 0.0,
        "min_rust_cpu_istft_speedup": min(istft_cpu_speedups) if istft_cpu_speedups else 0.0,
        "max_rust_cpu_stft_speedup": max(stft_cpu_speedups) if stft_cpu_speedups else 0.0,
        "max_rust_cpu_istft_speedup": max(istft_cpu_speedups) if istft_cpu_speedups else 0.0,
    }

    if stft_cuda_speedups:
        results["summary"].update({
            "mean_rust_cuda_stft_speedup": statistics.mean(stft_cuda_speedups),
            "mean_rust_cuda_istft_speedup": statistics.mean(istft_cuda_speedups),
            "min_rust_cuda_stft_speedup": min(stft_cuda_speedups),
            "min_rust_cuda_istft_speedup": min(istft_cuda_speedups),
            "max_rust_cuda_stft_speedup": max(stft_cuda_speedups),
            "max_rust_cuda_istft_speedup": max(istft_cuda_speedups),
            "cuda_vs_cpu_mean_stft_speedup": statistics.mean(
                rc_stft / (rust_cuda_times[wl][0] or 1.0)
                for wl, rc_stft, _ in [(w, rust_cpu_times[w][0], None) for w, *_ in WORKLOADS]
            ),
            "cuda_vs_cpu_mean_istft_speedup": statistics.mean(
                rc_istft / (rust_cuda_times[wl][1] or 1.0)
                for wl, rc_istft, _ in [(w, rust_cpu_times[w][1], None) for w, *_ in WORKLOADS]
            ),
        })

    return results


def _write_html(
    path: Path,
    python_times: dict[str, tuple[float, float]],
    rust_cpu_times: dict[str, tuple[float, float]],
    rust_cuda_times: Optional[dict[str, tuple[float, float]]],
    speedups: dict,
) -> None:
    """Write comprehensive HTML benchmark report."""
    speedup_rows = speedups["workloads"]
    summary = speedups["summary"]

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Three-Level Benchmark Report: Python vs Rust CPU vs Rust+CUDA</title>",
        "  <style>",
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }",
        "    h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }",
        "    h2 { color: #555; margin-top: 30px; }",
        "    .metric { background: white; padding: 15px; margin: 10px 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "    .metric-title { font-weight: bold; color: #333; }",
        "    .metric-value { font-size: 1.5em; color: #007bff; font-weight: bold; }",
        "    table { border-collapse: collapse; width: 100%; background: white; margin: 15px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "    th { background: #007bff; color: white; padding: 12px; text-align: left; font-weight: bold; }",
        "    td { padding: 10px 12px; border-bottom: 1px solid #ddd; }",
        "    tr:hover { background: #f9f9f9; }",
        "    .highlight-good { background: #d4edda; color: #155724; font-weight: bold; }",
        "    .highlight-warn { background: #fff3cd; color: #856404; font-weight: bold; }",
        "    .highlight-poor { background: #f8d7da; color: #721c24; font-weight: bold; }",
        "    .container { max-width: 1200px; margin: 0 auto; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <div class='container'>",
        "    <h1>Benchmark Report - Three Levels</h1>",
        "    <p>Comprehensive performance comparison: Original Python librosa -> Rust CPU -> Rust+CUDA</p>",
    ]

    # Summary metrics
    html_parts.extend([
        "    <h2>Summary Metrics</h2>",
        "    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;'>",
    ])

    metrics = [
        ("Mean Rust CPU STFT Speedup", f"{summary['mean_rust_cpu_stft_speedup']:.2f}x"),
        ("Mean Rust CPU iSTFT Speedup", f"{summary['mean_rust_cpu_istft_speedup']:.2f}x"),
    ]

    if "mean_rust_cuda_stft_speedup" in summary:
        metrics.extend([
            ("Mean Rust+CUDA STFT Speedup", f"{summary['mean_rust_cuda_stft_speedup']:.2f}x"),
            ("Mean Rust+CUDA iSTFT Speedup", f"{summary['mean_rust_cuda_istft_speedup']:.2f}x"),
        ])

    for title, value in metrics:
        html_parts.append(f"      <div class='metric'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>")

    html_parts.append("    </div>")

    # Detailed table
    html_parts.extend([
        "    <h2>Detailed Workload Results</h2>",
        "    <table>",
        "      <thead>",
        "        <tr>",
        "          <th>Workload</th>",
        "          <th colspan='2'>Python librosa (ms)</th>",
        "          <th colspan='2'>Rust CPU (ms)</th>",
        "          <th colspan='2'>Rust CPU vs Python</th>",
    ])

    if rust_cuda_times:
        html_parts.extend([
            "          <th colspan='2'>Rust+CUDA (ms)</th>",
            "          <th colspan='2'>Rust+CUDA vs Python</th>",
            "          <th colspan='2'>CUDA vs CPU</th>",
        ])

    html_parts.extend([
        "        </tr>",
        "        <tr>",
        "          <th></th>",
        "          <th>STFT</th>",
        "          <th>iSTFT</th>",
        "          <th>STFT</th>",
        "          <th>iSTFT</th>",
        "          <th>STFT</th>",
        "          <th>iSTFT</th>",
    ])

    if rust_cuda_times:
        html_parts.extend([
            "          <th>STFT</th>",
            "          <th>iSTFT</th>",
            "          <th>STFT</th>",
            "          <th>iSTFT</th>",
            "          <th>STFT</th>",
            "          <th>iSTFT</th>",
        ])

    html_parts.extend([
        "        </tr>",
        "      </thead>",
        "      <tbody>",
    ])

    for row in speedup_rows:
        workload = row["workload"]
        py_stft = row["python_stft_ms"]
        py_istft = row["python_istft_ms"]
        rc_stft = row["rust_cpu_stft_ms"]
        rc_istft = row["rust_cpu_istft_ms"]
        sp_stft_cpu = row["rust_cpu_stft_speedup"]
        sp_istft_cpu = row["rust_cpu_istft_speedup"]

        def color_class(speedup):
            if speedup >= 2.0:
                return "highlight-good"
            elif speedup >= 1.0:
                return "highlight-good"
            elif speedup >= 0.9:
                return "highlight-warn"
            else:
                return "highlight-poor"

        html_line = f"        <tr><td><strong>{workload}</strong></td>"
        html_line += f"<td>{py_stft:.3f}</td><td>{py_istft:.3f}</td>"
        html_line += f"<td>{rc_stft:.3f}</td><td>{rc_istft:.3f}</td>"
        html_line += f"<td class='{color_class(sp_stft_cpu)}'>{sp_stft_cpu:.2f}x</td>"
        html_line += f"<td class='{color_class(sp_istft_cpu)}'>{sp_istft_cpu:.2f}x</td>"

        if "rust_cuda_stft_ms" in row:
            ru_stft = row["rust_cuda_stft_ms"]
            ru_istft = row["rust_cuda_istft_ms"]
            sp_stft_cuda = row["rust_cuda_stft_speedup"]
            sp_istft_cuda = row["rust_cuda_istft_speedup"]
            cu_stft = row["cuda_vs_cpu_stft_speedup"]
            cu_istft = row["cuda_vs_cpu_istft_speedup"]

            html_line += f"<td>{ru_stft:.3f}</td><td>{ru_istft:.3f}</td>"
            html_line += f"<td class='{color_class(sp_stft_cuda)}'>{sp_stft_cuda:.2f}x</td>"
            html_line += f"<td class='{color_class(sp_istft_cuda)}'>{sp_istft_cuda:.2f}x</td>"
            html_line += f"<td class='{color_class(cu_stft)}'>{cu_stft:.2f}x</td>"
            html_line += f"<td class='{color_class(cu_istft)}'>{cu_istft:.2f}x</td>"

        html_line += "</tr>"
        html_parts.append(html_line)

    html_parts.extend([
        "      </tbody>",
        "    </table>",
        "    <hr>",
        "    <p style='text-align: center; color: #666; margin-top: 30px;'>",
        f"      Generated: {Path.cwd()} | Report covers {len(WORKLOADS)} workloads",
        "    </p>",
        "  </div>",
        "</body>",
        "</html>",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(html_parts), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-level benchmark report")
    ap.add_argument("--rounds", type=int, default=3, help="Measurement rounds per implementation")
    ap.add_argument("--repeats", type=int, default=3, help="Timed iterations per round")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    ap.add_argument("--skip-python", action="store_true", help="Skip Python librosa benchmark")
    ap.add_argument("--skip-cuda", action="store_true", help="Skip Rust+CUDA benchmark")
    ap.add_argument("--json-out", default=None, help="Write results JSON")
    ap.add_argument("--html-out", default=None, help="Write results HTML report")
    ap.add_argument("--md-out", default=None, help="Write results markdown")
    args = ap.parse_args()

    print("=" * 70, flush=True)
    print("THREE-LEVEL BENCHMARK: Python librosa -> Rust CPU -> Rust+CUDA", flush=True)
    print("=" * 70, flush=True)

    python_times = None
    rust_cpu_times = None
    rust_cuda_times = None

    if not args.skip_python:
        print("\n[1/3] Benchmarking Original Python librosa...", flush=True)
        try:
            python_times = _run_all(
                implementation="python",
                rounds=args.rounds,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            print("✓ Python librosa benchmark complete", flush=True)
        except Exception as e:
            print(f"✗ Python librosa benchmark failed: {e}", flush=True)
            python_times = None

    print("\n[2/3] Benchmarking Rust CPU...", flush=True)
    try:
        rust_cpu_times = _run_all(
            implementation="rust-cpu",
            rounds=args.rounds,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        print("✓ Rust CPU benchmark complete", flush=True)
    except Exception as e:
        print(f"✗ Rust CPU benchmark failed: {e}", flush=True)
        raise

    if not args.skip_cuda:
        print("\n[3/3] Benchmarking Rust+CUDA...", flush=True)
        os.environ["IRON_LIBROSA_RUST_DEVICE"] = "cuda-gpu"
        os.environ["IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL"] = "force-on"
        try:
            rust_cuda_times = _run_all(
                implementation="rust-cuda",
                rounds=args.rounds,
                repeats=args.repeats,
                warmup=args.warmup,
            )
            print("✓ Rust+CUDA benchmark complete", flush=True)
        except Exception as e:
            print(f"✗ Rust+CUDA benchmark failed (optional): {e}", flush=True)
            rust_cuda_times = None
        finally:
            if "IRON_LIBROSA_RUST_DEVICE" in os.environ:
                del os.environ["IRON_LIBROSA_RUST_DEVICE"]
            if "IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL" in os.environ:
                del os.environ["IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL"]

    print("\n" + "=" * 70, flush=True)

    # Compute speedups
    if python_times and rust_cpu_times:
        speedups = _compute_speedups(python_times, rust_cpu_times, rust_cuda_times)

        # Output JSON
        if args.json_out:
            payload = {
                "meta": {
                    "rounds": args.rounds,
                    "repeats": args.repeats,
                    "warmup": args.warmup,
                    "workloads": [label for label, *_ in WORKLOADS],
                },
                "timings": {
                    "python_librosa": {label: list(vals) for label, vals in (python_times or {}).items()},
                    "rust_cpu": {label: list(vals) for label, vals in (rust_cpu_times or {}).items()},
                    "rust_cuda": {label: list(vals) for label, vals in (rust_cuda_times or {}).items()} if rust_cuda_times else {},
                },
                "speedups": speedups,
            }
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"✓ JSON report: {out_path}", flush=True)

        # Output HTML
        if args.html_out:
            out_path = Path(args.html_out)
            _write_html(out_path, python_times, rust_cpu_times, rust_cuda_times, speedups)
            print(f"✓ HTML report: {out_path}", flush=True)

        # Print summary
        print("\nSUMMARY:")
        print(f"  Mean Rust CPU STFT speedup:  {speedups['summary']['mean_rust_cpu_stft_speedup']:.2f}x")
        print(f"  Mean Rust CPU iSTFT speedup: {speedups['summary']['mean_rust_cpu_istft_speedup']:.2f}x")
        if rust_cuda_times:
            print(f"  Mean Rust+CUDA STFT speedup:  {speedups['summary'].get('mean_rust_cuda_stft_speedup', 'N/A')}")
            print(f"  Mean Rust+CUDA iSTFT speedup: {speedups['summary'].get('mean_rust_cuda_istft_speedup', 'N/A')}")
        print()
    else:
        print("ERROR: Could not compute speedups (missing baseline data)")


if __name__ == "__main__":
    main()






