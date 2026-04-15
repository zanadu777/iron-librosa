"""
CUDA GPU Overhead Diagnostic Benchmark
=======================================
Runs iron-librosa STFT with IRON_LIBROSA_CUDA_PROFILE=1 and
IRON_LIBROSA_FFT_TIMING=1 enabled, collects stderr profile lines, and
produces a per-workload overhead budget table showing exactly where time
goes (H2D transfer, kernel compute, D2H+sync, plan creation).

Usage
-----
Build the wheel first:
    maturin develop --release

Then run (CUDA device must be present):
    python Benchmarks/scripts/benchmark_cuda_diagnostics.py

Environment knobs forwarded automatically:
    IRON_LIBROSA_RUST_DEVICE=cuda-gpu
    IRON_LIBROSA_CUDA_PROFILE=1
    IRON_LIBROSA_FFT_TIMING=1
    IRON_LIBROSA_CUDA_DEBUG=1  (optional, set VERBOSE=1 in env)
"""

from __future__ import annotations
import os
import re
import sys
import time
import statistics
import subprocess
import textwrap
import numpy as np

# ── env setup ────────────────────────────────────────────────────────────────
os.environ["IRON_LIBROSA_RUST_DEVICE"] = "cuda-gpu"
os.environ["IRON_LIBROSA_CUDA_PROFILE"] = "1"
os.environ["IRON_LIBROSA_FFT_TIMING"] = "1"
if os.environ.get("VERBOSE"):
    os.environ["IRON_LIBROSA_CUDA_DEBUG"] = "1"

try:
    from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
except ImportError:
    sys.exit("librosa Rust bridge not found — run `maturin develop --release` first.")

if not RUST_AVAILABLE or _rust_ext is None:
    sys.exit("Rust extension unavailable — ensure `librosa._rust` is built and dispatch is enabled.")

# ── workload definitions ──────────────────────────────────────────────────────
SAMPLE_RATE = 22050

WORKLOADS = [
    # name,           duration_s, n_fft, hop_length
    ("short_512",     1.0,        512,   128),
    ("short_1024",    1.0,        1024,  256),
    ("medium_512",    5.0,        512,   128),
    ("medium_1024",   5.0,        1024,  256),
    ("long_1024",    20.0,        1024,  256),
    ("long_2048",    20.0,        2048,  512),
]

N_WARMUP = 2
N_RUNS   = 5

# ── profile-line parser ───────────────────────────────────────────────────────
# Format emitted by Rust:
# [CUDA_PROFILE] op=r2c n=1024 frames=860 h2d_kb=3380.0 kernel_ms=0.012
#                d2h_kb=6762.5 sync_ms=0.543 total_ms=0.612 plan_created=false
_PROFILE_RE = re.compile(
    r"\[CUDA_PROFILE\]\s+"
    r"op=(\S+)\s+"
    r"n=(\d+)\s+"
    r"frames=(\d+)\s+"
    r"h2d_kb=([\d.]+)\s+"
    r"kernel_ms=([\d.]+)\s+"
    r"d2h_kb=([\d.]+)\s+"
    r"sync_ms=([\d.]+)\s+"
    r"total_ms=([\d.]+)\s+"
    r"plan_created=(\S+)"
)

# Format from stft_complex timing:
# [iron-librosa][stft_complex] mode=gpu ... total_ms=0.612 gpu_ms=0.590
_STFT_TIMING_RE = re.compile(
    r"\[iron-librosa\]\[stft_complex\].*?total_ms=([\d.]+).*?gpu_ms=([\d.]+)"
)

def parse_profile_lines(text: str) -> list[dict]:
    rows = []
    for m in _PROFILE_RE.finditer(text):
        rows.append(dict(
            op=m.group(1),
            n=int(m.group(2)),
            frames=int(m.group(3)),
            h2d_kb=float(m.group(4)),
            kernel_ms=float(m.group(5)),
            d2h_kb=float(m.group(6)),
            sync_ms=float(m.group(7)),
            total_ms=float(m.group(8)),
            plan_created=(m.group(9).lower() == "true"),
        ))
    return rows

def parse_stft_timing(text: str) -> tuple[float, float]:
    """Return (total_ms, gpu_ms) from the last stft_complex timing line."""
    matches = list(_STFT_TIMING_RE.finditer(text))
    if matches:
        m = matches[-1]
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0

# ── redirect stderr capture hack ─────────────────────────────────────────────
# Rust writes profile lines to stderr (fd 2).  We use a subprocess that
# executes just the benchmark call so we can capture fd 2 cleanly.

INNER_SCRIPT = textwrap.dedent("""\
import os, sys, numpy as np
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext

if (not RUST_AVAILABLE) or (_rust_ext is None):
    raise RuntimeError("Rust extension unavailable in subprocess")

y      = np.random.default_rng(42).random(int({dur} * 22050), dtype=np.float32)
n_fft  = {n_fft}
hop    = {hop}
nwarm  = {nwarm}
nrun   = {nrun}

for _ in range(nwarm):
    _rust_ext.stft_complex(y, n_fft=n_fft, hop_length=hop)

import time
times = []
for _ in range(nrun):
    t0 = time.perf_counter()
    _rust_ext.stft_complex(y, n_fft=n_fft, hop_length=hop)
    times.append((time.perf_counter() - t0) * 1000.0)

print(f"WALL_MS " + " ".join(f"{{t:.4f}}" for t in times))
""")

def run_workload(name, dur, n_fft, hop) -> dict:
    """Execute the workload in a subprocess and parse profile output."""
    script = INNER_SCRIPT.format(
        dur=dur, n_fft=n_fft, hop=hop, nwarm=N_WARMUP, nrun=N_RUNS
    )
    env = {**os.environ}  # inherit all env including CUDA_PROFILE etc.

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=False, env=env, timeout=120,
    )

    # stderr = Rust profile + timing lines
    stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")

    wall_times = []
    for line in stdout.splitlines():
        if line.startswith("WALL_MS "):
            wall_times = [float(x) for x in line.split()[1:]]

    profiles = parse_profile_lines(stderr)
    stft_total, stft_gpu = parse_stft_timing(stderr)

    # Aggregate profile rows (skip warmup runs — plan_created=true)
    warm_profiles = [p for p in profiles if not p["plan_created"]]

    def avg(key):
        vals = [p[key] for p in warm_profiles]
        return statistics.mean(vals) if vals else 0.0

    return dict(
        name=name,
        n_fft=n_fft,
        hop=hop,
        n_frames=warm_profiles[0]["frames"] if warm_profiles else 0,
        op=warm_profiles[0]["op"] if warm_profiles else "?",
        h2d_kb=avg("h2d_kb"),
        kernel_ms=avg("kernel_ms"),
        d2h_kb=avg("d2h_kb"),
        sync_ms=avg("sync_ms"),
        total_ms=avg("total_ms"),
        wall_ms_median=statistics.median(wall_times) if wall_times else 0.0,
        wall_ms_min=min(wall_times) if wall_times else 0.0,
        proc_returncode=result.returncode,
        plan_creates=sum(1 for p in profiles if p["plan_created"]),
        n_profile_rows=len(profiles),
        stdout_snippet=stdout[-400:] if stdout else "(none)",
        stderr_snippet=stderr[-800:] if stderr else "(none)",
    )

# ── CPU baseline ──────────────────────────────────────────────────────────────
def cpu_baseline(dur, n_fft, hop) -> float:
    """Return median wall time (ms) for CPU path."""
    old = os.environ.pop("IRON_LIBROSA_RUST_DEVICE", None)
    os.environ["IRON_LIBROSA_RUST_DEVICE"] = "cpu"
    script = INNER_SCRIPT.format(
        dur=dur, n_fft=n_fft, hop=hop, nwarm=N_WARMUP, nrun=N_RUNS
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=False,
        env={**os.environ},
        timeout=60,
    )
    if result.returncode != 0:
        err = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        out = (result.stdout or b"").decode("utf-8", errors="replace").strip()
        print("  [cpu-baseline] subprocess failed")
        print(f"    returncode={result.returncode}")
        if out:
            print(f"    stdout: {out[-300:]}")
        if err:
            print(f"    stderr: {err[-500:]}")
    os.environ["IRON_LIBROSA_RUST_DEVICE"] = "cuda-gpu"
    stdout = (result.stdout or b"").decode("utf-8", errors="replace").strip()
    for line in stdout.splitlines():
        if line.startswith("WALL_MS "):
            times = [float(x) for x in line.split()[1:]]
            return statistics.median(times)
    return 0.0

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*80}")
    print("  iron-librosa CUDA Overhead Diagnostic")
    print(f"  Warmup={N_WARMUP}  Runs={N_RUNS}  Device=cuda-gpu")
    print(f"{'='*80}\n")

    # Collect CPU baselines first
    print("Collecting CPU baselines ...")
    cpu_times = {}
    for name, dur, n_fft, hop in WORKLOADS:
        t = cpu_baseline(dur, n_fft, hop)
        cpu_times[name] = t
        print(f"  {name:<18} CPU median = {t:.3f} ms")

    print()

    # GPU runs
    results = []
    for name, dur, n_fft, hop in WORKLOADS:
        print(f"Running GPU workload: {name} ...")
        r = run_workload(name, dur, n_fft, hop)
        r["cpu_ms"] = cpu_times[name]
        results.append(r)

    # ── Print overhead budget table ───────────────────────────────────────────
    hdr = (
        f"{'Workload':<18} {'op':<18} {'frms':>5} "
        f"{'H2D_kb':>8} {'kern_ms':>8} {'D2H_kb':>8} {'sync_ms':>8} "
        f"{'total_ms':>9} {'wall_ms':>9} {'cpu_ms':>8} {'speedup':>8} "
        f"{'plans':>6}"
    )
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'-'*len(hdr)}")

    for r in results:
        speedup = r["cpu_ms"] / r["wall_ms_median"] if r["wall_ms_median"] > 0 else 0
        flag = "OK" if speedup >= 1.0 else "NO"
        print(
            f"{r['name']:<18} {r['op']:<18} {r['n_frames']:>5} "
            f"{r['h2d_kb']:>8.1f} {r['kernel_ms']:>8.3f} {r['d2h_kb']:>8.1f} {r['sync_ms']:>8.3f} "
            f"{r['total_ms']:>9.3f} {r['wall_ms_median']:>9.3f} {r['cpu_ms']:>8.3f} "
            f"{speedup:>7.3f}x {flag} {r['plan_creates']:>5}"
        )

    print(f"{'='*len(hdr)}\n")

    # ── Transfer efficiency analysis ──────────────────────────────────────────
    print("Transfer Efficiency Analysis")
    print(f"{'-'*60}")
    for r in results:
        total_transfer_kb = r["h2d_kb"] + r["d2h_kb"]
        overhead_ms = r["sync_ms"]
        pcie_eff = overhead_ms / r["wall_ms_median"] * 100 if r["wall_ms_median"] > 0 else 0
        n_frames = r["n_frames"]
        per_frame_kb = total_transfer_kb / n_frames if n_frames > 0 else 0
        print(
            f"  {r['name']:<18}  transfer={total_transfer_kb:>8.1f} KB  "
            f"PCIe%={pcie_eff:>5.1f}%  per_frame={per_frame_kb:.2f} KB"
        )

    print()
    print("Legend: sync_ms includes D2H async enqueue + cudaStreamSynchronize")
    print("        kernel_ms = time from enqueue cufftExec to async return (GPU may not be done)")
    print("        total_ms  = full Rust GPU call including all staging copies")
    print("        wall_ms   = Python-side perf_counter across the full stft_complex call")
    print()

    # ── Recommendations ───────────────────────────────────────────────────────
    print("Recommendations")
    print(f"{'-'*60}")
    for r in results:
        speedup = r["cpu_ms"] / r["wall_ms_median"] if r["wall_ms_median"] > 0 else 0
        if speedup < 1.0:
            if speedup > 0:
                print(
                    f"  {r['name']}: GPU is {1/speedup:.2f}x SLOWER than CPU. "
                    f"Transfer={r['h2d_kb']+r['d2h_kb']:.0f} KB dominates. "
                    f"Consider raising IRON_LIBROSA_CUDA_FFT_MIN_FRAMES or "
                    f"IRON_LIBROSA_CUDA_FFT_MIN_WORK_THRESHOLD."
                )
            else:
                print(
                    f"  {r['name']}: insufficient timing data (cpu_ms={r['cpu_ms']:.3f}, "
                    f"wall_ms={r['wall_ms_median']:.3f}). Check subprocess diagnostics below."
                )
        else:
            print(f"  {r['name']}: GPU wins ({speedup:.2f}x). OK")

    # ── Debug snippet for failures ────────────────────────────────────────────
    print()
    if any(r["n_profile_rows"] == 0 for r in results):
        print("WARNING: Some workloads produced no CUDA_PROFILE lines.")
        print("  Check that IRON_LIBROSA_RUST_DEVICE=cuda-gpu and CUDA is available.")
        for r in results:
            if r["n_profile_rows"] == 0:
                print(f"\n  --- subprocess diagnostics for {r['name']} ---")
                print(f"  returncode={r['proc_returncode']}")
                print("  stdout tail:")
                print(r["stdout_snippet"])
                print("  stderr tail:")
                print(r["stderr_snippet"])


if __name__ == "__main__":
    main()

