#!/usr/bin/env python
"""Phase 3 STFT benchmark: Rust fast-path vs forced Python fallback.

Benchmarks librosa.stft() across:
- dtype: float32, float64
- channels: mono (1), stereo (2)
- center: True, False

Output includes an append-ready markdown table.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import librosa
import librosa.core.spectrum as spectrum_mod


@dataclass
class Case:
    channels: int
    dtype: np.dtype
    center: bool


def _timeit(fn, warmup: int, runs: int, batches: int) -> float:
    batch_mins = []
    for _ in range(batches):
        for _ in range(warmup):
            fn()
        vals = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            vals.append(time.perf_counter() - t0)
        batch_mins.append(min(vals))
    return float(np.median(batch_mins))


def _make_signal(rng: np.random.Generator, channels: int, dtype: np.dtype, n: int) -> np.ndarray:
    if channels == 1:
        return rng.standard_normal(n).astype(dtype)
    return rng.standard_normal((channels, n)).astype(dtype)


def _bench_case(y: np.ndarray, center: bool, n_fft: int, hop_length: int, rust_enabled: bool, warmup: int, runs: int, batches: int) -> float:
    old_flag = spectrum_mod.RUST_AVAILABLE
    spectrum_mod.RUST_AVAILABLE = rust_enabled
    try:
        return _timeit(
            lambda: librosa.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                window="hann",
                center=center,
                pad_mode="constant",
            ),
            warmup=warmup,
            runs=runs,
            batches=batches,
        )
    finally:
        spectrum_mod.RUST_AVAILABLE = old_flag


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_samples = int(args.seconds * args.sr)

    cases = [
        Case(channels=1, dtype=np.float32, center=True),
        Case(channels=1, dtype=np.float64, center=True),
        Case(channels=2, dtype=np.float32, center=True),
        Case(channels=2, dtype=np.float64, center=True),
        Case(channels=1, dtype=np.float32, center=False),
        Case(channels=1, dtype=np.float64, center=False),
        Case(channels=2, dtype=np.float32, center=False),
        Case(channels=2, dtype=np.float64, center=False),
    ]

    rows = []
    for c in cases:
        y = _make_signal(rng, c.channels, c.dtype, n_samples)
        t_rust = _bench_case(
            y,
            center=c.center,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            rust_enabled=True,
            warmup=args.warmup,
            runs=args.runs,
            batches=args.batches,
        )
        t_py = _bench_case(
            y,
            center=c.center,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            rust_enabled=False,
            warmup=args.warmup,
            runs=args.runs,
            batches=args.batches,
        )
        speedup = t_py / t_rust if t_rust > 0 else float("inf")
        rows.append((c.center, c.channels, np.dtype(c.dtype).name, t_rust, t_py, speedup))

    print("# Phase 3 STFT Benchmark")
    print(
        f"seed={args.seed}, seconds={args.seconds}, sr={args.sr}, n_fft={args.n_fft}, "
        f"hop={args.hop_length}, warmup={args.warmup}, runs={args.runs}, batches={args.batches}"
    )
    print()
    print("| center | channels | dtype | rust_ms | py_ms | speedup_x |")
    print("|---|---:|---|---:|---:|---:|")
    for center, channels, dtype_name, t_rust, t_py, speedup in rows:
        print(
            f"| {center} | {channels} | {dtype_name} | {t_rust*1e3:.3f} | {t_py*1e3:.3f} | {speedup:.2f} |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

