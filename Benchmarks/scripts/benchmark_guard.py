from __future__ import annotations
__all__ = [
    "evaluate_speedup",
    "has_benchmark_payload_schema",
    "assert_benchmark_payload_schema",
    "REVIEW_SPEEDUP_THRESHOLD",
]
"""Simple performance guard for MFCC and onset benchmarks.

Runs a small subset of MFCC benchmarks and exits non-zero if speedups drop
below configurable thresholds.

Default thresholds are conservative to reduce false positives on noisy hosts.
"""

import argparse
import importlib
import sys
import time

import numpy as np
import librosa

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
N_MFCC = 20
ONSET_SHAPES = [(128, 2000), (256, 4000)]
REVIEW_SPEEDUP_THRESHOLD = 1.5
REQUIRED_BENCHMARK_PAYLOAD_KEYS = ("meta", "auto_review_cases", "rows")


def evaluate_speedup(speedup: float, threshold: float = REVIEW_SPEEDUP_THRESHOLD) -> dict[str, float | bool]:
    """Classify a speedup against the auto-review threshold."""
    return {
        "speedup": float(speedup),
        "review_threshold": float(threshold),
        "review_required": bool(speedup < threshold),
    }


def has_benchmark_payload_schema(payload: dict) -> bool:
    """Return True when payload contains the standard benchmark schema keys."""
    if not isinstance(payload, dict):
        return False

    for key in REQUIRED_BENCHMARK_PAYLOAD_KEYS:
        if key not in payload:
            return False

    if not isinstance(payload["meta"], dict):
        return False
    if not isinstance(payload["auto_review_cases"], list):
        return False
    if not isinstance(payload["rows"], list):
        return False

    return True


def assert_benchmark_payload_schema(payload: dict, benchmark_name: str) -> None:
    """Raise ValueError when payload does not match the benchmark schema."""
    if not has_benchmark_payload_schema(payload):
        raise ValueError(f"invalid benchmark payload schema for {benchmark_name}")


def _timeit(fn, runs: int, batches: int) -> float:
    """Return a robust runtime estimate as median(batch_min).

    Each batch executes one warmup call and `runs` timed calls.
    The batch minimum dampens outliers; the median across batches
    reduces host-level jitter.
    """
    batch_mins = []
    for _ in range(batches):
        fn()  # warmup
        vals = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            vals.append(time.perf_counter() - t0)
        batch_mins.append(min(vals))
    return float(np.median(batch_mins))


def _mfcc_y_min(mod, y: np.ndarray, runs: int, batches: int) -> float:
    return _timeit(
        lambda: mod.feature.mfcc(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP,
            n_mels=N_MELS,
            n_mfcc=N_MFCC,
            dct_type=2,
            norm="ortho",
            lifter=0,
        ),
        runs,
        batches,
    )


def _mfcc_s_min(mod, s: np.ndarray, runs: int, batches: int) -> float:
    return _timeit(
        lambda: mod.feature.mfcc(
            S=s,
            n_mfcc=N_MFCC,
            dct_type=2,
            norm="ortho",
            lifter=0,
        ),
        runs,
        batches,
    )


def _onset_multi_min(
    mod, s: np.ndarray, runs: int, batches: int, lag: int, max_size: int
) -> float:
    return _timeit(
        lambda: mod.onset.onset_strength_multi(
            S=s,
            lag=lag,
            max_size=max_size,
            aggregate=np.mean,
            center=False,
        ),
        runs,
        batches,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--durations", type=int, nargs="*", default=[5, 15, 30])
    parser.add_argument("--min-full-speedup", type=float, default=0.70)
    parser.add_argument("--min-dct-speedup", type=float, default=0.80)
    parser.add_argument("--min-onset-mean-speedup", type=float, default=0.95)
    parser.add_argument("--min-onset-maxfilter-speedup", type=float, default=0.90)
    parser.add_argument("--review-threshold", type=float, default=REVIEW_SPEEDUP_THRESHOLD)
    parser.add_argument(
        "--fail-on-review-required",
        action="store_true",
        help="Exit non-zero when any speedup is below the review threshold.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    iron_librosa = importlib.import_module("iron_librosa")

    failures: list[str] = []
    review_items: list[str] = []

    print("MFCC guard: full pipeline (y input)")
    for dur in args.durations:
        y = np.random.randn(SR * dur).astype(np.float32)
        t_py = _mfcc_y_min(librosa, y, args.runs, args.batches)
        t_rs = _mfcc_y_min(iron_librosa, y, args.runs, args.batches)
        speed = t_py / t_rs
        review = evaluate_speedup(speed, args.review_threshold)
        print(
            f"  {dur:2d}s  librosa={t_py*1e3:7.2f} ms  "
            f"iron={t_rs*1e3:7.2f} ms  speedup={speed:5.2f}x"
        )
        if review["review_required"]:
            review_items.append(
                f"full-mfcc {dur}s speedup {speed:.2f}x < review {args.review_threshold:.2f}x"
            )
        if speed < args.min_full_speedup:
            failures.append(
                f"full-mfcc {dur}s speedup {speed:.2f}x < {args.min_full_speedup:.2f}x"
            )

    print("\nMFCC guard: DCT-focused (S input, fast-path config)")
    for dur in args.durations:
        n_frames = 1 + (SR * dur) // HOP
        s = np.random.randn(N_MELS, n_frames).astype(np.float32)
        t_py = _mfcc_s_min(librosa, s, args.runs, args.batches)
        t_rs = _mfcc_s_min(iron_librosa, s, args.runs, args.batches)
        speed = t_py / t_rs
        review = evaluate_speedup(speed, args.review_threshold)
        print(
            f"  {dur:2d}s  librosa={t_py*1e3:7.3f} ms  "
            f"iron={t_rs*1e3:7.3f} ms  speedup={speed:5.2f}x"
        )
        if review["review_required"]:
            review_items.append(
                f"dct-mfcc {dur}s speedup {speed:.2f}x < review {args.review_threshold:.2f}x"
            )
        if speed < args.min_dct_speedup:
            failures.append(
                f"dct-mfcc {dur}s speedup {speed:.2f}x < {args.min_dct_speedup:.2f}x"
            )

    print("\nOnset guard: mean aggregation fast path (S input)")
    for n_bins, n_frames in ONSET_SHAPES:
        s = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32)
        t_py = _onset_multi_min(librosa, s, args.runs, args.batches, lag=1, max_size=1)
        t_rs = _onset_multi_min(
            iron_librosa, s, args.runs, args.batches, lag=1, max_size=1
        )
        speed = t_py / t_rs
        review = evaluate_speedup(speed, args.review_threshold)
        print(
            f"  {n_bins:3d}x{n_frames:<5d}  librosa={t_py*1e3:7.3f} ms  "
            f"iron={t_rs*1e3:7.3f} ms  speedup={speed:5.2f}x"
        )
        if review["review_required"]:
            review_items.append(
                f"onset-mean {n_bins}x{n_frames} speedup {speed:.2f}x < review {args.review_threshold:.2f}x"
            )
        if speed < args.min_onset_mean_speedup:
            failures.append(
                f"onset-mean {n_bins}x{n_frames} speedup {speed:.2f}x < "
                f"{args.min_onset_mean_speedup:.2f}x"
            )

    print("\nOnset guard: max-filter reference path (max_size=5)")
    for n_bins, n_frames in ONSET_SHAPES:
        s = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32)
        t_py = _onset_multi_min(librosa, s, args.runs, args.batches, lag=2, max_size=5)
        t_rs = _onset_multi_min(
            iron_librosa, s, args.runs, args.batches, lag=2, max_size=5
        )
        speed = t_py / t_rs
        review = evaluate_speedup(speed, args.review_threshold)
        print(
            f"  {n_bins:3d}x{n_frames:<5d}  librosa={t_py*1e3:7.3f} ms  "
            f"iron={t_rs*1e3:7.3f} ms  speedup={speed:5.2f}x"
        )
        if review["review_required"]:
            review_items.append(
                f"onset-maxfilter {n_bins}x{n_frames} speedup {speed:.2f}x < review {args.review_threshold:.2f}x"
            )
        if speed < args.min_onset_maxfilter_speedup:
            failures.append(
                f"onset-maxfilter {n_bins}x{n_frames} speedup {speed:.2f}x < "
                f"{args.min_onset_maxfilter_speedup:.2f}x"
            )

    if review_items:
        print("\nAUTO-REVIEW REQUIRED:")
        for item in review_items:
            print(f"  - {item}")

    if args.fail_on_review_required and review_items:
        print("\nFAIL: review threshold violations (strict mode)")
        return 2

    if failures:
        print("\nFAIL:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nPASS: all thresholds satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())

