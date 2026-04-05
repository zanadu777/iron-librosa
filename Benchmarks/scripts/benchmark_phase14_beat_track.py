"""Phase 14 benchmark: beat_track end-to-end timing and stage hints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import platform
import time
from typing import Dict, List

import numpy as np
import librosa
from librosa import beat as beat_mod
from benchmark_guard import (
    REVIEW_SPEEDUP_THRESHOLD,
    assert_benchmark_payload_schema,
    evaluate_speedup,
)


SR = 22050


@contextmanager
def _beat_backend(mode: str):
    old_numpy = beat_mod.FORCE_NUMPY_BEAT
    old_rust = beat_mod.FORCE_RUST_BEAT
    try:
        if mode == "numpy":
            beat_mod.FORCE_NUMPY_BEAT = True
            beat_mod.FORCE_RUST_BEAT = False
        elif mode == "rust":
            beat_mod.FORCE_NUMPY_BEAT = False
            beat_mod.FORCE_RUST_BEAT = True
        yield
    finally:
        beat_mod.FORCE_NUMPY_BEAT = old_numpy
        beat_mod.FORCE_RUST_BEAT = old_rust


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 14 beat_track benchmark")
    parser.add_argument(
        "--backend",
        choices=["auto", "numpy", "rust", "compare"],
        default="auto",
        help="Backend mode: auto, force numpy, force rust, or compare numpy/rust.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write benchmark output as JSON.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=REVIEW_SPEEDUP_THRESHOLD,
        help="Speedup threshold below which a case requires manual review.",
    )
    return parser.parse_args()


def _timeit(fn, repeats=5):
    times = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return last, float(np.mean(times)), float(np.min(times))


def _time_ms(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1e3


def _click_track(seconds: int, bpm: float) -> np.ndarray:
    n = seconds * SR
    y = np.zeros(n, dtype=np.float32)
    interval = int((60.0 / bpm) * SR)
    width = min(256, interval // 4)
    for idx in range(0, n, interval):
        y[idx : idx + width] += np.hanning(width).astype(np.float32)
    return y


def _noisy_music_like(seconds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = seconds * SR
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    y = 0.5 * np.sin(2.0 * np.pi * 110.0 * t) + 0.3 * np.sin(2.0 * np.pi * 220.0 * t)
    y += 0.2 * rng.standard_normal(n).astype(np.float32)
    return y.astype(np.float32)


def _bench_case(name: str, y: np.ndarray) -> dict:
    # Warm up numba-backed beat kernels before timing steady-state.
    librosa.beat.beat_track(y=y, sr=SR, hop_length=512)
    (_, beats), avg_ms, min_ms = _timeit(lambda: librosa.beat.beat_track(y=y, sr=SR, hop_length=512))
    print(f"{name:<16} samples={y.shape[0]:>8} beats={len(beats):>4} avg={avg_ms:8.3f} ms min={min_ms:8.3f} ms")

    # Stage-level timing to expose the dominant hotspot.
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    bpm = np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset_env, sr=SR, hop_length=512, start_bpm=120.0)
    )
    bpm_expanded = librosa.util.expand_to(bpm, ndim=onset_env.ndim, axes=range(bpm.ndim))
    frame_rate = float(SR) / 512.0
    tightness = 100.0

    # Warm up numba kernels used below before timing.
    norm = librosa.beat.__normalize_onsets(onset_env)
    frames_per_beat = np.round(frame_rate * 60.0 / bpm_expanded)
    localscore = librosa.beat.__beat_local_score(norm, frames_per_beat)
    backlink, cumscore = librosa.beat.__beat_track_dp(localscore, frames_per_beat, tightness)
    tail = librosa.beat.__last_beat(cumscore)
    beats_dense = np.zeros_like(onset_env, dtype=bool)
    librosa.beat.__dp_backtrack(backlink, tail, beats_dense)
    librosa.beat.__trim_beats(localscore, beats_dense, True)

    stage_acc: Dict[str, List[float]] = {
        "onset": [],
        "tempo": [],
        "local": [],
        "dp": [],
        "post": [],
    }

    repeats = 5
    for _ in range(repeats):
        onset_env, dt = _time_ms(lambda: librosa.onset.onset_strength(y=y, sr=SR, hop_length=512))
        stage_acc["onset"].append(dt)

        bpm, dt = _time_ms(
            lambda: np.atleast_1d(
                librosa.feature.tempo(
                    onset_envelope=onset_env,
                    sr=SR,
                    hop_length=512,
                    start_bpm=120.0,
                )
            )
        )
        stage_acc["tempo"].append(dt)

        bpm_expanded = librosa.util.expand_to(bpm, ndim=onset_env.ndim, axes=range(bpm.ndim))
        norm = librosa.beat.__normalize_onsets(onset_env)
        frames_per_beat = np.round(frame_rate * 60.0 / bpm_expanded)

        localscore, dt = _time_ms(lambda: librosa.beat.__beat_local_score(norm, frames_per_beat))
        stage_acc["local"].append(dt)

        (backlink, cumscore), dt = _time_ms(
            lambda: librosa.beat.__beat_track_dp(localscore, frames_per_beat, tightness)
        )
        stage_acc["dp"].append(dt)

        _, dt = _time_ms(
            lambda: _post_stage(onset_env, localscore, backlink, cumscore)
        )
        stage_acc["post"].append(dt)

    stage_avg = {k: float(np.mean(v)) for k, v in stage_acc.items()}
    total_stage = sum(stage_avg.values())
    print("  stage breakdown (avg ms, share):")
    for key in ("onset", "tempo", "local", "dp", "post"):
        pct = 100.0 * stage_avg[key] / total_stage if total_stage > 0 else 0.0
        print(f"    {key:<6} avg={stage_avg[key]:8.3f} ms  share={pct:6.1f}%")

    return {
        "name": name,
        "samples": int(y.shape[0]),
        "beats": int(len(beats)),
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "stage_avg_ms": stage_avg,
    }


def _post_stage(onset_env, localscore, backlink, cumscore):
    tail = librosa.beat.__last_beat(cumscore)
    beats_dense = np.zeros_like(onset_env, dtype=bool)
    librosa.beat.__dp_backtrack(backlink, tail, beats_dense)
    return librosa.beat.__trim_beats(localscore, beats_dense, True)


def _annotate_compare_reviews(results: list[dict], threshold: float) -> list[str]:
    """Attach speedup/review flags to compare-mode results and return flagged labels."""
    numpy_by_name = {
        item["name"]: item
        for item in results
        if item.get("backend") == "numpy"
    }

    flagged: list[str] = []
    for item in results:
        if item.get("backend") == "rust":
            base = numpy_by_name.get(item["name"])
            if base is None:
                item["speedup_vs_numpy"] = None
                item["review_required"] = None
                continue
            speedup = base["avg_ms"] / item["avg_ms"] if item["avg_ms"] > 0 else 0.0
            review = evaluate_speedup(speedup, threshold)
            item["speedup_vs_numpy"] = review["speedup"]
            item["review_required"] = review["review_required"]
            if review["review_required"]:
                flagged.append(f"{item['name']} ({speedup:.2f}x)")
        else:
            item["speedup_vs_numpy"] = 1.0
            item["review_required"] = False
    return flagged



def main() -> None:
    args = _parse_args()

    print("=" * 72)
    print("Phase 14 beat_track benchmark")
    print("=" * 72)

    cases = [
        ("click_120bpm_30s", _click_track(seconds=30, bpm=120.0)),
        ("noisy_30s", _noisy_music_like(seconds=30, seed=8801)),
        ("noisy_120s", _noisy_music_like(seconds=120, seed=8802)),
    ]

    results = []
    if args.backend == "compare":
        for mode in ("numpy", "rust"):
            print(f"-- backend: {mode} --")
            with _beat_backend(mode):
                for name, y in cases:
                    out = _bench_case(name, y)
                    out["backend"] = mode
                    results.append(out)
    else:
        mode = args.backend
        if mode in ("numpy", "rust"):
            print(f"-- backend: {mode} --")
        with _beat_backend(mode):
            for name, y in cases:
                out = _bench_case(name, y)
                out["backend"] = mode
                results.append(out)

    flagged_cases: list[str] = []
    if args.backend == "compare":
        flagged_cases = _annotate_compare_reviews(results, args.review_threshold)
        if flagged_cases:
            print("auto-review required (< threshold):")
            for case in flagged_cases:
                print(f"  - {case}")

    if args.json_out:
        payload = {
            "meta": {
                "benchmark": "phase14_beat_track",
                "backend": args.backend,
                "review_threshold": args.review_threshold,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "librosa": getattr(librosa, "__version__", "unknown"),
            },
            "auto_review_cases": flagged_cases,
            "rows": results,
            "cases": results,
        }
        assert_benchmark_payload_schema(payload, "phase14_beat_track")
        with open(args.json_out, "w", encoding="utf-8") as fdesc:
            json.dump(payload, fdesc, indent=2)
        print(f"wrote json report: {args.json_out}")


if __name__ == "__main__":
    main()

