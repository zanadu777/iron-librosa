"""Phase 12 benchmark: CQT/VQT baseline timings for medium/long workloads."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import json
import platform
import time
from typing import Any, Dict, List

import numpy as np
import librosa
from librosa.core import constantq as cqt_mod
from benchmark_guard import (
    REVIEW_SPEEDUP_THRESHOLD,
    assert_benchmark_payload_schema,
    evaluate_speedup,
)


SR = 22050


def _timeit(fn, repeats=3):
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    times = []
    out = fn()
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return out, float(np.mean(times)), float(np.min(times))


def _has_rust_backend() -> bool:
    return bool(cqt_mod.RUST_AVAILABLE and getattr(cqt_mod, "_rust_ext", None) is not None)


@contextmanager
def _rust_backend_enabled(enabled: bool):
    original = cqt_mod.RUST_AVAILABLE
    original_force_numpy = cqt_mod.FORCE_NUMPY_CQT_VQT
    original_force_rust = cqt_mod.FORCE_RUST_CQT_VQT
    try:
        cqt_mod.FORCE_NUMPY_CQT_VQT = not enabled
        cqt_mod.FORCE_RUST_CQT_VQT = bool(enabled and _has_rust_backend())
        cqt_mod.RUST_AVAILABLE = bool(enabled and _has_rust_backend())
        yield
    finally:
        cqt_mod.FORCE_NUMPY_CQT_VQT = original_force_numpy
        cqt_mod.FORCE_RUST_CQT_VQT = original_force_rust
        cqt_mod.RUST_AVAILABLE = original


def _time_backend(fn, repeats: int, *, use_rust: bool):
    cqt_mod._vqt_filter_fft_cache_clear()
    with _rust_backend_enabled(use_rust):
        return _timeit(fn, repeats=repeats)


@dataclass
class _StageStat:
    total_ms: float = 0.0
    calls: int = 0


class _StageProfiler:
    def __init__(self):
        self.stats = {}

    def _record(self, label: str, dt_ms: float) -> None:
        stat = self.stats.setdefault(label, _StageStat())
        stat.total_ms += dt_ms
        stat.calls += 1

    def wrap(self, label, fn):
        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            self._record(label, (time.perf_counter() - t0) * 1e3)
            return out

        return _wrapped

    @contextmanager
    def patch(self, targets):
        originals = []
        try:
            for obj, attr, label in targets:
                original = getattr(obj, attr)
                originals.append((obj, attr, original))
                setattr(obj, attr, self.wrap(label, original))
            yield
        finally:
            for obj, attr, original in reversed(originals):
                setattr(obj, attr, original)

    def as_rows(self) -> List[Dict[str, Any]]:
        total = sum(stat.total_ms for stat in self.stats.values())
        if total <= 0:
            return []
        rows: List[Dict[str, Any]] = []
        ranked = sorted(self.stats.items(), key=lambda kv: kv[1].total_ms, reverse=True)
        for label, stat in ranked:
            share = 100.0 * stat.total_ms / total
            avg = stat.total_ms / stat.calls if stat.calls else 0.0
            rows.append(
                {
                    "label": label,
                    "total_ms": float(stat.total_ms),
                    "calls": int(stat.calls),
                    "avg_ms": float(avg),
                    "share_pct": float(share),
                }
            )
        return rows

    def print_report(self, title: str) -> None:
        rows = self.as_rows()
        print(f"  stage profile: {title}")
        if not rows:
            print("    n/a")
            return
        for row in rows:
            print(
                f"    {row['label']:<24} total={row['total_ms']:8.3f} ms"
                f"  calls={row['calls']:3d}  avg={row['avg_ms']:7.3f} ms"
                f"  share={row['share_pct']:5.1f}%"
            )


def _profile_transform(name: str, fn, repeats: int = 2) -> Dict[str, Any]:
    targets = [
        (cqt_mod, "__early_downsample", "__early_downsample"),
        (cqt_mod, "__vqt_filter_fft", "__vqt_filter_fft"),
        (cqt_mod, "__cqt_response", "__cqt_response"),
        (cqt_mod, "__trim_stack", "__trim_stack"),
        (cqt_mod, "stft", "stft"),
        (cqt_mod.audio, "resample", "audio.resample"),
        (cqt_mod.filters, "wavelet_lengths", "filters.wavelet_lengths"),
        (cqt_mod.filters, "wavelet", "filters.wavelet"),
    ]
    profiler = _StageProfiler()
    cqt_mod._vqt_filter_fft_cache_clear()
    with profiler.patch(targets):
        for _ in range(repeats):
            fn()
    profiler.print_report(name)
    info = cqt_mod._vqt_filter_fft_cache_info()
    print(
        f"    cache __vqt_filter_fft     hits={info['hits']:3d} misses={info['misses']:3d} size={info['size']:3d}"
    )
    return {
        "name": name,
        "rows": profiler.as_rows(),
        "cache": info,
    }


def _signal(seconds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = seconds * SR
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    y = 0.6 * np.sin(2.0 * np.pi * 220.0 * t) + 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    y += 0.05 * rng.standard_normal(n).astype(np.float32)
    return y.astype(np.float32)


def _stereo(y: np.ndarray) -> np.ndarray:
    return np.stack([y, 0.8 * y], axis=0)


def _bench_case(name: str, y: np.ndarray, repeats: int) -> Dict[str, Any]:
    rust_ready = _has_rust_backend()

    def _format_summary(label: str, shape, rust_avg, rust_min, py_avg, py_min):
        if py_avg is None or py_min is None:
            print(
                f"{label}  {name:<12} shape={shape!s:<16} avg={rust_avg:8.3f} ms min={rust_min:8.3f} ms"
            )
            return

        print(
            f"{label}  {name:<12} shape={shape!s:<16} "
            f"rust avg/min={rust_avg:8.3f}/{rust_min:8.3f} ms  "
            f"py avg/min={py_avg:8.3f}/{py_min:8.3f} ms  "
            f"speedup avg/min={py_avg / rust_avg:5.3f}x/{py_min / rust_min:5.3f}x"
        )

    cqt_fn = lambda: librosa.cqt(y=y, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)
    vqt_fn = lambda: librosa.vqt(y=y, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)

    cqt_out, cqt_avg, cqt_min = _time_backend(cqt_fn, repeats, use_rust=rust_ready)
    cqt_py_avg = cqt_py_min = None
    if rust_ready:
        cqt_py_out, cqt_py_avg, cqt_py_min = _time_backend(cqt_fn, repeats, use_rust=False)
        if cqt_out.shape != cqt_py_out.shape:
            raise RuntimeError(f"CQT backend shape mismatch for {name}: {cqt_out.shape} vs {cqt_py_out.shape}")
    _format_summary("cqt", cqt_out.shape, cqt_avg, cqt_min, cqt_py_avg, cqt_py_min)

    vqt_out, vqt_avg, vqt_min = _time_backend(vqt_fn, repeats, use_rust=rust_ready)
    vqt_py_avg = vqt_py_min = None
    if rust_ready:
        vqt_py_out, vqt_py_avg, vqt_py_min = _time_backend(vqt_fn, repeats, use_rust=False)
        if vqt_out.shape != vqt_py_out.shape:
            raise RuntimeError(f"VQT backend shape mismatch for {name}: {vqt_out.shape} vs {vqt_py_out.shape}")
    _format_summary("vqt", vqt_out.shape, vqt_avg, vqt_min, vqt_py_avg, vqt_py_min)

    result = {
        "name": name,
        "shape_cqt": list(cqt_out.shape),
        "shape_vqt": list(vqt_out.shape),
        "rust_available": rust_ready,
        "cqt_avg_ms": cqt_avg,
        "cqt_min_ms": cqt_min,
        "vqt_avg_ms": vqt_avg,
        "vqt_min_ms": vqt_min,
    }

    if rust_ready:
        cqt_review = evaluate_speedup(cqt_py_avg / cqt_avg, REVIEW_SPEEDUP_THRESHOLD)
        vqt_review = evaluate_speedup(vqt_py_avg / vqt_avg, REVIEW_SPEEDUP_THRESHOLD)
        result.update(
            {
                "cqt_py_avg_ms": cqt_py_avg,
                "cqt_py_min_ms": cqt_py_min,
                "cqt_avg_speedup": cqt_py_avg / cqt_avg,
                "cqt_min_speedup": cqt_py_min / cqt_min,
                "cqt_review_required": cqt_review["review_required"],
                "vqt_py_avg_ms": vqt_py_avg,
                "vqt_py_min_ms": vqt_py_min,
                "vqt_avg_speedup": vqt_py_avg / vqt_avg,
                "vqt_min_speedup": vqt_py_min / vqt_min,
                "vqt_review_required": vqt_review["review_required"],
            }
        )

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 12 CQT/VQT baseline benchmark")
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[10, 30],
        help="Signal durations in seconds for workload matrix.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed repeats per transform.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write machine-readable benchmark output as JSON.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=REVIEW_SPEEDUP_THRESHOLD,
        help="Speedup threshold below which cases are flagged for review.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 72)
    print("Phase 13 CQT/VQT benchmark")
    print("=" * 72)

    case_results: List[Dict[str, Any]] = []
    for seconds in args.durations:
        y = _signal(seconds=seconds, seed=7000 + seconds)
        case_results.append(_bench_case(f"mono-{seconds:>2}s", y, repeats=args.repeats))
        case_results.append(_bench_case(f"stereo-{seconds:>2}s", _stereo(y), repeats=args.repeats))

    auto_review_cases: List[str] = []
    for case in case_results:
        if not case.get("rust_available"):
            continue

        cqt_review = evaluate_speedup(case["cqt_avg_speedup"], args.review_threshold)
        vqt_review = evaluate_speedup(case["vqt_avg_speedup"], args.review_threshold)
        case["cqt_review_required"] = cqt_review["review_required"]
        case["vqt_review_required"] = vqt_review["review_required"]

        if cqt_review["review_required"]:
            auto_review_cases.append(f"{case['name']}::cqt ({case['cqt_avg_speedup']:.2f}x)")
        if vqt_review["review_required"]:
            auto_review_cases.append(f"{case['name']}::vqt ({case['vqt_avg_speedup']:.2f}x)")

    if auto_review_cases:
        print("auto-review required (< threshold):")
        for item in auto_review_cases:
            print(f"  - {item}")

    print("-" * 72)
    print("Internal stage profile (reference: mono-30s)")
    print("-" * 72)
    y_ref = _signal(seconds=30, seed=7030)

    stage_profiles: List[Dict[str, Any]] = []
    with _rust_backend_enabled(_has_rust_backend()):
        # Warm once before collecting stage timings.
        librosa.cqt(y=y_ref, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)
        librosa.vqt(y=y_ref, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12)

        stage_profiles.append(
            _profile_transform(
                "cqt",
                lambda: librosa.cqt(y=y_ref, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12),
            )
        )
        stage_profiles.append(
            _profile_transform(
                "vqt",
                lambda: librosa.vqt(y=y_ref, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12),
            )
        )

    if args.json_out:
        payload = {
            "meta": {
                "benchmark": "phase12_cqt_vqt",
                "rust_available": _has_rust_backend(),
                "review_threshold": args.review_threshold,
                "sr": SR,
                "durations": args.durations,
                "repeats": args.repeats,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "librosa": getattr(librosa, "__version__", "unknown"),
            },
            "auto_review_cases": auto_review_cases,
            "rows": case_results,
            "cases": case_results,
            "stage_profiles": stage_profiles,
        }
        assert_benchmark_payload_schema(payload, "phase12_cqt_vqt")
        with open(args.json_out, "w", encoding="utf-8") as fdesc:
            json.dump(payload, fdesc, indent=2)
        print(f"wrote json report: {args.json_out}")


if __name__ == "__main__":
    main()

