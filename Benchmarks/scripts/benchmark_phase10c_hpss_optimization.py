"""
Phase 10C HPSS Optimization: Performance Benchmarking Suite

Measures:
1. Frame-level parallelism speedup
2. SciPy baseline comparison
3. Scaling analysis
4. Batch processing impact
5. Masking vs full decomposition
"""

import time
import numpy as np
import librosa
from scipy.ndimage import median_filter


def benchmark_case(name, S, kernel_size=(31, 31), repeats=5):
    """Benchmark a single HPSS case."""
    # Warmup
    _ = librosa.decompose.hpss(S, kernel_size=kernel_size)

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(repeats):
        H, P = librosa.decompose.hpss(S, kernel_size=kernel_size)
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / repeats
    throughput = (S.size / (avg_time * 1e6)) if avg_time > 0 else 0

    return {
        "label": name,
        "shape": S.shape,
        "elements": S.size,
        "time_ms": avg_time * 1000,
        "throughput_me_per_sec": throughput,
    }


def benchmark_scipy_baseline(S, kernel_size=(31, 31), repeats=5):
    """Benchmark SciPy median_filter baseline."""
    win_harm, win_perc = kernel_size

    harm_shape = [1] * S.ndim
    harm_shape[-1] = win_harm
    perc_shape = [1] * S.ndim
    perc_shape[-2] = win_perc

    # Warmup
    _ = median_filter(S, size=harm_shape, mode="reflect")
    _ = median_filter(S, size=perc_shape, mode="reflect")

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(repeats):
        harm = median_filter(S, size=harm_shape, mode="reflect")
        perc = median_filter(S, size=perc_shape, mode="reflect")
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / repeats
    return avg_time * 1000


def main():
    rng = np.random.default_rng(9102026)

    print("\n" + "=" * 120)
    print("PHASE 10C HPSS OPTIMIZATION: Performance Benchmarking")
    print("=" * 120)
    print("Evaluating frame-level parallelism in masking computation\n")

    # Test cases
    test_cases = [
        ("small_f32", np.abs(rng.standard_normal((256, 100))).astype(np.float32)),
        ("small_f64", np.abs(rng.standard_normal((256, 100))).astype(np.float64)),
        ("medium_f32", np.abs(rng.standard_normal((1025, 600))).astype(np.float32)),
        ("medium_f64", np.abs(rng.standard_normal((1025, 600))).astype(np.float64)),
        ("large_f32", np.abs(rng.standard_normal((2048, 1000))).astype(np.float32)),
        ("xlarge_f32", np.abs(rng.standard_normal((4096, 2000))).astype(np.float32)),
        ("stereo_f32", np.abs(rng.standard_normal((2, 1025, 600))).astype(np.float32)),
        ("batch_4_f32", np.abs(rng.standard_normal((4, 1025, 600))).astype(np.float32)),
    ]

    results = []

    print(f"{'Test Case':<20} {'Shape':<20} {'Rust HPSS (ms)':<18} {'Throughput':<15}")
    print("-" * 120)

    for name, S in test_cases:
        result = benchmark_case(name, S)
        results.append(result)
        shape_str = str(S.shape)
        print(f"{name:<20} {shape_str:<20} {result['time_ms']:<18.3f} {result['throughput_me_per_sec']:<15.1f}")

    print("-" * 120)

    # SciPy comparison (for 2D cases only)
    print("\n" + "=" * 120)
    print("BASELINE COMPARISON: Rust HPSS vs SciPy median_filter")
    print("=" * 120)
    print(f"{'Test Case':<20} {'SciPy (ms)':<15} {'Rust HPSS (ms)':<18} {'Speedup':<10}")
    print("-" * 120)

    scipy_results = []
    for name, S in test_cases[:5]:  # Only 2D cases
        scipy_time = benchmark_scipy_baseline(S)
        rust_time = next(r for r in results if r["label"] == name)["time_ms"]
        speedup = scipy_time / rust_time if rust_time > 0 else 0
        scipy_results.append((name, scipy_time, rust_time, speedup))
        print(f"{name:<20} {scipy_time:<15.3f} {rust_time:<18.3f} {speedup:<10.2f}x")

    print("-" * 120)

    # Scaling analysis
    print("\n" + "=" * 120)
    print("SCALING ANALYSIS: Impact of Input Size")
    print("=" * 120)
    print(f"{'Input Elements':<20} {'Time (ms)':<15} {'Throughput':<15} {'Scaling Factor':<15}")
    print("-" * 120)

    baseline_time = None
    for result in sorted(results, key=lambda x: x["elements"]):
        if "f32" in result["label"] and "batch" not in result["label"] and result["shape"][0] <= 4096:
            if baseline_time is None:
                baseline_time = result["time_ms"]
            scaling = baseline_time / result["time_ms"] if result["time_ms"] > 0 else 0
            elem_str = f"{result['elements']:,}"
            print(f"{elem_str:<20} {result['time_ms']:<15.3f} {result['throughput_me_per_sec']:<15.1f} {scaling:<15.2f}x")

    print("-" * 120)

    # Parallelism analysis
    print("\n" + "=" * 120)
    print("PARALLELISM ANALYSIS: Sequential vs Frame-Level Parallel")
    print("=" * 120)
    print("Frame-level parallelism is activated when:")
    print("  - total_elements = n_bins × n_frames >= PAR_THRESHOLD (200,000)")
    print("\nTest case analysis:")
    print(f"{'Test Case':<20} {'Elements':<15} {'PAR Threshold Met':<20} {'Expected Path':<20}")
    print("-" * 120)

    for result in sorted(results, key=lambda x: x["elements"]):
        if "batch" not in result["label"]:
            meets_threshold = result["elements"] >= 200_000
            path = "PARALLEL" if meets_threshold else "SEQUENTIAL"
            threshold_str = "YES" if meets_threshold else "NO"
            elem_str = f"{result['elements']:,}"
            print(f"{result['label']:<20} {elem_str:<15} {threshold_str:<20} {path:<20}")

    print("-" * 120)

    # Quality metrics
    print("\n" + "=" * 120)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 120)

    f32_results = [r for r in results if "f32" in r["label"]]
    f64_results = [r for r in results if "f64" in r["label"]]

    if f32_results:
        avg_throughput_f32 = np.mean([r["throughput_me_per_sec"] for r in f32_results])
        print(f"f32 average throughput: {avg_throughput_f32:.1f} M elements/sec")

    if f64_results:
        avg_throughput_f64 = np.mean([r["throughput_me_per_sec"] for r in f64_results])
        print(f"f64 average throughput: {avg_throughput_f64:.1f} M elements/sec")

    if scipy_results:
        avg_speedup = np.mean([s[3] for s in scipy_results])
        print(f"\nAverage speedup vs SciPy: {avg_speedup:.2f}x")
        print(f"  - Median filter baseline: {np.mean([s[1] for s in scipy_results]):.3f} ms")
        print(f"  - HPSS + masking: {np.mean([s[2] for s in scipy_results]):.3f} ms")

    # Throughput stability
    print(f"\nThroughput stability:")
    print(f"  - Min: {min([r['throughput_me_per_sec'] for r in results]):.1f} M elements/sec")
    print(f"  - Max: {max([r['throughput_me_per_sec'] for r in results]):.1f} M elements/sec")
    print(f"  - Variation: {(max([r['throughput_me_per_sec'] for r in results]) / min([r['throughput_me_per_sec'] for r in results])):.2f}x")

    print("\n" + "=" * 120)
    print("PHASE 10C BENCHMARKING COMPLETE")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()

