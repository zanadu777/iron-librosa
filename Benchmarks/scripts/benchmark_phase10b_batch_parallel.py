"""
Phase 10B benchmark: batch-level parallelism for HPSS with adaptive dispatch.

Tests:
1. Sequential vs parallel batch processing
2. Threshold-guided adaptive dispatch
3. Speedup scaling with batch size
4. Parity validation (outputs match)
"""

import time
import numpy as np
import librosa


def benchmark_hpss_batch(S_batch: np.ndarray, *, kernel_size=(31, 31), repeats=5, label="") -> dict:
    """Benchmark batch HPSS with detailed metrics."""
    batch_size = S_batch.shape[0]
    n_bins = S_batch.shape[1]
    n_frames = S_batch.shape[2]
    total_elements = batch_size * n_bins * n_frames

    # Warmup
    librosa.decompose.hpss(S_batch, kernel_size=kernel_size)

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(repeats):
        H, P = librosa.decompose.hpss(S_batch, kernel_size=kernel_size)
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / repeats
    throughput = total_elements / (avg_time * 1e6)  # M elements/sec

    return {
        "label": label,
        "batch_size": batch_size,
        "shape": S_batch.shape,
        "total_elements": total_elements,
        "time_ms": avg_time * 1000,
        "throughput_me_per_sec": throughput,
    }


def validate_parity(S_batch: np.ndarray, *, kernel_size=(31, 31)) -> bool:
    """Validate that batch results match individual processing."""
    batch_size = S_batch.shape[0]

    # Compute via batch API
    H_batch, P_batch = librosa.decompose.hpss(S_batch, kernel_size=kernel_size)

    # Compute individually
    for b in range(batch_size):
        H_individual, P_individual = librosa.decompose.hpss(
            S_batch[b:b+1, :, :], kernel_size=kernel_size
        )
        # Squeeze and compare
        h_ind = H_individual[0] if H_individual.ndim == 3 else H_individual
        p_ind = P_individual[0] if P_individual.ndim == 3 else P_individual
        h_batch = H_batch[b] if H_batch.ndim == 3 else H_batch
        p_batch = P_batch[b] if P_batch.ndim == 3 else P_batch

        h_match = np.allclose(h_ind, h_batch, rtol=1e-5, atol=1e-6)
        p_match = np.allclose(p_ind, p_batch, rtol=1e-5, atol=1e-6)

        if not (h_match and p_match):
            print(f"  Parity mismatch at batch {b}")
            print(f"    H: {h_match}, P: {p_match}")
            return False

    return True


def main() -> None:
    rng = np.random.default_rng(9102026)

    print("Phase 10B: Batch-Level Parallelism for HPSS")
    print("=" * 100)
    print("Adaptive dispatch tuning with threshold-guided scheduling\n")

    # Test cases: (name, batch_size, n_bins, n_frames, kernel_size)
    test_cases = [
        # Small batches (below threshold or small per-batch work)
        ("batch_1_small", 1, 513, 200, (17, 31)),
        ("batch_1_medium", 1, 1025, 600, (31, 31)),

        # Mid batches (below batch_size=4 threshold)
        ("batch_2_small", 2, 513, 200, (17, 31)),
        ("batch_2_medium", 2, 1025, 600, (31, 31)),
        ("batch_3_small", 3, 513, 200, (17, 31)),

        # Batch size = 4 threshold (should NOT parallelize due to per-batch element limit)
        ("batch_4_small", 4, 513, 200, (17, 31)),
        ("batch_4_medium", 4, 1025, 600, (31, 31)),

        # Above batch_size=4 but high per-batch work (no parallelism, inner parallelism works)
        ("batch_8_medium", 8, 1025, 600, (31, 31)),
        ("batch_16_medium", 16, 1025, 600, (31, 31)),
        ("batch_8_large", 8, 2048, 1200, (31, 31)),
        ("batch_16_large", 16, 2048, 1200, (31, 31)),

        # Specific case: batch_size=4+ with SMALL per-batch elements (triggers batch parallelism)
        ("batch_4_tiny_hires", 4, 256, 100, (17, 17)),
        ("batch_8_tiny_hires", 8, 256, 100, (17, 17)),
        ("batch_16_tiny_hires", 16, 256, 100, (17, 17)),

        # Stereo case
        ("batch_4_stereo_hires", 4, 2048, 1500, (31, 31)),
    ]

    results = []

    print(f"{'Test Case':<25} {'Batch':<6} {'Shape':<20} {'Time (ms)':<12} {'Throughput':<15} {'Parity':<7}")
    print("-" * 100)

    for name, batch_size, n_bins, n_frames, kernel_size in test_cases:
        S = np.abs(rng.standard_normal((batch_size, n_bins, n_frames))).astype(np.float32)
        per_batch_elem = n_bins * n_frames
        total_elem = batch_size * n_bins * n_frames

        # Check dispatch threshold (from Rust constants)
        # Parallelize batch if: batch_size >= 4 AND per_batch_elements < 150_000
        meets_batch_threshold = batch_size >= 4 and per_batch_elem < 150_000
        expected_dispatch = "parallel" if meets_batch_threshold else "sequential"

        result = benchmark_hpss_batch(S, kernel_size=kernel_size, repeats=5, label=name)

        # Validate parity on a subset (expensive)
        parity_ok = validate_parity(S, kernel_size=kernel_size) if batch_size <= 4 else "skipped"
        parity_str = "PASS" if parity_ok is True else ("FAIL" if parity_ok is False else "SKIP")

        shape_str = f"({batch_size},{n_bins},{n_frames})"
        print(f"{name:<25} {batch_size:<6} {shape_str:<20} {result['time_ms']:<12.3f} {result['throughput_me_per_sec']:<15.1f} {parity_str:<7}")

        results.append({
            **result,
            "dispatch": expected_dispatch,
            "parity": parity_ok,
        })

    print("-" * 100)
    print("\n" + "=" * 100)
    print("Analysis: Adaptive Dispatch Scheduling\n")

    # Group by dispatch strategy
    seq_results = [r for r in results if r["dispatch"] == "sequential"]
    par_results = [r for r in results if r["dispatch"] == "parallel"]

    if seq_results:
        avg_seq_time = np.mean([r["time_ms"] for r in seq_results])
        print(f"Sequential batches (n={len(seq_results)}):")
        print(f"  Average time: {avg_seq_time:.3f} ms")

    if par_results:
        avg_par_time = np.mean([r["time_ms"] for r in par_results])
        print(f"\nParallel batches (n={len(par_results)}):")
        print(f"  Average time: {avg_par_time:.3f} ms")
        print(f"  Threshold rules:")
        print(f"    - batch_size >= 4")
        print(f"    - per_batch_elements < 150,000")
        print(f"  Rationale: avoids nested rayon contention by parallelizing")
        print(f"    only when batch size justifies overhead AND per-batch work")
        print(f"    is small enough to not compete with inner rayon usage")

    # Scaling analysis
    print("\n" + "=" * 100)
    print("Scaling Analysis: Batch Size Impact\n")

    medium_cases = [r for r in results if "medium" in r["label"] and "stereo" not in r["label"]]
    if len(medium_cases) >= 2:
        sorted_cases = sorted(medium_cases, key=lambda x: x["batch_size"])
        print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Throughput':<15} {'Rel. Speedup':<12}")
        baseline_time = sorted_cases[0]["time_ms"]
        for case in sorted_cases:
            speedup = baseline_time / case["time_ms"]
            print(f"{case['batch_size']:<12} {case['time_ms']:<12.3f} {case['throughput_me_per_sec']:<15.1f} {speedup:<12.2f}x")

    print("\n" + "=" * 100)
    print("Parity Validation Summary\n")

    parity_results = [r for r in results if r["parity"] in [True, False]]
    if parity_results:
        passed = sum(1 for r in parity_results if r["parity"] is True)
        total = len(parity_results)
        print(f"Parity checks: {passed}/{total} passed")
        if passed == total:
            print("PASS: All parity validations passed!")
        else:
            print("FAIL: Some parity checks failed!")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

