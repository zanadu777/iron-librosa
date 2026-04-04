"""Phase 12 benchmark: phase_vocoder baseline and multichannel parity checks."""

from __future__ import annotations

import time

import numpy as np
import librosa


def _timeit(fn, repeats=5):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return out, float(np.mean(times)), float(np.min(times))


def _bench_case(name: str, d: np.ndarray, rate: float):
    # Warm up to remove first-run effects from timing output.
    librosa.phase_vocoder(d, rate=rate, hop_length=512)
    out, avg_ms, min_ms = _timeit(lambda: librosa.phase_vocoder(d, rate=rate, hop_length=512))
    print(f"{name:<18} rate={rate:<4.2f} in={d.shape!s:<16} out={out.shape!s:<16} avg={avg_ms:8.3f} ms min={min_ms:8.3f} ms")


def main() -> None:
    rng = np.random.default_rng(1204)

    print("=" * 72)
    print("Phase 12 phase_vocoder benchmark")
    print("=" * 72)

    n_bins, n_frames = 1025, 1200
    mono = (
        rng.standard_normal((n_bins, n_frames))
        + 1j * rng.standard_normal((n_bins, n_frames))
    ).astype(np.complex64)
    stereo = np.stack([mono, mono * np.complex64(0.9 + 0.1j)], axis=0)

    _bench_case("mono", mono, 0.75)
    _bench_case("mono", mono, 1.25)
    _bench_case("stereo", stereo, 0.75)
    _bench_case("stereo", stereo, 1.25)

    # Simple multichannel parity sanity: stacked mono channels should match
    ch0 = librosa.phase_vocoder(stereo[0], rate=1.1, hop_length=512)
    ch1 = librosa.phase_vocoder(stereo[1], rate=1.1, hop_length=512)
    stacked = librosa.phase_vocoder(stereo, rate=1.1, hop_length=512)
    parity = np.allclose(stacked[0], ch0, rtol=1e-6, atol=1e-8) and np.allclose(
        stacked[1], ch1, rtol=1e-6, atol=1e-8
    )
    print(f"multichannel parity: {'PASS' if parity else 'FAIL'}")


if __name__ == "__main__":
    main()

