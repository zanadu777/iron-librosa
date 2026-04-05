"""Compatibility entrypoint for the renamed Phase 14 beat benchmark script."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("benchmark_phase14_beat_track.py")
    runpy.run_path(str(target), run_name="__main__")

