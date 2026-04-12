#!/usr/bin/env python
"""Unit tests for GPU threshold calibration logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_gpu_work_threshold.py"
    module_name = "calibrate_gpu_work_threshold"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_recommend_threshold_none_when_gpu_never_wins():
    mod = _load_module()
    c1 = mod.BenchCase(1024, 300, 64)
    c2 = mod.BenchCase(2048, 800, 128)
    rows = [
        mod.BenchResult(c1, cpu_min_ms=1.0, gpu_min_ms=2.0, speedup_cpu_over_gpu=0.5),
        mod.BenchResult(c2, cpu_min_ms=2.0, gpu_min_ms=3.0, speedup_cpu_over_gpu=0.66),
    ]
    assert mod.recommend_threshold(rows) is None


def test_recommend_threshold_uses_last_losing_work_with_margin():
    mod = _load_module()
    c1 = mod.BenchCase(1024, 300, 64)
    c2 = mod.BenchCase(2048, 800, 128)
    c3 = mod.BenchCase(4096, 1200, 256)
    rows = [
        mod.BenchResult(c1, cpu_min_ms=1.0, gpu_min_ms=1.1, speedup_cpu_over_gpu=0.91),
        mod.BenchResult(c2, cpu_min_ms=2.0, gpu_min_ms=2.2, speedup_cpu_over_gpu=0.91),
        mod.BenchResult(c3, cpu_min_ms=4.0, gpu_min_ms=3.0, speedup_cpu_over_gpu=1.33),
    ]
    expected = int(c2.work * 1.10)
    assert mod.recommend_threshold(rows, safety_factor=1.10) == expected


