#!/usr/bin/env python

from __future__ import annotations

import pathlib
import sys


# Import benchmark_guard directly from Benchmarks/scripts.
ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Benchmarks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_guard  # noqa: E402


def test_review_boundary_below_threshold_requires_review():
    out = benchmark_guard.evaluate_speedup(1.49, threshold=1.5)
    assert out["review_required"] is True


def test_review_boundary_at_threshold_no_review():
    out = benchmark_guard.evaluate_speedup(1.50, threshold=1.5)
    assert out["review_required"] is False


def test_review_structure_fields_present():
    out = benchmark_guard.evaluate_speedup(2.0, threshold=1.5)
    assert out["speedup"] == 2.0
    assert out["review_threshold"] == 1.5
    assert out["review_required"] is False

