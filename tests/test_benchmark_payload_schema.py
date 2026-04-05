#!/usr/bin/env python

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Benchmarks" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_guard  # noqa: E402


def test_payload_schema_valid_minimal():
    payload = {
        "meta": {"benchmark": "demo"},
        "auto_review_cases": [],
        "rows": [],
    }
    assert benchmark_guard.has_benchmark_payload_schema(payload)


def test_payload_schema_rejects_missing_rows():
    payload = {
        "meta": {"benchmark": "demo"},
        "auto_review_cases": [],
    }
    assert not benchmark_guard.has_benchmark_payload_schema(payload)


def test_payload_schema_rejects_wrong_types():
    payload = {
        "meta": [],
        "auto_review_cases": {},
        "rows": "bad",
    }
    assert not benchmark_guard.has_benchmark_payload_schema(payload)

