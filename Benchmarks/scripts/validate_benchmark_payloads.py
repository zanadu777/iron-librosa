"""Validate benchmark JSON payloads against the shared schema contract.

Usage examples:
  python Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/*.json
  python Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/phase15_bench_*.json --require-files
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List

from benchmark_guard import has_benchmark_payload_schema

# Always resolve relative patterns against the project root, not the caller's CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))


def _expand_paths(patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and not os.path.isabs(pattern):
            # Fallback: try the pattern relative to the project root so the
            # script works regardless of the caller's working directory.
            matches = glob.glob(os.path.join(_PROJECT_ROOT, pattern))
        if matches:
            out.extend(matches)
        else:
            out.append(pattern)
    # Keep stable order and dedupe.
    seen = set()
    unique: List[str] = []
    for path in out:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _validate_one(path: str) -> tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"missing file: {path}"
    try:
        with open(path, "r", encoding="utf-8") as fdesc:
            payload = json.load(fdesc)
    except Exception as exc:
        return False, f"invalid json: {path} ({exc})"

    if not has_benchmark_payload_schema(payload):
        return False, f"schema mismatch: {path}"

    return True, f"ok: {path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark JSON payload schema")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="One or more file paths or glob patterns for benchmark JSON artifacts.",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Fail when a glob resolves to no files.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.paths)

    if args.require_files and not any(os.path.exists(p) for p in paths):
        print("FAIL: no files matched input patterns")
        return 2

    failures = 0
    for path in paths:
        ok, msg = _validate_one(path)
        print(msg)
        if not ok:
            failures += 1

    if failures:
        print(f"FAIL: {failures} payload(s) failed schema validation")
        return 1

    print("PASS: all payloads match schema")
    return 0


if __name__ == "__main__":
    sys.exit(main())

