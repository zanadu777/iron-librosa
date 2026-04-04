#!/usr/bin/env python
"""Generate a mel validation addendum from captured tmp_mel_* outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import List, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _find_pytest_summary(text: str) -> str:
    m = re.search(r"(\d+\s+passed[^\n]*)", text)
    return m.group(1).strip() if m else "n/a"


def _find_profile(text: str) -> str:
    m = re.search(r"Profile key:\s*([^\n]+)", text)
    return m.group(1).strip() if m else "n/a"


def _find_threshold(text: str) -> str:
    m = re.search(r"threshold=(\d[\d_]*)", text)
    if m:
        return m.group(1)
    m = re.search(r"threshold set to\s+([\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    return "n/a"


def _find_mel_speedups(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    out: List[Tuple[str, str]] = []
    current_case = ""
    for line in lines:
        case_match = re.search(r"case:\s*(.*)", line)
        if case_match:
            current_case = case_match.group(1).strip()
            continue
        speed_match = re.search(r"speedup \(min\)\s+([0-9.]+x)", line)
        if speed_match and current_case:
            out.append((current_case, speed_match.group(1)))
    return out


def _registry_entries(path: Path) -> List[Tuple[str, int]]:
    try:
        payload = json.loads(_read_text(path))
        thresholds = payload.get("thresholds", {})
        if not isinstance(thresholds, dict):
            return []
        items: List[Tuple[str, int]] = []
        for key in sorted(thresholds):
            value = thresholds[key]
            try:
                items.append((str(key), int(value)))
            except (TypeError, ValueError):
                continue
        return items
    except Exception:
        return []


def build_addendum(args: argparse.Namespace) -> str:
    policy_text = _read_text(args.policy)
    features_text = _read_text(args.features)
    calibrate_text = _read_text(args.calibrate)
    bench_text = _read_text(args.bench)

    policy_summary = _find_pytest_summary(policy_text)
    features_summary = _find_pytest_summary(features_text)
    profile_key = _find_profile(calibrate_text)
    threshold = _find_threshold(calibrate_text)
    speedups = _find_mel_speedups(bench_text)
    registry_items = _registry_entries(args.registry)

    today = dt.date.today().isoformat()
    lines: List[str] = []
    lines.append(f"Addendum (mel validation refresh - {today}, {args.host_label}):")
    lines.append("- Validation:")
    lines.append(
        f"  - `python -m pytest tests/test_mel_threshold_policy.py -q` -> {policy_summary}"
    )
    lines.append(
        f"  - `python -m pytest tests/test_features.py -q -k \"melspectrogram\"` -> {features_summary}"
    )
    lines.append("- Calibration refresh:")
    lines.append("  - `python calibrate_mel_threshold.py --dry-run --skip-registry`")
    lines.append(f"  - profile key: `{profile_key}`")
    lines.append(f"  - suggested threshold: `{threshold}`")
    lines.append("- Benchmark refresh (`python benchmark_melspectrogram.py`, speedup min):")
    if speedups:
        for case, speed in speedups[:3]:
            lines.append(f"  - {case}: `{speed}`")
    else:
        lines.append("  - n/a")
    lines.append("- Registry status:")
    if registry_items:
        for key, value in registry_items:
            lines.append(f"  - `{key}`: `{value}`")
    else:
        lines.append("  - n/a")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--calibrate", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("mel_threshold_registry.json"),
        help="Path to mel threshold registry JSON",
    )
    parser.add_argument("--host-label", default="host")
    parser.add_argument(
        "--append-file",
        type=Path,
        default=None,
        help="Optional file path to append generated addendum",
    )
    args = parser.parse_args()

    block = build_addendum(args)
    print(block, end="")

    if args.append_file is not None:
        args.append_file.write_text(
            args.append_file.read_text(encoding="utf-8", errors="replace") + "\n" + block,
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

