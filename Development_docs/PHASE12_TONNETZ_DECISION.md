# Phase 12 Tonnetz Decision

Date: 2026-04-04

## Current State

- Tonnetz API is available and functional.
- Workspace may not always include optional MSAF fixture files:
  - `tests/data/feature-tonnetz-chroma.npy`
  - `tests/data/feature-tonnetz-msaf.npy`

## Decision

Use a fixture-safe test policy:
- when fixtures exist: run full parity check against reference array.
- when fixtures are absent: skip test with explicit reason.

This keeps CI/workspace behavior deterministic while preserving parity checks where data exists.

## Evidence

- Updated `tests/test_features.py::test_tonnetz_msaf` to skip when fixtures are missing.

## Exit Mapping

- [x] Parity path defined and executable when fixtures are present.
- [x] Profile/decision documented for workspace portability.

