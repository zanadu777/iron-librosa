"""Unit tests for mel threshold resolution policy."""

from contextlib import contextmanager
import json
import os

import librosa.feature.spectral as spectral_mod


@contextmanager
def _env(overrides):
    prev = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_mel_threshold_falls_back_to_constant():
    with _env(
        {
            "IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD": None,
            "IRON_LIBROSA_MEL_PROFILE": None,
            "IRON_LIBROSA_MEL_THRESHOLD_FILE": None,
        }
    ):
        assert spectral_mod._resolve_mel_work_threshold() == spectral_mod._MEL_RUST_WORK_THRESHOLD


def test_mel_threshold_env_override_wins():
    with _env(
        {
            "IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD": "12345",
            "IRON_LIBROSA_MEL_PROFILE": "ignored-profile",
            "IRON_LIBROSA_MEL_THRESHOLD_FILE": None,
        }
    ):
        assert spectral_mod._resolve_mel_work_threshold() == 12345


def test_mel_threshold_profile_from_external_registry(tmp_path):
    registry_path = tmp_path / "mel_registry.json"
    payload = {"version": 1, "thresholds": {"ci-profile": 7777777}}
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with _env(
        {
            "IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD": None,
            "IRON_LIBROSA_MEL_PROFILE": "ci-profile",
            "IRON_LIBROSA_MEL_THRESHOLD_FILE": str(registry_path),
        }
    ):
        assert spectral_mod._resolve_mel_work_threshold() == 7777777


def test_mel_threshold_profile_from_builtin_registry(monkeypatch):
    monkeypatch.setattr(spectral_mod, "MEL_WORK_THRESHOLDS", {"builtin-profile": 4242424})

    with _env(
        {
            "IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD": None,
            "IRON_LIBROSA_MEL_PROFILE": "builtin-profile",
            "IRON_LIBROSA_MEL_THRESHOLD_FILE": None,
        }
    ):
        assert spectral_mod._resolve_mel_work_threshold() == 4242424


def test_mel_threshold_profile_missing_registry_falls_back(tmp_path):
    with _env(
        {
            "IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD": None,
            "IRON_LIBROSA_MEL_PROFILE": "missing-profile",
            "IRON_LIBROSA_MEL_THRESHOLD_FILE": str(tmp_path / "does_not_exist.json"),
        }
    ):
        assert spectral_mod._resolve_mel_work_threshold() == spectral_mod._MEL_RUST_WORK_THRESHOLD


def test_mel_threshold_invalid_env_override_falls_back():
    with _env(
        {
            "IRON_LIBROSA_MEL_RUST_WORK_THRESHOLD": "bad",
            "IRON_LIBROSA_MEL_PROFILE": None,
            "IRON_LIBROSA_MEL_THRESHOLD_FILE": None,
        }
    ):
        assert spectral_mod._resolve_mel_work_threshold() == spectral_mod._MEL_RUST_WORK_THRESHOLD

