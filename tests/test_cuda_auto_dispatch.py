#!/usr/bin/env python
"""Phase 21 auto-dispatch smoke tests for CUDA drop-in behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from librosa._rust_bridge import RUST_AVAILABLE


def _backend_info_from_subprocess(extra_env: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(extra_env)

    code = (
        "import json;"
        "from librosa._rust_bridge import RUST_AVAILABLE,_rust_ext;"
        "assert RUST_AVAILABLE and _rust_ext is not None;"
        "print(json.dumps(_rust_ext.rust_backend_info(), sort_keys=True))"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout.strip())


def _cuda_diagnostics_from_subprocess(extra_env: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(extra_env)

    code = (
        "import json;"
        "from librosa._rust_bridge import RUST_AVAILABLE,_rust_ext;"
        "assert RUST_AVAILABLE and _rust_ext is not None;"
        "print(json.dumps(_rust_ext.cuda_diagnostics(), sort_keys=True))"
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout.strip())


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_auto_mode_reports_reason_and_policy():
    info = _backend_info_from_subprocess(
        {
            "IRON_LIBROSA_RUST_DEVICE": "auto",
            "IRON_LIBROSA_CUDA_RUNTIME_FORCE": "0",
        }
    )
    assert info["requested"] == "auto"
    assert info["dispatch_policy"] == "auto_first_cpu_fallback"
    assert isinstance(info["resolved_reason"], str) and info["resolved_reason"]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cuda_request_falls_back_to_cpu_when_runtime_forced_off():
    info = _backend_info_from_subprocess(
        {
            "IRON_LIBROSA_RUST_DEVICE": "cuda-gpu",
            "IRON_LIBROSA_CUDA_RUNTIME_FORCE": "0",
        }
    )
    assert info["requested"] == "cuda-gpu"
    assert info["resolved"] == "cpu"
    assert "fallback" in info["resolved_reason"]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cuda_request_uses_cuda_when_feature_and_runtime_forced_on():
    info = _backend_info_from_subprocess(
        {
            "IRON_LIBROSA_RUST_DEVICE": "cuda-gpu",
            "IRON_LIBROSA_CUDA_RUNTIME_FORCE": "1",
        }
    )
    if info["cuda_gpu_feature_enabled"]:
        assert info["resolved"] == "cuda-gpu"
        assert info["resolved_reason"] == "requested_cuda_gpu_available"
    else:
        assert info["resolved"] == "cpu"
        assert "fallback" in info["resolved_reason"]


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cuda_diagnostics_reports_window_pack_helper_metadata():
    info = _cuda_diagnostics_from_subprocess(
        {
            "IRON_LIBROSA_RUST_DEVICE": "cuda-gpu",
            "IRON_LIBROSA_CUDA_RUNTIME_FORCE": "0",
        }
    )

    assert "cuda_window_pack_helper_built" in info
    assert "cuda_window_pack_helper_path" in info
    assert "cuda_fused_mel_helper_built" in info
    assert "cuda_fused_mel_helper_path" in info
    assert isinstance(info["cuda_window_pack_helper_built"], bool)
    assert isinstance(info["cuda_window_pack_helper_path"], str)
    assert isinstance(info["cuda_fused_mel_helper_built"], bool)
    assert isinstance(info["cuda_fused_mel_helper_path"], str)


