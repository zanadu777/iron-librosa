import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_phase21_module():
    script = Path(__file__).resolve().parents[1] / "Benchmarks" / "scripts" / "benchmark_phase21_cuda_baseline.py"
    spec = importlib.util.spec_from_file_location("phase21_cuda_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _timings(module, stft, istft):
    return {label: (stft, istft) for label, *_ in module.WORKLOADS}


def test_phase21_script_auto_mode_writes_backend_info(tmp_path):
    script = Path(__file__).resolve().parents[1] / "Benchmarks" / "scripts" / "benchmark_phase21_cuda_baseline.py"
    json_out = tmp_path / "phase21_auto_smoke.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--device",
            "auto",
            "--rounds",
            "1",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--json-out",
            str(json_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["meta"]["device"] == "auto"
    assert payload["meta"]["cuda_experimental"] is None
    assert "backend_info" in payload
    assert payload["backend_info"]["requested"] == "auto"
    assert payload["backend_info"]["resolved"] in {"cpu", "apple-gpu", "cuda-gpu"}
    assert "resolved_reason" in payload["backend_info"]
    assert "Phase 21 CUDA Benchmark" in proc.stdout


def test_phase21_promotion_gate_requires_large_workload_speedup():
    module = _load_phase21_module()
    baseline = _timings(module, 10.0, 10.0)
    current = _timings(module, 5.0, 5.0)

    # Force large workloads to underperform even while short workloads are faster.
    current["medium_512"] = (12.0, 12.0)
    current["medium_1024"] = (11.0, 11.0)
    current["long_1024"] = (10.5, 10.5)

    cmp = module._compare_with_baseline(current, baseline)

    assert cmp["promotion_gate"]["score_pass"]
    assert not cmp["promotion_gate"]["large_workload_gate"]
    assert cmp["promotion_gate"]["decision"] != "PROMOTE"


def test_phase21_promotion_gate_promotes_when_all_gates_pass():
    module = _load_phase21_module()
    baseline = _timings(module, 10.0, 10.0)
    current = _timings(module, 4.0, 4.0)

    cmp = module._compare_with_baseline(current, baseline)

    assert cmp["promotion_gate"]["score_pass"]
    assert cmp["promotion_gate"]["regression_gate"]
    assert cmp["promotion_gate"]["large_workload_gate"]
    assert cmp["promotion_gate"]["decision"] == "PROMOTE"


def test_phase21_promotion_gate_allows_near_parity_noise():
    module = _load_phase21_module()
    baseline = _timings(module, 10.0, 10.0)
    # 0.995x speedup lands inside 1% tolerance band.
    current = _timings(module, 10.0 / 0.995, 10.0 / 0.995)

    cmp = module._compare_with_baseline(current, baseline)

    assert cmp["summary"]["regressions_total"] == 0
    assert cmp["summary"]["stft_near_parity"] == len(module.WORKLOADS)
    assert cmp["summary"]["istft_near_parity"] == len(module.WORKLOADS)
    assert cmp["promotion_gate"]["regression_gate"]
    assert cmp["promotion_gate"]["large_workload_gate"]
    assert cmp["promotion_gate"]["decision"] == "PROMOTE"


def test_phase21_promotion_gate_blocks_true_regression():
    module = _load_phase21_module()
    baseline = _timings(module, 10.0, 10.0)
    # 0.95x is below the 1% tolerance floor and must count as regression.
    current = _timings(module, 10.0 / 0.95, 10.0 / 0.95)

    cmp = module._compare_with_baseline(current, baseline)

    assert cmp["summary"]["regressions_total"] > 0
    assert not cmp["promotion_gate"]["regression_gate"]
    assert cmp["promotion_gate"]["decision"] != "PROMOTE"


