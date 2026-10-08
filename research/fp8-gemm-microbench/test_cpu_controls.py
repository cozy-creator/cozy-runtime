"""CPU controls for numerical reports and real ChromeTrace parsing; no CUDA work."""

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).parent / "fp8_gemm_microbench"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantized_reference_factorization_and_finite_error():
    measure = load("measure")
    a = torch.tensor([[1.0, -0.5, 0.125], [-1.5, 0.25, 0.75]]).to(torch.float8_e4m3fn)
    b = torch.tensor([[0.25, -1.0, 2.0], [1.5, -0.125, 0.5]]).to(torch.float8_e4m3fn)
    rows = torch.tensor([[0.031], [0.12]], dtype=torch.float32)
    cols = torch.tensor([[0.04, 0.077]], dtype=torch.float32)
    scaled = ((a.float() @ b.float().t()) * rows) * cols
    dequantized = (a.float() * rows) @ (b.float() * cols.t()).t()
    torch.testing.assert_close(scaled, dequantized, atol=1e-8, rtol=1e-6)
    good = measure._error(torch, scaled, scaled)
    assert good["finite"] and good["max_abs"] == 0 and good["relative_l2"] == 0
    bad = measure._error(torch, scaled * float("nan"), scaled)
    assert not bad["finite"] and bad["rmse"] is None


def test_trace_summary_does_not_invent_gpu_events(tmp_path):
    probe = load("trace_probe")
    path = tmp_path / "cpu.json"
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with torch.profiler.record_function("probe/cpu-control"):
            value = torch.ones(16, 16) @ torch.ones(16, 16)
    assert value[0, 0] == 16
    profiler.export_chrome_trace(str(path))
    report = probe.summarize(path, ["probe/cpu-control", "probe/missing"], {})
    assert report["kernel_events"] == 0 and report["cuda_runtime_events"] == 0
    assert report["recorded_labels"] == ["probe/cpu-control"]
    assert report["missing_labels"] == ["probe/missing"]
    assert report["categories"]["cpu_op"] > 0
