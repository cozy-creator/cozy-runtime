"""Admitted job devices do not depend on constructing an inference model."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cozy_runtime.internal import executor
from cozy_runtime.internal.discovery import discover


class CUDA:
    def __init__(self, *, initialized: bool = False):
        self.initialized = initialized
        self.events: list[str] = []

    def is_initialized(self):
        return self.initialized

    def init(self):
        self.events.append("init")
        self.initialized = True

    def reset_peak_memory_stats(self):
        self.events.append("reset")

    def synchronize(self):
        self.events.append("synchronize")

    def max_memory_allocated(self):
        return 2048

    def current_stream(self):
        return self

    def default_stream(self):
        return self

    def query(self):
        self.events.append("query")
        return "synchronize" in self.events


def job(tmp_path: Path) -> tuple[executor.Executor, dict]:
    (tmp_path / "device_job.py").write_text("""
import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    device: str
@app.job
def compute(payload: Request, ctx: Context) -> Result:
    if ctx.device.type == "cuda":
        import torch
        assert torch.cuda.is_initialized(), "Runtime must initialize before the handler"
    return Result(str(ctx.device))
""")
    (tmp_path / "package.toml").write_text('[application]\nobject="device_job:app"\n')
    found = discover(tmp_path)
    process = executor.Executor(SimpleNamespace(send=lambda _message: None), tmp_path)
    process.device_kind = "cuda"
    process.discovered = found
    process.surface_names = {surface.name: surface for surface in found.surfaces}
    return process, {
        "job": "compute",
        "request_id": "job-device",
        "payload": {},
        "deadline_s": 30,
        "spool": str(tmp_path / "spool"),
        "budget": {"gpu_count": 1},
    }


def install_torch(monkeypatch: pytest.MonkeyPatch, cuda: CUDA) -> None:
    torch = ModuleType("torch")
    torch.cuda = cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_gpu_job_initializes_before_handler_and_proves_completion(tmp_path, monkeypatch):
    process, command = job(tmp_path)
    cuda = CUDA()
    install_torch(monkeypatch, cuda)
    result = process.run_job(command)
    assert result["outcome"]["terminal"] == "succeeded", result
    observed = json.loads((Path(command["spool"]) / executor.RESULT_DOCUMENT).read_bytes())
    assert observed == {"device": "cuda:0"}
    assert cuda.events[0] == "init"
    assert cuda.events.index("synchronize") < cuda.events.index("query")
    assert result["quiescent"] and result["metrics"]["peak_vram_bytes"] == 2048


def test_cpu_parent_never_imports_torch_or_opens_cuda(tmp_path, monkeypatch):
    process, command = job(tmp_path)
    command["budget"]["gpu_count"] = 0
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name != "torch" and not name.startswith("torch."), "CPU parent imported Torch"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = process.run_job(command)
    assert result["outcome"]["terminal"] == "succeeded", result
    observed = json.loads((Path(command["spool"]) / executor.RESULT_DOCUMENT).read_bytes())
    assert observed == {"device": "cpu"}
    assert process.torch is None and not process._device_ready
    assert result["metrics"]["gpu_count"] == 0 and result["quiescent"]


def test_gpu_job_refuses_cuda_initialized_outside_runtime(tmp_path, monkeypatch):
    process, command = job(tmp_path)
    cuda = CUDA(initialized=True)
    install_torch(monkeypatch, cuda)
    result = process.run_job(command)
    assert result["ok"] is False and result["code"] == "cuda_initialized_before_runtime"
    assert process.attempts == 0 and cuda.events == []
