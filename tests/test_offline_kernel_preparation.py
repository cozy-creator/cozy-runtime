"""Compile-only preparation never measures or launches a CUDA execution context."""

import socket
import sys
from dataclasses import replace
from types import SimpleNamespace

from cozy_runtime.internal import accel, attention, executor, kernel_cache, kernel_compile
from cozy_runtime.internal.executor_commands import Warm
from cozy_runtime.internal.seam import Channel


def test_architecture_query_uses_nvml_only(monkeypatch):
    def query(handle, major, minor):
        assert handle == "device"
        major._obj.value, minor._obj.value = 8, 9
        return 0

    monkeypatch.setattr(accel, "_nvml_library", lambda: SimpleNamespace(
        nvmlDeviceGetCudaComputeCapability=query,
    ))
    monkeypatch.setattr(accel, "_nvml_handle", lambda library, device: device)
    assert accel.device_architecture("device", "cuda") == 89
    assert accel.device_architecture("device", "cpu") is None


def test_import_only_executor_submits_fusion_with_gpu_hidden(monkeypatch, tmp_path):
    ours, theirs = socket.socketpair()
    owner = executor.Executor(Channel(ours), tmp_path)
    owner.imported, owner.started = True, False
    owner.torch = object()
    owner.device_kind = "cuda"
    monkeypatch.setattr(executor, "_sealed_entries", lambda: ("device",))
    monkeypatch.setattr(accel, "device_architecture", lambda *_: 89)
    monkeypatch.setattr(owner, "_model_classes", lambda: [SimpleNamespace(__fusion__="accept")])
    monkeypatch.setattr(
        owner, "_warm_rank", lambda: (_ for _ in ()).throw(AssertionError("launch"))
    )
    monkeypatch.setattr(attention, "prepare", lambda arch: [{"arch": arch, "status": "deferred"}])
    monkeypatch.setattr(kernel_cache, "configured", lambda: object())
    job = kernel_compile.Job("fusion", kernel_cache.Key.of("fusion", {"arch": 89}))
    monkeypatch.setitem(
        sys.modules, "cozy_runtime.internal.fusion", SimpleNamespace(job=lambda arch: job)
    )
    submitted = []
    monkeypatch.setattr(kernel_compile, "submit", lambda store, value: (
        submitted.append(value) or SimpleNamespace(line=lambda: "compiling")
    ))
    try:
        reply = owner.warm(Warm(compile_only=True))
        assert reply["ok"] and reply["ranks"][0]["compile_only"]
        assert not owner.started and not owner._device_ready
        assert submitted[0].key == job.key
        assert submitted[0].env == {"CUDA_VISIBLE_DEVICES": ""}
    finally:
        ours.close()
        theirs.close()


def test_attention_offline_preparation_does_not_launch_or_run_device_recipes(monkeypatch):
    base = attention.CANDIDATES[-1]
    monkeypatch.setattr(attention, "CANDIDATES", (
        replace(base, name="native", artifact="native"),
        replace(base, name="python", artifact="python"),
    ))
    monkeypatch.setattr(kernel_cache, "configured", lambda: object())
    monkeypatch.setattr(
        attention, "_resolve", lambda *_: (_ for _ in ()).throw(AssertionError("launch"))
    )
    monkeypatch.setattr(attention.kernel_sources, "job", lambda name, arch: kernel_compile.Job(
        "setup" if name == "native" else "python", kernel_cache.Key.of(name, {"arch": arch}),
    ))
    submitted = []
    monkeypatch.setattr(kernel_compile, "submit", lambda store, job: (
        submitted.append(job) or kernel_compile.State(job.key.kernel, "compiling")
    ))
    rows = attention.prepare(89)
    assert rows[0]["status"] == "deferred"
    assert [job.kind for job in submitted] == ["python"]
    assert submitted[0].env == {"CUDA_VISIBLE_DEVICES": ""}


def test_real_fusion_job_is_keyed_without_querying_cuda(monkeypatch, tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    from cozy_runtime.internal import fusion

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (
        _ for _ in ()
    ).throw(AssertionError("device query during offline preparation")))
    monkeypatch.setattr(triton.knobs.cache, "dir", str(tmp_path / "triton"))
    before = torch.cuda.is_initialized()
    job = fusion.job(89)  # Explicit compiler target, not a claim about any physical device.
    assert job.kind == "fusion" and job.spec["arch"] == 89
    assert job.spec["entries"]
    assert torch.cuda.is_initialized() == before
