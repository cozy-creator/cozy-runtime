"""Compile-only preparation never measures or launches a CUDA execution context."""

import socket
import sys
from ctypes import c_int
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pytest

from cozy_runtime.author import AttentionContext, Model, uses_components
from cozy_runtime.internal import (
    accel,
    attention,
    executor,
    kernel_cache,
    kernel_compile,
    kernel_sources,
)
from cozy_runtime.internal.executor_commands import Warm
from cozy_runtime.internal.seam import Channel


class _IntReference(Protocol):
    _obj: c_int


def test_architecture_query_uses_nvml_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def query(handle: str, major: _IntReference, minor: _IntReference) -> int:
        assert handle == "device"
        major._obj.value, minor._obj.value = 8, 9
        return 0

    monkeypatch.setattr(
        accel,
        "_nvml_library",
        lambda: SimpleNamespace(
            nvmlDeviceGetCudaComputeCapability=query,
        ),
    )
    monkeypatch.setattr(accel, "_nvml_handle", lambda library, device: device)
    assert accel.device_architecture("device", "cuda") == 89
    assert accel.device_architecture("device", "cpu") is None


def test_import_only_executor_submits_fusion_with_gpu_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ours, theirs = socket.socketpair()
    owner = executor.Executor(Channel(ours), tmp_path)
    owner.imported, owner.started = True, False
    owner.torch = object()
    owner.device_kind = "cuda"
    monkeypatch.setattr(executor, "_sealed_entries", lambda: ("device",))
    monkeypatch.setattr(accel, "device_architecture", lambda *_: 89)
    monkeypatch.setattr(
        owner,
        "_model_classes",
        lambda: [SimpleNamespace(__fusion__="accept", choose_attention=Model.choose_attention)],
    )
    monkeypatch.setattr(
        owner, "_warm_rank", lambda: (_ for _ in ()).throw(AssertionError("launch"))
    )
    monkeypatch.setattr(
        attention, "prepare", lambda arch, preferred=(): [{"arch": arch, "status": "deferred"}]
    )
    monkeypatch.setattr(kernel_cache, "configured", lambda: object())
    job = kernel_compile.Job("fusion", kernel_cache.Key.of("fusion", {"arch": 89}))
    monkeypatch.setitem(
        sys.modules, "cozy_runtime.internal.fusion", SimpleNamespace(job=lambda arch: job)
    )
    submitted: list[kernel_compile.Job] = []

    def submit(_store: object, value: kernel_compile.Job) -> SimpleNamespace:
        submitted.append(value)
        return SimpleNamespace(line=lambda: "compiling")

    monkeypatch.setattr(kernel_compile, "submit", submit)
    try:
        reply = owner.warm(Warm(compile_only=True))
        assert reply["ok"] and reply["ranks"][0]["compile_only"]
        assert not owner.started and not owner._device_ready
        assert submitted[0].key == job.key
        assert submitted[0].env == {"CUDA_VISIBLE_DEVICES": ""}
    finally:
        ours.close()
        theirs.close()


def test_attention_offline_preparation_does_not_launch_or_run_device_recipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = attention.CANDIDATES[-1]
    candidates = (
        replace(base, name="native", artifact="native"),
        replace(base, name="python", artifact="python"),
    )
    monkeypatch.setattr(attention, "CANDIDATES", candidates)
    monkeypatch.setattr(
        attention, "BY_NAME", {candidate.name: candidate for candidate in candidates}
    )
    monkeypatch.setattr(kernel_cache, "configured", lambda: object())
    monkeypatch.setattr(
        attention, "_resolve", lambda *_: (_ for _ in ()).throw(AssertionError("launch"))
    )
    monkeypatch.setattr(
        kernel_sources,
        "job",
        lambda name, arch: kernel_compile.Job(
            "setup" if name == "native" else "python",
            kernel_cache.Key.of(name, {"arch": arch}),
        ),
    )
    submitted: list[kernel_compile.Job] = []

    def submit(_store: object, job: kernel_compile.Job) -> kernel_compile.State:
        submitted.append(job)
        return kernel_compile.State(job.key.kernel, "compiling")

    monkeypatch.setattr(kernel_compile, "submit", submit)
    rows = attention.prepare(89)
    assert rows[0]["status"] == "deferred"
    assert [job.kind for job in submitted] == ["python"]
    assert submitted[0].env == {"CUDA_VISIBLE_DEVICES": ""}


def test_real_fusion_job_is_keyed_without_querying_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    from cozy_runtime.internal import fusion

    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_: (_ for _ in ()).throw(AssertionError("device query during offline preparation")),
    )
    monkeypatch.setattr(triton.knobs.cache, "dir", str(tmp_path / "triton"))
    before = torch.cuda.is_initialized()
    job = fusion.job(89)  # Explicit compiler target, not a claim about any physical device.
    assert job.kind == "fusion" and job.spec["arch"] == 89
    assert job.spec["entries"]
    assert torch.cuda.is_initialized() == before


def test_authored_native_preference_reaches_offline_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preferred = executor.Executor._preferred_attention
    compile_rank = executor.Executor._compile_rank

    class Family(Model[object]):
        @uses_components("dit")
        def run(self) -> None:
            pass

        def choose_attention(self, context: AttentionContext) -> tuple[str, ...]:
            assert (context.device_kind, context.device_name, context.sm) == ("cuda", "", 89)
            return ("sageattention", "sdpa")

    monkeypatch.setattr(executor, "_sealed_entries", lambda: ("sealed-device",))
    monkeypatch.setattr(accel, "device_architecture", lambda *_: 89)
    monkeypatch.setattr(kernel_cache, "configured", lambda: object())
    job = kernel_compile.Job(
        "setup",
        kernel_cache.Key.of("sageattention", {"arch": 89}),
        env={"CUDA_VISIBLE_DEVICES": "", "PYTORCH_NVML_BASED_CUDA_CHECK": "1"},
    )
    monkeypatch.setattr(kernel_sources, "job", lambda artifact, arch: job)
    submitted: list[kernel_compile.Job] = []

    def submit(_store: object, job: kernel_compile.Job) -> kernel_compile.State:
        submitted.append(job)
        return kernel_compile.State("sageattention", "compiling")

    monkeypatch.setattr(kernel_compile, "submit", submit)
    owner = SimpleNamespace(rank=0, device_kind="cuda", world=1, _model_classes=lambda: [Family])
    owner._preferred_attention = partial(preferred, cast(executor.Executor, owner))
    result = compile_rank(cast(executor.Executor, owner))
    assert len(submitted) == 1
    assert submitted[0].env["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    assert result["kernels"][0]["kernel"] == "sageattention"
