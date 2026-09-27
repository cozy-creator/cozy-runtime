from __future__ import annotations

import importlib.metadata
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import grpc
import pytest

from cozy_runtime.author import describe
from cozy_runtime.derive.operations import app
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.numerical_environment import NumericalEnvironmentRefusal, fingerprint
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.internal.worker.machine_child_target import Target, computation
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc


def quantize_key(root: Path, numerical: bytes) -> bytes:
    document = package_interface.build(
        Discovered(
            app,
            "cozy_runtime.derive.operations:app",
            root,
            sys.modules[app.get("quantize").fn.__module__],
            describe(app),
            {},
        )
    )
    rows: Any = document["jobs"]
    declaration = next(row for row in rows if row["name"] == "quantize")
    identity = declaration["invocable"]["operation_identity"]
    target = Target("installation-is-not-computation", "quantize", declaration, {}, None, identity)
    return computation(
        target,
        {
            "source": {"manifest": {"digest": "sha256:" + "a" * 64, "length": 164}},
            "plan": {"components": ["model"]},
            "encoding": "fp8-rowwise/1",
        },
        numerical,
    )


def test_actual_managed_key_ignores_runtime_inventory_but_tracks_declared_native_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = tmp_path / "cozy_runtime-0.9.0.dist-info"
    sdk.mkdir()
    (sdk / "METADATA").write_text("Name: cozy-runtime\nVersion: 0.9.0\n")
    (sdk / "RECORD").write_text("cozy_runtime/caller.py,sha256=first,128\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    installed = importlib.metadata.distribution("tensorfs")
    discover = importlib.metadata.distribution

    def no_sdk_inventory(name: str) -> importlib.metadata.Distribution:
        assert name.replace("_", "-") != "cozy-runtime", "the SDK is not a numerical dependency"
        return discover(name)

    monkeypatch.setattr(importlib.metadata, "distribution", no_sdk_inventory)
    policy = fingerprint(HostFacts(backend="cpu"), threads=1, inherited={})
    first = quantize_key(tmp_path, policy)
    assert len(first) == 32
    (sdk / "METADATA").write_text("Name: cozy-runtime\nVersion: 99.0\n")
    (sdk / "RECORD").write_text("cozy_runtime/unrelated_caller.py,sha256=changed,256\n")
    (sdk / "direct_url.json").write_text('{"dir_info":{"editable":true}}')
    assert (
        quantize_key(tmp_path, fingerprint(HostFacts(backend="cpu"), threads=1, inherited={}))
        == first
    )
    (sdk / "RECORD").unlink()
    assert quantize_key(tmp_path, policy) == first

    native = tmp_path / "tensorfs-999.0.dist-info"
    native.mkdir()
    metadata, wheel = installed.read_text("METADATA"), installed.read_text("WHEEL")
    assert metadata is not None and wheel is not None
    (native / "METADATA").write_text(
        metadata.replace(f"Version: {installed.version}\n", "Version: 999.0\n", 1)
    )
    (native / "WHEEL").write_text(wheel)
    assert quantize_key(tmp_path, policy) != first


def test_gpu_execution_policy_changes_keys_without_capacity_or_diagnostic_noise(
    tmp_path: Path,
) -> None:
    device = HostFacts(backend="cuda", gpu_sm=89, driver_version="580.1")

    def key(selected: HostFacts = device, threads: int = 4, **flags: str) -> bytes:
        return quantize_key(tmp_path, fingerprint(selected, threads=threads, inherited=flags))

    first = key()
    assert key(replace(device, gpu_name="another label", vram_total_bytes=128 << 30)) == first
    assert (
        key(
            COZY_BOOTSTRAP_CREDENTIAL="never-key-material",
            PYTORCH_CUDA_ALLOC_CONF="changed",
            CUDA_LAUNCH_BLOCKING="1",
            OMP_DISPLAY_ENV="TRUE",
        )
        == first
    )
    assert key(HostFacts(backend="cpu")) != first
    assert key(replace(device, gpu_sm=90)) != first
    # A provider's host driver update is not a numerical change.
    assert key(replace(device, driver_version="590.1")) == first
    assert key(replace(device, driver_version="")) == first
    assert key(NVIDIA_TF32_OVERRIDE="0") != first
    assert key(CUBLAS_WORKSPACE_CONFIG=":4096:8") != first
    assert key(threads=1) != first
    with pytest.raises(NumericalEnvironmentRefusal):
        key(replace(device, gpu_sm=0))


def test_numerical_environment_rpc_refuses_unproven_identity_without_detail_leak() -> None:
    def unproven() -> bytes:
        raise NumericalEnvironmentRefusal("private/path/or/config")

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_RuntimePreparationServicer_to_server(
        _PreparationServicer(None, None, None, None, numerical_environment=unproven), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            client = rpc.RuntimePreparationStub(channel)
            with pytest.raises(grpc.RpcError) as failure:
                client.NumericalEnvironment(pb.NumericalEnvironmentRequest(), timeout=5)
            assert failure.value.code() == grpc.StatusCode.FAILED_PRECONDITION
            assert "private/path" not in failure.value.details()
    finally:
        server.stop(None).wait()
