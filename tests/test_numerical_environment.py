from __future__ import annotations

import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import grpc
import pytest

from cozy_runtime.internal.base_observation import Observation
from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.numerical_environment import NumericalEnvironmentRefusal, fingerprint
from cozy_runtime.internal.worker.control import _PreparationServicer
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc


def test_probe_uses_selected_interpreter_and_ignores_install_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.internal import package_environment

    root = Path(__file__).parents[1]
    wheelhouse = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheelhouse)],
        cwd=root,
        check=True,
        capture_output=True,
    )
    (wheel,) = wheelhouse.glob("*.whl")
    interpreters = []
    for name in ("first", "relocated"):
        prefix = tmp_path / name
        python = prefix / "bin/python"
        copied_wheel = tmp_path / (name + "-wheels") / wheel.name
        copied_wheel.parent.mkdir()
        shutil.copyfile(wheel, copied_wheel)
        subprocess.run(
            ["uv", "venv", str(prefix), "--python", sys.executable], check=True, capture_output=True
        )
        subprocess.run(
            ["uv", "pip", "install", "--offline", "--python", str(python), str(copied_wheel)],
            check=True,
            capture_output=True,
        )
        interpreters.append((python, package_environment.observe_base(python)))

    def wrong_control(*args: object, **kwargs: object) -> None:
        raise AssertionError("control interpreter distribution discovery was used")

    monkeypatch.setattr("importlib.metadata.distribution", wrong_control)
    device = HostFacts(backend="cpu")
    identities = [
        fingerprint(base, device, threads=1, alloc_conf="", inherited={}, python=python)
        for python, base in interpreters
    ]
    assert len(identities[0]) == 32 and identities[0] == identities[1]
    python, base = interpreters[0]
    for distributions in (
        tuple(
            (name, "0.0.0" if name == "cozy-runtime" else version)
            for name, version in base.distributions
        ),
        (*base.distributions, ("torch", "2.14.0")),
    ):
        with pytest.raises(NumericalEnvironmentRefusal):
            fingerprint(
                replace(base, distributions=distributions),
                device,
                threads=1,
                alloc_conf="",
                inherited={},
                python=python,
            )


def test_installed_code_changes_reuse_identity_without_version_or_machine_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real distribution discovery reads these wheel metadata files; no probe is mocked.
    metadata = tmp_path / "cozy_runtime-0.9.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: cozy-runtime\nVersion: 0.9.0\n"
    )
    record = metadata / "RECORD"
    record.write_text("cozy_runtime/algorithm.py,sha256=first-implementation,128\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    base = Observation(
        "cp312",
        "linux/amd64",
        "glibc2.39",
        "cuda",
        "cu130",
        "2.13.0",
        "13.0",
        "0.10.0",
        (("cozy-runtime", "0.9.0"),),
        (),
        "3.12.12",
    )
    device = HostFacts(backend="cuda", gpu_sm=89, driver_version="580.1")

    def measure(selected: HostFacts = device, threads: int = 4) -> bytes:
        return fingerprint(
            base,
            selected,
            threads=threads,
            alloc_conf="expandable_segments:True",
            inherited={"COZY_BOOTSTRAP_CREDENTIAL": "never-part-of-identity"},
        )

    first = measure()
    assert len(first) == 32
    assert measure(replace(device, gpu_name="different label", vram_total_bytes=128 << 30)) == first
    assert measure(replace(device, driver_version="590.1")) != first
    assert measure(threads=8) != first
    record.write_text("cozy_runtime/algorithm.py,sha256=corrected-implementation,128\n")
    assert measure() != first
    (metadata / "direct_url.json").write_text('{"dir_info":{"editable":true}}')
    with pytest.raises(NumericalEnvironmentRefusal):
        measure()
    (metadata / "direct_url.json").unlink()
    record.unlink()
    with pytest.raises(NumericalEnvironmentRefusal):
        measure()


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
