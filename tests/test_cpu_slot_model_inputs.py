"""A device-less job root takes the CPU slot even when it holds derive-only Models or weights.

Run 1223: a client script with Model inputs and weights outputs held the device lane, so the
GPU child it called was refused as coexisting with a device ancestor.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from cozy_runtime.author import Invocation, attempt, describe, script_app
from cozy_runtime.internal import canonical, child_env, package_interface, static_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker import machine_slots
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

HOLDING = {"models": [{"path": "main.models.source"}], "weights_outputs": [{"output_id": "x"}]}


def test_the_device_grant_alone_decides_the_slot() -> None:
    assert machine_slots.admits(pb.JobDirective(orchestration=True), HOLDING)
    assert machine_slots.MODEL_INPUTS
    for granted in (
        pb.JobDirective(device_count=1),
        pb.JobDirective(resource_caps=pb.ResourceCaps(device_required=True)),
        pb.JobDirective(resource_caps=pb.ResourceCaps(max_device_memory_bytes=1)),
    ):
        assert not machine_slots.admits(granted, {})


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_a_root_with_models_and_weights_gets_its_own_cpu_lane() -> None:
    # A short root: the executor's control socket path must fit `sockaddr_un`.
    with tempfile.TemporaryDirectory(prefix="cpu") as short:
        root = Path(short)
        interface = canonical.write(
            {
                "application": "cpu_fixture:app",
                "format": package_interface.SCHEMA,
                "entrypoints": [],
                "jobs": [
                    {
                        "name": "main",
                        "request": {"fields": []},
                        "result": {"fields": []},
                        "publishes": False,
                        "accelerator": False,
                        "models": [
                            {"class": "Source", "component_use": {}, "path": "main.models.source"}
                        ],
                        "weights_outputs": [
                            {
                                "output_id": "model",
                                "mime_type": "application/vnd.cozy.model-manifest",
                                "max_bytes": 1 << 20,
                            }
                        ],
                    }
                ],
            }
        )
        path = root / "package-interface.json"
        path.write_bytes(interface)
        descriptor = package_interface.job_descriptor_id(json.loads(interface), "main")
        binding = JobBinding(
            job_descriptor_id=descriptor,
            installation_id="install-cpu",
            application="cpu_fixture:app",
            package_interface=str(path),
            python=sys.executable,
            job="main",
        )
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                child_base_env=tuple(
                    sorted(
                        (key, value)
                        for key, value in os.environ.items()
                        if not child_env.erased(key) and key != "PYTHONPATH"
                    )
                ),
            ),
            WorkerOptions(
                root=root / "worker",
                tensorfs_root=root / "store",
                devices="",
                accelerator_backend="none",
                python=sys.executable,
                worker_id="cpu",
            ),
            InMemoryControlHost(),
        )
        directive = pb.JobDirective(
            installation_id="install-cpu", job_descriptor_id=descriptor, orchestration=True
        )
        try:
            assert machine_slots.ensure(worker, "request-cpu", directive, binding)
            slot = worker.job_slots[machine_slots.key("request-cpu")]
            assert slot.ready() and slot.lane.lane_id.startswith("cpu-")
            granted = pb.JobDirective(
                installation_id="install-cpu", job_descriptor_id=descriptor, device_count=1
            )
            assert not machine_slots.ensure(worker, "request-gpu", granted, binding)
        finally:
            machine_slots.release(worker, "request-cpu")
            worker.shutdown()


def _script(tmp_path: Path, name: str, metadata: str, body: str) -> Path:
    project = tmp_path / name
    project.mkdir()
    (project / "pyproject.toml").write_text(f'[project]\nname="{name}"\nversion="0.1.0"\n')
    (project / "package.toml").write_text(f'[application]\nobject="{name}_entry:app"\n')
    (project / f"{name}_entry.py").write_text(
        f'from cozy_runtime.author import script_app\napp = script_app("{name}")\n'
    )
    (project / f"{name}.py").write_text(metadata + body)
    return project


@pytest.mark.parametrize(
    ("metadata", "declared"),
    [("", False), ("# /// script\n# [tool.cozy]\n# accelerator = true\n# ///\n", True)],
)
def test_a_script_declares_its_accelerator_in_both_readers(
    tmp_path: Path, metadata: str, declared: bool
) -> None:
    name = "accel_script_" + str(declared).lower()
    project = _script(tmp_path, name, metadata, "def main():\n    pass\n")
    sys.modules.pop(name, None)
    static = static_interface.build(project)
    imported = package_interface.build(discover(project))
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(imported)
    assert static["jobs"][0]["accelerator"] is declared


def test_a_cpu_script_that_uses_cuda_names_the_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # torch's own words when a process with no visible device asks for CUDA.
    (tmp_path / "cuda_script.py").write_text(
        "def main():\n    raise RuntimeError('No CUDA GPUs are available')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    app = script_app("cuda_script")
    describe(app)
    _, outcome, _ = attempt(
        app.get("main"), {}, Invocation("cuda", tmp_path / "out", time.monotonic() + 5)
    )
    assert outcome.terminal == "failed" and outcome.code == "job_needs_accelerator", outcome
    assert "accelerator = true" in outcome.message and "[tool.cozy]" in outcome.message
