"""Host RAM is a sizing fact, never a gate: work that outgrows memory pages to disk."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, child_env, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker import machine_slots
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

ROOT = Path(__file__).resolve().parents[1]
BUDGETED_VERIFY = """
import resource, sys
from pathlib import Path
from cozy_runtime.internal.worker import acquire
size = int(next(l for l in open("/proc/self/status") if l.startswith("VmSize")).split()[1])
budget = (size << 10) + (128 << 20)
resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
acquire._verify_file(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
print("verified", budget)
"""


def test_cached_artifact_larger_than_the_memory_budget_verifies(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    length = 512 << 20
    with artifact.open("wb") as stream:
        stream.truncate(length)
    with artifact.open("rb") as stream:
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    result = subprocess.run(
        [sys.executable, "-c", BUDGETED_VERIFY, str(artifact), digest, str(length)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    budget = int(result.stdout.split()[1])
    assert budget < length


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_cpu_composition_declaring_more_memory_than_the_host_is_admitted() -> None:
    # A short root: the executor's control socket path must fit `sockaddr_un`.
    with tempfile.TemporaryDirectory(prefix="ram") as short:
        _cpu_composition(Path(short))


def _cpu_composition(tmp_path: Path) -> None:
    interface = canonical.write(
        {
            "application": "ram_fixture:app",
            "format": package_interface.SCHEMA,
            "entrypoints": [],
            "jobs": [
                {
                    "name": "main",
                    "request": {"fields": []},
                    "result": {"fields": []},
                    "publishes": False,
                }
            ],
        }
    )
    path = tmp_path / "package-interface.json"
    path.write_bytes(interface)
    descriptor = package_interface.job_descriptor_id(json.loads(interface), "main")
    binding = JobBinding(
        job_descriptor_id=descriptor,
        installation_id="install-ram",
        application="ram_fixture:app",
        package_interface=str(path),
        python=sys.executable,
        job="main",
    )
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
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
            root=tmp_path / "worker",
            tensorfs_root=tmp_path / "store",
            devices="",
            accelerator_backend="none",
            python=sys.executable,
            worker_id="ram",
        ),
        InMemoryControlHost(),
    )
    # A declared cap far beyond any host: the OS pages, the worker does not refuse.
    directive = pb.JobDirective(
        installation_id="install-ram",
        job_descriptor_id=descriptor,
        orchestration=True,
        resource_caps=pb.ResourceCaps(max_rss_bytes=1 << 50),
    )
    try:
        assert machine_slots.ensure(worker, "request-ram", directive, binding)
        assert worker.job_slots[machine_slots.key("request-ram")].ready()
    finally:
        machine_slots.release(worker, "request-ram")
        worker.shutdown()
