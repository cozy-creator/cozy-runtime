"""A real job keeps its process and code while another package is prepared."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from conftest import image_python
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.job_slots import JobSlot
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR, _development, _measured_revision, _run, _wheel_rows

SOURCE = """from pathlib import Path
import time
import msgspec
from cozy_runtime.author import App, Context
app=App()
class Request(msgspec.Struct):
    entered: str = ""
    gate: str = ""
class Result(msgspec.Struct):
    value: int
@app.job
def JOB(ctx: Context, payload: Request) -> Result:
    if payload.entered:
        Path(payload.entered).write_text("entered")
        while not Path(payload.gate).exists():
            ctx.raise_if_cancelled()
            time.sleep(0.02)
    return Result(7)
"""


def package(root: Path, label: str) -> tuple[Path, pb.PrepareLocalPackageRequest]:
    project = root / ("project_" + label)
    project.mkdir()
    name, module = "prepare-" + label, "prepare_" + label
    (project / (module + ".py")).write_text(SOURCE.replace("JOB", label))
    (project / "pyproject.toml").write_text(
        f'[project]\nname="{name}"\nversion="1.0.0"\n'
        'requires-python=">=3.12,<3.13"\ndependencies=["cozy-runtime>=0.16.8,<1"]\n'
        f'[project.entry-points."cozy.application"]\ndefault="{module}:app"\n'
        '[build-system]\nrequires=["hatchling"]\nbuild-backend="hatchling.build"\n'
        f'[tool.hatch.build.targets.wheel]\nonly-include=["{module}.py"]\n'
    )
    environment = root / "environments"
    wheels = environment / ".stage" / label / "wheels"
    wheels.mkdir(parents=True)
    uv = shutil.which("uv")
    assert uv is not None
    _run(uv, "build", "--wheel", "--out-dir", str(wheels), str(project))
    (wheel,) = wheels.glob("*.whl")
    return environment, pb.PrepareLocalPackageRequest(
        operation_id=label,
        package=_development("local/" + name, "1.0.0"),
        wheels=_wheel_rows([wheel]),
        install_root=str(environment),
    )


def measure(worker: Worker, request: pb.PrepareLocalPackageRequest) -> None:
    with pytest.raises(PreparationRefusal) as missing:
        worker.prepare_local_package(request)
    request.package.local_revision_digest = _measured_revision(missing.value)


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_job_only_worker_preparation_preserves_running_job_and_exact_readiness() -> None:
    # Short socket paths and a real delegated executor; no device/model is needed.
    with tempfile.TemporaryDirectory(prefix="cz-job-prep.") as raw:
        root = Path(raw)
        _, first = package(root, "first")
        environment, other = package(root, "other")
        (root / "artifacts").mkdir()
        python = image_python()
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(
                sorted(
                    (k, v)
                    for k, v in os.environ.items()
                    if not child_env.erased(k)
                    and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
                )
            ),
        )
        request = LocalRequest(
            entrypoint="first",
            payload={"entered": str(root / "entered"), "gate": str(root / "gate")},
            outputs=(),
            kind="job",
            request_id="first-root",
            timeout_ms=60000,
        )
        owner = LocalRecordOwner(request, {}, root / "run", package_code_digest="")
        worker = Worker(
            config,
            WorkerOptions(
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(python),
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
            ),
            owner.host,
        )
        measure(worker, other)
        measure(worker, first)
        worker.prepare_local_package(first)
        # These fixture preparations precede control-service startup. Retire an
        # older Runtime's idle describe process before run's orphan sweep; the
        # regression below concerns the later preparation during actual work.
        if worker.supervision.current is not None:
            worker.supervision.retire_current(
                worker.supervision.current, "fixture begins control service"
            )
        plans = root / "home/job-plans"
        records = [json.loads(p.read_bytes()) for p in plans.glob("*/*.json")]
        plan = next(row for row in records if row.get("job") == "first")
        binding = JobBinding.read(plan)
        owner.request = dataclasses.replace(request, job_descriptor_id=binding.job_descriptor_id)
        owner.package_code_digest = binding.build_id
        observed: dict[str, Any] = {}
        failures: list[BaseException] = []

        def prepare_other() -> None:
            try:
                until = time.monotonic() + 45
                while not (root / "entered").exists():
                    if owner.done.is_set() or time.monotonic() > until:
                        raise AssertionError("the first author job never entered")
                    time.sleep(0.02)
                current = worker.supervision.current
                assert current is not None and current.alive()
                before = (current.epoch, current.pid)
                slot = JobSlot(
                    pb.JobDirective(), binding, worker.supervision, worker.lanes.envelope
                )
                assert slot.ready()
                worker.prepare_local_package(other)
                after = worker.supervision.current
                assert after is current and (after.epoch, after.pid) == before
                assert after.alive() and slot.ready()
                records = [json.loads(p.read_bytes()) for p in plans.glob("*/*.json")]
                changed = JobBinding.read(next(row for row in records if row.get("job") == "other"))
                wrong = JobSlot(
                    pb.JobDirective(), changed, worker.supervision, worker.lanes.envelope
                )
                assert not wrong.ready(), "a live process is not the newly selected code"
                worker.supervision.use_environment(
                    changed.python, changed.environment_content_digest
                )
                assert slot.ready() and not wrong.ready(), (
                    "readiness read mutable next-spawn selection"
                )
                worker.supervision.use_environment(
                    binding.python, binding.environment_content_digest
                )
                observed.update(epoch=before[0], pid=before[1], unchanged=True)
            except BaseException as exc:
                failures.append(exc)
            finally:
                (root / "gate").touch()

        thread = threading.Thread(target=prepare_other, daemon=True)
        worker.watches.tap = owner.tap_progress
        owner.watch(worker)
        thread.start()
        worker.run()
        thread.join(60)
        assert not thread.is_alive()
        assert not failures, failures
        assert observed.get("unchanged")
        assert owner.terminal is not None
        # The canonical outcome is already checked by LocalRecordOwner.
        from cozy_runtime.protocol import documents

        result = documents.read(owner.terminal.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        assert result["status"] == pb.OUTCOME_STATUS_SUCCEEDED, result
