"""A real worker keeps its desired state across harmless owner skew and scopes bad plans."""

from __future__ import annotations

import json
from pathlib import Path

from cozy_runtime.internal import job_plan
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions, desired_state_body
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb

DESCRIPTOR = "sha256:" + "d" * 64
INSTALLATION = "install-skew"


def _worker(root: Path) -> Worker:
    return Worker(
        RuntimeConfig(cozy_home=root / "home", credentials=Credentials()),
        WorkerOptions(root=root / "worker", tensorfs_root=root / "store", worker_id="skew"),
        InMemoryControlHost(),
    )


def test_same_revision_at_another_owner_minor_is_the_same_desired_state() -> None:
    newer = pb.DesiredWorkerState(revision=4, wire_minor=WIRE_MINOR)
    older = pb.DesiredWorkerState(revision=4, wire_minor=WIRE_MINOR - 1)
    assert desired_state_body(newer) == desired_state_body(older)
    changed = pb.DesiredWorkerState(revision=4, wire_minor=WIRE_MINOR, drain_grace_ms=1)
    assert desired_state_body(changed) != desired_state_body(newer)


def test_unreadable_job_plan_latches_only_that_job(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    try:
        plans = worker.config.cozy_home / "job-plans"
        # A plan from a Runtime whose record shape this one cannot build a job from.
        record = {"installation_id": INSTALLATION, "job_descriptor_id": DESCRIPTOR}
        job_plan.write(plans, record)
        worker.apply_job_directive(
            pb.JobDirective(
                installation_id=INSTALLATION,
                job_descriptor_id=DESCRIPTOR,
            )
        )
        assert worker.latched == "job_plan_unreadable"
        (fault,) = [row for row in worker.engine.faults if row.reason == "job_plan_unreadable"]
        assert fault.subject == DESCRIPTOR and "re-prepare" in fault.detail
        # The plan decodes once (JobBinding.read) and names the first field it lacks.
        assert "missing required field `application`" in fault.detail
        assert json.loads(job_plan.path(plans, INSTALLATION, DESCRIPTOR).read_text()) == record
    finally:
        worker.shutdown()
