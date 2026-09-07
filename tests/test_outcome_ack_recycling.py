"""Old terminal cleanup cannot recycle a replacement private job."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.job_slots import JobSlot
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb


@pytest.mark.parametrize("mode", ["duplicate", "replacement", "managed", "ordinary"])
def test_only_a_new_current_ordinary_terminal_ack_can_recycle(tmp_path: Path, mode: str) -> None:
    tensorfs.Store.init(tmp_path / "store")
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=tmp_path / "store", no_warm=True),
        InMemoryControlHost(),
    )
    try:
        job = JobBinding(
            "sha256:" + "11" * 32,
            "sha256:" + "12" * 32,
            "private_job:app",
            str(tmp_path / "interface.json"),
            sys.executable,
            "sha256:" + "13" * 32,
            "compute",
        )
        worker.jobs[job.job_descriptor_id] = job
        worker.reclaim_on_terminal = True
        record = AttemptRecord("old-child", 1, b"\x14" * 32, {}, kind="job", job=job)
        worker.engine.history[(record.request_id, record.attempt)] = record
        worker.engine.live[record.request_id] = record
        # An actual worker-created cancellation terminal before author execution.
        worker.engine.outcome(
            record,
            pb.OUTCOME_STATUS_CANCELED,
            pb.CAUSE_CODE_CLIENT_CANCEL,
            pb.CAUSE_ORIGIN_CLIENT,
            "owner stopped child",
        )
        ack = pb.AttemptOutcomeAck(
            request_id=record.request_id,
            attempt_ordinal=record.attempt,
            outcome_id=record.outcome_id,
            outcome_digest=record.outcome_digest,
        )
        if mode == "duplicate":
            ack.retain_work = True
            worker.on_outcome_ack(ack)
            assert record.state == "closed" and not worker.stop.is_set()
            ack.retain_work = False
        if mode == "replacement":
            worker.jobs[job.job_descriptor_id] = JobBinding(
                job.job_descriptor_id,
                "sha256:" + "15" * 32,
                job.application,
                job.package_interface,
                job.python,
                job.environment_content_digest,
                job.job,
            )
        if mode == "managed":
            worker.job_slots["envelope"] = JobSlot(
                pb.JobDirective(), job, worker.supervision, worker.lanes.envelope
            )
        worker.on_outcome_ack(ack)
        assert record.state == "closed"
        assert worker.stop.is_set() is (mode == "ordinary")
    finally:
        worker.shutdown()
