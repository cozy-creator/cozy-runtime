"""Compute parking frees idle child processes while durable admission remains available."""

from __future__ import annotations

import dataclasses
import time

import pytest

from cozy_runtime.internal.worker import machine_compute, machine_slots
from cozy_runtime.protocol import worker_pb2 as pb
from test_cpu_slot_model_inputs import _cpu_worker
from test_end_to_end import NO_EXECUTOR
from test_machine_execution import complete, offer


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_idle_executor_parks_and_next_request_reuses_machine_with_journal_intact() -> None:
    with _cpu_worker() as (worker, binding, descriptor):
        worker.options = dataclasses.replace(worker.options, sole_supervisor=True)
        worker.fence.record_owner_id = "owner"
        assert worker.executions is not None
        workspace_id = worker.executions.workspace_id
        directive = pb.JobDirective(
            installation_id="install-cpu", job_descriptor_id=descriptor, orchestration=True
        )
        assert machine_slots.ensure(worker, "first", directive, binding)
        first = worker.job_slots[machine_slots.key("first")].supervision.current
        assert first is not None
        machine_slots.release(worker, "first")
        assert not machine_compute.park_if_idle(worker)  # short gaps retain a warm process
        assert first.alive()

        worker.compute_idle_since = time.monotonic() - machine_compute.IDLE_SECONDS - 1
        with worker.preparation_lock:
            assert not machine_compute.park_if_idle(worker)
        worker.executions.submit(
            "owner",
            "pending",
            b"c" * 32,
            offer("pending"),
            expected_execution_workspace_id=workspace_id,
        )
        worker.compute_idle_since = time.monotonic() - machine_compute.IDLE_SECONDS - 1
        assert not machine_compute.park_if_idle(worker)  # durable queue wins without a unit
        assert first.alive()
        outcome = complete(worker.executions, "pending")

        worker.compute_idle_since = time.monotonic() - machine_compute.IDLE_SECONDS - 1
        assert machine_compute.park_if_idle(worker)
        assert not first.alive() and not worker.warm_cpu
        assert not worker.stop.is_set()
        assert worker.executions.workspace_id == workspace_id
        assert worker.executions.outcome("owner", "pending") == outcome

        assert machine_slots.ensure(worker, "second", directive, binding)
        second = worker.job_slots[machine_slots.key("second")].supervision.current
        assert second is not None and second.alive() and second.pid != first.pid
        machine_slots.release(worker, "second")
