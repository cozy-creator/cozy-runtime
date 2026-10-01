"""Identified explicit preparation (wire 73): disconnect detaches, cancel detaches one requester."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest
import tensorfs

import signed_claims
from cozy_runtime.internal.worker import activity, machine_model_defaults
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_machine_execution import settle
from test_model_preparation_progress import ProgressMachine, running_progress_machine
from test_model_runtime_closure import _ASSET

ASSET = hashlib.sha256(_ASSET).hexdigest()


@pytest.fixture
def machine(tmp_path: Path) -> Iterator[ProgressMachine]:
    yield from running_progress_machine(tmp_path)


def identified(machine: ProgressMachine, operation_id: str) -> pb.PreparePackageSetRequest:
    request = pb.PreparePackageSetRequest()
    request.CopyFrom(machine.request)
    request.operation_id = operation_id
    return request


def cancel(machine: ProgressMachine, operation_id: str) -> pb.CancelPreparationResult:
    call = pb.CancelPreparationCall(claim=signed_claims.claim(), operation_id=operation_id)
    answer: pb.CancelPreparationResult = machine.prepare.CancelPreparation(call)
    return answer


def cancelled(call: grpc.Future) -> bool:
    error = call.exception(timeout=60)
    return error is not None and ("cozy-error-code", "preparation_cancelled") in tuple(
        error.trailing_metadata() or ()
    )


def observers(machine: ProgressMachine) -> int:
    return machine.worker.model_preparation_observers.get(("proof/adapter", machine.manifest), 0)


def fetched_once(machine: ProgressMachine) -> bool:
    return all(machine.gets.count(digest) == 1 for digest in set(machine.gets) - {ASSET})


def test_cancelling_one_requester_leaves_the_shared_transfer_running(
    machine: ProgressMachine,
) -> None:
    first = machine.prepare.PreparePackageSet.future(identified(machine, "op-first"))
    assert machine.source_blocked.wait(10)
    second = machine.prepare.PreparePackageSet.future(identified(machine, "op-second"))
    settle(lambda: observers(machine) == 2 and machine.rows()[0].transferred_bytes > 0)
    assert not machine.worker.preparation_lock.locked()
    answer = cancel(machine, "op-second")
    assert answer.found and answer.model_progress[0].model.manifest == machine.manifest
    assert 0 < answer.model_progress[0].transferred_bytes < machine.total
    assert cancelled(second)
    assert machine.worker.explicit_preparations.live() == ["op-first"]
    assert not machine.aborted and machine.gets.count(ASSET) == 1
    machine.release_source.set()
    assert first.result(timeout=60).HasField("placement_set")
    assert len(machine.gets) == len(set(machine.gets)) == 3
    assert cancel(machine, "op-first") == pb.CancelPreparationResult()


def test_a_run_takes_over_the_transfer_its_explicit_leader_cancelled(
    machine: ProgressMachine,
) -> None:
    explicit = machine.prepare.PreparePackageSet.future(identified(machine, "op-explicit"))
    assert machine.source_blocked.wait(10)
    failures: list[BaseException] = []

    def run() -> None:
        try:
            machine_model_defaults.materialize(
                machine.worker,
                [
                    {
                        "parameter": "model",
                        "repository": "proof/adapter",
                        "manifest": {"digest": machine.manifest, "length": machine.manifest_length},
                        "public_origin": machine.request.hub,
                    }
                ],
                cancellation=tensorfs.PullCancellation(),
            )
        except BaseException as exc:
            failures.append(exc)

    writer = threading.Thread(target=run)
    writer.start()
    try:
        settle(lambda: observers(machine) == 2)
        assert cancel(machine, "op-explicit").found
        assert cancelled(explicit)
        settle(lambda: machine.aborted == [ASSET] and machine.gets.count(ASSET) == 2)
    finally:
        machine.release_source.set()
        writer.join(60)
    assert not writer.is_alive() and not failures
    assert fetched_once(machine)


def test_cancelling_the_last_requester_stops_fetches_and_a_new_request_resumes(
    machine: ProgressMachine,
) -> None:
    first = machine.prepare.PreparePackageSet.future(identified(machine, "op-cancelled"))
    assert machine.source_blocked.wait(10)
    settle(lambda: bool(machine.rows()) and machine.rows()[0].transferred_bytes > 0)
    answer = cancel(machine, "op-cancelled")
    assert answer.found and 0 < answer.model_progress[0].transferred_bytes < machine.total
    settle(lambda: machine.aborted == [ASSET])
    assert cancelled(first)
    assert not machine.worker.explicit_preparations.live()
    before = list(machine.gets)
    assert before.count(ASSET) == 1 and not observers(machine)
    machine.release_source.set()
    resumed = machine.prepare.PreparePackageSet(identified(machine, "op-resumed"), timeout=60)
    assert resumed.HasField("placement_set")
    assert machine.gets[len(before) :] == [ASSET] and fetched_once(machine)
    assert machine.rows()[0].transferred_bytes == machine.total


def test_a_disconnected_requester_attaches_again_to_its_work(
    machine: ProgressMachine, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = machine.worker.explicit_preparations
    submitted: list[str] = []
    submit = registry.submit

    def counted(request: pb.PreparePackageSetRequest) -> object:
        submitted.append(request.operation_id)
        return submit(request)

    monkeypatch.setattr(registry, "submit", counted)
    request = identified(machine, "op-detached")
    first = machine.prepare.PreparePackageSet.future(request)
    assert machine.source_blocked.wait(10)
    assert first.cancel()  # the requester disconnects; the identified work continues
    assert registry.live() == ["op-detached"] and observers(machine) == 1
    assert "preparation_operation:op-detached" in activity.holding(machine.worker)
    assert not machine.worker.request_idle_restart()
    with grpc.insecure_channel(machine.address) as reconnected:
        again = rpc.RuntimePreparationStub(reconnected).PreparePackageSet.future(request)
        settle(lambda: submitted == ["op-detached", "op-detached"])
        assert observers(machine) == 1 and not machine.aborted
        different = identified(machine, "op-detached")
        different.hub = machine.request.hub + "/other"
        with pytest.raises(grpc.RpcError, match="preparation_operation_conflict"):
            rpc.RuntimePreparationStub(reconnected).PreparePackageSet(different)
        machine.release_source.set()
        assert again.result(timeout=60).HasField("placement_set")
    assert len(machine.gets) == len(set(machine.gets)) == 3
    assert not registry.live() and not machine.worker.stop.is_set()
