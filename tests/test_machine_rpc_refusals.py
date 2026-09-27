"""Known preaccept validation is terminal; unexpected exceptions remain opaque."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import pytest

from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import offer


class Aborted(Exception):
    pass


CLAIM = pb.Claim(
    record_owner_id="owner", record_owner_epoch=1, wire_minor=WIRE_MINOR, proof=b"offline"
)


@pytest.fixture
def owned(tmp_path: Path) -> Iterator[Worker]:
    """A real in-process Worker claimed by `owner`; only the gRPC context is a stand-in."""
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home", credentials=Credentials(), bootstrap_credential="offline"
        ),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=tmp_path / "store"),
        InMemoryControlHost(),
    )
    try:
        worker.serve_stream(iter([pb.RecordOwnerFrame(claim=CLAIM)]), lambda _: None)
        yield worker
    finally:
        worker.shutdown()


def _context(observed: list[Any]) -> Any:
    def abort(code: Any, detail: str) -> None:
        observed.append((code, detail))
        raise Aborted

    return SimpleNamespace(abort=abort, set_trailing_metadata=lambda _: None)


def test_noncanonical_execution_payload_is_a_known_refusal(owned: Worker) -> None:
    rpc = MachineExecutionRPC(owned)
    observed: list[Any] = []
    with pytest.raises(Aborted):
        rpc._call(CLAIM, _context(observed), lambda: canonical.parse_canonical(b'{ "value": 7 }'))
    assert observed[0][0] == grpc.StatusCode.FAILED_PRECONDITION

    def unexpected() -> None:
        raise RuntimeError("private error contents")

    with pytest.raises(Aborted):
        rpc._call(CLAIM, _context(observed), unexpected)
    assert observed[1] == (grpc.StatusCode.UNAVAILABLE, "machine execution operation failed")


def test_lookup_distinguishes_absence_from_authority_workspace_and_storage_failure(
    owned: Worker,
) -> None:
    executions = owned.executions
    assert executions is not None
    rpc = MachineExecutionRPC(owned)
    observed: list[Any] = []
    context = _context(observed)
    query = pb.MachineExecutionQuery(
        claim=CLAIM, request_id="request", expected_execution_workspace_id=executions.workspace_id
    )
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop()[0] == grpc.StatusCode.NOT_FOUND
    query.expected_execution_workspace_id = "wrong-workspace"
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop()[0] == grpc.StatusCode.FAILED_PRECONDITION
    query.expected_execution_workspace_id = executions.workspace_id
    query.claim.record_owner_id = "other"
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop()[0] == grpc.StatusCode.FAILED_PRECONDITION
    query.claim.record_owner_id = "owner"
    executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        offer("request"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    assert rpc.GetMachineExecution(query, context).state == "queued"
    assert not observed
    # A failed read must never be interpreted as permission to submit fresh work.
    (executions.workspace.directory / "journal.sqlite3").write_bytes(b"corrupt database")
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop()[0] != grpc.StatusCode.NOT_FOUND
