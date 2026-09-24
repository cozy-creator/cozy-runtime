"""Known preaccept validation is terminal; unexpected exceptions remain opaque."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import grpc
import pytest

from cozy_runtime.internal import canonical
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb


class Aborted(Exception):
    pass


def test_noncanonical_execution_payload_is_a_known_refusal() -> None:
    rpc = MachineExecutionRPC(
        cast(
            Worker,
            SimpleNamespace(
                executions=object(),
                authorize_workspace=lambda _: "owner",
            ),
        )
    )
    observed: list[tuple[Any, str]] = []

    def abort(code: Any, detail: str) -> None:
        observed.append((code, detail))
        raise Aborted

    with pytest.raises(Aborted):
        rpc._call(
            pb.Claim(wire_minor=WIRE_MINOR),
            SimpleNamespace(abort=abort),
            lambda: canonical.parse_canonical(b'{ "value": 7 }'),
        )
    assert observed[0][0] == grpc.StatusCode.FAILED_PRECONDITION

    def unexpected() -> None:
        raise RuntimeError("private error contents")

    with pytest.raises(Aborted):
        rpc._call(pb.Claim(wire_minor=WIRE_MINOR), SimpleNamespace(abort=abort), unexpected)
    assert observed[1] == (grpc.StatusCode.UNAVAILABLE, "machine execution operation failed")


def test_lookup_distinguishes_absence_from_authority_workspace_and_storage_failure(
    tmp_path: Path,
) -> None:
    from test_machine_execution import offer

    executions = Executions(Workspace(tmp_path / "store"))
    observed: list[Any] = []

    def authorize(claim: pb.Claim) -> str:
        if claim.record_owner_id != "owner":
            raise WorkspaceRefusal("owner is not authorized")
        return claim.record_owner_id

    def abort(code: Any, _detail: str) -> None:
        observed.append(code)
        raise Aborted

    worker = cast(
        Worker,
        SimpleNamespace(
            executions=executions,
            authorize_workspace=authorize,
            execution_status=lambda claim, request: executions.status(
                claim.record_owner_id, request
            ),
            options=SimpleNamespace(worker_id="worker"),
            fence=SimpleNamespace(worker_boot_id="boot"),
        ),
    )
    rpc = MachineExecutionRPC(worker)
    context = SimpleNamespace(abort=abort, set_trailing_metadata=lambda _: None)
    query = pb.MachineExecutionQuery(
        claim=pb.Claim(record_owner_id="owner", wire_minor=WIRE_MINOR),
        request_id="request",
        expected_execution_workspace_id=executions.workspace_id,
    )
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop() == grpc.StatusCode.NOT_FOUND
    query.expected_execution_workspace_id = "wrong-workspace"
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop() == grpc.StatusCode.FAILED_PRECONDITION
    query.expected_execution_workspace_id = executions.workspace_id
    query.claim.record_owner_id = "other"
    with pytest.raises(Aborted):
        rpc.GetMachineExecution(query, context)
    assert observed.pop() == grpc.StatusCode.FAILED_PRECONDITION
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
    assert observed.pop() != grpc.StatusCode.NOT_FOUND
