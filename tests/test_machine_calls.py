"""Actual workspace ownership fences durable call intents and effect send markers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import complete, offer


def running(executions: Executions, request: str = "parent") -> pb.AttemptOffer:
    selected = executions.offer("owner", request)
    accepted = pb.AttemptAccepted(
        request_id=request,
        attempt_ordinal=selected.attempt_ordinal,
        invocation_spec_digest=selected.invocation_spec_digest,
    )
    executions.workspace.mark_running("owner", accepted)
    executions.observe("owner", pb.WorkerFrame(attempt_accepted=accepted))
    return selected


def request(parent: pb.AttemptOffer, *, value: int = 1) -> pb.ChildCallRequest:
    interface = b"i" * 32
    intent = {
        "interface_digest": documents.spell(interface),
        "module": "library",
        "export": "publish",
        "request": {"value": value},
    }
    return pb.ChildCallRequest(
        parent_request_id=parent.request_id,
        parent_attempt_ordinal=parent.attempt_ordinal,
        parent_invocation_spec_digest=parent.invocation_spec_digest,
        call_index=0,
        interface_digest=interface,
        module="library",
        export="publish",
        request_canonical_bytes=canonical_json.encode({"value": value}),
        intent_digest=hashlib.sha256(canonical_json.encode(intent)).digest(),
    )


def test_call_replay_preserves_send_marker_and_fences_prior_parent_generation(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit("owner", "parent", b"c" * 32, offer("parent"))
    first = request(running(executions))
    calls = Calls(workspace)
    accepted = calls.accept("owner", first)
    assert calls.accept("owner", first) == accepted
    with pytest.raises(WorkspaceRefusal, match="different frozen intent"):
        calls.accept("owner", request(running(executions), value=2))
    calls.before_write("owner", accepted)
    assert Calls(Workspace(tmp_path / "store")).get("owner", "parent", 0).sent
    complete(executions, "parent", status=pb.OUTCOME_STATUS_FAILED)
    executions.control("owner", "parent", "resume", 1, "resume")
    resumed = calls.accept("owner", request(running(executions)))
    assert resumed.child_request == accepted.child_request and resumed.sent
    assert resumed.parent_ordinal == 2 and resumed.parent_generation == 2
    with pytest.raises(WorkspaceRefusal, match="generation"):
        calls.before_write("owner", accepted)
    # A committed remote effect's late response is a fact, even after its originating
    # parent stopped. Reconciliation preserves it rather than blindly writing again.
    calls.complete("owner", accepted, b"null")
    assert calls.get("owner", "parent", 0).result == b"null"
    calls.complete("owner", resumed, b"null")
    with pytest.raises(WorkspaceRefusal, match="prior result"):
        calls.complete("owner", resumed, b"false")
    with pytest.raises(WorkspaceRefusal, match="settlement"):
        calls.before_write("owner", resumed)


def test_cancel_before_send_performs_no_write_or_forged_completion(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit("owner", "parent", b"c" * 32, offer("parent"))
    calls = Calls(workspace)
    accepted = calls.accept("owner", request(running(executions)))
    executions.control("owner", "parent", "cancel", 1, "cancel")
    with pytest.raises(WorkspaceRefusal, match="generation"):
        calls.before_write("owner", accepted)
    with pytest.raises(WorkspaceRefusal, match="authorized write"):
        calls.complete("owner", accepted, b"{}")
    with pytest.raises(WorkspaceRefusal, match="accepted intent"):
        calls.get("foreign-owner", "parent", 0)
    assert not calls.get("owner", "parent", 0).sent
