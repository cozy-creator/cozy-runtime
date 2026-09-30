"""Actual workspace ownership fences durable call intents and effect send markers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import msgspec
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._executor_requests import CallProgress, CallState, ChildPoll
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import complete, offer


def running(executions: Executions, request: str = "parent") -> pb.AttemptOffer:
    """The parent's attempt is on its executor (dispatched once, however often asked)."""
    selected = executions.offer("owner", request)
    if executions.status("owner", request).state == "queued":
        executions.dispatched("owner", request, selected.attempt_ordinal)
    return selected


def request(
    parent: pb.AttemptOffer, *, value: int = 1, module: str = "library"
) -> pb.ChildCallRequest:
    intent = {
        "module": module,
        "export": "publish",
        "request": {"value": value},
    }
    return pb.ChildCallRequest(
        parent_request_id=parent.request_id,
        parent_attempt_ordinal=parent.attempt_ordinal,
        parent_invocation_spec_digest=parent.invocation_spec_digest,
        call_index=0,
        module=module,
        export="publish",
        request_canonical_bytes=canonical_json.encode({"value": value}),
        intent_digest=hashlib.sha256(canonical_json.encode(intent)).digest(),
    )


def test_call_replay_preserves_send_marker_and_fences_prior_parent_generation(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
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


def test_child_progress_poll_reuses_journal_and_fences_owner_parent_and_attempt(
    tmp_path: Path,
) -> None:
    from cozy_runtime.internal.worker.calls import Calls as Transport
    from cozy_runtime.internal.worker.calls import _Pending

    root = tmp_path / "store"
    workspace = Workspace(root)
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    incoming = request(running(executions))
    child = Calls(workspace).accept("owner", incoming)
    executions.submit(
        "owner",
        child.child_request,
        b"c" * 32,
        offer(child.child_request),
        expected_execution_workspace_id=executions.workspace_id,
    )
    running(executions, child.child_request)
    document = {
        "type": "progress",
        "payload": {
            "stage": "denoise",
            "position": 3,
            "total": 8,
            "stage_fraction": 0.375,
            "overall_fraction": 0.4,
        },
    }
    executions.progress("owner", child.child_request, 1, document)
    pending = _Pending(incoming, pb.ChildCallResult(child_request_id=child.child_request))
    transport = Transport(
        lambda *_: pb.ChildCallResult(), workspace=Workspace(root), owner=lambda: "owner"
    )
    observed = transport._progress(pending)
    assert observed is not None
    assert observed.payload["call_request"] == child.child_request
    assert observed.payload["call_attempt"] == 1
    assert {
        key: value
        for key, value in observed.payload.items()
        if key not in {"call_request", "call_attempt"}
    } == document["payload"]
    assert transport._progress(pending) == observed  # polling never adds journal rows
    incoming.parent_attempt_ordinal = 2
    assert transport._progress(pending) is None
    incoming.parent_attempt_ordinal = 1
    incoming.parent_request_id = "another-parent"
    assert transport._progress(pending) is None
    incoming.parent_request_id = "parent"
    transport.owner = lambda: "other-owner"
    assert transport._progress(pending) is None
    transport.owner = lambda: "owner"
    complete(executions, child.child_request, status=pb.OUTCOME_STATUS_FAILED)
    executions.control("owner", child.child_request, "retry", 1, "resume")
    assert transport._progress(pending) is None


def test_preparation_start_is_visible_without_fabricated_fraction(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.internal.worker.machine_serving import Serving

    emitted = []
    worker = SimpleNamespace(
        workspace=None,
        emit_progress=lambda *args: emitted.append(args),
        calls=SimpleNamespace(progress_label=lambda *args: "Shot 1 of 7"),
    )
    serving = Serving(cast(Any, worker))
    call = SimpleNamespace(parent_request="parent", child_request="child", call_index=0)
    incoming = pb.ChildCallRequest(parent_attempt_ordinal=1)
    with (
        pytest.raises(ValueError),
        serving.phase(cast(Any, call), incoming, "Downloading model weights"),
    ):
        assert emitted[-1] == (
            "parent",
            1,
            {"kind": "progress", "stage": "Shot 1 of 7 / Downloading model weights"},
        )
        raise ValueError("download failed")
    assert emitted[-1][2]["fields"]["completed"] is False
    serving.close()


def test_optional_child_progress_cannot_turn_a_valid_result_into_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from typing import Any, cast

    from cozy_runtime.internal.worker import calls as transport

    done = pb.ChildCallResult(
        state=pb.CHILD_CALL_STATE_SUCCEEDED,
        child_request_id="child",
        result_canonical_bytes=b'"done"',
    )
    held = transport._Pending(pb.ChildCallRequest(), done, byte_grants=[])
    broker = transport.Calls(lambda *_: done)
    broker.pending[("parent", 1, 0)] = held
    expected = CallState(ok=True, state="succeeded", child_request_id="child", result='"done"')
    monkeypatch.setattr(transport, "MAX_FRAME", len(msgspec.json.encode(expected)) + 128)
    monkeypatch.setattr(
        broker, "_progress", lambda _: CallProgress(sequence=1, payload={"stage": "x" * 120})
    )
    attempt = SimpleNamespace(request_id="parent", attempt=1, state="running", canceling=False)
    assert broker.handle(cast(Any, attempt), ChildPoll(call_index=0)) == expected


def test_cancel_before_send_performs_no_write_or_forged_completion(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
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


def test_release_baseline_freezes_before_send_and_survives_reopen(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    accepted = calls.accept("owner", request(running(executions)))
    prepared = b'{"baseline_revision":1,"desired":{"bf16":"checkpoint"}}'
    accepted = calls.freeze("owner", accepted, prepared)
    calls.before_write("owner", accepted)
    recovered = Calls(Workspace(tmp_path / "store")).get("owner", "parent", 0)
    assert recovered.prepared == prepared and recovered.sent
    assert calls.freeze("owner", recovered, prepared) == recovered
    with pytest.raises(WorkspaceRefusal, match="frozen baseline"):
        calls.freeze("owner", recovered, b'{"baseline_revision":2}')
    executions.control("owner", "parent", "cancel", 1, "cancel")
    calls.complete("owner", recovered, b'{"revision":2}')
    assert calls.get("owner", "parent", 0).result == b'{"revision":2}'


def test_read_only_convergence_does_not_fabricate_a_send(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    accepted = calls.accept("owner", request(running(executions)))
    calls.observe_result("owner", accepted, b'{"observation":"observed_convergence"}')
    recovered = calls.get("owner", "parent", 0)
    assert recovered.result and not recovered.sent
    with pytest.raises(WorkspaceRefusal, match="settlement"):
        calls.before_write("owner", accepted)


def test_root_cancel_fences_descendant_writes_and_retains_uncertain_effect(tmp_path: Path) -> None:
    from cozy_runtime.internal.effect_interfaces import MODULE

    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    child = calls.accept("owner", request(running(executions))).child_request
    executions.submit(
        "owner",
        child,
        b"d" * 32,
        offer(child),
        expected_execution_workspace_id=executions.workspace_id,
    )
    effect = calls.accept("owner", request(running(executions, child), module=MODULE))
    effect = calls.freeze("owner", effect, b'{"baseline_revision":0}')
    calls.before_write("owner", effect)
    assert calls.has_unsettled_effects("owner", "parent")
    complete(executions, "parent", status=pb.OUTCOME_STATUS_FAILED)
    executions.control("owner", "parent", "cancel", 1, "cancel")
    with pytest.raises(WorkspaceRefusal, match="ancestor"):
        calls.before_write("owner", effect)
    assert not executions.release("owner", "parent")
    assert executions.retention_required("owner")
    calls.complete("owner", effect, b'{"revision":1}')
    assert not calls.has_unsettled_effects("owner", "parent")
    assert executions.release("owner", "parent")


def test_accepted_publication_authority_is_immutable_and_survives_reopen(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    grant = "019aaaab-0000-7000-8000-000000000003"
    accepted = executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        publication_authorization_id=grant,
        expected_execution_workspace_id=executions.workspace_id,
    )
    assert accepted.publication_authorization_id == grant
    reopened = Executions(Workspace(tmp_path / "store"))
    assert reopened.publication_authorization("owner", "parent") == grant
    assert (
        reopened.submit(
            "owner",
            "parent",
            b"c" * 32,
            offer("parent"),
            publication_authorization_id=grant,
            expected_execution_workspace_id=reopened.workspace_id,
        )
        == accepted
    )
    for changed in ("", "019aaaab-0000-7000-8000-000000000004"):
        with pytest.raises(WorkspaceRefusal, match="identity changed"):
            reopened.submit(
                "owner",
                "parent",
                b"c" * 32,
                offer("parent"),
                publication_authorization_id=changed,
                expected_execution_workspace_id=reopened.workspace_id,
            )
    with pytest.raises(WorkspaceRefusal, match="not held"):
        reopened.publication_authorization("another-owner", "parent")


def test_descendant_cannot_borrow_or_broaden_another_publication_grant(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    grant = "019aaaab-0000-7000-8000-000000000003"
    other_grant = "019aaaab-0000-7000-8000-000000000004"
    executions.submit(
        "owner",
        "root",
        b"c" * 32,
        offer("parent"),
        publication_authorization_id=grant,
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    child = calls.accept("owner", request(running(executions))).child_request
    executions.submit(
        "owner",
        child,
        b"d" * 32,
        offer(child),
        expected_execution_workspace_id=executions.workspace_id,
    )
    assert executions.publication_authorization("owner", child) == grant
    executions.submit(
        "owner",
        "other",
        b"e" * 32,
        offer("other"),
        publication_authorization_id=other_grant,
        expected_execution_workspace_id=executions.workspace_id,
    )
    assert executions.publication_authorization("owner", child) != other_grant


def test_refusal_cannot_overwrite_an_accepted_child_or_uncertain_send(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions)))
    calls.before_write("owner", call)
    with pytest.raises(WorkspaceRefusal, match="accepted work"):
        calls.refuse("owner", call, "test.refusal", "rejected")
    assert calls.get("owner", "parent", 0).sent
