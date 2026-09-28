"""Publication absence refuses before a send, without polluting another call."""

import threading
from pathlib import Path

from cozy_runtime.internal import effect_interfaces
from cozy_runtime.internal.worker.machine_effects import Effects
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.supervisor import Unit
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from test_machine_calls import request, running
from test_machine_execution import offer


def test_ungranted_effect_is_durably_refused_without_external_send(tmp_path: Path) -> None:
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
    call = calls.accept("owner", request(running(executions), module=effect_interfaces.MODULE))
    effects = Effects(workspace, executions, None)
    try:
        effects.run(Unit("effect", threading.Event()), "owner", call)  # settles at once
        value = calls.get("owner", "parent", 0)
        assert value.safe_code == "publication.authority_absent"
        assert not value.sent and not value.result
        assert not calls.has_unsettled_effects("owner", "parent")
        assert not calls.unsettled_effects("owner")
    finally:
        effects.close()


def test_failed_effect_remains_reconcilable_after_parent_stops(tmp_path: Path) -> None:
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
    call = calls.accept("owner", request(running(executions), module=effect_interfaces.MODULE))
    calls.before_write("owner", call)
    calls.unresolved("owner", call, "publication.outcome_unknown")
    executions.control("owner", "parent", "cancel", 1, "cancel")
    failed = calls.get("owner", "parent", 0)
    assert failed.safe_code == "publication.outcome_unknown"
    assert calls.has_unsettled_effects("owner", "parent")
    assert calls.unsettled_effects("owner") == [failed.child_request]
    calls.complete("owner", call, b'{"revision":1}')
    observed = calls.get("owner", "parent", 0)
    assert observed.result and not observed.safe_code
    assert not calls.has_unsettled_effects("owner", "parent")


def test_invalid_worker_authority_is_visible(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        publication_authorization_id="019aaaab-0000-7000-8000-000000000003",
        expected_execution_workspace_id=executions.workspace_id,
    )
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions), module=effect_interfaces.MODULE))
    effects = Effects(
        workspace,
        executions,
        PublicationAuthority(
            "https://example.test",
            "worker-test",
            "",
        ),
    )
    try:
        effects.run(Unit("effect", threading.Event()), "owner", call)  # settles at once
        observed = calls.get("owner", "parent", 0)
        assert observed.safe_code == "publication.worker_authority_invalid"
        assert not observed.sent and not observed.result
    finally:
        effects.close()
