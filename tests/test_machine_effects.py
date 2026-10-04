"""Publication absence refuses before a send, without polluting another call."""

from pathlib import Path

from cozy_runtime.internal import effect_interfaces
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from test_machine_calls import request, running
from test_machine_execution import offer


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
