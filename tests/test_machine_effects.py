"""Publication absence refuses before a send, without polluting another call."""

import hashlib
import time
from pathlib import Path

from cozy_runtime import canonical_json
from cozy_runtime.internal import effect_interfaces
from cozy_runtime.internal.worker.machine_effects import Effects
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from test_machine_calls import request, running
from test_machine_execution import offer


def test_ungranted_effect_is_durably_refused_without_external_send(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit("owner", "parent", b"c" * 32, offer("parent"))
    calls = Calls(workspace)
    calls.accept("owner", request(running(executions), module=effect_interfaces.MODULE))
    effects = Effects(workspace, executions, None)
    try:
        until = time.monotonic() + 5
        while not calls.get("owner", "parent", 0).safe_code:
            effects.drive("owner")
            assert time.monotonic() < until
            time.sleep(0.01)
        value = calls.get("owner", "parent", 0)
        assert value.safe_code == "publication.authority_absent"
        assert not value.sent and not value.result
        assert not calls.has_unsettled_effects("owner", "parent")
        assert not calls.pending_effects("another-owner")
    finally:
        effects.close()


def test_failed_effect_remains_reconcilable_after_parent_stops(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit("owner", "parent", b"c" * 32, offer("parent"))
    calls = Calls(workspace)
    call = calls.accept("owner", request(running(executions), module=effect_interfaces.MODULE))
    calls.before_write("owner", call)
    calls.unresolved("owner", call, "publication.outcome_unknown")
    executions.control("owner", "parent", "cancel", 1, "cancel")
    failed = calls.get("owner", "parent", 0)
    assert failed.safe_code == "publication.outcome_unknown"
    assert calls.has_unsettled_effects("owner", "parent")
    assert calls.pending_effects("owner") == [failed]
    calls.complete("owner", call, b'{"revision":1}')
    observed = calls.get("owner", "parent", 0)
    assert observed.result and not observed.safe_code
    assert not calls.has_unsettled_effects("owner", "parent")


def test_invalid_worker_authority_is_visible_and_background_threads_are_pollers(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "store")
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        publication_authorization_id="019aaaab-0000-7000-8000-000000000003",
    )
    call = request(running(executions), module=effect_interfaces.MODULE)
    call.interface_digest = documents.raw(effect_interfaces.DIGEST)
    call.intent_digest = hashlib.sha256(
        canonical_json.encode(
            {
                "interface_digest": effect_interfaces.DIGEST,
                "module": call.module,
                "export": call.export,
                "request": {"value": 1},
            }
        )
    ).digest()
    calls = Calls(workspace)
    calls.accept("owner", call)
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
        until = time.monotonic() + 5
        while not calls.get("owner", "parent", 0).safe_code:
            effects.drive("owner")
            assert time.monotonic() < until
            time.sleep(0.01)
        observed = calls.get("owner", "parent", 0)
        assert observed.safe_code == "publication.worker_authority_invalid"
        assert not observed.sent and not observed.result
        assert effects.pollers
    finally:
        effects.close()
    assert not effects.pollers
