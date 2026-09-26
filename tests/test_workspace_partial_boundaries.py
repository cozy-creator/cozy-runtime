"""Exact native adoption refusals at identity and cancellation boundaries."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.numerical_environment import fingerprint as numerical_fingerprint
from cozy_runtime.internal.weights_sink import work_fingerprint
from cozy_runtime.internal.worker import workspace_finalize
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_partial import accepted, failed, fixture, intent


@pytest.mark.parametrize("boundary", ["before-native", "after-native"])
def test_recipient_cancel_during_native_adoption_cannot_resurrect_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    store, workspace, spec, first, declaration, *_ = fixture(tmp_path)
    failed(workspace, "first", spec)
    spec, second = accepted(workspace, "second")
    entered, resume = threading.Event(), threading.Event()
    original_native = Journal.native
    errors: list[Exception] = []

    def paused_native(self: Journal, operation: Callable[[], Any]) -> Any:
        if getattr(operation, "__name__", "") != "acquire":
            return original_native(self, operation)

        def acquire() -> Any:
            result = operation() if boundary == "after-native" else None
            entered.set()
            assert resume.wait(10), "the adoption fault boundary was not released"
            return result if boundary == "after-native" else operation()

        return original_native(self, acquire)

    monkeypatch.setattr(Journal, "native", paused_native)

    def start() -> None:
        try:
            workspace.begin_weights("owner", intent("second", spec, second, declaration))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=start)
    thread.start()
    try:
        assert entered.wait(10), "native adoption was not reached"
        competing = Workspace(Path(store.root))
        failed(competing, "second", spec)
        with competing.locked() as db:
            invocation = bytes(
                db.execute(
                    "SELECT invocation FROM attempts WHERE owner='owner' AND request='second'"
                ).fetchone()[0]
            )
        canceled = workspace_finalize.finalize(
            competing,
            "owner",
            pb.WeightsFinalizeRequest(
                owner_authority_scope="owner",
                request_id="second",
                invocation_spec_digest=spec,
                invocation_spec_canonical_bytes=invocation,
                output_slot="model",
                disposition=pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED,
            ),
        )
        assert canceled.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ABANDONED
    finally:
        resume.set()
        thread.join(10)
    assert not thread.is_alive() and len(errors) == 1
    if boundary == "after-native":
        assert isinstance(errors[0], WorkspaceRefusal)
        assert "changed during native adoption" in str(errors[0])
    else:
        assert getattr(errors[0], "code", None) == "TRANSACTION_CONFLICT"
    row = workspace.weights_row("owner", second)
    assert row["state"] == "released" and row["checkpoint"] == b"" and not row["ready"]
    assert store.derived_lookup(second) == {"state": "abandoned"}
    with pytest.raises(WorkspaceRefusal, match="live accepted invocation"):
        workspace.begin_weights("owner", intent("second", spec, second, declaration))
    # The canceled recipient cannot damage the donor or a later authorized consumer.
    assert store.derived_lookup(first)["state"] == "open"
    monkeypatch.setattr(Journal, "native", original_native)
    third_spec, third = accepted(workspace, "third")
    adopted = workspace.begin_weights("owner", intent("third", third_spec, third, declaration))
    assert adopted["adoption_source"] == first and adopted["checkpoint"]
    store.derived_abandon(first)
    tensorfs.gc(str(store.root))
    assert store.derived_lookup(third)["state"] == "open"


@pytest.mark.parametrize("change", ["parameter", "input", "numerical", "order", "plan"])
def test_each_partial_work_identity_mismatch_misses_and_native_refuses(
    tmp_path: Path, change: str
) -> None:
    spec_body: dict[str, Any] = {
        "job": {
            "installation_id": "sha256:" + "21" * 32,
            "job_descriptor_id": "sha256:" + "22" * 32,
        },
        "payload_digest": "sha256:" + "23" * 32,
        "inputs": [{"input_id": "source", "digest": "sha256:" + "24" * 32, "length": 2048}],
    }
    numerical = numerical_fingerprint(HostFacts(backend="cpu"), threads=1, inherited={})
    fingerprint = work_fingerprint(spec_body, numerical)
    store, workspace, spec, first, original, targets, order, _ = fixture(
        tmp_path, fingerprint=fingerprint
    )
    failed(workspace, "first", spec)
    before = workspace.weights_row("owner", first)
    checkpoint = pb.CheckpointRef.FromString(before["checkpoint"])
    changed_spec, changed_targets, changed_order = (
        deepcopy(spec_body),
        deepcopy(targets),
        list(order),
    )
    if change == "parameter":
        changed_spec["payload_digest"] = "sha256:" + "31" * 32
    elif change == "input":
        changed_spec["inputs"][0]["digest"] = "sha256:" + "32" * 32
    elif change == "numerical":
        numerical = numerical_fingerprint(HostFacts(backend="cpu"), threads=2, inherited={})
    elif change == "order":
        changed_order.reverse()
    else:
        changed_targets["model"]["add"]["b"]["shape"] = [256]
        changed_targets["model"]["add"]["b"]["parts"]["value"]["shape"] = [256]
    changed_fingerprint = work_fingerprint(changed_spec, numerical)
    declaration = store.derived_declaration(
        {}, changed_targets, {}, changed_order, 4096, work_fingerprint=changed_fingerprint
    )
    assert declaration != original
    next_spec, second = accepted(workspace, "second")
    row = workspace.begin_weights("owner", intent("second", next_spec, second, declaration))
    assert row["checkpoint"] == b"" and not row["adoption_source"]
    with pytest.raises(Exception) as refused:
        store.adopt_derived_checkpoint(
            second,
            first,
            declaration,
            documents.spell(checkpoint.head.digest),
            checkpoint.head.length,
            operation_id="second",
            slot="model",
        )
    assert getattr(refused.value, "code", None) == "TRANSACTION_CONFLICT"
    assert store.derived_lookup(second) == {"state": "absent"}
    after = workspace.weights_row("owner", first)
    assert after["checkpoint"] == before["checkpoint"] and after["declaration"] == original
    # A matching consumer still gets exactly the donor's original generation.
    third_spec, third = accepted(workspace, "third")
    result = workspace.begin_weights("owner", intent("third", third_spec, third, original))
    assert result["adoption_source"] == first
    assert result["adoption_checkpoint"] == before["checkpoint"]
    assert hashlib.sha256(result["declaration"]).digest() == before["declaration_digest"]
