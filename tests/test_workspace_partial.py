"""Fresh accepted requests adopt real native groups without inheriting a writer."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def accepted(workspace: Workspace, request: str, *, memoize: bool = True) -> tuple[bytes, str]:
    raw = documents.canonical_bytes(
        pb.InvocationSpec(
            outputs=[
                pb.OutputBinding(
                    output_id="model",
                    mime_type="application/vnd.cozy.model-manifest",
                    max_bytes=4096,
                )
            ],
            job=pb.JobInvocationSpec(
                build_id="sha256:" + "22" * 32,
                job_descriptor_id="sha256:" + "33" * 32,
            ),
        )
    )
    spec = hashlib.sha256(raw).digest()
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=raw,
        ),
        memoize=memoize,
    )
    return spec, weights_transaction_id("owner", request, documents.spell(spec), "model")


def intent(
    request: str, spec: bytes, transaction: str, declaration: bytes
) -> pb.WeightsIntentFrame:
    return pb.WeightsIntentFrame(
        request_id=request,
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        output_slot="model",
        weights_transaction_id=transaction,
        tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
        tensorfs_declaration_canonical_bytes=declaration,
    )


def failed(workspace: Workspace, request: str, spec: bytes) -> None:
    body = documents.canonical_bytes(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_FAILED,
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="failed-" + request,
            outcome_canonical_bytes=body,
            outcome_digest=hashlib.sha256(body).digest(),
        ),
    )


def fixture(
    tmp_path: Path, *, memoize: bool = True, fingerprint: str = "sha256:" + "45" * 32
) -> tuple[Any, ...]:
    store = tensorfs.Store.init(tmp_path / "store")
    workspace = Workspace(Path(store.root))
    spec, transaction = accepted(workspace, "first", memoize=memoize)
    plain = next(value for alias, value in tensorfs.seed_digests() if alias == "plain/1")
    tensor = {
        "logical_dtype": "f32",
        "shape": [512],
        "encoding": plain,
        "parts": {"value": {"dtype": "f32", "shape": [512]}},
    }
    targets = {"model": {"add": {"a": tensor, "b": tensor}, "drop": []}}
    order = [("model", "a"), ("model", "b")]
    declaration = store.derived_declaration(
        {}, targets, {}, order, 4096, work_fingerprint=fingerprint
    )
    workspace.begin_weights("owner", intent("first", spec, transaction, declaration))
    writer = store.begin_derived(
        transaction, 1, {}, targets, {}, order, 4096, work_fingerprint=fingerprint
    )
    writer.add_part("model", "a", "value", io.BytesIO(b"\x31" * 2048))
    facts = writer.checkpoint("first", "model")
    length, index, stored = facts["head_length"], facts["index"], facts["bytes"]
    assert isinstance(length, int) and isinstance(index, int) and isinstance(stored, int)
    checkpoint = pb.CheckpointRef(
        head=pb.Ref(digest=documents.raw(str(facts["head"])), length=length),
        plan_digest=documents.raw(str(facts["plan_digest"])),
        index=index,
        bytes=stored,
    )
    workspace.checkpoint(
        "owner",
        pb.WeightsCheckpointFrame(
            request_id="first",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            output_slot="model",
            weights_transaction_id=transaction,
            writer_epoch=1,
            tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
            checkpoint=checkpoint,
        ),
    )
    writer.fence()
    return store, workspace, spec, transaction, declaration, targets, order, fingerprint


def test_fresh_script_adopts_stopped_work_and_keeps_independent_custody(tmp_path: Path) -> None:
    store, workspace, spec, first, declaration, targets, order, fingerprint = fixture(tmp_path)
    failed(workspace, "first", spec)
    spec, second = accepted(workspace, "second")
    request = intent("second", spec, second, declaration)
    row = workspace.begin_weights("owner", request)
    checkpoint = pb.CheckpointRef.FromString(row["checkpoint"])
    assert row["adoption_source"] == first
    assert row["ready"] == 0 and checkpoint.head.length > 0
    assert store.derived_lookup(first) == {"state": "open"}
    store.derived_abandon(first)
    tensorfs.gc(str(store.root))
    restarted = Workspace(Path(store.root))
    assert restarted.begin_weights("owner", request)["checkpoint"] == row["checkpoint"]
    ready = restarted.ready_weights(
        "owner",
        pb.WeightsIntentReadyRequest(
            attempt_ordinal=1,
            weights=pb.WeightsCheckpointSubject(
                request_id="second",
                invocation_spec_digest=spec,
                output_slot="model",
                weights_transaction_id=second,
                writer_epoch=1,
                tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
            ),
            checkpoint=checkpoint,
        ),
    )
    assert ready["ready"] == 1
    writer = store.begin_derived(
        second,
        1,
        {},
        targets,
        {},
        order,
        4096,
        work_fingerprint=fingerprint,
        checkpoint=(documents.spell(checkpoint.head.digest), checkpoint.head.length),
    )
    assert writer.completed_parts() == [("model", "a", "value")]
    writer.add_part("model", "b", "value", io.BytesIO(b"\x32" * 2048))
    result = writer.commit()
    clean = store.begin_derived(
        "sha256:" + "99" * 32, 1, {}, targets, {}, order, 4096, work_fingerprint=fingerprint
    )
    clean.add_part("model", "a", "value", io.BytesIO(b"\x31" * 2048))
    clean.add_part("model", "b", "value", io.BytesIO(b"\x32" * 2048))
    assert clean.commit()["manifest"] == result["manifest"]


@pytest.mark.parametrize(
    "reason", ["active", "source-not-memoized", "recipient-not-memoized", "changed-code"]
)
def test_incompatible_or_unqualified_partial_work_is_not_adopted(
    tmp_path: Path, reason: str
) -> None:
    store, workspace, spec, first, declaration, targets, order, _ = fixture(
        tmp_path, memoize=reason != "source-not-memoized"
    )
    if reason != "active":
        failed(workspace, "first", spec)
    spec, second = accepted(workspace, "second", memoize=reason != "recipient-not-memoized")
    if reason == "changed-code":
        declaration = store.derived_declaration(
            {}, targets, {}, order, 4096, work_fingerprint="sha256:" + "ff" * 32
        )
    row = workspace.begin_weights("owner", intent("second", spec, second, declaration))
    assert row["adoption_source"] == "" and row["checkpoint"] == b""
    assert store.derived_lookup(second) == {"state": "absent"}
    assert store.derived_lookup(first)["state"] == "open"


def test_native_commit_before_journal_result_replays_original_adoption(tmp_path: Path) -> None:
    store, workspace, spec, first, declaration, _, _, _ = fixture(tmp_path)
    failed(workspace, "first", spec)
    spec, second = accepted(workspace, "second")
    request = intent("second", spec, second, declaration)
    result = workspace.begin_weights("owner", request)
    # Reproduce a native commit before the final workspace SQL update.
    with workspace.locked() as db:
        db.execute("UPDATE weights SET checkpoint=x'' WHERE id=?", (second,))
    store.derived_abandon(first)
    tensorfs.gc(str(store.root))
    assert (
        Workspace(Path(store.root)).begin_weights("owner", request)["checkpoint"]
        == result["checkpoint"]
    )


def test_schema_two_migrates_conservatively_and_future_schema_refuses(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    with workspace.locked() as db:
        for table in ("execution_calls", "execution_commands", "execution_events", "executions"):
            db.execute("DROP TABLE " + table)
        db.execute("ALTER TABLE workspace_state DROP COLUMN execution_workspace_id")
        db.execute("DROP TABLE byte_outputs")
        db.execute("DROP TABLE native_calls")
        db.execute("DROP TABLE source_access")
        db.execute("ALTER TABLE holds DROP COLUMN kind")
        db.execute("ALTER TABLE attempts DROP COLUMN memoize")
        for column in ("adoption_source", "adoption_epoch", "adoption_checkpoint"):
            db.execute(f"ALTER TABLE weights DROP COLUMN {column}")
        db.execute("PRAGMA user_version=2")
    with workspace.locked() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 8
        assert "memoize" in {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
    with sqlite3.connect(workspace.directory / "journal.sqlite3") as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(WorkspaceRefusal, match="schema"), workspace.locked():
        pass
