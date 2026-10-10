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
from cozy_runtime.internal.worker import workspace_finalize, workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def spec_bytes() -> bytes:
    return documents.canonical_bytes(
        pb.InvocationSpec(
            outputs=[
                pb.OutputBinding(
                    output_id="model",
                    mime_type="application/vnd.cozy.model-manifest",
                    max_bytes=4096,
                )
            ],
            job=pb.JobInvocationSpec(
                installation_id="sha256:" + "22" * 32,
                job_descriptor_id="sha256:" + "33" * 32,
            ),
        )
    )


def accepted(workspace: Workspace, request: str, *, memoize: bool = True) -> tuple[bytes, str]:
    raw = spec_bytes()
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


def failed(
    workspace: Workspace,
    request: str,
    spec: bytes,
    status: pb.OutcomeStatus = pb.OUTCOME_STATUS_FAILED,
) -> pb.AttemptOutcomeAck:
    body = documents.canonical_bytes(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=status,
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
    return pb.AttemptOutcomeAck(
        request_id=request,
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="failed-" + request,
        outcome_digest=hashlib.sha256(body).digest(),
        retain_work=False,
    )


def released(workspace: Workspace, request: str, spec: bytes, status: pb.OutcomeStatus) -> None:
    """The owner's ordinary end of a stopped attempt: abandon its output, release it."""
    ack = failed(workspace, request, spec, status)
    workspace_finalize.finalize(
        workspace,
        "owner",
        pb.WeightsFinalizeRequest(
            owner_authority_scope="owner",
            request_id=request,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=spec_bytes(),
            output_slot="model",
            disposition=pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED,
        ),
    )
    workspace.acknowledge("owner", ack)


TARGETS = {
    "model": {
        "add": {
            name: {
                "logical_dtype": "f32",
                "shape": [512],
                "encoding": next(v for alias, v in tensorfs.seed_digests() if alias == "plain/1"),
                "parts": {"value": {"dtype": "f32", "shape": [512]}},
            }
            for name in ("a", "b")
        },
        "drop": [],
    }
}
ORDER = [("model", "a"), ("model", "b")]


def checkpointed(
    store: Any, workspace: Workspace, request: str, *, memoize: bool, fingerprint: str
) -> tuple[bytes, str, bytes]:
    """A real native writer that checkpointed part a and then stopped."""
    spec, transaction = accepted(workspace, request, memoize=memoize)
    declaration = store.derived_declaration(
        {}, TARGETS, {}, ORDER, 4096, work_fingerprint=fingerprint
    )
    workspace.begin_weights("owner", intent(request, spec, transaction, declaration))
    writer = store.begin_derived(
        transaction, 1, {}, TARGETS, {}, ORDER, 4096, work_fingerprint=fingerprint
    )
    writer.add_part("model", "a", "value", io.BytesIO(b"\x31" * 2048))
    facts = writer.checkpoint(request, "model")
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
            request_id=request,
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
    return spec, transaction, declaration


def fixture(
    tmp_path: Path, *, memoize: bool = True, fingerprint: str = "sha256:" + "45" * 32
) -> tuple[Any, ...]:
    store = tensorfs.Store.init(tmp_path / "store")
    workspace = Workspace(Path(store.root))
    spec, transaction, declaration = checkpointed(
        store, workspace, "first", memoize=memoize, fingerprint=fingerprint
    )
    return store, workspace, spec, transaction, declaration, TARGETS, ORDER, fingerprint


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


@pytest.mark.parametrize("version", [13, 99])
def test_a_journal_another_runtime_wrote_converges_and_keeps_its_rows(
    tmp_path: Path, version: int
) -> None:
    """An older journal lacks a table and a column; a newer one holds one this Runtime does
    not know. Either opens with this Runtime's schema added, and its work continues."""
    workspace = Workspace(tmp_path / "store")
    accepted(workspace, "before")
    with sqlite3.connect(workspace.directory / "journal.sqlite3") as db:
        # Old journals had neither these derived views nor the removed schema.
        for (name,) in db.execute("SELECT name FROM sqlite_schema WHERE type='view'").fetchall():
            db.execute('DROP VIEW "' + name.replace('"', '""') + '"')
        db.execute("DROP TABLE holds")
        db.execute("ALTER TABLE executions DROP COLUMN owner_memo")
        db.execute("CREATE TABLE newer_runtime (id TEXT PRIMARY KEY) STRICT")
        db.execute("ALTER TABLE attempts ADD COLUMN newer_column TEXT NOT NULL DEFAULT ''")
        db.execute(f"PRAGMA user_version={version}")
    reopened = Workspace(tmp_path / "store")
    accepted(reopened, "after")
    with reopened.locked() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == max(version, 14)
        assert {row[0] for row in db.execute("SELECT request FROM attempts")} == {"before", "after"}
        assert db.execute("SELECT count(*) FROM holds").fetchone()[0] == 0
        assert "owner_memo" in {row[1] for row in db.execute("PRAGMA table_info(executions)")}
        assert db.execute("SELECT count(*) FROM newer_runtime").fetchone()[0] == 0


def test_a_journal_missing_a_required_column_refuses_by_name(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    with workspace.locked():
        pass
    with sqlite3.connect(workspace.directory / "journal.sqlite3") as db:
        for (name,) in db.execute("SELECT name FROM sqlite_schema WHERE type='view'").fetchall():
            db.execute('DROP VIEW "' + name.replace('"', '""') + '"')
        db.execute("ALTER TABLE attempts DROP COLUMN spec")
        db.execute("PRAGMA user_version=13")
    with (
        pytest.raises(WorkspaceRefusal, match="lacks column spec"),
        Workspace(tmp_path / "store").locked(),
    ):
        pass


@pytest.mark.parametrize("status", [pb.OUTCOME_STATUS_CANCELED, pb.OUTCOME_STATUS_FAILED])
def test_released_memoized_work_is_retired_and_a_rerun_adopts_it(
    tmp_path: Path, status: pb.OutcomeStatus
) -> None:
    store, workspace, spec, first, declaration, targets, order, fingerprint = fixture(tmp_path)
    released(workspace, "first", spec, status)
    # The owner holds nothing, yet the checkpointed part survives the release.
    assert store.derived_lookup(first)["state"] == "open"
    tensorfs.gc(str(store.root))
    spec, second = accepted(workspace, "second")
    row = workspace.begin_weights("owner", intent("second", spec, second, declaration))
    checkpoint = pb.CheckpointRef.FromString(row["checkpoint"])
    assert row["adoption_source"] == first and checkpoint.head.length > 0
    # Adoption spends the retired donor; the recipient keeps its own copy.
    assert store.derived_lookup(first)["state"] != "open"
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM weights WHERE id=?", (first,)).fetchone()[0] == (
            "released"
        )
    tensorfs.gc(str(store.root))
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


def test_released_unmemoized_work_is_destroyed(tmp_path: Path) -> None:
    store, workspace, spec, first, declaration, _, _, _ = fixture(tmp_path, memoize=False)
    released(workspace, "first", spec, pb.OUTCOME_STATUS_CANCELED)
    assert store.derived_lookup(first)["state"] != "open"
    spec, second = accepted(workspace, "second")
    row = workspace.begin_weights("owner", intent("second", spec, second, declaration))
    assert row["adoption_source"] == "" and row["checkpoint"] == b""


def test_storage_pressure_evicts_the_oldest_retired_work_first(tmp_path: Path) -> None:
    store, workspace, spec, first, _, _, _, _ = fixture(tmp_path)
    released(workspace, "first", spec, pb.OUTCOME_STATUS_CANCELED)
    spec, newer, _ = checkpointed(
        store, workspace, "newer", memoize=True, fingerprint="sha256:" + "46" * 32
    )
    released(workspace, "newer", spec, pb.OUTCOME_STATUS_CANCELED)
    result = workspace_memo.reclaim_unused(workspace, target_bytes=1)
    assert result.removed_entries >= 1 and result.reclaimed_bytes > 0
    assert store.derived_lookup(first)["state"] != "open"
    assert store.derived_lookup(newer)["state"] == "open"
