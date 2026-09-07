"""Real staged/native migration, authenticated claims and replay without replacement services."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tensorfs
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from cozy_runtime import canonical_json
from cozy_runtime.internal.weights_sink import protocol_receipt, weights_transaction_id
from cozy_runtime.internal.worker.downloads import verify_claim_proof
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_import import activate, receive
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_custody import produced, retention


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def digest(raw: bytes) -> str:
    return "sha256:" + raw.hex() if raw else ""


class Authority:
    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.claim = pb.Claim(
            record_owner_id="owner", record_owner_epoch=1, worker_id="worker", worker_boot_id="boot"
        )
        self.claim.proof = self.key.sign(
            documents.canonical_bytes(
                pb.ClaimProof(
                    record_owner_epoch=1,
                    worker_boot_id="boot",
                    worker_id="worker",
                    worker_tls_certificate_digest=b"t" * 32,
                )
            )
        )

    def __call__(self, claim: pb.Claim) -> str:
        verify_claim_proof(
            claim,
            self.key.public_key().public_bytes_raw(),
            worker_id="worker",
            worker_boot_id="boot",
            worker_tls_certificate_digest="sha256:" + "74" * 32,
        )
        if claim.record_owner_id != "owner":
            raise WorkspaceRefusal("claim owner changed")
        return claim.record_owner_id


def frames(
    records: list[tuple[str, dict[str, Any]]], auth: Authority
) -> list[pb.WorkspaceImportFrame]:
    payloads = [(kind, canonical_json.encode(row)) for kind, row in records]
    root = hashlib.sha256()
    for kind, raw in payloads:
        root.update(kind.encode() + b"\0" + raw + b"\n")
    result = [
        pb.WorkspaceImportFrame(
            begin=pb.WorkspaceImportBegin(
                claim=auth.claim,
                migration_id=documents.spell(root.digest()),
                record_count=len(records),
                source_schema=19,
            )
        )
    ]
    for index, (kind, raw) in enumerate(payloads):
        result.append(
            pb.WorkspaceImportFrame(
                record=pb.WorkspaceImportRecord(
                    index=index, kind=kind, length=len(raw), digest=hashlib.sha256(raw).digest()
                )
            )
        )
        for offset in range(0, len(raw), 48 * 1024):
            result.append(
                pb.WorkspaceImportFrame(
                    chunk=pb.WorkspaceImportChunk(
                        index=index, offset=offset, data=raw[offset : offset + 48 * 1024]
                    )
                )
            )
    result.append(
        pb.WorkspaceImportFrame(
            commit=pb.WorkspaceImportCommit(digest=root.digest(), record_count=len(records))
        )
    )
    return result


def native_export(
    tmp_path: Path,
) -> tuple[tensorfs.Store, Workspace, list[tuple[str, dict[str, Any]]]]:
    store, prior, receipt = produced(tmp_path)
    held = retention(receipt, 101)
    prior.retain("owner", held)
    with prior.locked() as db:
        a = dict(db.execute("SELECT * FROM attempts").fetchone())
        w = dict(db.execute("SELECT * FROM weights").fetchone())
    reference, _, _ = protocol_receipt(
        receipt,
        owner_scope="owner",
        request_id="producer",
        invocation_spec_digest=documents.spell(a["spec"]),
    )
    outcome_body = pb.AttemptOutcomeBody(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=documents.spell(a["spec"]),
        status=pb.OUTCOME_STATUS_SUCCEEDED,
        weights_receipts=[reference],
    )
    raw = documents.canonical_bytes(outcome_body)
    source = {
        "request_id": "producer",
        "attempt_ordinal": 1,
        "invocation_spec_digest": digest(a["spec"]),
        "outcome_id": "out-original",
        "outcome_digest": documents.spell(hashlib.sha256(raw).digest()),
        "outcome_canonical_bytes": b64(raw),
    }
    attempt = {
        "owner_scope": "owner",
        **source,
        "invocation_spec_canonical_bytes": b64(a["invocation"]),
        "state": "outcome",
        "retain_work": True,
    }
    manifest = json.loads(receipt.tensorfs_receipt)["manifest"]
    objects = {
        "sha256:" + manifest["sha256"]: {
            "object_id": "sha256:" + manifest["sha256"],
            "length": manifest["length"],
            "source_ref": "tensorfs-manifest:sha256:" + manifest["sha256"],
        }
    }
    for obj in store.walk("sha256:" + manifest["sha256"]):
        objects[obj["id"]] = {
            "object_id": obj["id"],
            "length": obj["length"],
            "source_ref": "tensorfs-object:" + obj["id"],
        }
    weights = {
        "transaction_id": w["id"],
        "owner_scope": "owner",
        "request_id": "producer",
        "attempt_ordinal": 1,
        "invocation_spec_digest": digest(w["spec"]),
        "output_slot": w["slot"],
        "declaration_digest": digest(w["declaration_digest"]),
        "declaration_canonical_bytes": b64(w["declaration"]),
        "writer_epoch": w["epoch"],
        "state": "receipt",
        "intent_ready": True,
        "restore_checkpoint": "",
        "checkpoint": "",
        "receipt_digest": digest(w["receipt_digest"]),
        "receipt_canonical_bytes": b64(w["receipt"]),
        "manifest_digest": digest(w["manifest"]),
        "manifest_length": w["manifest_length"],
        "objects": [objects[key] for key in sorted(objects)],
    }
    hold = {
        "owner_scope": "owner",
        "transaction_id": w["id"],
        "retention_id": held.retention_id,
        "native_receipt_digest": receipt.tensorfs_receipt_digest,
        "state": "retained",
    }
    memo_hold = {
        key: hold[key] for key in ("transaction_id", "retention_id", "native_receipt_digest")
    }
    memo_hold.update(
        manifest_digest=weights["manifest_digest"], manifest_length=weights["manifest_length"]
    )
    memo = {
        "cache_id": "memo-original",
        "owner_scope": "owner",
        "computation_digest": "sha256:" + "11" * 32,
        "sequence": 17,
        "state": "ready",
        "source": source,
        "holds": [memo_hold],
    }
    # Preserve the fixture's old custody evidence separately. Only native Store
    # bytes/roots are retained at the configured path for the incoming authority.
    prior.directory.rename(tmp_path / "prior-authority-evidence")
    return (
        store,
        Workspace(Path(store.root)),
        [("attempt", attempt), ("weights", weights), ("hold", hold), ("cache", memo)],
    )


def test_native_import_stays_frozen_until_activation_and_replay_keeps_new_state(
    tmp_path: Path,
) -> None:
    store, workspace, records = native_export(tmp_path)
    auth = Authority()
    stream = frames(records, auth)
    reply = receive(workspace, stream, auth)
    assert reply.imported and reply.record_count == 4
    with pytest.raises(WorkspaceRefusal, match="not active"), workspace.locked():
        pass
    activate(
        workspace,
        pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
        auth,
    )
    store.derived_dispose(records[1][1]["transaction_id"])
    tensorfs.gc(store.root)
    assert store.document(hashlib.sha256(b"\x31" * 2048).hexdigest(), 2048) == b"\x31" * 2048
    with workspace.locked() as db:
        db.execute("UPDATE operation_cache SET sequence=999")
        assert db.execute("SELECT epoch,objects FROM weights").fetchone()[0] == 1
    assert receive(workspace, stream, auth) == reply
    with workspace.locked() as db:
        assert db.execute("SELECT sequence FROM operation_cache").fetchone()[0] == 999
    assert (
        activate(
            workspace,
            pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
            auth,
        )
        == reply
    )


def test_disconnect_after_record_recovers_with_fresh_process(tmp_path: Path) -> None:
    _, workspace, records = native_export(tmp_path)
    auth = Authority()
    stream = frames(records, auth)
    with pytest.raises(WorkspaceRefusal, match="before commit"):
        receive(workspace, stream[:3], auth)
    program = """from pathlib import Path
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
import sys
w=Workspace(Path(sys.argv[1]))
try:
 with w.locked(): pass
except WorkspaceRefusal: pass
else: raise AssertionError('partial import became active')
with w.locked(allow_frozen=True) as db:
 assert db.execute('SELECT count(*) FROM workspace_import_records').fetchone()[0]==1
 assert db.execute('SELECT count(*) FROM attempts').fetchone()[0]==0
"""
    subprocess.run([sys.executable, "-c", program, str(workspace.store_root)], check=True)
    assert receive(Workspace(workspace.store_root), stream, auth).imported


@pytest.mark.parametrize("fault", ["offset", "digest", "kind", "trailing", "owner", "schema"])
def test_invalid_stream_never_activates_or_imports_attempts(tmp_path: Path, fault: str) -> None:
    _, workspace, records = native_export(tmp_path)
    auth = Authority()
    stream = frames(records, auth)
    if fault == "offset":
        stream[2].chunk.offset = 1
    elif fault == "digest":
        stream[1].record.digest = b"x" * 32
    elif fault == "kind":
        stream[1].record.kind = "python_pickle"
    elif fault == "trailing":
        stream.append(stream[-1])
    elif fault == "owner":
        stream[0].begin.claim.record_owner_id = "other"
    else:
        stream[0].begin.source_schema = 18
    with pytest.raises((WorkspaceRefusal, ValueError)):
        receive(workspace, stream, auth)
    with workspace.locked(allow_frozen=True) as db:
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0


def test_duplicate_hold_refuses_before_releasing_real_custody(tmp_path: Path) -> None:
    store, workspace, records = native_export(tmp_path)
    auth = Authority()
    released = dict(records[2][1], state="released")
    records.insert(3, ("hold", released))
    with pytest.raises(WorkspaceRefusal, match="duplicate"):
        receive(workspace, frames(records, auth), auth)
    original = records[2][1]
    assert (
        store.retain_derived_result(
            original["transaction_id"], original["native_receipt_digest"], original["retention_id"]
        )["released"]
        is False
    )


def test_released_tombstone_import_needs_no_payload_bytes(tmp_path: Path) -> None:
    store, workspace, records = native_export(tmp_path)
    auth = Authority()
    hold = records[2][1]
    store.release_derived_retention(
        hold["transaction_id"], hold["native_receipt_digest"], hold["retention_id"]
    )
    store.derived_dispose(hold["transaction_id"])
    tensorfs.gc(store.root)
    assert not store.contains(hashlib.sha256(b"\x31" * 2048).hexdigest())
    stream = frames([("hold", dict(hold, state="released"))], auth)
    reply = receive(workspace, stream, auth)
    assert reply.imported
    activate(
        workspace,
        pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
        auth,
    )
    with workspace.locked() as db:
        assert db.execute("SELECT state FROM holds").fetchone()[0] == "released"


def test_missing_active_payload_refuses_without_claiming_imported(tmp_path: Path) -> None:
    store, workspace, records = native_export(tmp_path)
    auth = Authority()
    digest_value = hashlib.sha256(b"\x31" * 2048).hexdigest()
    # Deliberate disk-loss fixture; never interpreted as a successfully imported hit.
    matches = list(Path(store.root).glob("**/" + digest_value))
    assert matches
    matches[0].unlink()
    with pytest.raises(tensorfs.errors.Refusal):
        receive(workspace, frames(records, auth), auth)
    with workspace.locked(allow_frozen=True) as db:
        assert db.execute("SELECT state FROM workspace_state").fetchone()[0] == "frozen"
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0


def test_running_writer_is_fenced_and_its_partial_checkpoint_resumes(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    invocation = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                build_id="sha256:" + "22" * 32, job_descriptor_id="sha256:" + "33" * 32
            )
        )
    )
    spec = hashlib.sha256(invocation).digest()
    transaction = weights_transaction_id("owner", "partial", documents.spell(spec), "model")
    plain = next(value for alias, value in tensorfs.seed_digests() if alias == "plain/1")
    tensor = {
        "logical_dtype": "f32",
        "shape": [512],
        "encoding": plain,
        "parts": {"value": {"dtype": "f32", "shape": [512]}},
    }
    targets = {"model": {"add": {"a": tensor, "b": tensor}, "drop": []}}
    order = [("model", "a"), ("model", "b")]
    fingerprint = "sha256:" + "45" * 32
    declaration = store.derived_declaration(
        {}, targets, {}, order, 4096, work_fingerprint=fingerprint
    )
    writer = store.begin_derived(
        transaction, 43, {}, targets, {}, order, 4096, work_fingerprint=fingerprint
    )
    writer.add_part("model", "a", "value", io.BytesIO(b"\x31" * 2048))
    checkpoint = writer.checkpoint("partial", "model")
    head_length, link_index, stored_bytes = (
        checkpoint["head_length"],
        checkpoint["index"],
        checkpoint["bytes"],
    )
    assert (
        isinstance(head_length, int)
        and isinstance(link_index, int)
        and isinstance(stored_bytes, int)
    )
    ref = pb.CheckpointRef(
        head=pb.Ref(digest=documents.raw(str(checkpoint["head"])), length=head_length),
        plan_digest=documents.raw(str(checkpoint["plan_digest"])),
        index=link_index,
        bytes=stored_bytes,
    )
    attempt = {
        "owner_scope": "owner",
        "request_id": "partial",
        "attempt_ordinal": 7,
        "invocation_spec_digest": documents.spell(spec),
        "invocation_spec_canonical_bytes": b64(invocation),
        "state": "running",
        "outcome_id": "",
        "outcome_digest": "",
        "outcome_canonical_bytes": "",
        "retain_work": False,
    }
    weights = {
        "transaction_id": transaction,
        "owner_scope": "owner",
        "request_id": "partial",
        "attempt_ordinal": 7,
        "invocation_spec_digest": documents.spell(spec),
        "output_slot": "model",
        "declaration_digest": documents.spell(hashlib.sha256(declaration).digest()),
        "declaration_canonical_bytes": b64(declaration),
        "writer_epoch": 43,
        "state": "intent",
        "intent_ready": True,
        "restore_checkpoint": "",
        "checkpoint": b64(ref.SerializeToString()),
        "receipt_digest": "",
        "receipt_canonical_bytes": "",
        "manifest_digest": "",
        "manifest_length": 0,
        "objects": [],
    }
    workspace = Workspace(Path(store.root))
    auth = Authority()
    reply = receive(workspace, frames([("attempt", attempt), ("weights", weights)], auth), auth)
    assert store.derived_lookup(transaction) == {"state": "open"}
    with pytest.raises(tensorfs.errors.Refusal):
        writer.add_part("model", "b", "value", io.BytesIO(b"\x32" * 2048))
    activate(
        workspace,
        pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
        auth,
    )
    with workspace.locked() as db:
        historical = dict(db.execute("SELECT * FROM attempts").fetchone())
    recovered = documents.read(historical["outcome"], pb.AttemptOutcomeBody)
    assert historical["state"] == "outcome" and historical["fenced"] == 1
    assert recovered["status"] == pb.OUTCOME_STATUS_ABANDONED
    assert recovered["cause"]["code"] == pb.CAUSE_CODE_EXECUTOR_INVALIDATED
    assert recovered["cause"]["origin"] == pb.CAUSE_ORIGIN_WORKER
    assert recovered["execution_started"] is True
    assert recovered["attempt_ordinal"] == 7
    activate(
        workspace,
        pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
        auth,
    )
    with workspace.locked() as db:
        assert dict(db.execute("SELECT * FROM attempts").fetchone()) == historical
    workspace.accept(
        "owner",
        pb.AttemptOffer(
            request_id="partial",
            attempt_ordinal=8,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=invocation,
        ),
    )
    next_writer = workspace.begin_weights(
        "owner",
        pb.WeightsIntentFrame(
            request_id="partial",
            attempt_ordinal=8,
            invocation_spec_digest=spec,
            output_slot="model",
            weights_transaction_id=transaction,
            tensorfs_declaration_digest=hashlib.sha256(declaration).digest(),
            tensorfs_declaration_canonical_bytes=declaration,
        ),
    )
    assert next_writer["epoch"] == 44
    resumed = store.begin_derived(
        transaction, 44, {}, targets, {}, order, 4096, work_fingerprint=fingerprint
    )
    assert resumed.completed_parts() == [("model", "a", "value")]
    resumed.add_part("model", "b", "value", io.BytesIO(b"\x32" * 2048))
    completed = resumed.commit()
    assert completed["transaction_id"] == transaction


def test_unqualified_legacy_scalar_history_does_not_invent_owner(tmp_path: Path) -> None:
    _, workspace, records = native_export(tmp_path)
    auth = Authority()
    row = dict(records[0][1], owner_scope="")
    stream = frames([("attempt", row)], auth)
    reply = receive(workspace, stream, auth)
    activate(
        workspace,
        pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
        auth,
    )
    with workspace.locked() as db:
        assert db.execute("SELECT owner FROM attempts").fetchone()[0] == ""
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 0


@pytest.mark.parametrize("state,started", [("received", False), ("running", True)])
def test_activation_recovers_unqualified_interrupted_history_once(
    tmp_path: Path, state: str, started: bool
) -> None:
    _, workspace, records = native_export(tmp_path)
    auth = Authority()
    attempt = dict(
        records[0][1],
        owner_scope="",
        state=state,
        outcome_id="",
        outcome_digest="",
        outcome_canonical_bytes="",
    )
    reply = receive(workspace, frames([("attempt", attempt)], auth), auth)
    call = pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id)
    activate(workspace, call, auth)
    with workspace.locked() as db:
        row = dict(db.execute("SELECT * FROM attempts").fetchone())
    body = documents.read(row["outcome"], pb.AttemptOutcomeBody)
    assert row["owner"] == ""
    assert row["outcome_id"] == "out-" + hashlib.sha256(row["outcome"]).hexdigest()[:24]
    assert body["request_id"] == attempt["request_id"]
    assert body["invocation_spec_digest"] == attempt["invocation_spec_digest"]
    assert body["status"] == pb.OUTCOME_STATUS_ABANDONED
    assert body.get("execution_started", False) is started
    # Repeat the entire stream/activation after a new process opens the journal.
    resumed = Workspace(workspace.store_root)
    receive(resumed, frames([("attempt", attempt)], auth), auth)
    activate(resumed, call, auth)
    with resumed.locked() as db:
        assert dict(db.execute("SELECT * FROM attempts").fetchone()) == row
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 0


def test_recovery_terminal_preserves_committed_slots_without_claiming_success(
    tmp_path: Path,
) -> None:
    _, workspace, records = native_export(tmp_path)
    auth = Authority()
    previous_terminal = records[0][1]["outcome_canonical_bytes"]
    attempt = dict(
        records[0][1],
        state="running",
        outcome_id="",
        outcome_digest="",
        outcome_canonical_bytes="",
    )
    reply = receive(workspace, frames([("attempt", attempt), records[1], records[2]], auth), auth)
    activate(
        workspace,
        pb.WorkspaceActivationCall(claim=auth.claim, migration_id=reply.migration_id),
        auth,
    )
    with workspace.locked() as db:
        row = dict(db.execute("SELECT * FROM attempts").fetchone())
    recovered = documents.read(row["outcome"], pb.AttemptOutcomeBody)
    previous = documents.read(base64.b64decode(previous_terminal), pb.AttemptOutcomeBody)
    assert recovered["status"] == pb.OUTCOME_STATUS_ABANDONED
    assert recovered["weights_receipts"] == previous["weights_receipts"]


def test_import_never_overwrites_an_existing_workspace(tmp_path: Path) -> None:
    _, workspace, _ = produced(tmp_path)
    auth = Authority()
    with pytest.raises(WorkspaceRefusal, match="cannot be overwritten"):
        receive(workspace, frames([], auth), auth)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM weights").fetchone()[0] == 1


def test_unqualified_attempt_cannot_supply_a_memo_mapping(tmp_path: Path) -> None:
    _, workspace, records = native_export(tmp_path)
    auth = Authority()
    source = dict(records[0][1], owner_scope="")
    memo = dict(records[3][1], holds=[])
    # A self-consistent scalar terminal remains historical when its prior owner
    # is unknown; a new mapping cannot assign that authority retroactively.
    body = pb.AttemptOutcomeBody(
        request_id="producer",
        attempt_ordinal=1,
        invocation_spec_digest=source["invocation_spec_digest"],
        status=pb.OUTCOME_STATUS_SUCCEEDED,
    )
    raw = documents.canonical_bytes(body)
    source.update(
        outcome_canonical_bytes=b64(raw),
        outcome_digest=documents.spell(hashlib.sha256(raw).digest()),
    )
    memo["source"] = {key: source[key] for key in memo["source"]}
    with pytest.raises(WorkspaceRefusal, match="qualified imported attempt"):
        receive(workspace, frames([("attempt", source), ("cache", memo)], auth), auth)
    with workspace.locked(allow_frozen=True) as db:
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 0
