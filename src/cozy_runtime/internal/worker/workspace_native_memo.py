"""Typed native-service terminals and source roots in the common operation index."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import derived_retention
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def native_receipt_owned(
    db: Journal,
    owner: str,
    request: pb.DerivedRetentionRequest,
    kind: str,
) -> bool | None:
    rows = db.execute(
        "SELECT operation,native_receipt FROM native_calls WHERE owner=? AND native_owner=? "
        "AND state='complete'",
        (owner, request.weights_transaction_id),
    ).fetchall()
    for row in rows:
        actual = "derived" if row["operation"] == "convert_cozytensors" else "tree"
        if (
            actual == kind
            and hashlib.sha256(row["native_receipt"]).digest() == request.tensorfs_receipt_digest
        ):
            return True
    return None


def change_native_hold(
    workspace: Workspace,
    db: Journal,
    owner: str,
    request: pb.DerivedRetentionRequest,
    kind: str,
    *,
    release: bool,
) -> pb.DerivedRetentionResult:
    if kind == "derived":
        return db.native(
            lambda: derived_retention.change(
                request, tensorfs_root=workspace.store_root, release=release
            )
        )
    if kind != "tree":
        raise WorkspaceRefusal("unknown native retention kind")
    donors = [request.weights_transaction_id]
    donors.extend(
        row[0]
        for row in db.execute(
            "SELECT id FROM holds WHERE owner=? AND transaction_id=? AND native_digest=? "
            "AND kind='tree' AND state='held' ORDER BY id",
            (owner, request.weights_transaction_id, request.tensorfs_receipt_digest),
        )
    )

    def change() -> pb.DerivedRetentionResult:
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        expected = documents.spell(request.tensorfs_receipt_digest)
        own = store.tree_root(request.retention_id)
        donor = None
        for identity in donors:
            root = store.tree_root(identity)
            if not root or not root["complete"] or root.get("receipt_digest") != expected:
                continue
            if release or not root["released"]:
                donor = identity
                break
        if donor is None and own and own.get("receipt_digest") == expected:
            donor = request.retention_id
        if donor is None:
            raise WorkspaceRefusal("source-tree retention has no verified native donor")
        if release:
            store.release_tree_retention(donor, expected, request.retention_id)
            result = store.tree_root(request.retention_id)
        else:
            result = store.retain_tree_root(donor, request.retention_id)
        if not result or result["receipt_digest"] != expected or result["released"] != release:
            raise WorkspaceRefusal("source-tree retention differs from its expected native receipt")
        # This is an internal retention projection, never a fabricated DerivedReceipt.
        return pb.DerivedRetentionResult(
            weights_transaction_id=request.weights_transaction_id,
            retention_id=request.retention_id,
            tensorfs_receipt_digest=request.tensorfs_receipt_digest,
            released=release,
            manifest=pb.Ref(
                digest=documents.raw(result["manifest_digest"]), length=result["manifest_length"]
            ),
        )

    return db.native(change)


def _row(db: Journal, owner: str, service: str) -> Any:
    row = db.execute(
        "SELECT * FROM native_calls WHERE owner=? AND service_id=?", (owner, service)
    ).fetchone()
    if row is None:
        raise WorkspaceRefusal("native memo call has no accepted service intent")
    return row


def _payload(db: Journal, owner: str, service: str) -> dict[str, Any]:
    row = _row(db, owner, service)
    if row["state"] != "complete" or not row["native_receipt"] or not row["result"]:
        raise WorkspaceRefusal("native memo result has no successful retained terminal")
    value = canonical_json.decode(row["result"])
    receipt = canonical_json.decode(row["native_receipt"])
    digest = documents.spell(hashlib.sha256(row["native_receipt"]).digest())
    manifest = {
        "digest": "sha256:" + receipt["manifest"]["sha256"],
        "length": receipt["manifest"]["length"],
    }
    if value.get("manifest") != manifest or value.get("tensorfs_receipt_digest") != digest:
        raise WorkspaceRefusal("native memo result differs from its admitted receipt")
    return {
        "source": {
            "kind": "native",
            "service_id": service,
            "operation": row["operation"],
            "parent_request": row["parent_request"],
            "parent_ordinal": row["parent_ordinal"],
            "result_digest": documents.spell(hashlib.sha256(row["result"]).digest()),
            "receipt_digest": digest,
            "result": base64.b64encode(row["result"]).decode(),
            "native_receipt": base64.b64encode(row["native_receipt"]).decode(),
        },
        "holds": [
            {
                "kind": "derived" if row["operation"] == "convert_cozytensors" else "tree",
                "transaction_id": row["native_owner"],
                "native_receipt_digest": digest,
                "manifest_digest": manifest["digest"],
                "manifest_length": manifest["length"],
            }
        ],
        "schema": {"input": "model" if row["operation"] == "convert_cozytensors" else "source"},
    }


def compare_native(payload: dict[str, Any]) -> bytes:
    value = canonical_json.decode(base64.b64decode(payload["source"]["result"], validate=True))
    if set(value) != {"producer_request_id", "output_slot", "manifest", "tensorfs_receipt_digest"}:
        raise WorkspaceRefusal("native memo result is not a closed artifact value")
    (hold,) = payload["holds"]
    if value["manifest"] != {"digest": hold["manifest_digest"], "length": hold["manifest_length"]}:
        raise WorkspaceRefusal("native memo artifact changed its manifest")
    return canonical_json.encode({"kind": hold["kind"], "manifest": value["manifest"]})


def record_native(workspace: Workspace, owner: str, service_id: str) -> bool:
    from . import workspace_memo

    with workspace.locked() as db:
        row = _row(db, owner, service_id)
        key = row["computation_digest"]
        expected = workspace_memo._source_identity(_payload(db, owner, service_id)["source"])
    return workspace_memo._record(
        workspace, owner, key, expected, lambda db: _payload(db, owner, service_id)
    )


def retain_native_result(workspace: Workspace, owner: str, service_id: str) -> bytes:
    """Give the receiving request its own access before optional cache insertion."""
    from . import workspace_memo

    with workspace.locked() as db:
        row = _row(db, owner, service_id)
        payload = _payload(db, owner, service_id)
        (hold,) = payload["holds"]
        access = db.execute(
            "SELECT a.*,h.kind FROM source_access a LEFT JOIN holds h ON h.id=a.retention_id "
            "WHERE a.owner=? AND a.request=? AND a.call_index=? AND a.producer=? "
            "AND a.native_digest=?",
            (
                owner,
                row["parent_request"],
                row["call_index"],
                hold["transaction_id"],
                documents.raw(hold["native_receipt_digest"]),
            ),
        ).fetchone()
        lookup = db.execute(
            "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?",
            (owner, service_id),
        ).fetchone()
        if access is not None:
            if access["state"] != "held" or access["kind"] != hold["kind"]:
                raise WorkspaceRefusal("native source access was changed or released")
            hold["retention_id"] = access["retention_id"]
        elif lookup is not None and lookup["state"] != "miss":
            if lookup["key"] != row["computation_digest"] or lookup["state"] not in (
                "ready",
                "acknowledged",
            ):
                raise WorkspaceRefusal("native result lookup was changed or released")
            prior = canonical_json.decode(lookup["body"])
            workspace_memo._require_kind(prior, "native")
            (prior_hold,) = prior["holds"]
            if any(prior_hold.get(key) != value for key, value in hold.items()):
                raise WorkspaceRefusal("native result lookup changed its exact subject")
            hold["retention_id"] = prior_hold["retention_id"]
        else:
            from .workspace_sources import native_owner

            if hold["transaction_id"] != native_owner(owner, service_id) or (
                lookup is not None and lookup["key"] != row["computation_digest"]
            ):
                raise WorkspaceRefusal("native result has no matching producer or owned lookup")
            hold["retention_id"] = workspace_memo._hold_id(
                "native-result",
                owner,
                service_id,
                row["computation_digest"],
                hold["transaction_id"],
            )
        retained = workspace._change_hold(
            db, owner, workspace_memo._request(hold), kind=hold["kind"], release=False
        )
        current = _row(db, owner, service_id)
        if current["state"] != "complete" or current["result"] != row["result"]:
            workspace._change_hold(
                db, owner, workspace_memo._request(hold), kind=hold["kind"], release=True
            )
            raise WorkspaceRefusal("native result was canceled during recipient retention")
        db.execute(
            "INSERT INTO source_access(owner,request,call_index,producer,native_digest,"
            "manifest,manifest_length,retention_id,state) VALUES(?,?,?,?,?,?,?,?,'held') "
            "ON CONFLICT(retention_id) DO NOTHING",
            (
                owner,
                row["parent_request"],
                row["call_index"],
                hold["transaction_id"],
                retained.tensorfs_receipt_digest,
                retained.manifest.digest,
                retained.manifest.length,
                hold["retention_id"],
            ),
        )
        access = db.execute(
            "SELECT * FROM source_access WHERE retention_id=?", (hold["retention_id"],)
        ).fetchone()
        if access["state"] != "held" or access["producer"] != hold["transaction_id"]:
            raise WorkspaceRefusal("native source access was changed or released")
        return bytes(row["result"])


def lookup_native(
    workspace: Workspace, owner: str, service_id: str, computation_digest: bytes
) -> dict[str, Any] | None:
    from . import workspace_memo

    with workspace.locked() as db:
        row = _row(db, owner, service_id)
        parent = db.execute(
            "SELECT spec,state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, row["parent_request"], row["parent_ordinal"]),
        ).fetchone()
        if (
            row["state"] not in ("accepted", "resolved", "executing")
            or parent is None
            or parent["spec"] != row["parent_spec"]
            or parent["state"] != "running"
            or parent["fenced"]
        ):
            raise WorkspaceRefusal("native memo recipient has no current accepted call")
        if row["computation_digest"] not in (b"", computation_digest):
            raise WorkspaceRefusal("native memo call changed its effective computation")
        db.execute(
            "UPDATE native_calls SET computation_digest=? WHERE owner=? AND service_id=?",
            (computation_digest, owner, service_id),
        )
    found = workspace_memo._lookup(workspace, owner, computation_digest, service_id, "native")
    if found is None:
        return None
    payload, _ = found
    return {
        "result": base64.b64decode(payload["source"]["result"], validate=True),
        "native_receipt": base64.b64decode(payload["source"]["native_receipt"], validate=True),
        "producer_service": payload["source"]["service_id"],
        "holds": payload["holds"],
    }


def record_reuse(workspace: Workspace, owner: str, service_id: str) -> bytes:
    """Deliver only the owned lookup observation, preserving original producer history."""
    from . import workspace_memo

    with workspace.locked() as db:
        row = _row(db, owner, service_id)
        lookup = db.execute(
            "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?",
            (owner, service_id),
        ).fetchone()
        if (
            lookup is None
            or lookup["state"] != "ready"
            or lookup["key"] != row["computation_digest"]
        ):
            raise WorkspaceRefusal("native reuse has no completed owned lookup")
        payload = canonical_json.decode(lookup["body"])
        workspace_memo._require_kind(payload, "native")
        result = base64.b64decode(payload["source"]["result"], validate=True)
        receipt = base64.b64decode(payload["source"]["native_receipt"], validate=True)
        (hold,) = payload["holds"]
        native = db.execute("SELECT * FROM holds WHERE id=?", (hold["retention_id"],)).fetchone()
        if (
            native is None
            or native["owner"] != owner
            or native["state"] != "held"
            or native["transaction_id"] != hold["transaction_id"]
            or documents.spell(native["native_digest"]) != hold["native_receipt_digest"]
        ):
            raise WorkspaceRefusal("native reuse lost its recipient custody")
        parent = db.execute(
            "SELECT spec,state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, row["parent_request"], row["parent_ordinal"]),
        ).fetchone()
        if (
            row["state"] == "canceled"
            or parent is None
            or parent["fenced"]
            or parent["spec"] != row["parent_spec"]
            or parent["state"] != "running"
        ):
            raise WorkspaceRefusal("native reuse parent stopped before result delivery")
        if row["result"] and (row["result"] != result or row["native_receipt"] != receipt):
            raise WorkspaceRefusal("native reused result changed after acceptance")
        db.execute("BEGIN IMMEDIATE")
        try:
            if hold["kind"] == "tree":
                db.execute(
                    "INSERT INTO source_access(owner,request,call_index,producer,native_digest,"
                    "manifest,manifest_length,retention_id,state) VALUES(?,?,?,?,?,?,?,?,'held') "
                    "ON CONFLICT(retention_id) DO NOTHING",
                    (
                        owner,
                        row["parent_request"],
                        row["call_index"],
                        hold["transaction_id"],
                        native["native_digest"],
                        native["manifest"],
                        native["manifest_length"],
                        hold["retention_id"],
                    ),
                )
            db.execute(
                "UPDATE native_calls SET result=?,native_receipt=?,native_owner=?,state='complete' "
                "WHERE owner=? AND service_id=?",
                (result, receipt, hold["transaction_id"], owner, service_id),
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise
    workspace_memo.delivered(workspace, owner, service_id)
    return result


def release_native_lookup(workspace: Workspace, owner: str, service_id: str) -> None:
    from . import workspace_memo

    with workspace.locked() as db:
        lookup = db.execute(
            "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?",
            (owner, service_id),
        ).fetchone()
        if lookup is None:
            return
        payload = canonical_json.decode(lookup["body"])
        workspace_memo._require_kind(payload, "native")
        db.execute(
            "UPDATE operation_lookups SET state='released' WHERE owner=? AND consumer=?",
            (owner, service_id),
        )
        for hold in payload["holds"]:
            workspace._change_hold(
                db, owner, workspace_memo._request(hold), kind=hold["kind"], release=True
            )
            db.execute(
                "UPDATE source_access SET state='released' WHERE retention_id=? AND owner=?",
                (hold["retention_id"], owner),
            )


def release_native_result(workspace: Workspace, owner: str, service_id: str) -> None:
    """Release this call's recipient access; original producer/cache roots are separate."""
    from . import workspace_memo
    from .workspace_sources import native_owner

    release_native_lookup(workspace, owner, service_id)
    with workspace.locked() as db:
        row = _row(db, owner, service_id)
        access = db.execute(
            "SELECT a.*,h.kind FROM source_access a JOIN holds h ON h.id=a.retention_id "
            "WHERE a.owner=? AND a.request=? AND a.call_index=? AND a.state<>'released'",
            (owner, row["parent_request"], row["call_index"]),
        ).fetchall()
        if (
            not access
            and row["state"] == "complete"
            and row["native_owner"] == native_owner(owner, service_id)
        ):
            # A cancel may arrive before the first recipient is created. Fence its
            # deterministic identity even when no source_access row exists yet.
            (hold,) = _payload(db, owner, service_id)["holds"]
            hold["retention_id"] = workspace_memo._hold_id(
                "native-result",
                owner,
                service_id,
                row["computation_digest"],
                hold["transaction_id"],
            )
            workspace._change_hold(
                db, owner, workspace_memo._request(hold), kind=hold["kind"], release=True
            )
        for held in access:
            db.execute(
                "UPDATE source_access SET state='releasing' WHERE retention_id=?",
                (held["retention_id"],),
            )
            workspace._change_hold(
                db,
                owner,
                pb.DerivedRetentionRequest(
                    weights_transaction_id=held["producer"],
                    retention_id=held["retention_id"],
                    tensorfs_receipt_digest=held["native_digest"],
                ),
                kind=held["kind"],
                release=True,
            )
            db.execute(
                "UPDATE source_access SET state='released' WHERE retention_id=?",
                (held["retention_id"],),
            )
