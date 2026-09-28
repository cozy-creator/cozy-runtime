"""Recover work only after its originating Runtime process birth has ended."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, proctree, weights_sink
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.workspace import Journal, Workspace


def process_identity() -> bytes:
    process = proctree.process_identity(os.getpid())
    return canonical_json.encode(
        {"boot": _boot(), "pid": process.pid, "started_ticks": process.started_ticks}
    )


def _boot() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_ended(identity: bytes) -> bool:
    value = canonical_json.decode(identity)
    if not isinstance(value, dict) or not {"boot", "pid", "started_ticks"} <= set(value):
        raise WorkspaceRefusal("workspace process identity is invalid")
    boot, pid, ticks = value["boot"], value["pid"], value["started_ticks"]
    if (
        not isinstance(boot, str)
        or len(boot) != 36
        or type(pid) is not int
        or type(ticks) is not int
        or pid <= 0
        or ticks < 0
    ):
        raise WorkspaceRefusal("workspace process birth is invalid")
    if boot != _boot():
        return True
    expected = proctree.ProcessIdentity(pid, ticks)
    try:
        actual = proctree.process_identity(pid)
    except ProcessLookupError:
        # The process helper may also report an unreadable /proc entry. Only
        # confirmed absence permits recovery; an unreadable live process does not.
        try:
            Path(f"/proc/{pid}").stat()
        except FileNotFoundError:
            return True
        except OSError as exc:
            raise WorkspaceRefusal("previous Runtime process state is unavailable") from exc
        return False
    return actual != expected or proctree.process_state(actual) == "Z"


def native_result(workspace: Workspace, row: dict[str, Any]) -> pb.WeightsReceiptFrame | None:
    store = fill.store(workspace.store_root)
    found = store.derived_lookup(row["id"])
    if found["state"] == "open":
        epoch = found.get("writer_session_id")
        if epoch:
            if epoch > row["epoch"]:
                raise WorkspaceRefusal("native writer advanced beyond its interrupted execution")
            store.derived_fence(row["id"], epoch)
        return None
    if found["state"] != "committed" or row["receipt"]:
        return None
    if found.get("disposition", {}).get("kind") == "released":
        return None
    native = found["receipt"]
    admitted = canonical_json.decode(row["declaration"])
    declared = native.get("declaration")
    if declared is None:
        changed = (
            weights_sink.declaration_digest(native) != hashlib.sha256(row["declaration"]).digest()
        )
    else:
        # Another TensorFS version may add declaration facts; only a changed shared fact counts.
        changed = not isinstance(admitted, dict) or any(
            declared.get(key) != value for key, value in admitted.items() if key in declared
        )
    if changed:
        raise WorkspaceRefusal("interrupted native result changed its admitted declaration")
    raw = canonical_json.encode(native)
    outer, digest = documents.identity(
        pb.WeightsReceipt(
            owner_authority_scope=row["owner"],
            request_id=row["request"],
            invocation_spec_digest=documents.spell(row["spec"]),
            output_slot=row["slot"],
            weights_transaction_id=row["id"],
            tensorfs_receipt_digest=documents.spell(hashlib.sha256(raw).digest()),
            tensorfs_receipt_canonical_bytes=raw,
        )
    )
    manifest = native["manifest"]
    manifest_id = "sha256:" + manifest["sha256"]
    store.inspect_derived_source(
        manifest_id, manifest["length"], sorted(admitted["components"]), []
    )
    lease = store.acquire_cozytensors(manifest_id)
    try:
        objects = {
            manifest_id: pb.WeightsObjectSource(
                object_id=manifest_id,
                length=manifest["length"],
                source_ref="tensorfs-manifest:" + manifest_id,
            )
        }
        for obj in store.walk(manifest_id):
            objects[obj["id"]] = pb.WeightsObjectSource(
                object_id=obj["id"], length=obj["length"], source_ref="tensorfs-object:" + obj["id"]
            )
    finally:
        lease.release()
    return pb.WeightsReceiptFrame(
        request_id=row["request"],
        attempt_ordinal=row["ordinal"],
        invocation_spec_digest=row["spec"],
        output_slot=row["slot"],
        weights_transaction_id=row["id"],
        writer_epoch=row["epoch"],
        tensorfs_declaration_digest=row["declaration_digest"],
        weights_receipt=pb.WeightsReceiptRef(
            weights_receipt_digest=digest, weights_receipt_canonical_bytes=outer
        ),
        manifest=pb.Ref(digest=bytes.fromhex(manifest["sha256"]), length=manifest["length"]),
        objects=[objects[key] for key in sorted(objects)],
    )


def terminal(db: Journal, attempt: Any, detail: str) -> None:
    rows = db.execute(
        "SELECT * FROM weights WHERE owner=? AND request=? AND ordinal=? AND spec=? ORDER BY slot",
        (attempt["owner"], attempt["request"], attempt["ordinal"], attempt["spec"]),
    ).fetchall()
    body = pb.AttemptOutcomeBody(
        request_id=attempt["request"],
        attempt_ordinal=attempt["ordinal"],
        invocation_spec_digest=documents.spell(attempt["spec"]),
        status=pb.OUTCOME_STATUS_ABANDONED,
        safe_message=detail,
        cause=pb.OutcomeCause(
            code=pb.CAUSE_CODE_EXECUTOR_INVALIDATED, origin=pb.CAUSE_ORIGIN_INFRA, detail=detail
        ),
        execution_started=attempt["state"] == "running" or bool(rows),
    )
    for row in rows:
        if row["state"] == "receipt":
            body.weights_receipts.append(
                pb.WeightsReceiptRef(
                    weights_receipt_digest=row["receipt_digest"],
                    weights_receipt_canonical_bytes=row["receipt"],
                )
            )
    raw, digest = documents.identity(body)
    db.execute(
        "UPDATE attempts SET state='outcome',fenced=1,outcome_id=?,outcome_digest=?,outcome=? "
        "WHERE owner=? AND request=? AND ordinal=?",
        (
            "out-" + digest.hex()[:24],
            digest,
            raw,
            attempt["owner"],
            attempt["request"],
            attempt["ordinal"],
        ),
    )
