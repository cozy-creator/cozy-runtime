"""The existing native finalizer, rooted in the common workspace lifetime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker import weights_finalize
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.workspace import Workspace


def finalize(
    workspace: Workspace,
    owner: str,
    request: pb.WeightsFinalizeRequest,
    *,
    legacy_root: Path | None = None,
    on_abandon: Callable[[str], None] | None = None,
) -> pb.WeightsFinalizeResult:
    decision = canonical_json.encode(weights_finalize._decision(request, owner))
    transaction = weights_transaction_id(
        owner,
        request.request_id,
        documents.spell(request.invocation_spec_digest),
        request.output_slot,
    )
    directory = workspace.directory / "weights-finalizations"
    directory.mkdir(mode=0o700, exist_ok=True)
    name = transaction.removeprefix("sha256:")
    legacy: dict[str, bytes] = {}
    if legacy_root is not None and legacy_root != workspace.directory:
        for suffix in ("intent", "result"):
            value = weights_finalize._read(
                legacy_root / "weights-finalizations" / (name + "." + suffix)
            )
            if value is not None:
                legacy[suffix] = value
    with workspace.locked() as db:

        def attempts() -> list[AttemptRecord]:
            rows = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=?", (owner, request.request_id)
            ).fetchall()
            return [
                AttemptRecord(
                    request_id=row["request"],
                    attempt=row["ordinal"],
                    digest=row["spec"],
                    spec=documents.read(row["invocation"], pb.InvocationSpec)
                    if row["invocation"]
                    else {},
                    state="closed"
                    if row["state"] == "released"
                    else "outcome"
                    if row["fenced"]
                    else row["state"],
                )
                for row in rows
            ]

        weights_finalize._authorize(request, attempts(), False)
        observed = db.native(
            lambda: (
                fill.tensorfs_module()
                .Store.open(str(workspace.store_root))
                .derived_lookup(transaction)
            )
        )
        if request.disposition != pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED:
            if observed.get("state") != "committed":
                raise weights_finalize.FinalizationRefusal("finalization requires a native receipt")
            weights_finalize._receipt(request, transaction, observed.get("receipt"))
        current_attempts = attempts()
        weights_finalize._authorize(
            request, current_attempts, bool(observed.get("writer_session_id"))
        )
        # Import legacy decisions and choose the immutable winner while holding
        # the SAME journal lock used by all local/private Runtime processes.
        if legacy.get("intent") is not None:
            weights_finalize._record(directory / (name + ".intent"), legacy["intent"])
        previous = weights_finalize._read(directory / (name + ".intent"))
        if previous is not None and previous != decision:
            raise weights_finalize.FinalizationRefusal(
                "weights finalization conflicts with its first-wins decision"
            )
        weights_finalize._record(directory / (name + ".intent"), decision)
        if legacy.get("result") is not None:
            weights_finalize._record(directory / (name + ".result"), legacy["result"])
        if request.disposition != pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT:
            db.execute(
                "UPDATE weights SET state='released' WHERE owner=? AND id=?", (owner, transaction)
            )
        return db.native(
            lambda: weights_finalize.finalize(
                request,
                owner_scope=owner,
                attempts=current_attempts,
                tensorfs_root=workspace.store_root,
                journal_root=workspace.directory,
                on_abandon=on_abandon,
            )
        )
