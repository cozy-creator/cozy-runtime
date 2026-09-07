"""The existing native finalizer, rooted in the common workspace lifetime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

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
    weights_finalize._decision(request, owner)
    transaction = weights_transaction_id(
        owner,
        request.request_id,
        documents.spell(request.invocation_spec_digest),
        request.output_slot,
    )
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT * FROM attempts WHERE owner=? AND request=?", (owner, request.request_id)
        ).fetchall()
        attempts = [
            AttemptRecord(
                request_id=row["request"],
                attempt=row["ordinal"],
                digest=row["spec"],
                spec=documents.read(row["invocation"], pb.InvocationSpec),
                state="closed"
                if row["state"] == "released"
                else "outcome"
                if row["fenced"]
                else row["state"],
            )
            for row in rows
        ]
        weights_finalize._authorize(request, attempts, False)
        if request.disposition != pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT:
            db.execute(
                "UPDATE weights SET state='released' WHERE owner=? AND id=?", (owner, transaction)
            )

        def native() -> pb.WeightsFinalizeResult:
            # Preserve exact first-wins decisions in the fixed Runtime root
            # when a private machine upgrades from the previous worker journal.
            if legacy_root is not None and legacy_root != workspace.directory:
                destination = workspace.directory / "weights-finalizations"
                destination.mkdir(mode=0o700, exist_ok=True)
                for suffix in ("intent", "result"):
                    name = transaction.removeprefix("sha256:") + "." + suffix
                    data = weights_finalize._read(legacy_root / "weights-finalizations" / name)
                    if data is not None:
                        weights_finalize._record(destination / name, data)
            return weights_finalize.finalize(
                request,
                owner_scope=owner,
                attempts=attempts,
                tensorfs_root=workspace.store_root,
                journal_root=workspace.directory,
                on_abandon=on_abandon,
            )

        return db.native(native)
