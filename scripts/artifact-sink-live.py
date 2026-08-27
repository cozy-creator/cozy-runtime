#!/usr/bin/env python3
"""cr-030 live proof over one real TensorFS store; no fake store or mock transport.

The caller supplies a tiny real source snapshot (tfs-019's `checkpoint write` is enough).
This run crosses Runtime's exact worker/executor transaction seam in-process, because the
seam protocol itself is not the subject here: exact grant closure, copy/derive/create,
commit recovery, fenced replay, and both final dispositions are.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensorfs

from cozy_runtime.author import (
    ArtifactPart,
    ArtifactValue,
    CopyComponent,
    CopyConfig,
    CreateComponent,
    DeriveComponent,
    Tree,
)
from cozy_runtime.author._artifact_sink import _ArtifactRequest
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal import artifact_transactions
from cozy_runtime.internal.journal import Journal


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def source_header(store: Any, snapshot: str) -> dict[str, Any]:
    header_bytes = store.snapshot(snapshot)["header"]
    unbound = tensorfs.parse_header(header_bytes)
    closure = tensorfs.Closure()
    for ref in unbound["encodings"]:
        closure.insert(store.document(str(ref["sha256"])))
    return tensorfs.parse_header(header_bytes, closure)


def part_bytes(dtype: str, shape: list[int]) -> int:
    total = int(tensorfs.DTYPES[dtype])
    for extent in shape:
        total *= int(extent)
    return total


def added_value(entry: dict[str, Any], key: str) -> ArtifactValue:
    logical = entry["logical"]
    parts = tuple(
        ArtifactPart(
            role=str(role),
            dtype=str(part["dtype"]),
            shape=tuple(int(extent) for extent in part["shape"]),
            data=io.BytesIO(bytes(part_bytes(str(part["dtype"]), part["shape"]))),
        )
        for role, part in sorted(entry["parts"].items())
    )
    return ArtifactValue(
        key=key,
        logical_dtype=str(logical["dtype"]),
        shape=tuple(int(extent) for extent in logical["shape"]),
        encoding=str(entry["encoding"]),
        parts=parts,
    )


def request(snapshot: str, header: dict[str, Any], component: str) -> _ArtifactRequest:
    source_values = header["components"][component]
    source_key = sorted(source_values)[0]
    entry = source_values[source_key]
    derived_key = "proof.derived"
    created_key = "proof.created"
    return _ArtifactRequest(
        "artifact",
        {"source": Tree("proof-source", digest=snapshot)},
        {
            "copied": CopyComponent("source", component),
            "created": CreateComponent((added_value(entry, created_key),)),
            "derived": DeriveComponent(
                "source",
                component,
                drop=(source_key,),
                add=(added_value(entry, derived_key),),
            ),
        },
        {name: CopyConfig("source", name) for name in sorted(header["configs"])},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    store_root = Path(args.store).resolve()
    snapshot = args.snapshot
    store = tensorfs.Store.open(str(store_root))
    snapshot_length = len(store.document(snapshot))
    header = source_header(store, snapshot)
    component = sorted(header["components"])[0]
    first = request(snapshot, header, component)
    produced_bytes = sum(
        part_bytes(part.dtype, list(part.shape))
        for target in first.targets.values()
        for value in (
            target.add
            if isinstance(target, DeriveComponent)
            else target.values
            if isinstance(target, CreateComponent)
            else ()
        )
        for part in value.parts
    )
    invocation = digest("cr-030-live-invocation")
    grant_sources = {"proof-source": (snapshot, snapshot_length)}
    grant_outputs = {"artifact": produced_bytes}

    with tempfile.TemporaryDirectory(prefix="cozy-artifact-sink-live-") as temporary:
        root = Path(temporary)
        spool = root / "spool"
        spool.mkdir()
        journal_path = root / "journal.log"
        journal = Journal(journal_path)
        coordinator = artifact_transactions.ArtifactCoordinator(journal, store_root)

        def exchange(
            request_id: str, attempt: int, generation: int, crash: bool = False
        ) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
            def durable(kind: str, body: dict[str, Any]) -> dict[str, Any]:
                if kind == "artifact_open":
                    return {
                        "ok": True,
                        **coordinator.open(
                            owner_authority_scope="proof-owner",
                            request_id=request_id,
                            invocation_spec_digest=invocation,
                            attempt_ordinal=attempt,
                            executor_generation=generation,
                            spool=spool,
                            staged_name=str(body["staged_name"]),
                            staged_digest=str(body["staged_digest"]),
                            staged_length=int(body["staged_length"]),
                            granted_sources=grant_sources,
                            granted_outputs=grant_outputs,
                        ),
                    }
                if kind == "artifact_commit" and crash:
                    raise RuntimeError("planted crash after TensorFS commit")
                if kind == "artifact_commit":
                    return {
                        "ok": True,
                        **coordinator.commit(
                            transaction=str(body["transaction_id"]),
                            writer_session_id=int(body["writer_session_id"]),
                            spool=spool,
                            staged_name=str(body["staged_name"]),
                            staged_digest=str(body["staged_digest"]),
                            staged_length=int(body["staged_length"]),
                        ),
                    }
                raise AssertionError(kind)

            return durable

        sink = artifact_transactions.ExecutorArtifactSink(
            store_root=store_root,
            spool=spool,
            granted_sources=grant_sources,
            granted_outputs=grant_outputs,
            durable=exchange("normal", 1, 1),
        )
        committed = sink.commit(first)
        replayed = sink.commit(request(snapshot, header, component))
        assert committed.snapshot_digest == replayed.snapshot_digest and replayed.replayed
        normal_transaction = artifact_transactions.transaction_id(
            "proof-owner", "normal", invocation, "artifact"
        )
        normal_receipt = coordinator.state.receipts[normal_transaction]
        finalized, replay = coordinator.finalize(
            transaction=normal_transaction,
            decision="adopt",
            receipt_digest=normal_receipt.receipt_digest,
            scratch_root="proof-adopted",
        )
        assert finalized["status"] == "adopted" and not replay
        assert coordinator.finalize(
            transaction=normal_transaction,
            decision="adopt",
            receipt_digest=normal_receipt.receipt_digest,
            scratch_root="proof-adopted",
        )[1]
        print(
            "PASS copy/create/derive + replay + adopt",
            committed.snapshot_digest,
            f"new_bytes_cap={produced_bytes}",
        )

        crash_sink = artifact_transactions.ExecutorArtifactSink(
            store_root=store_root,
            spool=spool,
            granted_sources=grant_sources,
            granted_outputs=grant_outputs,
            durable=exchange("crash-gap", 2, 2, crash=True),
        )
        try:
            crash_sink.commit(request(snapshot, header, component))
        except CapabilityError as exc:
            assert "planted crash" in str(exc)
        else:
            raise AssertionError("the planted commit/journal crash did not fire")
        journal.close()
        journal = Journal(journal_path)
        coordinator = artifact_transactions.ArtifactCoordinator(journal, store_root)
        recovered = coordinator.settle(attempt_ordinal=2, executor_generation=2)
        assert len(recovered) == 1
        coordinator.finalize(
            transaction=recovered[0].transaction_id,
            decision="abandon",
            receipt_digest=recovered[0].receipt_digest,
        )
        print("PASS commit-before-journal recovered exact receipt + released root")

        uncommitted = request(snapshot, header, component)
        prepared = artifact_transactions.prepare_declaration(
            uncommitted, granted_sources=grant_sources
        )
        staged_name, staged_digest = artifact_transactions.stage_exact(
            spool, "declaration", prepared.canonical_bytes
        )
        opened = coordinator.open(
            owner_authority_scope="proof-owner",
            request_id="uncommitted",
            invocation_spec_digest=invocation,
            attempt_ordinal=3,
            executor_generation=3,
            spool=spool,
            staged_name=staged_name,
            staged_digest=staged_digest,
            staged_length=len(prepared.canonical_bytes),
            granted_sources=grant_sources,
            granted_outputs=grant_outputs,
        )
        abandoned, _ = coordinator.finalize(
            transaction=str(opened["transaction_id"]), decision="abandon_uncommitted"
        )
        assert abandoned["status"] == "abandoned"
        print("PASS receipt-free uncommitted abandon tombstoned the semantic identity")
        journal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
