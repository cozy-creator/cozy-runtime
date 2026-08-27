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
    ArtifactCommit,
    ArtifactComponentMetadata,
    ArtifactInspection,
    ArtifactSink,
    ArtifactSourceConfig,
    ArtifactValue,
    CopyComponent,
    CreateComponent,
    CreateConfig,
    DeriveComponent,
    DeriveConfig,
    Tree,
)
from cozy_runtime.author._artifact_sink import _ArtifactRequest
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._services import Attempt
from cozy_runtime.internal import artifact_transactions
from cozy_runtime.internal.journal import Journal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

PROOF_CONFIG = b'{"proof":true}'


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def source_header(store: Any, snapshot: str) -> dict[str, Any]:
    header_bytes = store.snapshot(snapshot)["header"]
    unbound = tensorfs.parse_header(header_bytes)
    closure = tensorfs.Closure()
    for ref in unbound["encodings"]:
        closure.insert(store.document(str(ref["sha256"])))
    parsed = tensorfs.parse_header(header_bytes, closure)
    if not isinstance(parsed, dict):
        raise TypeError("TensorFS parse_header returned a non-object")
    return parsed


def part_bytes(dtype: str, shape: list[int]) -> int:
    total = int(tensorfs.DTYPES[dtype])
    for extent in shape:
        total *= int(extent)
    return total


def added_value(value: ArtifactValue, key: str) -> ArtifactValue:
    return ArtifactValue(
        key=key,
        logical_dtype=value.logical_dtype,
        shape=value.shape,
        encoding=value.encoding,
        parts=value.parts,
    )


def request(
    snapshot: str,
    component: ArtifactComponentMetadata,
    source_config: ArtifactSourceConfig,
) -> _ArtifactRequest:
    source_value = component.tensors[0]
    derived_key = "proof.derived"
    created_key = "proof.created"
    return _ArtifactRequest(
        "artifact",
        {"source": Tree("proof-source", digest=snapshot)},
        {
            "copied": CopyComponent("source", component.name),
            "created": CreateComponent((added_value(source_value, created_key),)),
            "derived": DeriveComponent(
                "source",
                component.name,
                drop=(source_value.key,),
                add=(added_value(source_value, derived_key),),
            ),
        },
        {
            "proof": CreateConfig("application/json", len(PROOF_CONFIG)),
            "proof-derived": DeriveConfig(
                "source",
                source_config.name,
                source_config.media_type,
                len(source_config.data),
            ),
        },
    )


def drive(
    sink: ArtifactSink,
    artifact_request: _ArtifactRequest,
    source_component: ArtifactComponentMetadata,
    source_config: ArtifactSourceConfig,
) -> ArtifactCommit:
    with sink.open(
        artifact_request.output_slot,
        sources=artifact_request.sources,
        targets=artifact_request.targets,
        configs=artifact_request.configs,
    ) as writer:
        if writer.replayed:
            return writer.commit()
        source_value = source_component.tensors[0]
        source_part = source_value.parts[0]
        sample = bytearray(min(part_bytes(source_part.dtype, list(source_part.shape)), 32))
        writer.source_read_into(
            "source",
            source_component.name,
            source_value.key,
            source_part.role,
            0,
            memoryview(sample),
        )
        assert sample, "the grant-bound source reader returned no bytes"
        for component, target in artifact_request.targets.items():
            values = (
                target.add
                if isinstance(target, DeriveComponent)
                else target.values
                if isinstance(target, CreateComponent)
                else ()
            )
            for value in values:
                for part in value.parts:
                    length = part_bytes(part.dtype, list(part.shape))
                    writer.add_part(component, value.key, part.role, io.BytesIO(bytes(length)))
        writer.add_config("proof", io.BytesIO(PROOF_CONFIG))
        writer.add_config("proof-derived", io.BytesIO(source_config.data))
        return writer.commit()


def finalize(
    coordinator: artifact_transactions.ArtifactCoordinator,
    *,
    request_id: str,
    disposition: pb.ArtifactFinalizeDisposition,
    receipt_digest: str = "",
    scratch_root: str = "",
) -> pb.ArtifactFinalizeResultFrame:
    decision = pb.ArtifactFinalizeDecision(
        owner_authority_scope="proof-owner",
        request_id=request_id,
        invocation_spec_digest=digest("cr-030-live-invocation"),
        output_slot="artifact",
        disposition=disposition,
        artifact_receipt_digest=receipt_digest,
        scratch_root_id=scratch_root,
    )
    decision_bytes, decision_digest = documents.identity(decision)
    return coordinator.finalize_request(
        pb.ArtifactFinalizeRequest(
            request_id=request_id,
            invocation_spec_digest=documents.raw(digest("cr-030-live-invocation")),
            output_slot="artifact",
            decision_digest=decision_digest,
            decision_canonical_bytes=decision_bytes,
        ),
        "proof-owner",
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
    component_name = sorted(header["components"])[0]
    config_name = sorted(header["configs"])[0]
    invocation = digest("cr-030-live-invocation")
    grant_sources = {"proof-source": (snapshot, snapshot_length)}
    grant_outputs = {"artifact": 0}

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

        backend = artifact_transactions.ExecutorArtifactSink(
            store_root=store_root,
            spool=spool,
            granted_sources=grant_sources,
            granted_outputs=grant_outputs,
            durable=exchange("normal", 1, 1),
        )
        author_attempt = Attempt("artifact-proof", spool)
        sink = ArtifactSink(author_attempt, backend.inspect, backend.open)
        inspection: ArtifactInspection = sink.inspect(
            Tree("proof-source", digest=snapshot),
            components=(component_name,),
            configs=(config_name,),
        )
        component = inspection.components[0]
        source_config = inspection.configs[0]
        first = request(snapshot, component, source_config)
        produced_bytes = (
            len(PROOF_CONFIG)
            + len(source_config.data)
            + sum(
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
        )
        grant_outputs["artifact"] = produced_bytes
        committed = drive(sink, first, component, source_config)
        replayed = drive(
            sink,
            request(snapshot, component, source_config),
            component,
            source_config,
        )
        assert committed.snapshot_digest == replayed.snapshot_digest and replayed.replayed
        retry_sink = artifact_transactions.ExecutorArtifactSink(
            store_root=store_root,
            spool=spool,
            granted_sources=grant_sources,
            granted_outputs=grant_outputs,
            durable=exchange("normal", 4, 4),
        )
        retry_public = ArtifactSink(author_attempt, retry_sink.inspect, retry_sink.open)
        assert drive(
            retry_public,
            request(snapshot, component, source_config),
            component,
            source_config,
        ).replayed
        assert (
            len(coordinator.settle(request_id="normal", attempt_ordinal=4, executor_generation=4))
            == 1
        )
        normal_transaction = artifact_transactions.transaction_id(
            "proof-owner", "normal", invocation, "artifact"
        )
        normal_receipt = coordinator.state.receipts[normal_transaction]
        outer = documents.read(normal_receipt.artifact_receipt_bytes, pb.ArtifactReceipt)
        assert outer["artifact_transaction_id"] == normal_transaction
        assert outer["tensorfs_receipt_digest"] == normal_receipt.tensorfs_receipt_digest
        outcome_bytes, _ = documents.identity(
            pb.AttemptOutcomeBody(
                request_id="normal",
                attempt_ordinal=1,
                invocation_spec_digest=invocation,
                status=pb.OUTCOME_STATUS_SUCCEEDED,
                execution_started=True,
                cause=pb.OutcomeCause(
                    code=pb.CAUSE_CODE_UNSPECIFIED,
                    origin=pb.CAUSE_ORIGIN_RUNTIME,
                ),
                artifact_receipts=[
                    pb.ArtifactReceiptRef(
                        artifact_receipt_digest=documents.raw(
                            normal_receipt.artifact_receipt_digest
                        ),
                        artifact_receipt_canonical_bytes=normal_receipt.artifact_receipt_bytes,
                    )
                ],
            )
        )
        assert documents.read(outcome_bytes, pb.AttemptOutcomeBody)["artifact_receipts"]
        finalized = finalize(
            coordinator,
            request_id="normal",
            disposition=pb.ARTIFACT_FINALIZE_DISPOSITION_ADOPT,
            receipt_digest=normal_receipt.artifact_receipt_digest,
            scratch_root=artifact_transactions.scratch_root_id("proof-owner", "normal", "artifact"),
        )
        finalized_doc = documents.read(finalized.result_canonical_bytes, pb.ArtifactFinalizeResult)
        assert finalized_doc["outcome"] == pb.ARTIFACT_FINALIZE_OUTCOME_ADOPTED
        replayed_final = finalize(
            coordinator,
            request_id="normal",
            disposition=pb.ARTIFACT_FINALIZE_DISPOSITION_ADOPT,
            receipt_digest=normal_receipt.artifact_receipt_digest,
            scratch_root=artifact_transactions.scratch_root_id("proof-owner", "normal", "artifact"),
        )
        assert replayed_final.result_canonical_bytes == finalized.result_canonical_bytes
        print(
            "PASS public inspect/read + copy/create/derive config+tensor + retry replay + adopt",
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
        crash_public = ArtifactSink(author_attempt, crash_sink.inspect, crash_sink.open)
        try:
            drive(
                crash_public,
                request(snapshot, component, source_config),
                component,
                source_config,
            )
        except CapabilityError as exc:
            assert "planted crash" in str(exc)
        else:
            raise AssertionError("the planted commit/journal crash did not fire")
        journal.close()
        journal = Journal(journal_path)
        coordinator = artifact_transactions.ArtifactCoordinator(journal, store_root)
        recovered = coordinator.settle(
            request_id="crash-gap", attempt_ordinal=2, executor_generation=2
        )
        assert len(recovered) == 1
        finalize(
            coordinator,
            request_id="crash-gap",
            disposition=pb.ARTIFACT_FINALIZE_DISPOSITION_ABANDON,
            receipt_digest=recovered[0].artifact_receipt_digest,
        )
        print("PASS commit-before-journal recovered exact receipt + released root")

        uncommitted = request(snapshot, component, source_config)
        prepared = artifact_transactions.prepare_declaration(
            uncommitted, granted_sources=grant_sources
        )
        staged_name, staged_digest = artifact_transactions.stage_exact(
            spool, "declaration", prepared.canonical_bytes
        )
        coordinator.open(
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
        abandoned = finalize(
            coordinator,
            request_id="uncommitted",
            disposition=pb.ARTIFACT_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED,
        )
        abandoned_doc = documents.read(abandoned.result_canonical_bytes, pb.ArtifactFinalizeResult)
        assert abandoned_doc["outcome"] == pb.ARTIFACT_FINALIZE_OUTCOME_ABANDONED
        print("PASS receipt-free uncommitted abandon tombstoned the semantic identity")
        journal.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
