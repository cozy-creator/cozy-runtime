"""Bounded publication work on Runtime's existing durable Python-call journal."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import effect_interfaces, fill
from cozy_runtime.internal.worker import machine_models
from cozy_runtime.internal.worker.machine_artifacts import model_retention
from cozy_runtime.internal.worker.machine_publication import (
    PublicationClient,
    PublicationRefusal,
    PublicationTLS,
    apply_release,
    prepare_release,
    upload_checkpoint,
)
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call, Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class Effects:
    def __init__(self, workspace: Workspace, executions: Executions, tls: PublicationTLS | None):
        self.workspace, self.executions, self.tls = workspace, executions, tls
        self.journal = Calls(workspace)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="publication")
        self.lock = threading.Lock()
        self.active: dict[tuple[str, str], Future[None]] = {}
        self.retry_after: dict[tuple[str, str], float] = {}
        self.stopped = False

    def close(self) -> None:
        with self.lock:
            self.stopped = True
        self.pool.shutdown(wait=True, cancel_futures=True)

    def schedule(self, owner: str, call: Call) -> None:
        """Queue metadata work, never perform network IO on the worker control lock."""
        key = owner, call.child_request
        with self.lock:
            if self.stopped or key in self.active or len(self.active) >= 2:
                return
            if self.retry_after.get(key, 0) > time.monotonic():
                return
            self.active[key] = self.pool.submit(self._run, owner, call)

    def drive(self, owner: str) -> None:
        # Sent calls stay in this query after the parent stops or the client goes
        # away. Their exact remote state must be reconciled before retention ends.
        with self.lock:
            for key, deadline in list(self.retry_after.items()):
                if deadline <= time.monotonic():
                    del self.retry_after[key]
            for key, future in list(self.active.items()):
                if future.done():
                    del self.active[key]
                    if future.exception() is not None:
                        self.retry_after[key] = time.monotonic() + 1
            if self.stopped:
                return
        for call in self.journal.pending_effects(owner):
            self.schedule(owner, call)
        self._cleanup(owner)

    def _before_write(self, owner: str, call: Call) -> None:
        with self.lock:
            if self.stopped:
                raise WorkspaceRefusal("publication worker is stopping")
        self.journal.before_write(owner, call)

    def _run(self, owner: str, original: Call) -> None:
        call = self.journal.get(owner, original.parent_request, original.call_index)
        try:
            if call.result or call.safe_code:
                return
            with self.workspace.locked() as db:
                pure = db.execute(
                    "SELECT memoize FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, call.parent_request, call.parent_ordinal),
                ).fetchone()
            if pure is None or pure["memoize"]:
                raise PublicationRefusal("publication.impure_call_forbidden")
            grant = self.executions.publication_authorization(owner, call.parent_request)
            if not grant or self.tls is None:
                raise PublicationRefusal("publication.authority_absent", status=403)
            intent = canonical_json.decode(call.intent)
            if (
                intent["module"] != effect_interfaces.MODULE
                or intent["interface_digest"] != effect_interfaces.DIGEST
            ):
                raise PublicationRefusal("publication.interface_changed")
            client = self.tls.client(grant)
            operation = intent["export"]
            if operation == "publish_release":
                request = msgspec.convert(
                    intent["request"], type=effect_interfaces.Publish, strict=True
                )
                if not call.prepared:
                    frozen = prepare_release(client, **msgspec.structs.asdict(request))
                    call = self.journal.freeze(owner, call, frozen)
                result: Any = apply_release(
                    client,
                    call.prepared,
                    sent=call.sent,
                    before_write=lambda: self._before_write(owner, call),
                )
            elif operation == "upload_checkpoint":
                request_upload = msgspec.convert(
                    intent["request"], type=effect_interfaces.Upload, strict=True
                )
                result = self._upload(owner, call, request_upload, client)
            else:
                raise PublicationRefusal("publication.operation_unavailable")
            raw = canonical_json.encode(msgspec.to_builtins(result))
            latest = self.journal.get(owner, call.parent_request, call.call_index)
            if latest.sent:
                self.journal.complete(owner, call, raw)
            else:
                self.journal.observe_result(owner, call, raw)
        except (PublicationRefusal, WorkspaceRefusal, msgspec.ValidationError, ValueError) as exc:
            latest = self.journal.get(owner, call.parent_request, call.call_index)
            transient = isinstance(exc, PublicationRefusal) and (
                exc.status in {408, 425, 429} or exc.status >= 500
            )
            if not latest.sent and not transient:
                code = (
                    exc.code
                    if isinstance(exc, PublicationRefusal)
                    else "publication.intent_refused"
                )
                self.journal.refuse(owner, latest, code, "publication was refused before any write")
                return
            # An interrupted/revoked authority does not establish whether a prior
            # send committed. Keep the obligation and exact baseline for readback.
            self.executions.progress(
                owner,
                call.parent_request,
                call.parent_ordinal,
                {
                    "type": "progress",
                    "payload": {
                        "operation": "publication",
                        "call_index": call.call_index,
                        "phase": "reconciling" if latest.sent else "retrying",
                        "code": exc.code
                        if isinstance(exc, PublicationRefusal)
                        else "publication.parent_stopped",
                    },
                },
            )
            raise

    def _upload(
        self, owner: str, call: Call, request: effect_interfaces.Upload, client: PublicationClient
    ) -> CheckpointRef:
        artifact = request.artifact
        source = model_retention(self.workspace, owner, call.child_request, "artifact", artifact)
        serialized = documents.body(source)
        # Preparation survives a crash before retaining or before the first upload.
        if call.prepared:
            prepared = canonical_json.decode(call.prepared)
            if prepared.get("source") != serialized:
                raise WorkspaceRefusal("publication input changed its retained native subject")
        else:
            self.journal.check_active(owner, call)
            call = self.journal.freeze(owner, call, canonical_json.encode({"source": serialized}))
        registered, _ = machine_models.retain(
            self.workspace, owner, call.child_request, "artifact", artifact
        )
        if registered != source:
            raise WorkspaceRefusal("publication input changed its semantic custody")
        manifest = pb.Ref(
            digest=documents.raw(artifact.manifest.digest), length=artifact.manifest.length
        )
        with self.workspace.held_model(owner, source, manifest):
            store = fill.tensorfs_module().Store.open(str(self.workspace.store_root))
            objects = {artifact.manifest.digest: artifact.manifest.length}
            for row in store.walk_cozytensors(artifact.manifest.digest):
                key, length = str(row["id"]), int(row["length"])
                if key in objects and objects[key] != length:
                    raise WorkspaceRefusal("publication closure changed its object length")
                objects[key] = length
            # The native manifest fixes this inventory; no second file/catalog
            # journal or multi-megabyte inventory copy lives in execution_calls.
            checkpoint = CheckpointRef(
                request.destination,
                artifact.manifest.digest,
                artifact.manifest,
                call.child_request,
                "",
            )

            def push(key: str, length: int, grant: str) -> None:
                store.checkpoint_push(key, length, key == artifact.manifest.digest, grant)

            return upload_checkpoint(
                client,
                operation=call.child_request,
                checkpoint=checkpoint,
                objects=objects,
                before_write=lambda: self._before_write(owner, call),
                push=push,
            )

    def _cleanup(self, owner: str) -> None:
        with self.workspace.locked() as db:
            rows = db.execute(
                "SELECT c.prepared FROM execution_calls c JOIN holds h ON "
                "h.id=json_extract(CASE WHEN json_valid(CAST(c.prepared AS TEXT)) "
                "THEN CAST(c.prepared AS TEXT) ELSE '{}' END,'$.source.retention_id') "
                "WHERE c.owner=? AND h.owner=c.owner AND h.state<>'released' "
                "AND (c.result<>x'' OR c.safe_code<>'' OR (c.sent=0 AND EXISTS "
                "(SELECT 1 FROM executions e WHERE e.owner=c.owner "
                "AND e.request=c.parent_request AND e.desired='cancel'))) "
                "AND json_valid(CAST(c.prepared AS TEXT)) "
                "AND json_extract(CAST(c.intent AS TEXT),'$.module')=? LIMIT 32",
                (owner, effect_interfaces.MODULE),
            ).fetchall()
        for row in rows:
            value = canonical_json.decode(row["prepared"])["source"]
            source = pb.DerivedRetentionRequest(
                weights_transaction_id=value["weights_transaction_id"],
                tensorfs_receipt_digest=documents.raw(value["tensorfs_receipt_digest"]),
                retention_id=value["retention_id"],
            )
            # The same semantic model registry is used by ordinary child outputs.
            # Its native releases are exact and idempotent after a lost reply.
            self.workspace.retain(owner, source, release=True)
