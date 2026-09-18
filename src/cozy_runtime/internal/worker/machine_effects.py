"""Bounded publication work on Runtime's existing durable Python-call journal."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._assets import asset_dec_hook
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import effect_interfaces, fill
from cozy_runtime.internal.worker import machine_models
from cozy_runtime.internal.worker.machine_artifacts import model_retention
from cozy_runtime.internal.worker.machine_publication import (
    PublicationAuthority,
    PublicationClient,
    PublicationRefusal,
    apply_release,
    attach_assessment,
    prepare_release,
    upload_checkpoint,
)
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call, Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class Effects:
    def __init__(
        self, workspace: Workspace, executions: Executions, authority: PublicationAuthority | None
    ):
        self.workspace, self.executions, self.authority = workspace, executions, authority
        self.journal = Calls(workspace)
        self.lock = threading.Lock()
        self.thread_ids: set[int] = set()
        self.pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="publication", initializer=self._register_poller
        )
        self.active: dict[tuple[str, str], Future[None]] = {}
        self.retry_after: dict[tuple[str, str], float] = {}
        self.cleanup_after: dict[str, tuple[int, str]] = {}
        self.stopped = False

    def _register_poller(self) -> None:
        with self.lock:
            self.thread_ids.add(threading.get_native_id())

    @property
    def pollers(self) -> set[int]:
        with self.lock:
            return set(self.thread_ids)

    def close(self) -> None:
        with self.lock:
            self.stopped = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.thread_ids.clear()

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
        if self.authority is None:
            return
        with self.lock:
            key = owner, "cleanup"
            if not self.stopped and len(self.active) < 2 and key not in self.active:
                self.active[key] = self.pool.submit(self._cleanup, owner)

    def _before_write(self, owner: str, call: Call) -> None:
        with self.lock:
            if self.stopped:
                raise WorkspaceRefusal("publication worker is stopping")
        self.journal.before_write(owner, call)

    def _run(self, owner: str, original: Call) -> None:
        call = self.journal.get(owner, original.parent_request, original.call_index)
        try:
            if call.result or (call.safe_code and not call.sent):
                return
            with self.workspace.locked() as db:
                pure = db.execute(
                    "SELECT memoize FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, call.parent_request, call.parent_ordinal),
                ).fetchone()
            if pure is None or pure["memoize"]:
                raise PublicationRefusal("publication.impure_call_forbidden")
            grant = self.executions.publication_authorization(owner, call.parent_request)
            if not grant or self.authority is None:
                raise PublicationRefusal("publication.authority_absent", status=403)
            intent = canonical_json.decode(call.intent)
            if (
                intent["module"] != effect_interfaces.MODULE
                or intent["interface_digest"] != effect_interfaces.DIGEST
            ):
                raise PublicationRefusal("publication.interface_changed")
            client = self.authority.client(grant)
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
            elif operation == "attach_assessment":
                attachment = msgspec.convert(
                    intent["request"],
                    type=effect_interfaces.AttachAssessment,
                    strict=True,
                    dec_hook=asset_dec_hook,
                )
                result = self._assessment(owner, call, attachment, client)
            else:
                raise PublicationRefusal("publication.operation_unavailable")
            raw = canonical_json.encode(msgspec.to_builtins(result))
            latest = self.journal.get(owner, call.parent_request, call.call_index)
            if latest.sent:
                self.journal.complete(owner, call, raw)
            else:
                self.journal.observe_result(owner, call, raw)
        except Exception as exc:
            if not isinstance(
                exc, (PublicationRefusal, WorkspaceRefusal, msgspec.ValidationError, ValueError)
            ):
                # Native failures and credential IO must not silently retry in a
                # Future. Preserve only stable identifiers, never exception text
                # that could contain signed URLs, headers or private arguments.
                native = getattr(exc, "code", "")
                if native == "STORE_BUSY":
                    exc = PublicationRefusal("publication.storage_busy", status=503)
                else:
                    code = "publication.adapter_failed." + type(exc).__name__
                    if (
                        isinstance(native, str)
                        and 0 < len(native) <= 80
                        and all(c.isascii() and (c.isalnum() or c == "_") for c in native)
                    ):
                        code += "." + native
                    exc = PublicationRefusal(code[:128])
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
            if latest.sent and isinstance(exc, PublicationRefusal) and not transient:
                self.journal.unresolved(owner, latest, exc.code)
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
        after = self.cleanup_after.get(owner, (0, ""))
        with self.workspace.locked() as db:
            rows = db.execute(
                "SELECT c.*,e.desired FROM execution_calls c JOIN executions e "
                "ON e.owner=c.owner AND e.request=c.parent_request "
                "WHERE c.owner=? AND (c.created_ms,c.child_request)>(?,?) "
                "AND json_extract(CAST(c.intent AS TEXT),'$.module')=? "
                "ORDER BY c.created_ms,c.child_request LIMIT 32",
                (owner, *after, effect_interfaces.MODULE),
            ).fetchall()
        for row in rows:
            if not (
                row["result"]
                or (not row["sent"] and (row["safe_code"] or row["desired"] == "cancel"))
            ):
                continue
            with self.workspace.locked() as db:
                registry = db.execute(
                    "SELECT retention FROM execution_model_holds WHERE owner=? AND recipient=?",
                    (owner, row["child_request"]),
                ).fetchall()
                held = any(
                    db.execute(
                        "SELECT 1 FROM holds WHERE owner=? AND id=? AND state<>'released'",
                        (
                            owner,
                            pb.DerivedRetentionRequest.FromString(entry["retention"]).retention_id,
                        ),
                    ).fetchone()
                    for entry in registry
                )
            if held:
                machine_models.release(self.workspace, owner, row["child_request"])
        # This cursor only bounds background scanning; custody remains native and
        # release is idempotent. Cycling revisits calls that finish after this page.
        self.cleanup_after[owner] = (
            (rows[-1]["created_ms"], rows[-1]["child_request"]) if rows else (0, "")
        )

    def _assessment(
        self,
        owner: str,
        call: Call,
        request: effect_interfaces.AttachAssessment,
        client: PublicationClient,
    ) -> Any:
        from . import machine_assessment

        report, producer = machine_assessment.read_file(
            self.workspace,
            owner,
            call.parent_request,
            call.child_request,
            "report",
            request.report.ref,
        )
        workloads, workload_producer = machine_assessment.read_file(
            self.workspace,
            owner,
            call.parent_request,
            call.child_request,
            "workloads",
            request.workloads.ref,
        )
        if producer != workload_producer:
            raise WorkspaceRefusal("assessment report and workload have different producers")
        if not call.prepared:
            verdict = machine_assessment.verify(
                self.workspace,
                owner,
                producer,
                call.child_request,
                request.checkpoint,
                report,
                workloads,
            )
            call = self.journal.freeze(
                owner,
                call,
                canonical_json.encode(
                    {
                        "kind": "assessment",
                        "report": request.report.ref,
                        "workloads": request.workloads.ref,
                        "verdict": verdict,
                        "producer": producer,
                    }
                ),
            )
        else:
            frozen = canonical_json.decode(call.prepared)
            if (
                frozen.get("report") != request.report.ref
                or frozen.get("workloads") != request.workloads.ref
                or frozen.get("producer") != producer
            ):
                raise WorkspaceRefusal("assessment changed its frozen source identity")
            verdict = frozen["verdict"]
        return attach_assessment(
            client, request.checkpoint, report, verdict, lambda: self._before_write(owner, call)
        )
