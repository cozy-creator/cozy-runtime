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
    PUBLICATION_UPLOADS,
    PublicationAuthority,
    PublicationClient,
    PublicationRefusal,
    PublicationUncommitted,
    apply_release,
    attach_assessment,
    prepare_release,
    reconcile_checkpoint,
    upload_checkpoint,
)
from cozy_runtime.internal.worker.stage_progress import Progress, StageProgress
from cozy_runtime.internal.worker.workspace import (
    Workspace,
    WorkspaceBusy,
    WorkspaceInactive,
    WorkspaceRefusal,
)
from cozy_runtime.internal.worker.workspace_calls import Call, CallFenced, Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

# Native codes for IO that may succeed when repeated; others are refusals of this closure.
_TRANSIENT_NATIVE = frozenset({"STORE_BUSY", "TRANSFER_FAILED", "DEADLINE_EXCEEDED"})
# An uncommitted effect whose consecutive attempts land nothing new refuses; any landed
# object resets the count. Backoff doubles per stalled attempt up to its ceiling.
_STALLED_ATTEMPTS = 10
_MAX_BACKOFF_SECONDS = 60.0
# Hub's durable finalization job is progressing on its own: poll it at most 8 s apart.
_PENDING_POLL_STALLS = 3


def _transient(exc: Exception) -> bool:
    return isinstance(exc, PublicationRefusal) and (
        exc.status in {408, 425, 429} or exc.status >= 500
    )


def _refusal(exc: Exception) -> Exception:
    """Map native and credential failures to stable codes, never their text.

    Exception text can contain signed URLs, headers or private arguments.
    """
    if isinstance(exc, (PublicationRefusal, WorkspaceRefusal, msgspec.ValidationError, ValueError)):
        return exc
    native = getattr(exc, "code", "")
    if native == "STORE_BUSY":
        return PublicationRefusal("publication.storage_busy", status=503)
    code = "publication.adapter_failed." + type(exc).__name__
    if (
        isinstance(native, str)
        and 0 < len(native) <= 80
        and all(c.isascii() and (c.isalnum() or c == "_") for c in native)
    ):
        code += "." + native
    return PublicationRefusal(code[:128], status=503 if native in _TRANSIENT_NATIVE else 409)


class Effects:
    def __init__(
        self,
        workspace: Workspace,
        executions: Executions,
        authority: PublicationAuthority | None,
        *,
        allow_local: bool = False,
        progress: Progress | None = None,
    ):
        self.workspace, self.executions, self.authority = workspace, executions, authority
        self.allow_local, self.report = allow_local, progress
        self.journal = Calls(workspace)
        self.lock = threading.Lock()
        self.thread_ids: set[int] = set()
        self.pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="publication", initializer=self._register_poller
        )
        self.active: dict[tuple[str, str], Future[None]] = {}
        self.retry_after: dict[tuple[str, str], float] = {}
        self.landed: dict[tuple[str, str], int] = {}
        self.stalls: dict[tuple[str, str], int] = {}
        # The stage each effect is in; it outlives retries, so its timing covers them.
        self.stages: dict[tuple[str, str], StageProgress] = {}
        self.cleanup_after: dict[str, tuple[int, str]] = {}
        self.stopped = False

    def _stalls(self, key: tuple[str, str], landed: int) -> int:
        """Consecutive attempts, this one included, that landed nothing; caller holds the lock."""
        return 0 if self.landed.get(key, 0) > landed else self.stalls.get(key, 0) + 1

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
                        self.retry_after[key] = time.monotonic() + min(
                            2.0 ** self.stalls.get(key, 0), _MAX_BACKOFF_SECONDS
                        )
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

    def _stage(self, key: tuple[str, str], call: Call, name: str) -> StageProgress:
        """The effect's current stage, ending its previous stage as completed."""
        with self.lock:
            current = self.stages.get(key)
        if current is not None and current.stage == name:
            return current
        if current is not None:
            current.finish(True)

        def emit(frame: dict[str, Any]) -> None:
            if self.report is not None:
                self.report(call.parent_request, call.parent_ordinal, call.call_index, frame)

        stage = StageProgress(name, emit)
        with self.lock:
            self.stages[key] = stage
        return stage

    def _before_stage(self, owner: str, call: Call) -> None:
        with self.lock:
            if self.stopped:
                raise WorkspaceRefusal("publication worker is stopping")
        self.journal.check_active(owner, call)

    def _before_write(self, owner: str, call: Call) -> None:
        with self.lock:
            if self.stopped:
                raise WorkspaceRefusal("publication worker is stopping")
        self.journal.before_write(owner, call)

    def _run(self, owner: str, original: Call) -> None:
        key = owner, original.child_request
        with self.lock:
            landed = self.landed.get(key, 0)
        call = self.journal.get(owner, original.parent_request, original.call_index)
        operation = ""
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
            operation = intent["export"]
            if intent["module"] != effect_interfaces.MODULE:
                raise PublicationRefusal("publication.interface_changed")
            client = self.authority.client(grant)
            if operation == "publish_release":
                request = msgspec.convert(
                    intent["request"], type=effect_interfaces.Publish, strict=True
                )
                if not call.prepared:
                    frozen = prepare_release(client, **msgspec.structs.asdict(request))
                    call = self.journal.freeze(owner, call, frozen)
                self._stage(key, call, "Publishing release")
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
            self._settled(key, True)
        except Exception as caught:
            exc = _refusal(caught)
            pending = isinstance(exc, PublicationRefusal) and (
                exc.code == "publication.finalization_pending"
            )
            with self.lock:
                stopping = self.stopped
                stalls = self._stalls(key, landed)
                self.stalls[key] = min(stalls, _PENDING_POLL_STALLS) if pending else stalls
            if stopping or isinstance(caught, (CallFenced, WorkspaceInactive, WorkspaceBusy)):
                # This attempt may not act now. The parent's next generation, its
                # cancellation or the next worker decides; none of those is a refusal.
                raise
            latest = self.journal.get(owner, call.parent_request, call.call_index)
            transient = _transient(exc)
            if not latest.sent and (not transient or stalls >= _STALLED_ATTEMPTS):
                # Nothing committed, so no remote readback is owed: settle and release.
                code = (
                    exc.code
                    if isinstance(exc, PublicationRefusal)
                    else "publication.intent_refused"
                )
                self.journal.refuse(
                    owner,
                    latest,
                    code,
                    "publication stopped making progress before its commit"
                    if transient
                    else "publication was refused before its commit",
                )
                self._settled(key, False)
                return
            if latest.sent and isinstance(exc, PublicationUncommitted):
                # Hub durably reports the commit did not land: settle like a refusal.
                self.journal.uncommitted(owner, latest, exc.code, "publication commit did not land")
                self._settled(key, False)
                return
            if (
                latest.sent
                and isinstance(exc, PublicationRefusal)
                and not transient
                and (
                    operation != "upload_checkpoint" or exc.code == "publication.checkpoint_changed"
                )
            ):
                # A sent upload stays pending until Hub's finalization status decides it;
                # only a durable answer contradicting the frozen closure is surfaced.
                self.journal.unresolved(owner, latest, exc.code)
            # An interrupted/revoked authority does not establish whether a prior
            # send committed. Keep the obligation and exact baseline for readback.
            phase = "reconciling" if latest.sent else "retrying"
            with self.lock:
                stage = self.stages.get(key)
            self.executions.progress(
                owner,
                call.parent_request,
                call.parent_ordinal,
                {
                    "type": "progress",
                    "payload": {
                        **({"stage": f"{stage.stage} / {phase}"} if stage is not None else {}),
                        "operation": "publication",
                        "call_index": call.call_index,
                        "phase": phase,
                        "code": exc.code
                        if isinstance(exc, PublicationRefusal)
                        else "publication.parent_stopped",
                    },
                },
            )
            raise

    def _settled(self, key: tuple[str, str], completed: bool) -> None:
        with self.lock:
            self.landed.pop(key, None)
            self.stalls.pop(key, None)
            stage = self.stages.pop(key, None)
        if stage is not None:
            stage.finish(completed)

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
        manifest = pb.Ref(
            digest=documents.raw(artifact.manifest.digest), length=artifact.manifest.length
        )
        # The first attempt retains the closure. Later attempts and reconcile polls only
        # check that hold: retaining again re-verifies every closure object.
        if not self.workspace.model_held(owner, source, manifest):
            registered, _ = machine_models.retain(
                self.workspace, owner, call.child_request, "artifact", artifact
            )
            if registered != source:
                raise WorkspaceRefusal("publication input changed its semantic custody")
        slot = owner, call.child_request
        # The retention above keeps the closure; the native manifest fixes this inventory,
        # so no second file/catalog journal or inventory copy lives in execution_calls.
        store = fill.store(self.workspace.store_root)
        objects = {artifact.manifest.digest: artifact.manifest.length}
        for row in store.walk_cozytensors(artifact.manifest.digest):
            key, length = str(row["id"]), int(row["length"])
            if key in objects and objects[key] != length:
                raise WorkspaceRefusal("publication closure changed its object length")
            objects[key] = length
        checkpoint = CheckpointRef(
            request.destination, artifact.manifest.digest, artifact.manifest, call.child_request, ""
        )
        total = sum(objects.values())
        if call.sent:
            # A sent finalize is decided by Hub's durable job, without re-staging bytes.
            self._stage(slot, call, "Publishing checkpoint")
            settled = reconcile_checkpoint(
                client, operation=call.child_request, checkpoint=checkpoint, objects=objects
            )
            if settled is not None:
                return settled
            # The finalize never reached Hub. A live parent stages and sends it again;
            # a canceled one never will, so nothing can land.
            if self.journal.canceled(owner, call):
                raise PublicationUncommitted("publication.finalize_absent")
        # The retention keeps the closure from collection. Each push leases only its own
        # object and re-checks that hold, so an attempt never opens objects it does not send.
        digest = artifact.manifest.digest

        def push(key: str, length: int, grant: str) -> None:
            lease = None if key == digest else store.acquire(digest, [(key, length)])
            try:
                if not self.workspace.model_held(owner, source, manifest):
                    raise WorkspaceRefusal("publication hold was released during acquisition")
                store.checkpoint_push(
                    key, length, key == digest, grant, allow_local=self.allow_local
                )
            finally:
                if lease is not None:
                    lease.release()

        def landed(count: int, held: int) -> None:
            with self.lock:
                self.landed[slot] = max(self.landed.get(slot, 0), count)
            if held < total:
                self._stage(slot, call, "Uploading checkpoint").update(held, total)
            else:
                # Every closure byte is held; Hub's finalization publishes it.
                self._stage(slot, call, "Publishing checkpoint")

        with self.lock:
            before = self.landed.get(slot, 0)
        try:
            return upload_checkpoint(
                client,
                operation=call.child_request,
                checkpoint=checkpoint,
                objects=objects,
                before_stage=lambda: self._before_stage(owner, call),
                before_write=lambda: self._before_write(owner, call),
                push=push,
                landed=landed,
                window=fill.tensorfs_module().transfer_streams(PUBLICATION_UPLOADS),
            )
        except Exception as caught:
            exc = _refusal(caught)
            with self.lock:
                stalled = self._stalls(slot, before) >= _STALLED_ATTEMPTS
            if (
                call.sent
                and isinstance(exc, PublicationRefusal)
                and not isinstance(exc, PublicationUncommitted)
                and (stalled or not _transient(exc))
                and reconcile_checkpoint(
                    client, operation=call.child_request, checkpoint=checkpoint, objects=objects
                )
                is None
            ):
                # Hub still holds no finalize from this call, and staging it again was refused
                # or stopped landing objects: it cannot land, so its hold is released.
                raise PublicationUncommitted(exc.code) from caught
            raise

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
