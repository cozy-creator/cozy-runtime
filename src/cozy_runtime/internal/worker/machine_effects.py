"""Bounded publication work on Runtime's existing durable Python-call journal."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._assets import asset_dec_hook
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import effect_interfaces, fill
from cozy_runtime.internal.worker import machine_models
from cozy_runtime.internal.worker.machine_artifacts import model_retention
from cozy_runtime.internal.worker.machine_publication import (
    FINALIZATION_ABSENT,
    PublicationAuthority,
    PublicationClient,
    PublicationRefusal,
    PublicationUncommitted,
    apply_release,
    attach_assessment,
    prepare_release,
    reconcile_checkpoint,
    settle_checkpoint,
    upload_checkpoint,
)
from cozy_runtime.internal.worker.stage_progress import Progress, StageProgress
from cozy_runtime.internal.worker.supervisor import Unit
from cozy_runtime.internal.worker.workspace import (
    Workspace,
    WorkspaceBusy,
    WorkspaceRefusal,
)
from cozy_runtime.internal.worker.workspace_calls import Call, CallFenced, Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

# Native codes for IO that may succeed when repeated; others are refusals of this closure.
_TRANSIENT_NATIVE = frozenset({"STORE_BUSY", "TRANSFER_FAILED", "DEADLINE_EXCEEDED"})
# An effect settles after this many consecutive tries land nothing new (any landed object
# resets the count). The doubling backoff between tries only paces a remote.
_STALLED_ATTEMPTS = 10
_MAX_BACKOFF_SECONDS = 60.0
# Hub's durable finalization job is progressing on its own: poll it at most 8 s apart.
_PENDING_POLL_STALLS = 3
# The machine's own publication authority no longer answers: only its owner can read Hub.
AUTHORITY_LOST = frozenset(
    {
        "publication.authority_absent",
        "publication.authority_refused",
        "publication.authority_expired",
    }
)


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


def _publication(call: Call) -> dict[str, Any] | None:
    """The Hub publication a checkpoint upload call names, or None for other effects."""
    intent = canonical_json.decode(call.intent)
    if (
        intent.get("module") != effect_interfaces.MODULE
        or intent.get("export") != "upload_checkpoint"
    ):
        return None
    return {
        "call_index": call.call_index,
        "publication": call.child_request,
        "destination": intent["request"]["destination"],
    }


def _document(body: bytes) -> dict[str, Any] | None:
    try:
        value = canonical_json.decode(body)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


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
        #: two effects move bytes at a time; the rest wait their turn on their own units
        self.slots = threading.Semaphore(2)
        self.landed: dict[tuple[str, str], int] = {}
        self.stalls: dict[tuple[str, str], int] = {}
        # The stage each effect is in; it outlives retries, so its timing covers them.
        self.stages: dict[tuple[str, str], StageProgress] = {}
        # Sent publications whose awaiting-owner notice this process already recorded.
        self.noticed: set[tuple[str, str]] = set()
        self.stopped = False
        #: told (owner, call) when an effect settles: its parent and root act on it
        self.settled: Callable[[str, Call], None] = lambda owner, call: None

    def _stalls(self, key: tuple[str, str], landed: int) -> int:
        """Consecutive attempts, this one included, that landed nothing; caller holds the lock."""
        return 0 if self.landed.get(key, 0) > landed else self.stalls.get(key, 0) + 1

    def close(self) -> None:
        with self.lock:
            self.stopped = True

    def run(self, unit: Unit, owner: str, call: Call) -> None:
        """One effect call, on its own unit, until it settles: committed, refused, or left
        for its owner to reconcile. A try that fails is tried again only while tries keep
        landing objects (measured progress), at the backoff its stalled tries earned."""
        key = owner, call.child_request
        while True:
            latest = self.journal.get(owner, call.parent_request, call.call_index)
            with self.lock:
                awaiting_owner = key in self.noticed
            if latest.result or (latest.safe_code and not latest.sent) or awaiting_owner:
                break
            with self.slots:
                try:
                    self._run(owner, latest)
                    continue
                except CallFenced:
                    with self.lock:
                        stopping = self.stopped
                    if stopping or not latest.sent:
                        break  # no one runs it now; a sent one is still read back
                except Exception:
                    pass  # recorded by `_run`; tried again below
            with self.lock:
                stalls = self.stalls.get(key, 0)
            unit.wait(time.monotonic() + min(2.0**stalls, _MAX_BACKOFF_SECONDS))
        self._release(owner, self.journal.get(owner, call.parent_request, call.call_index))
        self.settled(owner, call)

    def _release(self, owner: str, call: Call) -> None:
        """A settled effect's Model holds go: it will read its artifact no more. So do an
        unsent one's whose parent stopped: nothing will send it."""
        if call.result or (
            not call.sent and (call.safe_code or self.journal.canceled(owner, call))
        ):
            with self.workspace.locked() as db:
                held = db.execute(
                    "SELECT 1 FROM execution_model_holds WHERE owner=? AND recipient=? LIMIT 1",
                    (owner, call.child_request),
                ).fetchone()
            if held is not None:
                machine_models.release(self.workspace, owner, call.child_request)

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

    def source_client(self, owner: str, request: str, ordinal: int) -> PublicationClient:
        """An upload source operation publishes under the same gates as an effect call."""
        with self.workspace.locked() as db:
            pure = db.execute(
                "SELECT memoize FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, request, ordinal),
            ).fetchone()
        if pure is None or pure["memoize"]:
            raise PublicationRefusal("publication.impure_call_forbidden")
        grant = self.executions.publication_authorization(owner, request)
        if not grant or self.authority is None:
            raise PublicationRefusal("publication.authority_absent", status=403)
        return self.authority.client(grant)

    def _before_stage(self, owner: str, call: Call) -> None:
        with self.lock:
            if self.stopped:
                raise CallFenced("publication worker is stopping")
        self.journal.check_active(owner, call)

    def _before_write(self, owner: str, call: Call) -> None:
        with self.lock:
            if self.stopped:
                raise CallFenced("publication worker is stopping")
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
            intent = canonical_json.decode(call.intent)
            operation = intent["export"]
            # A memoized root may upload only its own result, to the destination its
            # owner granted; its memo identity never includes that upload.
            granted = operation == "upload_checkpoint" and intent["request"].get(
                "destination"
            ) == self.executions.destination(owner, call.parent_request)
            if pure is None or (pure["memoize"] and not granted):
                raise PublicationRefusal("publication.impure_call_forbidden")
            grant = self.executions.publication_authorization(owner, call.parent_request)
            if not grant or self.authority is None:
                raise PublicationRefusal("publication.authority_absent", status=403)
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
                # The owner settles a destination from these, at least once per output.
                self.executions.record(
                    owner,
                    call.parent_request,
                    "checkpoint",
                    {
                        "destination": result.destination,
                        "checkpoint": result.checkpoint,
                        "output_slot": request_upload.artifact.output_slot,
                        "observation": result.observation,
                    },
                )
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
                self._notice_settled(owner, latest, True)
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
            if stopping or isinstance(caught, (CallFenced, WorkspaceBusy)):
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
                self._notice_settled(owner, latest, False)
                self._settled(key, False)
                return
            target = _publication(latest)
            if (
                latest.sent
                and target is not None
                and isinstance(exc, PublicationRefusal)
                and exc.code in AUTHORITY_LOST
            ):
                # Only the owner's own Hub credential can now read whether it committed.
                self._notice_unresolved(owner, latest, target, exc.code)
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
                        **(target or {}),
                        "phase": phase,
                        "code": exc.code
                        if isinstance(exc, PublicationRefusal)
                        else "publication.parent_stopped",
                    },
                },
            )
            raise

    def _notice_unresolved(self, owner: str, call: Call, target: dict[str, Any], code: str) -> None:
        key = owner, call.child_request
        with self.lock:
            if key in self.noticed:
                return
            self.noticed.add(key)
        self.executions.notice(
            owner, call.parent_request, "publication_unresolved", {**target, "code": code}
        )

    def _notice_settled(self, owner: str, call: Call, committed: bool) -> None:
        target = _publication(call)
        if target is None:
            return
        with self.lock:
            self.noticed.discard((owner, call.child_request))
        self.executions.notice(
            owner, call.parent_request, "publication_settled", {**target, "committed": committed}
        )

    def reconcile(
        self, owner: str, request: str, call_index: int, status: int, body: bytes
    ) -> None:
        """Settle one sent publication from its owner's finalization read.

        Refused while this machine's own authority still reads Hub: then it reconciles the
        call itself. The owner's document must name this call's operation, checkpoint and
        objects; a queued or running finalization is refused as pending.
        """
        call = self.journal.get(owner, request, call_index)
        target = _publication(call)
        if target is None:
            raise WorkspaceRefusal("call is not a checkpoint publication")
        if call.result or not call.sent:
            return
        upload = msgspec.convert(
            canonical_json.decode(call.intent)["request"],
            type=effect_interfaces.Upload,
            strict=True,
        )
        checkpoint, objects = self._closure(call, upload)
        try:
            grant = self.executions.publication_authorization(owner, call.parent_request)
            if not grant or self.authority is None:
                raise PublicationRefusal("publication.authority_absent", status=403)
            reconcile_checkpoint(
                self.authority.client(grant),
                operation=call.child_request,
                checkpoint=checkpoint,
                objects=objects,
            )
        except Exception as caught:
            exc = _refusal(caught)
            if not isinstance(exc, PublicationRefusal) or exc.code not in AUTHORITY_LOST:
                raise WorkspaceRefusal(
                    "machine publication authority still reads Hub; it reconciles this call"
                ) from caught
        else:
            raise WorkspaceRefusal(
                "machine publication authority still reads Hub; it reconciles this call"
            )
        document = _document(body)
        committed, raw, code, detail = False, b"", "", ""
        if 200 <= status < 300 and document is not None:
            try:
                settled = settle_checkpoint(
                    document, operation=call.child_request, checkpoint=checkpoint, objects=objects
                )
            except PublicationUncommitted as exc:
                code, detail = exc.code, "the owner's Hub reports the commit did not land"
            except PublicationRefusal as exc:
                raise WorkspaceRefusal(
                    f"owner finalization read does not settle: {exc.code}"
                ) from exc
            else:
                committed = True
                raw = canonical_json.encode(msgspec.to_builtins(settled))
        elif (
            status == 404
            and document is not None
            and isinstance(document.get("error"), dict)
            and document["error"].get("code") in FINALIZATION_ABSENT
        ):
            code = "publication.finalize_absent"
            detail = "the owner's Hub holds no finalization for this publication"
        else:
            raise WorkspaceRefusal("owner finalization read is not a settlement")
        try:
            if committed:
                self.journal.complete(owner, call, raw)
            else:
                self.journal.uncommitted(owner, call, code, detail)
        except WorkspaceRefusal:
            latest = self.journal.get(owner, request, call_index)
            if latest.result or not latest.sent:
                return  # settled concurrently
            raise
        self._notice_settled(owner, call, committed)
        self._settled((owner, call.child_request), committed)

    def _closure(
        self, call: Call, upload: effect_interfaces.Upload
    ) -> tuple[CheckpointRef, dict[str, int]]:
        """The frozen checkpoint and its exact object inventory from the native manifest."""
        artifact = upload.artifact
        store = fill.store(self.workspace.store_root)
        objects = {artifact.manifest.digest: artifact.manifest.length}
        for row in store.walk_cozytensors(artifact.manifest.digest):
            key, length = str(row["id"]), int(row["length"])
            if key in objects and objects[key] != length:
                raise WorkspaceRefusal("publication closure changed its object length")
            objects[key] = length
        checkpoint = CheckpointRef(
            upload.destination, artifact.manifest.digest, artifact.manifest, call.child_request, ""
        )
        return checkpoint, objects

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
        checkpoint, objects = self._closure(call, request)
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
                streams=fill.tensorfs_module().transfer_streams(len(objects)),
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
