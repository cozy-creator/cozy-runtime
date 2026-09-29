"""Asynchronous execution of the three accepted native source operations.

The existing control stream carries acceptance/results. Native work runs in a
separate killable process so heartbeats and parent cancellation remain responsive.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import signal
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import get_args, get_type_hints

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef, SourceArtifact
from cozy_runtime.author._errors import AuthorError
from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
from cozy_runtime.internal import fill, source_interfaces, storage_admission
from cozy_runtime.internal.source_interfaces import (
    Civitai,
    CommitFile,
    HuggingFace,
    SourceFiles,
    UploadCivitai,
    UploadHuggingFace,
)
from cozy_runtime.internal.worker import (
    commit_files,
    source_steps,
    source_upload,
    source_views,
    upload_plan,
    workspace_byte_outputs,
    workspace_memo,
    workspace_partial,
    workspace_sources,
)
from cozy_runtime.internal.worker.machine_owner_memo import OwnerMemo
from cozy_runtime.internal.worker.machine_publication import PublicationClient, PublicationRefusal
from cozy_runtime.internal.worker.source_steps import (
    Answer,
    Launch,
    Pin,
    Produced,
    Refused,
    Resolved,
    Sample,
    Selection,
    SourceRefusal,
    SourceStep,
    UploadStep,
)
from cozy_runtime.internal.worker.stage_progress import Progress, StageProgress
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_sources import NativeCall
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

_LOG = logging.getLogger(__name__)

SOURCE_WORKER = "cozy_runtime.internal.worker.source_worker"
UPLOAD_CHILD = "cozy_runtime.internal.upload_child"
STAGES = {"download": "Downloading source", "convert": "Converting to cozytensors"}
# The part of a child's stderr tail a child-call failure detail carries (authors see at
# most 1 KiB of detail).
LOG_TAIL_CHARS = 4000
DETAIL_TAIL_CHARS = 600
_FRAME = re.compile(r'File "(?:[^"]*/)?([^"/]+)", line (\d+),? in ')
_URL_QUERY = re.compile(r"(\w+://[^\s?#'\"]+)[?#][^\s'\"]*")
_SECRET = re.compile(
    r"(?i)\b(bearer|token|authorization|signature|credential)([\s:=\"']+)[^\s'\",;]+"
)


def diagnosis(returncode: int, stderr: str, credential: str) -> str:
    """Why a native child ended, with its redacted stderr tail; never argv or env."""
    for secret in {credential, *credential.split()}:
        if len(secret) >= 8:
            stderr = stderr.replace(secret, "[redacted]")
    stderr = _SECRET.sub(r"\1\2[redacted]", _URL_QUERY.sub(r"\1?[redacted]", stderr))
    stderr = _FRAME.sub(r"\1:\2 in ", stderr)  # stack frames without host paths
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        cause = f"source worker was killed by {name}"
        if name == "SIGKILL":
            cause += " (the kernel's out-of-memory killer sends SIGKILL)"
    elif returncode:
        cause = f"source worker exited with status {returncode}"
    else:
        cause = "source worker ended without a bounded result"
    lines = " | ".join(line.strip() for line in stderr.splitlines() if line.strip())
    return f"{cause}; stderr: {_clip(lines, LOG_TAIL_CHARS)}" if lines else cause


def _clip(text: str, limit: int) -> str:
    """Keep the headline (a fatal error or panic names itself first) and the newest end."""
    if len(text) <= limit:
        return text
    head = text.partition(" | ")[0][:160]
    return f"{head} | … {text[len(head) - limit + 5 :]}"


def brief(detail: str) -> str:
    cause, _, tail = detail.partition("; stderr: ")
    return f"{cause}; stderr: {_clip(tail, DETAIL_TAIL_CHARS)}" if tail else cause


def _answer[A: Answer](output: bytes, stderr: str, credential: str, kind: type[A]) -> A:
    """A child's answer; a refusal crosses only as a known native or reviewed code."""
    answer = source_steps.answer(output, kind)
    if isinstance(answer, Refused):
        code = answer.code
        if code not in fill.tensorfs_module().errors.CODES and code not in source_steps.CODES:
            code = "native_source_failed"
        raise NativeSourceFailure(code, diagnosis(0, stderr, credential) if stderr.strip() else "")
    return answer


def _after[**P](first: Callable[[], None], then: Callable[P, None]) -> Callable[P, None]:
    def both(*args: P.args, **kwargs: P.kwargs) -> None:
        first()
        then(*args, **kwargs)

    return both


def _selection(value: Selection) -> pb.NativeSourceSelection:
    return pb.NativeSourceSelection(
        canonical=value.canonical,
        selection_digest=bytes.fromhex(value.selection_sha256),
        content_manifest=pb.Ref(
            digest=documents.raw(value.content_manifest_digest),
            length=value.content_manifest_length,
        ),
        members=[
            pb.NativeSourceMember(
                member=member, object=pb.Ref(digest=documents.raw(digest), length=length), url=url
            )
            for member, digest, length, url in value.members
        ],
        allowed_hosts=value.allowed_hosts,
        credential_hosts=value.credential_hosts,
    )


def _native_selection(value: pb.NativeSourceSelection) -> Pin:
    return Pin(
        [
            (row.member, documents.spell(row.object.digest), row.object.length, row.url)
            for row in value.members
        ],
        list(value.allowed_hosts),
        list(value.credential_hosts),
    )


def _resolution(request: object, access: source_steps.Access) -> source_steps.ResolveSource:
    if isinstance(request, HuggingFace):
        return source_steps.ResolveSource(
            uri=f"hf://{request.repository}@{request.revision}",
            carriers=list(request.carriers),
            profiles=list(request.profiles),
            files=list(request.files),
            access=access,
        )
    if isinstance(request, Civitai):
        return source_steps.ResolveSource(
            uri=f"civitai://{request.version}",
            carriers=[request.file] if request.file else [],
            profiles=[],
            files=[],
            access=access,
        )
    raise WorkspaceRefusal("conversion does not resolve a foreign source")


def _profiles(request: source_interfaces.Convert) -> tuple[str, ...]:
    """The reviewed profiles one conversion composes, in their declared order."""
    if not request.profiles or len(set(request.profiles)) != len(request.profiles):
        raise SourceRefusal("model_source_profiles_invalid", "name distinct profiles")
    return request.profiles


# The CozyTensors format this TensorFS writes, as its typed header declares it.
OUTPUT_FORMAT: str = get_args(get_type_hints(fill.tensorfs_module().Header)["format"])[0]


def computation_key(
    operation: str, effective: Mapping[str, object], output_format: str = OUTPUT_FORMAT
) -> bytes:
    """The native memo key: the operation, its declared output version, TensorFS's output
    format and its real inputs.

    A Runtime or TensorFS release that keeps an operation's output meaning and format keeps
    the key. A TensorFS writing another format misses; a changed meaning bumps
    `source_interfaces.OPERATION_VERSIONS` by hand.
    """
    return hashlib.sha256(
        canonical_json.encode(
            {
                "operation": operation,
                "version": source_interfaces.OPERATION_VERSIONS[operation],
                "format": output_format,
                "effective": effective,
            }
        )
    ).digest()


class NativeSourceFailure(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code, self.detail = code, detail
        super().__init__(code)


class SourceCalls:
    def __init__(
        self,
        workspace: Workspace,
        owner: Callable[[], str],
        observe: Callable[[pb.NativeSourceStatus], None],
        current: Callable[[pb.NativeSourceCommand], bool],
        *,
        endpoints: dict[str, str] | None = None,
        native_registry: bytes | None = None,
        spool_root: Path | None = None,
        progress: Progress | None = None,
        publication: Callable[[str, str, int], PublicationClient] | None = None,
        memo: OwnerMemo | None = None,
    ):
        self.workspace, self.owner, self.observe, self.current = workspace, owner, observe, current
        self.memo = memo
        self.progress = progress
        # The latest failure detail per native call, for the machine child-call result.
        self.details: dict[str, str] = {}
        # The stages each native call ran, for the `call` record its root keeps.
        self.ran: dict[str, list[StageProgress]] = {}
        self.endpoints = endpoints or {}
        self.native_registry = native_registry
        self.spool_root = spool_root
        self.publication = publication
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen[bytes] | None] = {}
        self.tasks: dict[str, threading.Event] = {}
        #: told the service id of every native task that ends, whatever its end
        self.finished: Callable[[str], None] = lambda service_id: None
        self.parents: dict[str, str] = {}
        self.commands: dict[str, pb.NativeSourceCommand] = {}
        self.process_owners: dict[str, str] = {}
        self.canceled: set[str] = set()

    def handle(self, command: pb.NativeSourceCommand) -> None:
        owner = self.owner()
        if command.phase == pb.NATIVE_SOURCE_PHASE_CANCEL:
            workspace_sources.stopped(self.workspace, owner, command)
            with self.lock:
                self.canceled.add(command.service_id)
                process = self.processes.get(command.service_id)
                if process is not None:
                    process.kill()
            self._wake()
            return
        if (
            command.operation
            in (pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES, pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE)
            and command.phase != pb.NATIVE_SOURCE_PHASE_EXECUTE
        ):
            raise WorkspaceRefusal("source view has no foreign resolution phase")
        row = workspace_sources.accepted(self.workspace, owner, command)
        if row.state not in ("stopped", "failed", "releasing", "released"):
            with self.lock:
                self.canceled.discard(command.service_id)
        if command.phase not in (pb.NATIVE_SOURCE_PHASE_RESOLVE, pb.NATIVE_SOURCE_PHASE_EXECUTE):
            raise WorkspaceRefusal("unknown native source execution phase")
        if row.state == "stopped":
            self._status(command, pb.NATIVE_SOURCE_STATE_CANCELED)
            return
        if command.phase == pb.NATIVE_SOURCE_PHASE_EXECUTE and command.operation in (
            pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
            pb.NATIVE_SOURCE_OPERATION_CIVITAI,
        ):
            workspace_sources.record_pin(self.workspace, owner, command, command.selection)
        copied = pb.NativeSourceCommand()
        copied.CopyFrom(command)
        with self.lock:
            self.commands[command.service_id] = copied
            self.process_owners[command.service_id] = owner
            if command.service_id in self.processes:
                return
            self.processes[command.service_id] = None
            self.details.pop(command.service_id, None)
            self.parents[command.service_id] = command.parent_call.parent_request_id
            done = threading.Event()
            self.tasks[command.service_id] = done
        threading.Thread(
            target=self._run, args=(owner, copied, row, done), daemon=True, name="native-source"
        ).start()

    def cancel_parent(
        self,
        request_id: str,
        *,
        ordinal: int | None = None,
        spec: bytes | None = None,
        through: int | None = None,
    ) -> None:
        with self.lock:
            for service, parent in self.parents.items():
                command = self.commands.get(service)
                if command is None:
                    continue
                call = command.parent_call
                if ordinal is not None and (
                    call.parent_attempt_ordinal != ordinal
                    or call.parent_invocation_spec_digest != spec
                ):
                    continue
                if through is not None and call.parent_attempt_ordinal > through:
                    continue
                if parent == request_id:
                    self.canceled.add(service)
                    process = self.processes.get(service)
                    if process is not None:
                        process.kill()
        self._wake()

    def _wake(self) -> None:
        """A call parked on its owner's memo answer sees its cancellation."""
        if self.memo is not None:
            self.memo.wake()

    def _latest(self, command: pb.NativeSourceCommand) -> pb.NativeSourceCommand:
        with self.lock:
            return self.commands.get(command.service_id, command)

    def failure_detail(self, service_id: str) -> str:
        with self.lock:
            return self.details.get(service_id, "")

    def stages(self, service_id: str) -> list[StageProgress]:
        """The stages a native call ran, handed over once."""
        with self.lock:
            return self.ran.pop(service_id, [])

    def _stage(self, command: pb.NativeSourceCommand, name: str) -> StageProgress:
        stage = StageProgress(name, lambda frame: self._emit(command, frame))
        with self.lock:
            self.ran.setdefault(command.service_id, []).append(stage)
            while len(self.ran) > 256:
                self.ran.pop(next(iter(self.ran)))
        return stage

    def _emit(self, command: pb.NativeSourceCommand, frame: dict[str, object]) -> None:
        if self.progress is not None:
            call = self._latest(command).parent_call
            self.progress(
                call.parent_request_id, call.parent_attempt_ordinal, call.call_index, frame
            )

    def fence_owner(self, owner: str) -> None:
        with self.lock:
            for service, expected in self.process_owners.items():
                if expected != owner:
                    self.canceled.add(service)
                    process = self.processes.get(service)
                    if process is not None and process.poll() is None:
                        process.kill()
        self._wake()

    def _status(
        self,
        command: pb.NativeSourceCommand,
        state: pb.NativeSourceState,
        *,
        result: bytes = b"",
        receipt: bytes = b"",
        computation: bytes = b"",
        selection: pb.NativeSourceSelection | None = None,
        code: str = "",
        detail: str = "",
        memo_hit: bool = False,
        byte_output: pb.NativeByteTreeRef | None = None,
        byte_output_attempt: int = 0,
        byte_output_spec: bytes = b"",
    ) -> None:
        command = self._latest(command)
        if not self.current(command):
            return
        call = command.parent_call
        status = pb.NativeSourceStatus(
            parent_request_id=call.parent_request_id,
            parent_attempt_ordinal=call.parent_attempt_ordinal,
            parent_invocation_spec_digest=call.parent_invocation_spec_digest,
            call_index=call.call_index,
            intent_digest=call.intent_digest,
            service_id=command.service_id,
            state=state,
            result_canonical_bytes=result,
            native_receipt_canonical_bytes=receipt,
            computation_digest=computation,
            safe_code=code,
            safe_detail=detail,
            memo_hit=memo_hit,
            byte_output_attempt_ordinal=byte_output_attempt,
            byte_output_invocation_spec_digest=byte_output_spec,
        )
        if selection is not None:
            status.selection.CopyFrom(selection)
        if byte_output is not None:
            status.byte_output.CopyFrom(byte_output)
        self.observe(status)

    def active(self) -> set[str]:
        """Native calls running now; pressure reclamation never touches their bytes."""
        with self.lock:
            return set(self.tasks)

    def _live_uploads(self) -> set[str]:
        with self.workspace.locked() as db:
            return workspace_sources.live_uploads(db)

    def _stopped(self, service_id: str) -> bool:
        with self.lock:
            return service_id in self.canceled

    def _started(self, service_id: str, process: subprocess.Popen[bytes]) -> None:
        with self.lock:
            self.processes[service_id] = process
            if service_id in self.canceled:
                process.kill()

    def _child[A: Answer](
        self, command: pb.NativeSourceCommand, launch: Launch[UploadStep], answer: type[A]
    ) -> A:
        """One killable upload step. A cancellation kills it and surfaces as ``Canceled``."""
        status, output, stderr = source_steps.run(
            UPLOAD_CHILD, launch, started=partial(self._started, command.service_id)
        )
        if self._stopped(command.service_id):
            raise source_upload.Canceled()
        if status or not output:
            raise NativeSourceFailure(
                "native_source_failed", diagnosis(status, stderr, command.credential)
            )
        return _answer(output, stderr, command.credential, answer)

    def _converters(self, profiles: Sequence[str]) -> list[list[str]]:
        """The versioned converters the profiles bind, each with its pinned reference."""
        rows = fill.tensorfs_module().source_profile_converters(
            fill.store(self.workspace.store_root), list(profiles), registry=self.native_registry
        )
        return [[name, reference or ""] for name, reference in rows]

    def _upload(
        self,
        owner: str,
        command: pb.NativeSourceCommand,
        row: NativeCall,
        request: upload_plan.Request,
        phase: str,
        finish: Callable[[], None],
    ) -> None:
        """Resolve, or stream the pinned source through conversion into its publication."""
        try:
            self._upload_steps(owner, command, row, request, phase, finish)
        except (AuthorError, PublicationRefusal) as exc:
            # A reviewed or Hub refusal keeps its stable code; its text stays private.
            raise NativeSourceFailure(exc.code) from exc

    def _upload_steps(
        self,
        owner: str,
        command: pb.NativeSourceCommand,
        row: NativeCall,
        request: upload_plan.Request,
        phase: str,
        finish: Callable[[], None],
    ) -> None:
        if self.publication is None:
            raise NativeSourceFailure("publication_authority_absent")
        call = command.parent_call
        terminal = _after(finish, self._status)
        stage: StageProgress | None = None

        def report(name: str, position: int, total: int) -> None:
            nonlocal stage
            if stage is None or stage.stage != name:
                if stage is not None:
                    stage.finish(True)
                stage = self._stage(command, name)
            stage.update(position, total)

        def child[A: Answer](launch: Launch[UploadStep], answer: type[A]) -> A:
            return self._child(command, launch, answer)

        upload = source_upload.Upload(
            store_root=self.workspace.store_root,
            native_owner=row.native_owner,
            service_id=command.service_id,
            request=request,
            access=source_steps.access(self.endpoints, command.credential),
            registry=self.native_registry,
            client=self.publication(owner, call.parent_request_id, call.parent_attempt_ordinal),
            hooks=source_upload.Hooks(
                child=child,
                canceled=lambda: self._stopped(command.service_id),
                active=self.active,
                live=self._live_uploads,
                evict_retired=lambda target: workspace_partial.reclaim(self.workspace, target),
                consume=partial(workspace_sources.consume_partial, self.workspace, owner),
                progress=report,
            ),
            directory=self.workspace.directory,
        )
        try:
            if phase == "resolve":
                resolved = _selection(upload.resolve())
                workspace_sources.record_pin(self.workspace, owner, command, resolved)
                terminal(command, pb.NATIVE_SOURCE_STATE_RESOLVED, selection=resolved)
                return
            content = workspace_sources.selection_content(command.selection)
            pin = _native_selection(command.selection)
            if not upload.slots:
                upload.select(pin)
            if any(profile == upload_plan.AS_IS for _, profile in upload.slots):
                members = [m.member for m in command.selection.members]
                note = {"members": [m for m in members if upload_plan.carrier(m)]}
                self._emit(
                    command,
                    {
                        "kind": "log",
                        "name": upload_plan.AS_IS_NOTE,
                        "value": "warning",
                        "fields": note,
                    },
                )
            effective: dict[str, object] = {
                "source": content["content_manifest"],
                "slots": upload.slots,
                "converters": self._converters([profile for _, profile in upload.slots]),
                "recipe": upload.recipe.name if upload.recipe is not None else "",
                "destination": request.destination,
            }
            if self.native_registry is not None:
                effective["registry"] = hashlib.sha256(self.native_registry).hexdigest()
            configs = [m for m in content["members"] if not upload_plan.carrier(m["member"])]
            if upload.recipe is None and configs:
                # A Diffusers source's configs are inputs: they become the model's configs.
                effective["configs"] = configs
            computation = computation_key(row.operation, effective)
            hit = (
                self.memo.lookup(
                    owner,
                    call,
                    row.operation,
                    computation,
                    upload.destination,
                    lambda: self._stopped(command.service_id),
                )
                if self.memo is not None
                else None
            )
            if hit is not None:
                workspace_sources.record_upload(self.workspace, owner, command, hit, computation)
                terminal(
                    command,
                    pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                    result=hit,
                    computation=computation,
                    memo_hit=True,
                )
                return
            manifest = content["content_manifest"]
            checkpoint = upload.run(
                pin,
                ObjectRef(manifest["digest"], manifest["length"]),
                documents.spell(computation),
                call.parent_attempt_ordinal,
            )
        except source_upload.Canceled:
            if stage is not None:
                stage.finish(False)
            with self.workspace.locked() as db:
                db.execute(
                    "UPDATE native_calls SET state='failed' WHERE owner=? "
                    "AND service_id=? AND state IN ('accepted','resolved','executing')",
                    (owner, command.service_id),
                )
            terminal(command, pb.NATIVE_SOURCE_STATE_CANCELED)
            return
        if stage is not None:
            stage.finish(True)
        result = canonical_json.encode(msgspec.to_builtins(checkpoint))
        workspace_sources.record_upload(self.workspace, owner, command, result, computation)
        if self.memo is not None:
            self.memo.record(owner, call, row.operation, computation, result)
        try:
            upload.finish()
        except Exception:
            # The checkpoint is committed; the parent's release and pressure retry cleanup.
            _LOG.exception("upload cleanup deferred")
        terminal(command, pb.NATIVE_SOURCE_STATE_SUCCEEDED, result=result, computation=computation)

    def _write_bounds(self, step: SourceStep) -> tuple[storage_admission.Write, ...]:
        if not storage_admission.pressure_enabled():
            return ()
        root = self.workspace.store_root
        store = fill.store(root)
        match step:
            case source_steps.Download():
                # Native presence excludes complete members. Partial prefixes remain
                # conservatively charged until native exposes verified remaining bytes.
                members = [
                    row
                    for row in step.pin.members
                    if not store.contains(row[1].removeprefix("sha256:"))
                ]
                payload = sum(row[2] for row in members)
                return (storage_admission.native_write(root, payload, len(members)),)
            case source_steps.Commit():
                payload = step.request.size_bytes
                return (
                    storage_admission.native_write(root, payload),
                    storage_admission.Write(Path(step.staging), payload, 3),
                )
            case source_steps.Convert() if (
                store.derived_lookup(step.native_owner).get("state") != "committed"
            ):
                retained = store.tree_root(step.source_owner)
                if retained is None:
                    raise WorkspaceRefusal("source conversion lost its retained source")
                # Source conversion admits identity/permutation classes only. Their
                # body bytes cannot exceed the verified carrier roster; no quantizer
                # or value-expanding converter runs on this native source path.
                # This child runs all native passes before returning, unlike the
                # incremental PrepareModelSource RPC, so its hold covers the whole body.
                rows = store.walk(retained["manifest_digest"])
                payload = sum(row["length"] for row in rows)
                return (storage_admission.native_write(root, payload, len(rows)),)
        # Profile selection stages sparse header carriers and Store metadata.
        return (storage_admission.native_write(root),)

    def _run(
        self,
        owner: str,
        command: pb.NativeSourceCommand,
        row: NativeCall,
        done: threading.Event,
    ) -> None:
        def finish() -> None:
            # A retry can arrive immediately after the terminal frame. Detach this
            # exact completed task first, and never erase a new task's registration.
            with self.lock:
                if self.tasks.get(command.service_id) is done:
                    self.processes.pop(command.service_id, None)
                    self.parents.pop(command.service_id, None)
                    self.commands.pop(command.service_id, None)
                    self.process_owners.pop(command.service_id, None)
                    self.tasks.pop(command.service_id, None)
            done.set()
            self.finished(command.service_id)

        terminal = _after(finish, self._status)
        view = command.operation == pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES
        file = command.operation == pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE
        native_bytes = view or file
        stage: StageProgress | None = None
        try:
            if row.state == "complete":
                view_fact = (
                    workspace_byte_outputs.replay_native(self.workspace, owner, command)
                    if file
                    else source_views.replay(self.workspace, owner, command)
                    if view
                    else None
                )
                if not native_bytes and row.operation not in source_interfaces.UPLOADS:
                    workspace_memo.retain_native_result(self.workspace, owner, command.service_id)
                terminal(
                    command,
                    pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                    result=row.result,
                    receipt=row.native_receipt,
                    computation=row.computation_digest,
                    byte_output=view_fact[0] if view_fact else None,
                    byte_output_attempt=view_fact[1] if view_fact else 0,
                    byte_output_spec=view_fact[2] if view_fact else b"",
                )
                return
            request = msgspec.json.decode(
                row.accepted, type=source_interfaces.TYPES[row.operation][0]
            )
            phase = "resolve" if command.phase == pb.NATIVE_SOURCE_PHASE_RESOLVE else "execute"
            if isinstance(request, (UploadHuggingFace, UploadCivitai)):
                self._upload(owner, command, row, request, phase, finish)
                return
            call = command.parent_call
            access = source_steps.access(self.endpoints, command.credential)
            computation = b""
            step: SourceStep
            if phase == "resolve":
                step = _resolution(request, access)
            elif isinstance(request, CommitFile):
                effective = {
                    "digest": request.digest,
                    "size_bytes": request.size_bytes,
                    "media_type": request.media_type,
                }
                computation = computation_key(row.operation, effective)
                step = commit_files.prepare(
                    self.workspace, owner, command, computation, self.spool_root
                )
            elif isinstance(request, SourceFiles):
                effective = {
                    "source": msgspec.to_builtins(request.source.manifest),
                    "max_bytes": SOURCE_FILES_MAX_BYTES,
                }
                computation = computation_key(row.operation, effective)
                step = source_views.prepare(self.workspace, owner, command, computation)
            else:
                convert = isinstance(request, source_interfaces.Convert)
                if isinstance(request, source_interfaces.Convert):
                    source_owner = self._retention(owner, command, request.source)
                    profiles = _profiles(request)
                    source = msgspec.to_builtins(request.source.manifest)
                    effective = (
                        {"source": source, "profile": profiles[0]}
                        if len(profiles) == 1
                        else {"source": source, "profiles": list(profiles)}
                    )
                    if self.native_registry is not None:
                        effective["registry"] = hashlib.sha256(self.native_registry).hexdigest()
                    # The versioned converters (and their pinned references) are the meaning.
                    effective["converters"] = self._converters(profiles)
                else:
                    content = workspace_sources.selection_content(command.selection)
                    effective = {"source": content["content_manifest"]}
                computation = computation_key(row.operation, effective)
                hit = workspace_memo.lookup_native(
                    self.workspace, owner, command.service_id, computation
                )
                if hit is not None:
                    workspace_memo.record_reuse(self.workspace, owner, command.service_id)
                    terminal(
                        command,
                        pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                        result=hit["result"],
                        receipt=hit["native_receipt"],
                        computation=computation,
                        memo_hit=True,
                    )
                    return
                if convert:
                    step = source_steps.Convert(
                        native_owner=row.native_owner,
                        service_id=command.service_id,
                        source_owner=source_owner,
                        slots=upload_plan.slots(profiles),
                        registry=self.native_registry,
                        adopt=workspace_sources.conversion_predecessor(
                            self.workspace, owner, command.service_id
                        ),
                        writer_epoch=call.parent_attempt_ordinal,
                        computation_digest=documents.spell(computation),
                        access=access,
                    )
                else:
                    workspace_sources.adopt_download_progress(
                        self.workspace, owner, command.service_id
                    )
                    step = source_steps.Download(
                        native_owner=row.native_owner,
                        service_id=command.service_id,
                        pin=_native_selection(command.selection),
                        access=access,
                    )

            def observe(sample: Sample) -> None:
                nonlocal stage
                label = STAGES[sample.stage]
                if stage is None or stage.stage != label:
                    if stage is not None:
                        stage.finish(True)
                    stage = self._stage(command, label)
                stage.update(sample.position, sample.total)

            launch = Launch(store=str(self.workspace.store_root), parent_pid=os.getpid(), step=step)
            # Admission is held until the native child can no longer write.
            with storage_admission.admit(*self._write_bounds(step)):
                status, output, stderr = source_steps.run(
                    SOURCE_WORKER,
                    launch,
                    observe=observe,
                    started=partial(self._started, command.service_id),
                )
            with self.lock:
                canceled = command.service_id in self.canceled
            if canceled:
                with self.workspace.locked() as db:
                    # A parent interruption is retryable. Explicit logical source
                    # cancellation already wrote stopped and must remain permanent.
                    db.execute(
                        "UPDATE native_calls SET state='failed' WHERE owner=? "
                        "AND service_id=? AND state IN ('accepted','resolved','executing')",
                        (owner, command.service_id),
                    )
                if stage is not None:
                    stage.finish(False)
                terminal(command, pb.NATIVE_SOURCE_STATE_CANCELED)
                return
            if status or not output:
                raise NativeSourceFailure(
                    "native_source_failed", diagnosis(status, stderr, command.credential)
                )
            command = self._latest(command)
            if self.owner() != owner or not self.current(command):
                return
            if isinstance(step, source_steps.ResolveSource):
                answer = _answer(output, stderr, command.credential, Resolved)
                resolved = _selection(answer.selection)
                workspace_sources.record_pin(self.workspace, owner, command, resolved)
                terminal(command, pb.NATIVE_SOURCE_STATE_RESOLVED, selection=resolved)
                return
            produced = _answer(output, stderr, command.credential, Produced)
            result, receipt = canonical_json.encode(produced.result), produced.native_receipt
            view_fact = None
            if isinstance(step, source_steps.Commit):
                view_fact = workspace_byte_outputs.complete_native(
                    self.workspace, owner, command, result, computation, receipt, step.view_owner
                )
            elif isinstance(step, source_steps.View):
                view_fact = source_views.complete(
                    self.workspace, owner, command, result, computation, receipt, step.view_owner
                )
            else:
                workspace_sources.record_complete(
                    self.workspace, owner, command, result, computation, receipt
                )
                if isinstance(step, source_steps.Convert) and step.adopt:
                    workspace_sources.consume_partial(self.workspace, owner, step.adopt)
                workspace_memo.retain_native_result(self.workspace, owner, command.service_id)
                # The memo is an optional cache: a row it cannot record (for example one
                # written by another Runtime version) never fails a completed call.
                try:
                    workspace_memo.record_native(self.workspace, owner, command.service_id)
                except Exception:
                    _LOG.exception("native memo was not recorded")
            if stage is not None:
                stage.finish(True)
            terminal(
                command,
                pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                result=result,
                receipt=receipt,
                computation=computation,
                byte_output=view_fact[0] if view_fact else None,
                byte_output_attempt=view_fact[1] if view_fact else 0,
                byte_output_spec=view_fact[2] if view_fact else b"",
            )
        except Exception as exc:
            # Only a known native refusal code crosses; arbitrary exception text stays
            # private. A child's redacted exit status and stderr tail is its diagnosis.
            coded = (
                NativeSourceFailure,
                SourceRefusal,
                storage_admission.StorageRefusal,
                commit_files.UnsupportedCommitRequest,
            )
            code = exc.code if isinstance(exc, coded) else "native_source_failed"
            detail = (
                exc.detail
                if isinstance(exc, NativeSourceFailure)
                else str(exc)
                if isinstance(exc, (SourceRefusal, commit_files.UnsupportedCommitRequest))
                else ""
            )
            with self.workspace.locked() as db:
                db.execute(
                    "UPDATE native_calls SET state='failed',native_error=? "
                    "WHERE owner=? AND service_id=? "
                    "AND state IN ('accepted','resolved','executing')",
                    (code, owner, command.service_id),
                )
            if stage is not None:
                stage.finish(False)
            if detail:
                _LOG.warning("native source %s failed: %s", command.service_id, detail)
                with self.lock:
                    self.details[command.service_id] = brief(detail)
                    while len(self.details) > 256:
                        self.details.pop(next(iter(self.details)))
                self._emit(
                    command,
                    {
                        "kind": "log",
                        "name": "native source failed",
                        "value": "error",
                        "fields": {"code": code, "detail": detail},
                    },
                )
            terminal(command, pb.NATIVE_SOURCE_STATE_FAILED, code=code, detail=brief(detail))
        finally:
            if view:
                try:
                    source_views.abort_unfinished(self.workspace, owner, command)
                except Exception:
                    # Durable release intent is retried by the ordinary parent ACK path.
                    _LOG.warning("Native source view cleanup deferred to parent release")
            finish()

    def _retention(
        self, owner: str, command: pb.NativeSourceCommand, artifact: SourceArtifact
    ) -> str:
        """The byte-tree custody this parent was granted of the conversion's source."""
        with self.workspace.locked() as db:
            access = db.execute(
                "SELECT retention_id FROM source_access WHERE owner=? AND request=? "
                "AND native_digest=? AND manifest=? AND manifest_length=? AND state='held'",
                (
                    owner,
                    command.parent_call.parent_request_id,
                    documents.raw(artifact.tensorfs_receipt_digest),
                    documents.raw(artifact.manifest.digest),
                    artifact.manifest.length,
                ),
            ).fetchone()
        if access is None:
            raise WorkspaceRefusal("source conversion lacks granted byte-tree custody")
        return str(access["retention_id"])
