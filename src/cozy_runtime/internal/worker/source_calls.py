"""Asynchronous execution of the three accepted native source operations.

The existing control stream carries acceptance/results. Native work runs in a
separate killable process so heartbeats and parent cancellation remain responsive.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import IO, Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._errors import AuthorError
from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
from cozy_runtime.internal import fill, source_interfaces, storage_admission
from cozy_runtime.internal.worker import (
    source_upload,
    source_views,
    source_worker,
    upload_plan,
    workspace_partial,
    workspace_sources,
)
from cozy_runtime.internal.worker.machine_publication import PublicationClient, PublicationRefusal
from cozy_runtime.internal.worker.stage_progress import Progress, StageProgress
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

_LOG = logging.getLogger(__name__)

STAGES = {"download": "Downloading source", "convert": "Converting to cozytensors"}
# The retained stderr tail, and the part of it a child-call failure detail carries
# (authors see at most 1 KiB of detail).
TAIL_BYTES = 8 << 10
LOG_TAIL_CHARS = 4000
DETAIL_TAIL_CHARS = 600
_FRAME = re.compile(r'File "(?:[^"]*/)?([^"/]+)", line (\d+),? in ')
_URL_QUERY = re.compile(r"(\w+://[^\s?#'\"]+)[?#][^\s'\"]*")
_SECRET = re.compile(
    r"(?i)\b(bearer|token|authorization|signature|credential)([\s:=\"']+)[^\s'\",;]+"
)


class _Tail:
    """The last stderr bytes of a child, drained continuously so it never blocks."""

    def __init__(self, stream: IO[bytes]) -> None:
        self.data = bytearray()
        self.thread = threading.Thread(
            target=self._drain, args=(stream,), daemon=True, name="native-source-stderr"
        )
        self.thread.start()

    def _drain(self, stream: IO[bytes]) -> None:
        while chunk := stream.read(65536):
            self.data += chunk
            del self.data[:-TAIL_BYTES]

    def text(self) -> str:
        self.thread.join()
        return self.data.decode(errors="replace")


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


def _exchange(
    process: subprocess.Popen[bytes], command: bytes, observe: Callable[[dict[str, Any]], None]
) -> bytes:
    """Send the command; stream progress lines; return the raw final result line."""
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(command)
        process.stdin.close()
    except BrokenPipeError:
        pass  # a killed child; its exit status says why
    answer = b""
    while line := process.stdout.readline((2 << 20) + 2):
        if line.startswith(b'{"progress":') and line.endswith(b"\n"):
            document = canonical_json.decode(line)
            if isinstance(document, dict) and document.keys() == {"progress"}:
                observe(document["progress"])
                continue
        answer += line
        if len(answer) > (2 << 20) + 1:
            process.kill()
            break
    return answer


def _selection(value: dict[str, Any]) -> pb.NativeSourceSelection:
    return pb.NativeSourceSelection(
        canonical=value["canonical"],
        selection_digest=bytes.fromhex(value["selection_sha256"]),
        content_manifest=pb.Ref(
            digest=documents.raw(value["content_manifest_digest"]),
            length=value["content_manifest_length"],
        ),
        members=[
            pb.NativeSourceMember(
                member=member, object=pb.Ref(digest=documents.raw(digest), length=length), url=url
            )
            for member, digest, length, url in value["members"]
        ],
        allowed_hosts=value["allowed_hosts"],
        credential_hosts=value["credential_hosts"],
    )


def _native_selection(value: pb.NativeSourceSelection) -> dict[str, Any]:
    return {
        "members": [
            (row.member, documents.spell(row.object.digest), row.object.length, row.url)
            for row in value.members
        ],
        "allowed_hosts": list(value.allowed_hosts),
        "credential_hosts": list(value.credential_hosts),
    }


def computation_key(operation: str, effective: dict[str, Any]) -> bytes:
    """The native memo key: the operation, its declared output version and its real inputs.

    A Runtime or TensorFS release that keeps an operation's output meaning keeps the key;
    one that changes it bumps `source_interfaces.OPERATION_VERSIONS` by hand.
    """
    return hashlib.sha256(
        canonical_json.encode(
            {
                "operation": operation,
                "version": source_interfaces.OPERATION_VERSIONS[operation],
                "effective": effective,
            }
        )
    ).digest()


def refusal_frame(command: pb.NativeSourceCommand, code: str) -> pb.WorkerFrame:
    """An authenticated rejected command keeps its caller's original correlation identity."""
    call = command.parent_call
    return pb.WorkerFrame(
        native_source_status=pb.NativeSourceStatus(
            parent_request_id=call.parent_request_id,
            parent_attempt_ordinal=call.parent_attempt_ordinal,
            parent_invocation_spec_digest=call.parent_invocation_spec_digest,
            call_index=call.call_index,
            intent_digest=call.intent_digest,
            service_id=command.service_id,
            state=pb.NATIVE_SOURCE_STATE_FAILED,
            safe_code=code,
        )
    )


class NativeSourceFailure(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code, self.detail = code, detail
        super().__init__(code)


class SourceCalls:
    def __init__(
        self,
        workspace: Workspace,
        owner: Callable[[], str],
        send: Callable[[pb.WorkerFrame], None],
        current: Callable[[pb.NativeSourceCommand], bool],
        *,
        endpoints: dict[str, str] | None = None,
        native_registry: bytes | None = None,
        spool_root: Path | None = None,
        progress: Progress | None = None,
        publication: Callable[[str, str, int], PublicationClient] | None = None,
    ):
        self.workspace, self.owner, self.send, self.current = workspace, owner, send, current
        self.progress = progress
        # The latest failure detail per native call, for the machine child-call result.
        self.details: dict[str, str] = {}
        self.endpoints = endpoints or {}
        self.native_registry = native_registry
        self.spool_root = spool_root
        self.publication = publication
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen[bytes] | None] = {}
        self.tasks: dict[str, threading.Event] = {}
        self.parents: dict[str, str] = {}
        self.commands: dict[str, pb.NativeSourceCommand] = {}
        self.process_owners: dict[str, str] = {}
        self.canceled: set[str] = set()
        self.finalizers: set[tuple[str, int]] = set()

    def handle(self, command: pb.NativeSourceCommand) -> None:
        owner = self.owner()
        if command.phase == pb.NATIVE_SOURCE_PHASE_CANCEL:
            workspace_sources.stopped(self.workspace, owner, command)
            with self.lock:
                self.canceled.add(command.service_id)
                process = self.processes.get(command.service_id)
                if process is not None:
                    process.kill()
            return
        if (
            command.operation
            in (pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES, pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE)
            and command.phase != pb.NATIVE_SOURCE_PHASE_EXECUTE
        ):
            raise WorkspaceRefusal("source view has no foreign resolution phase")
        row = workspace_sources.accepted(self.workspace, owner, command)
        if row["state"] not in ("stopped", "failed", "releasing", "released"):
            with self.lock:
                self.canceled.discard(command.service_id)
        if command.phase not in (pb.NATIVE_SOURCE_PHASE_RESOLVE, pb.NATIVE_SOURCE_PHASE_EXECUTE):
            raise WorkspaceRefusal("unknown native source execution phase")
        if row["state"] == "stopped":
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

    def acknowledge(self, ack: pb.AttemptOutcomeAck, finished: Callable[[], None]) -> bool:
        """Drain native processes and finalize off the heartbeat/control thread."""
        if ack.retain_work:
            return False
        owner = self.owner()
        with self.workspace.locked() as db:
            found = db.execute(
                "SELECT 1 FROM native_calls WHERE owner=? AND parent_request=? "
                "AND state<>'released' LIMIT 1",
                (owner, ack.request_id),
            ).fetchone()
            if found is None:
                return False
            parent = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, ack.request_id, ack.attempt_ordinal),
            ).fetchone()
            if parent is not None:
                if (
                    parent["state"] not in ("outcome", "released")
                    or parent["spec"] != ack.invocation_spec_digest
                    or parent["outcome_id"] != ack.outcome_id
                    or parent["outcome_digest"] != ack.outcome_digest
                ):
                    raise WorkspaceRefusal("native finalization changed its terminal ACK")
            elif (
                db.execute(
                    "SELECT 1 FROM native_calls WHERE owner=? AND parent_request=? "
                    "AND release_ack=? AND state='releasing' LIMIT 1",
                    (owner, ack.request_id, workspace_sources.ack_identity(ack)),
                ).fetchone()
                is None
            ):
                raise WorkspaceRefusal("native finalization has no recorded terminal ACK")
        key = (ack.request_id, ack.attempt_ordinal)
        copied = pb.AttemptOutcomeAck()
        copied.CopyFrom(ack)
        with self.lock:
            if key in self.finalizers:
                return True
            self.finalizers.add(key)

        def finalize() -> None:
            try:
                self.cancel_parent(copied.request_id, through=copied.attempt_ordinal)
                with self.lock:
                    tasks = [
                        done
                        for service, done in self.tasks.items()
                        if self.parents.get(service) == copied.request_id
                        and self.commands[service].parent_call.parent_attempt_ordinal
                        <= copied.attempt_ordinal
                    ]
                for done in tasks:
                    done.wait()
                self.workspace.acknowledge(owner, copied)
                finished()
            finally:
                with self.lock:
                    self.finalizers.discard(key)

        threading.Thread(target=finalize, daemon=True, name="native-source-release").start()
        return True

    def _latest(self, command: pb.NativeSourceCommand) -> pb.NativeSourceCommand:
        with self.lock:
            return self.commands.get(command.service_id, command)

    def failure_detail(self, service_id: str) -> str:
        with self.lock:
            return self.details.get(service_id, "")

    def _emit(self, command: pb.NativeSourceCommand, frame: dict[str, Any]) -> None:
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
        self.send(pb.WorkerFrame(native_source_status=status))

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

    def _child(self, command: pb.NativeSourceCommand, envelope: dict[str, Any]) -> dict[str, Any]:
        """One killable upload step. A cancellation kills it and surfaces as ``Canceled``."""
        encoded = canonical_json.encode(envelope)
        if len(encoded) > 2 << 20:
            raise WorkspaceRefusal("upload step exceeds bounded transfer")
        process = subprocess.Popen(
            [sys.executable, "-m", "cozy_runtime.internal.upload_child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self.lock:
            self.processes[command.service_id] = process
            if command.service_id in self.canceled:
                process.kill()
        assert process.stderr is not None
        tail = _Tail(process.stderr)
        try:
            output = _exchange(process, encoded, lambda _sample: None)
        finally:
            # Admission is held by the caller until this child can no longer write.
            if process.poll() is None:
                process.kill()
            process.wait()
        stderr = tail.text()
        if self._stopped(command.service_id):
            raise source_upload.Canceled()
        if process.returncode or not output or len(output) > 2 << 20:
            raise NativeSourceFailure(
                "native_source_failed", diagnosis(process.returncode, stderr, command.credential)
            )
        answer = canonical_json.decode(output)
        if not answer.get("ok"):
            code = answer.get("code")
            if not isinstance(code, str) or (
                code not in fill.tensorfs_module().errors.CODES and code not in upload_plan.CODES
            ):
                code = "native_source_failed"
            raise NativeSourceFailure(
                code, diagnosis(0, stderr, command.credential) if stderr.strip() else ""
            )
        return dict(answer)

    def _upload(
        self,
        owner: str,
        command: pb.NativeSourceCommand,
        row: dict[str, Any],
        arguments: dict[str, Any],
        phase: str,
        terminal: Callable[..., None],
    ) -> None:
        """Resolve, or stream the pinned source through conversion into its publication."""
        try:
            self._upload_steps(owner, command, row, arguments, phase, terminal)
        except upload_plan.UploadRefusal as exc:
            # Reviewed refusals name what the caller changes; their text holds no secret.
            raise NativeSourceFailure(exc.code, str(exc)) from exc
        except (AuthorError, PublicationRefusal) as exc:
            # A reviewed or Hub refusal keeps its stable code; its text stays private.
            raise NativeSourceFailure(exc.code) from exc

    def _upload_steps(
        self,
        owner: str,
        command: pb.NativeSourceCommand,
        row: dict[str, Any],
        arguments: dict[str, Any],
        phase: str,
        terminal: Callable[..., None],
    ) -> None:
        upload_plan.require_bindings(fill.tensorfs_module(), row["operation"])
        if self.publication is None:
            raise NativeSourceFailure("publication_authority_absent")
        call = command.parent_call
        stage: StageProgress | None = None

        def report(name: str, position: int, total: int) -> None:
            nonlocal stage
            if stage is None or stage.stage != name:
                if stage is not None:
                    stage.finish(True)
                stage = StageProgress(name, lambda frame: self._emit(command, frame))
            stage.update(position, total)

        upload = source_upload.Upload(
            store_root=self.workspace.store_root,
            native_owner=row["native_owner"],
            service_id=command.service_id,
            operation=row["operation"],
            arguments=arguments,
            envelope={
                "store": str(self.workspace.store_root),
                "parent_pid": os.getpid(),
                "operation": row["operation"],
                "arguments": arguments,
                "native_owner": row["native_owner"],
                "credential": command.credential,
                **self.endpoints,
                "allow_local": bool(self.endpoints),
            },
            registry=self.native_registry,
            client=self.publication(owner, call.parent_request_id, call.parent_attempt_ordinal),
            hooks=source_upload.Hooks(
                child=lambda envelope: self._child(command, envelope),
                canceled=lambda: self._stopped(command.service_id),
                active=self.active,
                live=self._live_uploads,
                evict_retired=lambda target: workspace_partial.reclaim(self.workspace, target),
                consume=partial(workspace_sources.consume_partial, self.workspace, owner),
                progress=report,
            ),
            allow_local=bool(self.endpoints),
            directory=self.workspace.directory,
        )
        try:
            if phase == "resolve":
                resolved = _selection(upload.resolve())
                workspace_sources.record_pin(self.workspace, owner, command, resolved)
                terminal(command, pb.NATIVE_SOURCE_STATE_RESOLVED, selection=resolved)
                return
            content = workspace_sources.selection_content(command.selection)
            effective = {
                "source": content["content_manifest"],
                "slots": upload.slots,
                "recipe": upload.recipe.name if upload.recipe is not None else "",
                "destination": arguments["destination"],
            }
            if self.native_registry is not None:
                effective["registry"] = hashlib.sha256(self.native_registry).hexdigest()
            configs = [m for m in content["members"] if not upload_plan.carrier(m["member"])]
            if upload.recipe is None and configs:
                # A Diffusers source's configs are inputs: they become the model's configs.
                effective["configs"] = configs
            computation = computation_key(row["operation"], effective)
            checkpoint = upload.run(
                _native_selection(command.selection),
                content,
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
        try:
            upload.finish()
        except Exception:
            # The checkpoint is committed; the parent's release and pressure retry cleanup.
            _LOG.exception("upload cleanup deferred")
        terminal(command, pb.NATIVE_SOURCE_STATE_SUCCEEDED, result=result, computation=computation)

    def _write_bounds(self, command: dict[str, Any]) -> tuple[storage_admission.Write, ...]:
        if not storage_admission.pressure_enabled():
            return ()
        if command["phase"] == "resolve":
            # Profile selection stages sparse header carriers and Store metadata.
            return (storage_admission.native_write(self.workspace.store_root),)
        operation, args = command["operation"], command["arguments"]
        payload, objects = 0, 1
        if operation in ("download_huggingface", "download_civitai"):
            members = command["selection"]["members"]
            store = fill.store(self.workspace.store_root)
            # Native presence excludes complete members. Partial prefixes remain
            # conservatively charged until native exposes verified remaining bytes.
            members = [row for row in members if not store.contains(row[1].removeprefix("sha256:"))]
            payload, objects = sum(row[2] for row in members), len(members)
        elif operation == "commit_file":
            payload = args["size_bytes"]
            return (
                storage_admission.native_write(self.workspace.store_root, payload),
                storage_admission.Write(Path(command["file_staging"]), payload, 3),
            )
        elif operation == "convert_cozytensors":
            store = fill.store(self.workspace.store_root)
            if store.derived_lookup(command["native_owner"]).get("state") == "committed":
                return (storage_admission.native_write(self.workspace.store_root),)
            root = store.tree_root(command["source_owner"])
            # Source conversion admits identity/permutation classes only. Their
            # body bytes cannot exceed the verified carrier roster; no quantizer
            # or value-expanding converter runs on this native source path.
            # This child runs all native passes before returning, unlike the
            # incremental PrepareModelSource RPC, so its hold covers the whole body.
            members = store.walk(root["manifest_digest"])
            payload, objects = sum(int(row["length"]) for row in members), len(members)
        return (storage_admission.native_write(self.workspace.store_root, payload, objects),)

    def _run(
        self,
        owner: str,
        command: pb.NativeSourceCommand,
        row: dict[str, Any],
        done: threading.Event,
    ) -> None:
        from cozy_runtime.internal.worker import (
            commit_files,
            workspace_byte_outputs,
            workspace_memo,
        )

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

        def terminal(
            current: pb.NativeSourceCommand, state: pb.NativeSourceState, **fields: Any
        ) -> None:
            finish()
            self._status(current, state, **fields)

        view = command.operation == pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES
        file = command.operation == pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE
        native_bytes = view or file
        stage: StageProgress | None = None
        try:
            if row["state"] == "complete":
                view_fact = (
                    workspace_byte_outputs.replay_native(self.workspace, owner, command)
                    if file
                    else source_views.replay(self.workspace, owner, command)
                    if view
                    else None
                )
                if not native_bytes and row["operation"] not in source_interfaces.UPLOADS:
                    workspace_memo.retain_native_result(self.workspace, owner, command.service_id)
                terminal(
                    command,
                    pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                    result=row["result"],
                    receipt=row["native_receipt"],
                    computation=row["computation_digest"],
                    byte_output=view_fact[0] if view_fact else None,
                    byte_output_attempt=view_fact[1] if view_fact else 0,
                    byte_output_spec=view_fact[2] if view_fact else b"",
                )
                return
            request = source_interfaces.TYPES[row["operation"]][0]
            arguments = msgspec.to_builtins(
                msgspec.convert(canonical_json.decode(row["accepted"]), type=request, strict=True)
            )
            phase = "resolve" if command.phase == pb.NATIVE_SOURCE_PHASE_RESOLVE else "execute"
            if row["operation"] in source_interfaces.UPLOADS:
                self._upload(owner, command, row, arguments, phase, terminal)
                return
            envelope = {
                "store": str(self.workspace.store_root),
                "parent_pid": os.getpid(),
                "operation": row["operation"],
                "arguments": arguments,
                "phase": phase,
                "native_owner": row["native_owner"],
                "service_id": command.service_id,
                "credential": command.credential,
                **self.endpoints,
                "allow_local": bool(self.endpoints),
            }
            computation = b""
            if phase == "execute":
                if file:
                    effective = {
                        key: arguments[key] for key in ("digest", "size_bytes", "media_type")
                    }
                elif view:
                    effective = {
                        "source": arguments["source"]["manifest"],
                        "max_bytes": SOURCE_FILES_MAX_BYTES,
                    }
                elif command.operation == pb.NATIVE_SOURCE_OPERATION_CONVERT:
                    artifact = arguments["source"]
                    with self.workspace.locked() as db:
                        access = db.execute(
                            "SELECT retention_id FROM source_access WHERE owner=? AND request=? "
                            "AND native_digest=? AND manifest=? AND manifest_length=? "
                            "AND state='held'",
                            (
                                owner,
                                command.parent_call.parent_request_id,
                                documents.raw(artifact["tensorfs_receipt_digest"]),
                                documents.raw(artifact["manifest"]["digest"]),
                                artifact["manifest"]["length"],
                            ),
                        ).fetchone()
                        if access is None:
                            raise WorkspaceRefusal(
                                "source conversion lacks granted byte-tree custody"
                            )
                    envelope["source_owner"] = access["retention_id"]
                    profiles = source_worker.profiles_of(arguments)
                    effective = (
                        {"source": artifact["manifest"], "profile": profiles[0]}
                        if len(profiles) == 1
                        else {"source": artifact["manifest"], "profiles": list(profiles)}
                    )
                    if self.native_registry is not None:
                        envelope["registry"] = base64.b64encode(self.native_registry).decode()
                        effective["registry"] = hashlib.sha256(self.native_registry).hexdigest()
                else:
                    envelope["selection"] = _native_selection(command.selection)
                    effective = {
                        "source": workspace_sources.selection_content(command.selection)[
                            "content_manifest"
                        ]
                    }
                computation = computation_key(row["operation"], effective)
                hit = (
                    None
                    if native_bytes
                    else workspace_memo.lookup_native(
                        self.workspace, owner, command.service_id, computation
                    )
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
                if file:
                    envelope.update(
                        commit_files.prepare(
                            self.workspace, owner, command, computation, self.spool_root
                        )
                    )
                elif view:
                    envelope.update(
                        source_views.prepare(self.workspace, owner, command, computation)
                    )
                elif command.operation == pb.NATIVE_SOURCE_OPERATION_CONVERT:
                    prior = workspace_sources.conversion_predecessor(
                        self.workspace, owner, command.service_id
                    )
                    if prior is not None:
                        envelope["adopt_operation"] = prior
                else:
                    workspace_sources.adopt_download_progress(
                        self.workspace, owner, command.service_id
                    )
                envelope["computation_digest"] = documents.spell(computation)
                envelope["writer_epoch"] = command.parent_call.parent_attempt_ordinal
            encoded = canonical_json.encode(envelope)
            if len(encoded) > 2 << 20:
                raise WorkspaceRefusal("source execution command exceeds bounded transfer")

            def observe(sample: dict[str, Any]) -> None:
                nonlocal stage
                label = STAGES.get(str(sample.get("stage")))
                if label is None:
                    return
                if stage is None or stage.stage != label:
                    if stage is not None:
                        stage.finish(True)
                    stage = StageProgress(label, lambda frame: self._emit(command, frame))
                stage.update(int(sample["position"]), int(sample["total"]))

            with storage_admission.admit(*self._write_bounds(envelope)):
                process = subprocess.Popen(
                    [sys.executable, "-m", "cozy_runtime.internal.worker.source_worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                with self.lock:
                    self.processes[command.service_id] = process
                    if command.service_id in self.canceled:
                        process.kill()
                assert process.stderr is not None
                tail = _Tail(process.stderr)
                try:
                    output = _exchange(process, encoded, observe)
                finally:
                    # Do not free admission while an exceptional IPC exit leaves
                    # the privileged native child writing in the background.
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                stderr = tail.text()
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
            if process.returncode or not output or len(output) > 2 << 20:
                raise NativeSourceFailure(
                    "native_source_failed",
                    diagnosis(process.returncode, stderr, command.credential),
                )
            command = self._latest(command)
            if self.owner() != owner or not self.current(command):
                return
            answer = canonical_json.decode(output)
            if not answer.get("ok"):
                code = answer.get("code")
                if not isinstance(code, str) or code not in fill.tensorfs_module().errors.CODES:
                    code = "native_source_failed"
                raise NativeSourceFailure(
                    code, diagnosis(0, stderr, command.credential) if stderr.strip() else ""
                )
            if phase == "resolve":
                resolved = _selection(answer["selection"])
                workspace_sources.record_pin(self.workspace, owner, command, resolved)
                terminal(command, pb.NATIVE_SOURCE_STATE_RESOLVED, selection=resolved)
                return
            result = canonical_json.encode(answer["result"])
            receipt = base64.b64decode(answer["native_receipt"], validate=True)
            view_fact = None
            if file:
                view_fact = workspace_byte_outputs.complete_native(
                    self.workspace,
                    owner,
                    command,
                    result,
                    computation,
                    receipt,
                    str(envelope["view_owner"]),
                )
            elif view:
                view_fact = source_views.complete(
                    self.workspace,
                    owner,
                    command,
                    result,
                    computation,
                    receipt,
                    str(envelope["view_owner"]),
                )
            else:
                workspace_sources.record_complete(
                    self.workspace, owner, command, result, computation, receipt
                )
                if adopted := envelope.get("adopt_operation"):
                    workspace_sources.consume_partial(self.workspace, owner, str(adopted))
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
            code = (
                exc.code
                if isinstance(
                    exc,
                    (
                        NativeSourceFailure,
                        storage_admission.StorageRefusal,
                        commit_files.UnsupportedCommitRequest,
                    ),
                )
                else "native_source_failed"
            )
            detail = (
                exc.detail
                if isinstance(exc, NativeSourceFailure)
                else str(exc)
                if isinstance(exc, commit_files.UnsupportedCommitRequest)
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
