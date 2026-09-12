"""Asynchronous execution of the three accepted native source operations.

The existing control stream carries acceptance/results. Native work runs in a
separate killable process so heartbeats and parent cancellation remain responsive.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
from cozy_runtime.internal import fill, source_interfaces, storage_admission
from cozy_runtime.internal.worker import source_views, workspace_sources
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

_LOG = logging.getLogger(__name__)


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


def implementation_digest(store: Any) -> str:
    import cozy_runtime

    root = Path(cozy_runtime.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = str(path.relative_to(root)).encode()
        body = path.read_bytes()
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(len(body).to_bytes(8, "little"))
        digest.update(body)
    digest.update(store.implementation_sha256.encode())
    extension = importlib.import_module("tensorfs._ext")
    extension_path = extension.__file__
    if extension_path is None:
        raise WorkspaceRefusal("native implementation has no immutable extension artifact")
    with Path(extension_path).open("rb") as binary:
        for chunk in iter(lambda: binary.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
    def __init__(self, code: str) -> None:
        self.code = code
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
    ):
        self.workspace, self.owner, self.send, self.current = workspace, owner, send, current
        self.endpoints = endpoints or {}
        self.native_registry = native_registry
        self.spool_root = spool_root
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
            memo_hit=memo_hit,
            byte_output_attempt_ordinal=byte_output_attempt,
            byte_output_invocation_spec_digest=byte_output_spec,
        )
        if selection is not None:
            status.selection.CopyFrom(selection)
        if byte_output is not None:
            status.byte_output.CopyFrom(byte_output)
        self.send(pb.WorkerFrame(native_source_status=status))

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
            store = fill.tensorfs_module().Store.open(str(self.workspace.store_root))
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
            store = fill.tensorfs_module().Store.open(str(self.workspace.store_root))
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
        try:
            if row["state"] == "complete":
                view_fact = (
                    workspace_byte_outputs.replay_native(self.workspace, owner, command)
                    if file
                    else source_views.replay(self.workspace, owner, command)
                    if view
                    else None
                )
                if not native_bytes:
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
            store = fill.tensorfs_module().Store.open(str(self.workspace.store_root))
            binding = source_interfaces.bindings()[
                (source_interfaces.DIGEST, source_interfaces.MODULE, row["operation"])
            ]
            arguments = msgspec.to_builtins(
                msgspec.convert(
                    canonical_json.decode(row["accepted"]), type=binding.request, strict=True
                )
            )
            phase = "resolve" if command.phase == pb.NATIVE_SOURCE_PHASE_RESOLVE else "execute"
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
                    effective = {"source": artifact["manifest"], "profile": arguments["profile"]}
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
                computation = hashlib.sha256(
                    canonical_json.encode(
                        {
                            "interface": source_interfaces.DIGEST,
                            "operation": row["operation"],
                            "implementation": implementation_digest(store),
                            "effective": effective,
                        }
                    )
                ).digest()
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
            with storage_admission.admit(*self._write_bounds(envelope)):
                process = subprocess.Popen(
                    [sys.executable, "-m", "cozy_runtime.internal.worker.source_worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                with self.lock:
                    self.processes[command.service_id] = process
                    if command.service_id in self.canceled:
                        process.kill()
                try:
                    output, _ = process.communicate(encoded)
                finally:
                    # Do not free admission while an exceptional IPC exit leaves
                    # the privileged native child writing in the background.
                    if process.poll() is None:
                        process.kill()
                    process.wait()
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
                terminal(command, pb.NATIVE_SOURCE_STATE_CANCELED)
                return
            if process.returncode or len(output) > 2 << 20:
                raise WorkspaceRefusal("native source execution ended without bounded result")
            command = self._latest(command)
            if self.owner() != owner or not self.current(command):
                return
            answer = canonical_json.decode(output)
            if not answer.get("ok"):
                code = answer.get("code")
                if not isinstance(code, str) or code not in fill.tensorfs_module().errors.CODES:
                    code = "native_source_failed"
                raise NativeSourceFailure(code)
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
                workspace_memo.retain_native_result(self.workspace, owner, command.service_id)
                workspace_memo.record_native(self.workspace, owner, command.service_id)
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
            # Only a known native refusal code crosses; arbitrary exception text stays private.
            code = (
                exc.code
                if isinstance(exc, (NativeSourceFailure, storage_admission.StorageRefusal))
                else "native_source_failed"
            )
            with self.workspace.locked() as db:
                db.execute(
                    "UPDATE native_calls SET state='failed',native_error=? "
                    "WHERE owner=? AND service_id=? "
                    "AND state IN ('accepted','resolved','executing')",
                    (code, owner, command.service_id),
                )
            terminal(command, pb.NATIVE_SOURCE_STATE_FAILED, code=code)
        finally:
            if view:
                try:
                    source_views.abort_unfinished(self.workspace, owner, command)
                except Exception:
                    # Durable release intent is retried by the ordinary parent ACK path.
                    _LOG.warning("Native source view cleanup deferred to parent release")
            finish()
