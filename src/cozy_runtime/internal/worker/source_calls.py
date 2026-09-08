"""Asynchronous execution of the three accepted native source operations.

The existing control stream carries acceptance/results. Native work runs in a
separate killable process so heartbeats and parent cancellation remain responsive.
"""

from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, source_interfaces
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


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
    return "sha256:" + digest.hexdigest()


class SourceCalls:
    def __init__(
        self,
        workspace: Workspace,
        owner: Callable[[], str],
        send: Callable[[pb.WorkerFrame], None],
        current: Callable[[pb.NativeSourceCommand], bool],
        *,
        endpoints: dict[str, str] | None = None,
    ):
        self.workspace, self.owner, self.send, self.current = workspace, owner, send, current
        self.endpoints = endpoints or {}
        self.lock = threading.Lock()
        self.processes: dict[str, subprocess.Popen[bytes] | None] = {}
        self.parents: dict[str, str] = {}
        self.canceled: set[str] = set()

    def handle(self, command: pb.NativeSourceCommand) -> None:
        owner = self.owner()
        row = workspace_sources.accepted(self.workspace, owner, command)
        if command.phase == pb.NATIVE_SOURCE_PHASE_CANCEL:
            with self.lock:
                self.canceled.add(command.service_id)
                process = self.processes.get(command.service_id)
                if process is not None:
                    process.kill()
            with self.workspace.locked() as db:
                db.execute(
                    "UPDATE native_calls SET state='canceled' "
                    "WHERE owner=? AND service_id=? AND state<>'complete'",
                    (owner, command.service_id),
                )
            return
        if command.phase not in (pb.NATIVE_SOURCE_PHASE_RESOLVE, pb.NATIVE_SOURCE_PHASE_EXECUTE):
            raise WorkspaceRefusal("unknown native source execution phase")
        if row["state"] == "canceled":
            self._status(command, pb.NATIVE_SOURCE_STATE_CANCELED)
            return
        if row["state"] == "complete":
            self._status(
                command,
                pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                result=row["result"],
                receipt=row["native_receipt"],
                computation=row["computation_digest"],
            )
            return
        if (
            command.phase == pb.NATIVE_SOURCE_PHASE_EXECUTE
            and command.operation != pb.NATIVE_SOURCE_OPERATION_CONVERT
        ):
            workspace_sources.record_pin(self.workspace, owner, command, command.selection)
        copied = pb.NativeSourceCommand()
        copied.CopyFrom(command)
        with self.lock:
            if command.service_id in self.processes:
                return
            self.processes[command.service_id] = None
            self.parents[command.service_id] = command.parent_call.parent_request_id
        threading.Thread(
            target=self._run, args=(owner, copied, row), daemon=True, name="native-source"
        ).start()

    def cancel_parent(self, request_id: str) -> None:
        with self.lock:
            for service, parent in self.parents.items():
                if parent == request_id:
                    self.canceled.add(service)
                    process = self.processes.get(service)
                    if process is not None:
                        process.kill()

    def _status(
        self,
        command: pb.NativeSourceCommand,
        state: int,
        *,
        result: bytes = b"",
        receipt: bytes = b"",
        computation: bytes = b"",
        selection: pb.NativeSourceSelection | None = None,
        code: str = "",
        memo_hit: bool = False,
    ) -> None:
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
        )
        if selection is not None:
            status.selection.CopyFrom(selection)
        self.send(pb.WorkerFrame(native_source_status=status))

    def _run(self, owner: str, command: pb.NativeSourceCommand, row: dict[str, Any]) -> None:
        from cozy_runtime.internal.worker import workspace_memo

        released = False
        try:
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
                if command.operation == pb.NATIVE_SOURCE_OPERATION_CONVERT:
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
                hit = workspace_memo.lookup_native(
                    self.workspace, owner, command.service_id, computation
                )
                if hit is not None:
                    workspace_memo.record_reuse(self.workspace, owner, command.service_id)
                    self._status(
                        command,
                        pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                        result=hit["result"],
                        receipt=hit["native_receipt"],
                        computation=computation,
                        memo_hit=True,
                    )
                    return
                envelope["computation_digest"] = documents.spell(computation)
                envelope["writer_epoch"] = command.parent_call.parent_attempt_ordinal
            encoded = canonical_json.encode(envelope)
            if len(encoded) > 2 << 20:
                raise WorkspaceRefusal("source execution command exceeds bounded transfer")
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
            output, _ = process.communicate(encoded)
            with self.lock:
                canceled = command.service_id in self.canceled
            if canceled:
                self._status(command, pb.NATIVE_SOURCE_STATE_CANCELED)
                return
            if process.returncode or len(output) > 2 << 20:
                raise WorkspaceRefusal("native source execution ended without bounded result")
            answer = canonical_json.decode(output)
            if not answer.get("ok"):
                raise WorkspaceRefusal(str(answer.get("code", "native_source_failed")))
            if phase == "resolve":
                resolved = _selection(answer["selection"])
                workspace_sources.record_pin(self.workspace, owner, command, resolved)
                with self.lock:
                    self.processes.pop(command.service_id, None)
                    self.parents.pop(command.service_id, None)
                released = True
                self._status(command, pb.NATIVE_SOURCE_STATE_RESOLVED, selection=resolved)
                return
            result = canonical_json.encode(answer["result"])
            receipt = base64.b64decode(answer["native_receipt"], validate=True)
            workspace_sources.record_complete(
                self.workspace, owner, command, result, computation, receipt
            )
            workspace_memo.retain_native_result(self.workspace, owner, command.service_id)
            workspace_memo.record_native(self.workspace, owner, command.service_id)
            self._status(
                command,
                pb.NATIVE_SOURCE_STATE_SUCCEEDED,
                result=result,
                receipt=receipt,
                computation=computation,
            )
        except Exception:
            # No arbitrary native exception text or source/credential data crosses control.
            self._status(command, pb.NATIVE_SOURCE_STATE_FAILED, code="native_source_failed")
        finally:
            if not released:
                with self.lock:
                    self.processes.pop(command.service_id, None)
                    self.parents.pop(command.service_id, None)
