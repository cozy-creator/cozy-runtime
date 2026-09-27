"""Route accepted machine calls through Runtime's existing native source service."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING

from cozy_runtime import canonical_json
from cozy_runtime.internal import source_interfaces
from cozy_runtime.protocol import worker_pb2 as pb

from . import (
    commit_files,
    machine_models,
    workspace_byte_outputs,
    workspace_memo,
    workspace_sources,
)
from .workspace import Journal, WorkspaceRefusal
from .workspace_calls import Call
from .workspace_calls import Calls as JournalCalls

if TYPE_CHECKING:
    from .session import Worker

OPERATIONS = {
    "download_huggingface": pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
    "download_civitai": pb.NATIVE_SOURCE_OPERATION_CIVITAI,
    "convert_cozytensors": pb.NATIVE_SOURCE_OPERATION_CONVERT,
    "source_files": pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES,
    "commit_file": pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE,
    # An upload is a provider source operation whose result is a publication.
    "upload_huggingface": pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
    "upload_civitai": pb.NATIVE_SOURCE_OPERATION_CIVITAI,
}
PROVIDER_EXPORTS = ("download_huggingface", "download_civitai", *source_interfaces.UPLOADS)
PROVIDERS = {
    pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE: "huggingface",
    pb.NATIVE_SOURCE_OPERATION_CIVITAI: "civitai",
}


class Sources:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.lock = threading.Lock()
        self.selections: dict[str, pb.NativeSourceSelection] = {}
        # The owner's provider credentials per (owner, root execution), in memory only.
        self.credentials: dict[tuple[str, str], dict[pb.NativeSourceOperation, str]] = {}

    def hold(self, owner: str, request: str, given: Iterable[pb.SourceCredential]) -> None:
        values = {row.provider: row.credential for row in given if row.provider in PROVIDERS}
        if values:
            with self.lock:
                self.credentials.setdefault((owner, request), {}).update(values)

    def declare(self, owner: str, request: str) -> None:
        """Journal the providers this accepted execution holds a credential for."""
        with self.lock:
            held = {PROVIDERS[p] for p in self.credentials.get((owner, request), {})}
        if held and (workspace := self.worker.workspace) is not None:
            with workspace.locked() as db:
                names = ",".join(sorted(held | self._declared(db, owner, request)))
                db.execute(
                    "UPDATE executions SET source_providers=? WHERE owner=? AND request=?",
                    (names, owner, request),
                )

    def forget(self, owner: str, request: str) -> None:
        with self.lock:
            self.credentials.pop((owner, request), None)

    def awaiting(self, owner: str, request: str) -> list[pb.NativeSourceOperation]:
        """Declared providers whose credential this process no longer holds (a restart)."""
        assert self.worker.workspace is not None
        with self.worker.workspace.locked() as db:
            declared = self._declared(db, owner, request)
        with self.lock:
            held = self.credentials.get((owner, request), {})
        return [p for p, name in PROVIDERS.items() if name in declared and p not in held]

    @staticmethod
    def _declared(db: Journal, owner: str, request: str) -> set[str]:
        row = db.execute(
            "SELECT source_providers FROM executions WHERE owner=? AND request=?", (owner, request)
        ).fetchone()
        return set(filter(None, row[0].split(","))) if row is not None else set()

    def observe(self, status: pb.NativeSourceStatus) -> None:
        # Delivery URLs stay in memory. The source service already journals the exact
        # immutable pin and compares it when a restarted machine resolves URLs again.
        with self.lock:
            if status.state == pb.NATIVE_SOURCE_STATE_RESOLVED:
                selected = pb.NativeSourceSelection()
                selected.CopyFrom(status.selection)
                self.selections[status.service_id] = selected
            else:
                self.selections.pop(status.service_id, None)

    def command(self, owner: str, request: pb.ChildCallRequest) -> pb.NativeSourceCommand:

        command = pb.NativeSourceCommand(
            service_id=workspace_sources.identity(
                owner, request.parent_request_id, request.call_index
            ),
            parent_call=request,
            operation=OPERATIONS[request.export],
            phase=pb.NATIVE_SOURCE_PHASE_EXECUTE,
        )
        self.worker.stamp(command)
        return command

    def drive(self, owner: str, call: Call, request: pb.ChildCallRequest) -> None:
        service, workspace = self.worker.source_calls, self.worker.workspace
        if service is None or workspace is None:
            raise WorkspaceRefusal("native source execution requires its retained workspace")
        command = self.command(owner, request)
        if not call.prepared:
            JournalCalls(workspace).freeze(owner, call, canonical_json.encode({"kind": "source"}))
        with workspace.locked() as db:
            row = db.execute(
                "SELECT state,parent_ordinal FROM native_calls WHERE owner=? AND service_id=?",
                (owner, command.service_id),
            ).fetchone()
            if (
                row is not None
                and row["parent_ordinal"] == request.parent_attempt_ordinal
                and row["state"] in ("complete", "failed", "stopped", "released", "releasing")
            ):
                return
        if request.export in PROVIDER_EXPORTS:
            assert self.worker.executions is not None
            root = self.worker.executions.scheduling_root(owner, request.parent_request_id)[0]
            if command.operation in self.awaiting(owner, root):
                return  # the owner's credential is supplied again before this call runs
            with self.lock:
                command.credential = self.credentials.get((owner, root), {}).get(
                    command.operation, ""
                )
                selection = self.selections.get(command.service_id)
            if selection is None:
                command.phase = pb.NATIVE_SOURCE_PHASE_RESOLVE
            else:
                command.selection.CopyFrom(selection)
        service.handle(command)

    def cancel(self, owner: str, request: pb.ChildCallRequest) -> None:
        if self.worker.source_calls is not None:
            command = self.command(owner, request)
            command.phase = pb.NATIVE_SOURCE_PHASE_CANCEL
            self.worker.source_calls.handle(command)

    def result(
        self, owner: str, call: Call, request: pb.ChildCallRequest, result: pb.ChildCallResult
    ) -> pb.ChildCallResult:
        workspace = self.worker.workspace
        assert workspace is not None
        command = self.command(owner, request)
        with workspace.locked() as db:
            row = db.execute(
                "SELECT * FROM native_calls WHERE owner=? AND service_id=?",
                (owner, command.service_id),
            ).fetchone()
        if row is None or row["parent_ordinal"] != request.parent_attempt_ordinal:
            return result
        if row["state"] in ("failed", "stopped"):
            result.state = pb.CHILD_CALL_STATE_FAILED
            result.safe_code = row["native_error"] or "native_source_failed"
            if self.worker.source_calls is not None:
                result.safe_detail = self.worker.source_calls.failure_detail(command.service_id)
            if (
                not result.safe_detail
                and result.safe_code == commit_files.UnsupportedCommitRequest.code
            ):
                result.safe_detail = commit_files.unsupported_detail(row["accepted"])
            return result
        if row["state"] != "complete":
            return result
        raw = row["result"]
        if request.export in ("download_huggingface", "download_civitai", "convert_cozytensors"):
            workspace_memo.retain_native_result(workspace, owner, command.service_id)
        if request.export == "convert_cozytensors":
            machine_models.retain_result(
                workspace,
                owner,
                call.parent_request,
                f"call.{call.call_index}.result",
                canonical_json.decode(raw),
                {"input": "model"},
            )
        if request.export in ("source_files", "commit_file"):
            native, _, _ = workspace_byte_outputs.replay_native(workspace, owner, command)
            grant = machine_models.retain_bytes(
                workspace, owner, call.parent_request, f"call.{call.call_index}.bytes", native
            )
            grant.output_id = "files" if request.export == "source_files" else "file"
            result.byte_result_grants.append(grant)
        if not call.result:
            JournalCalls(workspace).observe_result(owner, call, raw)
        result.state, result.result_canonical_bytes = pb.CHILD_CALL_STATE_SUCCEEDED, raw
        return result
