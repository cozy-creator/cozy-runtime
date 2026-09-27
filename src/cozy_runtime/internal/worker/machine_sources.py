"""Route accepted machine calls through Runtime's existing native source service."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from cozy_runtime import canonical_json
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_models, workspace_byte_outputs, workspace_memo, workspace_sources
from .workspace import WorkspaceRefusal
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
}


class Sources:
    def __init__(self, worker: Worker):
        self.worker = worker
        self.lock = threading.Lock()
        self.selections: dict[str, pb.NativeSourceSelection] = {}

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
        if request.export in ("download_huggingface", "download_civitai"):
            with self.lock:
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
