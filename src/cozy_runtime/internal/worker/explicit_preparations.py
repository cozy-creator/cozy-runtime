"""Explicit preparations identified by their requester's operation_id (wire 73).

The RPC only waits: a disconnect detaches it, and the same operation_id attaches again.
CancelPreparation cancels this requester's own TensorFS token. A shared native flight
continues for every other requester, and landed objects stay.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import Future, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cozy_runtime.internal import fill
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_materialization, machine_model_defaults
from .package_prepare import PreparationRefusal

if TYPE_CHECKING:
    from .session import Worker

_OPERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


@dataclass(eq=False)
class _Operation:
    request: bytes
    models: list[tuple[str, str]]
    cancellation: machine_model_defaults.Cancellation
    future: Future[pb.PreparePackageSetResult] = field(default_factory=Future)


class ExplicitPreparations:
    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.lock = threading.Lock()
        self.operations: dict[str, _Operation] = {}

    def submit(self, request: pb.PreparePackageSetRequest) -> Future[pb.PreparePackageSetResult]:
        """Start the identified work, or attach to it while it is in flight."""
        if not _OPERATION_ID.fullmatch(request.operation_id):
            raise PreparationRefusal("preparation_operation_invalid", "invalid operation_id")
        raw = request.SerializeToString(deterministic=True)
        models = [
            (row.model, row.manifest)
            for row in machine_materialization.models(request.download_delegation)
        ]
        with self.lock:
            operation = self.operations.get(request.operation_id)
            if operation is not None:
                if operation.request != raw:
                    raise PreparationRefusal(
                        "preparation_operation_conflict",
                        "this operation_id is preparing a different request",
                    )
                return operation.future
            operation = _Operation(raw, models, fill.tensorfs_module().PullCancellation())
            if self.worker.transfers.cancelled:
                operation.cancellation.cancel()
            self.operations[request.operation_id] = operation
        threading.Thread(
            target=self._run,
            args=(request, operation),
            name="explicit-preparation",
            daemon=True,
        ).start()
        return operation.future

    def _run(self, request: pb.PreparePackageSetRequest, operation: _Operation) -> None:
        try:
            result = self.worker.prepare_package_set(request, operation.cancellation)
        except BaseException as exc:
            self._end(request.operation_id, operation)
            operation.future.set_exception(exc)
        else:
            self._end(request.operation_id, operation)
            operation.future.set_result(result)

    def _end(self, operation_id: str, operation: _Operation) -> None:
        with self.lock:
            if self.operations.get(operation_id) is operation:
                del self.operations[operation_id]

    def cancel(self, claim: pb.Claim, operation_id: str) -> pb.CancelPreparationResult:
        """Detach one requester; answer its last per-model rows before its token stops."""
        self.worker.authorize_workspace(claim)
        with self.lock:
            operation = self.operations.get(operation_id)
        if operation is None:
            return pb.CancelPreparationResult()
        rows = [
            row
            for row in machine_materialization.snapshot(self.worker)
            if (row.model.model, row.model.manifest) in operation.models
        ]
        operation.cancellation.cancel()
        return pb.CancelPreparationResult(found=True, model_progress=rows)

    def live(self) -> list[str]:
        with self.lock:
            return sorted(self.operations)

    def cancel_all(self) -> None:
        with self.lock:
            for operation in self.operations.values():
                operation.cancellation.cancel()

    def drain(self) -> None:
        with self.lock:
            pending = [operation.future for operation in self.operations.values()]
        wait(pending)
