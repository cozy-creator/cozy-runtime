"""The worker-hosted control transport: the ONE seam between the worker and its owner.

The worker SERVES `WorkerControl` and the record-plane owner DIALS it (worker-protocol/01,
decisions #436/#454) — the 2026-08-25 re-landing inverted the old worker-dials arrangement.
The worker speaks `cozy.worker.v1` and nothing else; WHO carries those frames — a gRPC
server on a Unix socket (loopback TCP on win32), or a pair of in-process queues — is a
transport detail, and this module is the whole of it. §8's one-kernel rule is why the
in-memory host exists: `run` and `job` drive the SAME worker through the SAME per-stream
conversation and change nothing else.

Two durability classes, kept apart here exactly as they are on the wire:

* **Control** — the durable bidi stream. Frames are never shed; terminal authority lives
  only here. Each accepted stream runs ONE conversation (`ControlPlane.serve_stream`), and
  the worker's ownership fence — not this module — decides which stream is live.
* **WatchProgress** — bounded and LOSSY, opened by the owner on a physically separate
  connection. Saturating it can never block Control: a different RPC, a different queue,
  a different thread, no shared buffer for backpressure to cross.

Address discovery is a FILE HANDOFF: the host binds first (an ephemeral loopback port is
only knowable after bind) and atomically writes the dialable address to `control.addr` in
the worker root. The launcher that spawned this worker watches for that file; nothing is
ever guessed and no port is chosen by the dialer.
"""

from __future__ import annotations

import collections
import os
import queue
import re
import sys
import threading
from collections.abc import Callable, Iterator
from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

import grpc

from cozy_runtime.internal.numerical_environment import NumericalEnvironmentRefusal
from cozy_runtime.internal.worker.derived_retention import RetentionRefusal
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.model_source_prepare import ModelSourceRefusal
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal
from cozy_runtime.internal.worker.servicer_context import ServicerContext
from cozy_runtime.internal.worker.workspace_rpc import Service as WorkspaceService
from cozy_runtime.internal.worker.workspace_rpc import WorkspaceRPC
from cozy_runtime.protocol import MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

if TYPE_CHECKING:
    from .session import Worker


class ControlPlane(Protocol):
    """What the worker exposes to a host: one conversation per accepted stream, and the
    lossy watch. The worker's fence (record_owner_epoch / control_stream_epoch / worker_boot_id)
    lives behind `serve_stream`; hosts route bytes and decide nothing."""

    def serve_stream(
        self, inbound: Iterator[pb.RecordOwnerFrame], send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """Run one control stream to its end. Returns when the stream closes or is fenced."""

    def watch(
        self,
        opened: pb.ProgressOpen,
        *,
        on_cancel: Callable[[Callable[[], None]], bool] | None = None,
    ) -> Iterator[pb.AttemptProgress]:
        """The lossy lane for one WatchProgress call. Ends when the stream is fenced."""


class ControlHost(Protocol):
    """One way of hosting the plane. `persistent` says whether losing every stream ends
    the worker: a gRPC host outlives its streams (an owner reconnects and re-claims); the
    in-memory host IS its one stream, and the stream ending is the run ending."""

    persistent: bool

    def serve(self, plane: ControlPlane) -> None:
        """Block until `stop()`. Streams are handled on the host's own threads."""

    def stop(self) -> None: ...


def write_addr(addr_file: Path, address: str) -> None:
    """Atomically publish the dialable address. The launcher reads exactly this file."""
    addr_file.parent.mkdir(parents=True, exist_ok=True)
    staged = addr_file.with_suffix(".staging")
    staged.write_text(address + "\n")
    os.replace(staged, addr_file)


# --------------------------------------------------------------------------- gRPC


class _Servicer(MachineExecutionRPC):
    # Deliberately not a subclass: the generated base is untyped (Any), and registration
    # only needs the two method names.
    def __init__(self, plane: ControlPlane) -> None:
        super().__init__(cast("Worker", plane))
        self.plane = plane
        self._readers: list[threading.Thread] = []
        self._readers_lock = threading.Lock()

    def wait_closed(self) -> None:
        """Called after RPC handlers stop, so no reader can register after this snapshot."""
        with self._readers_lock:
            readers = list(self._readers)
        for reader in readers:
            reader.join()

    def Control(
        self, request_iterator: Iterator[pb.RecordOwnerFrame], context: ServicerContext
    ) -> Iterator[pb.WorkerFrame]:
        out: queue.Queue[pb.WorkerFrame | None] = queue.Queue()

        def conversation() -> None:
            try:
                self.plane.serve_stream(request_iterator, out.put)
            except grpc.RpcError:
                if context.is_active():
                    raise
            finally:
                out.put(None)

        reader = threading.Thread(target=conversation, daemon=True, name="control-stream")
        with self._readers_lock:
            self._readers = [thread for thread in self._readers if thread.is_alive()]
            self._readers.append(reader)
            reader.start()
        while True:
            frame = out.get()
            if frame is None:
                return
            yield frame

    def WatchProgress(
        self, request: pb.ProgressOpen, context: ServicerContext
    ) -> Iterator[pb.AttemptProgress]:
        yield from self.plane.watch(request, on_cancel=context.add_callback)


class RuntimeWeightsHost(Protocol):
    """A bounded native object upload; custody observations use WorkerControl."""

    def upload(self, request: pb.WeightsUploadRequest) -> pb.WeightsUploadResult: ...


class _RuntimeWeightsServicer:
    # Not a subclass for the same reason as `_Servicer`: the generated base is untyped.
    def __init__(self, weights: RuntimeWeightsHost) -> None:
        self.weights = weights

    def Upload(
        self, request: pb.WeightsUploadRequest, context: ServicerContext
    ) -> pb.WeightsUploadResult:
        return self.weights.upload(request)


class GrpcControlHost:
    """The production host: one gRPC server on the worker's `host:port` (`host:0` binds an
    ephemeral port). The REAL bound address is published to `addr_file` after bind."""

    persistent = True

    def __init__(
        self,
        listen: str,
        addr_file: Path,
        *,
        tls_cert: Path | None = None,
        tls_key: Path | None = None,
        on_bound: Callable[[str], None] | None = None,
        preparer: Callable[[pb.PreparePackageSetRequest], pb.PreparePackageSetResult] | None = None,
        model_source_preparer: (
            Callable[[pb.PrepareModelSourceRequest], pb.PrepareModelSourceResult] | None
        ) = None,
        model_source_releaser: Callable[[str], None] | None = None,
        derived_retainer: Callable[[pb.DerivedRetentionRequest, bool], pb.DerivedRetentionResult]
        | None = None,
        derived_result_releaser: Callable[
            [pb.DerivedResultReleaseRequest], pb.DerivedResultReleaseResult
        ]
        | None = None,
        checkpoint_pager: (
            Callable[[pb.CheckpointPageRequest], pb.CheckpointPageResult] | None
        ) = None,
        checkpoint_transfer: (
            Callable[[pb.CheckpointTransferRequest], pb.CheckpointTransferStatus] | None
        ) = None,
        weights_checkpoint_validator: (
            Callable[[pb.ValidateWeightsCheckpointRequest], pb.ValidateWeightsCheckpointResult]
            | None
        ) = None,
        local_package_preparer: (
            Callable[[pb.PrepareLocalPackageRequest], pb.PreparePackageSetResult] | None
        ) = None,
        unpublished_placement_preparer: (
            Callable[[pb.PreparePrivatePlacementRequest], pb.PreparePackageSetResult] | None
        ) = None,
        numerical_environment: Callable[[], bytes] | None = None,
        store_collector: Callable[[], pb.CollectStoreGarbageResult] | None = None,
        weights: RuntimeWeightsHost | None = None,
        workspace_service: WorkspaceService | None = None,
    ) -> None:
        self.listen = listen
        self.addr_file = addr_file
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.on_bound = on_bound
        self.preparer = preparer
        self.model_source_preparer = model_source_preparer
        self.model_source_releaser = model_source_releaser
        self.derived_retainer = derived_retainer
        self.derived_result_releaser = derived_result_releaser
        self.numerical_environment = numerical_environment
        self.store_collector = store_collector
        self.workspace_service = workspace_service
        self.checkpoint_pager = checkpoint_pager
        self.checkpoint_transfer = checkpoint_transfer
        self.weights_checkpoint_validator = weights_checkpoint_validator
        self.local_package_preparer = local_package_preparer
        self.unpublished_placement_preparer = unpublished_placement_preparer
        #: The loopback weights seam (proto-025), served only when a pod supervisor is the
        #: peer. Absent on a local run, where there is no supervisor to open `Exchange`.
        self.weights = weights
        # grpcio is untyped in the check venv, so mypy sees this as Any.
        self.server: grpc.Server | None = None
        self.stopped = threading.Event()

    def serve(self, plane: ControlPlane) -> None:
        # The worker is a serial machine behind one device lease; a small thread pool is
        # about stream lifetimes (control + watches), never about parallel dispatch.

        executor = futures.ThreadPoolExecutor(max_workers=8)
        control = _Servicer(plane)
        self.server = server = grpc.server(executor)
        pb_grpc.add_WorkerControlServicer_to_server(control, server)
        loopback_services = (
            self.workspace_service is not None
            or self.preparer is not None
            or self.model_source_preparer is not None
            or self.model_source_releaser is not None
            or self.derived_retainer is not None
            or self.derived_result_releaser is not None
            or self.numerical_environment is not None
            or self.store_collector is not None
            or self.checkpoint_pager is not None
            or self.checkpoint_transfer is not None
            or self.local_package_preparer is not None
            or self.unpublished_placement_preparer is not None
            or self.weights is not None
            or self.weights_checkpoint_validator is not None
        )
        if loopback_services and not _loopback_preparation_listener(self.listen):
            raise RuntimeError("the loopback Runtime services can only bind a loopback listener")
        # Every worker exposes the static protocol probe, including local workers
        # without optional preparation handlers. Absent handlers still refuse calls.
        pb_grpc.add_RuntimePreparationServicer_to_server(
            _PreparationServicer(
                self.preparer,
                self.model_source_preparer,
                self.local_package_preparer,
                self.unpublished_placement_preparer,
                self.checkpoint_pager,
                self.checkpoint_transfer,
                self.weights_checkpoint_validator,
                self.model_source_releaser,
                self.derived_retainer,
                self.derived_result_releaser,
                self.numerical_environment,
                self.store_collector,
                self.workspace_service,
            ),
            server,
        )
        if self.weights is not None:
            # Loopback-only, exactly like RuntimePreparation: the supervisor dials it on the
            # same 127.0.0.1 listener and it is never reachable from the external leg.
            pb_grpc.add_RuntimeWeightsServicer_to_server(
                _RuntimeWeightsServicer(self.weights), server
            )
        if self.tls_cert is not None and self.tls_key is not None:
            # THE REMOTE LEG (cl-015 / #445): one-way TLS with the worker's own cert —
            # self-signed on a pod, the owner pins it from the provision record — plus the
            # RecordOwner's signed Claim. Never mTLS (#445).
            credentials = grpc.ssl_server_credentials(
                [(self.tls_key.read_bytes(), self.tls_cert.read_bytes())]
            )
            port = server.add_secure_port(self.listen, credentials)
            host = self.listen.rsplit(":", 1)[0]
            address = f"{host}:{port}"
        else:
            port = server.add_insecure_port(self.listen)
            host = self.listen.rsplit(":", 1)[0]
            address = f"{host}:{port}"
        try:
            server.start()
            write_addr(self.addr_file, address)
            if self.on_bound is not None:
                self.on_bound(address)
            self.stopped.wait()
        finally:
            server.stop(grace=0).wait()
            executor.shutdown(wait=True)
            control.wait_closed()
            self.stopped.set()

    def stop(self) -> None:
        if self.server is not None:
            # A control reader may request stop. Only serve() joins readers and RPC
            # handlers, after this caller can return and release its own conversation.
            self.server.stop(grace=0)
        self.stopped.set()


_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}")


#: TensorFS refusals that describe the store's momentary state, not the request.
_TRANSIENT_STORE_CODES = frozenset(
    {
        "CAPACITY_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "DURABILITY_UNPROVEN",
        "FD_HEADROOM",
        "IO_FAILED",
        "LOCK_CONTENDED",
        "STORE_BUSY",
    }
)


def _custody_failure(
    context: ServicerContext, exc: BaseException, what: str, *permanent: type[BaseException]
) -> NoReturn:
    """Refuse only what retrying cannot change; everything else is UNAVAILABLE (retry).

    A typed Runtime refusal and a TensorFS refusal about the request's data are verdicts. A
    busy or momentarily failing store, an OSError, or any unexpected failure is not: the
    owner retries with its own backoff. Details stay generic; the type names the fault.
    """
    # A TensorFS refusal implies its module is loaded; the worker never imports TensorFS (cr-067).
    store_errors = sys.modules.get("tensorfs.errors")
    store_code = (
        exc.code if store_errors is not None and isinstance(exc, store_errors.Refusal) else None
    )
    if isinstance(exc, permanent) or (
        store_code is not None and store_code not in _TRANSIENT_STORE_CODES
    ):
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"{what} was refused")
    context.abort(
        grpc.StatusCode.UNAVAILABLE,
        f"{what} is temporarily unavailable ({store_code or type(exc).__name__}); retry",
    )


def _refuse(context: ServicerContext, exc: BaseException) -> NoReturn:
    """Answer one failed preparation call, saying WHICH KIND of failure it was.

    The host on the other end of this loopback journals a refusal as the permanent answer
    to an immutable request, and it decides whether to do that from the status code here.
    So the status is not decoration: it is the difference between "this request cannot be
    prepared" and "I failed at it this time".

    Every arm of this servicer used to be `except Exception: abort(FAILED_PRECONDITION,
    str(exc))`, which said "verdict" about everything. A missing scratch directory, a
    TensorFS store still opening, a lease revoked mid-read and a genuinely incompatible
    package set all left this process under the one code whose meaning is "do not retry
    until the system state is fixed" — and the host wrote it down. On 2026-09-03 that
    turned a transient fault into a pod that answered the identical request from a cached
    refusal, instantly and with no work attempted, for its whole billed life.

    Three answers, and the middle one is the one that did not exist:

      FAILED_PRECONDITION  a typed PreparationRefusal. This request is refused, and the
                           identical bytes cannot be answered differently later.
      INTERNAL             this Runtime failed AT the request. The owner is answered so
                           nothing redials, and nothing is written down, so the next ask
                           actually runs.
      UNAVAILABLE          the work could not start or was cut. No answer at all; ask again.

    The polarity is an ALLOW-LIST of verdicts defaulting to not-a-verdict, which is the
    opposite default from a retry classifier and correct for the same reason: the safe
    default is the one whose mistake is recoverable.
    """
    if isinstance(exc, PreparationRefusal):
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:1024])
    # The type is part of the diagnosis and was being discarded: `PermissionError` and
    # `FileNotFoundError` send an operator to two different places, and `str(exc)` alone
    # frequently names neither. A typed failure's own stable code rides beside it.
    code = getattr(exc, "code", None)
    if isinstance(code, str) and _ERROR_CODE.fullmatch(code):
        context.set_trailing_metadata((("cozy-error-code", code),))
    context.abort(grpc.StatusCode.INTERNAL, f"{type(exc).__name__}: {exc}"[:1024])


class _PreparationServicer(WorkspaceRPC):
    def __init__(
        self,
        package_handler: (
            Callable[[pb.PreparePackageSetRequest], pb.PreparePackageSetResult] | None
        ),
        model_source_handler: (
            Callable[[pb.PrepareModelSourceRequest], pb.PrepareModelSourceResult] | None
        ),
        local_package_handler: (
            Callable[[pb.PrepareLocalPackageRequest], pb.PreparePackageSetResult] | None
        ),
        unpublished_placement_handler: (
            Callable[[pb.PreparePrivatePlacementRequest], pb.PreparePackageSetResult] | None
        ),
        checkpoint_pager: (
            Callable[[pb.CheckpointPageRequest], pb.CheckpointPageResult] | None
        ) = None,
        checkpoint_transfer: (
            Callable[[pb.CheckpointTransferRequest], pb.CheckpointTransferStatus] | None
        ) = None,
        weights_checkpoint_validator: (
            Callable[[pb.ValidateWeightsCheckpointRequest], pb.ValidateWeightsCheckpointResult]
            | None
        ) = None,
        model_source_releaser: Callable[[str], None] | None = None,
        derived_retainer: Callable[[pb.DerivedRetentionRequest, bool], pb.DerivedRetentionResult]
        | None = None,
        derived_result_releaser: Callable[
            [pb.DerivedResultReleaseRequest], pb.DerivedResultReleaseResult
        ]
        | None = None,
        numerical_environment: Callable[[], bytes] | None = None,
        store_collector: Callable[[], pb.CollectStoreGarbageResult] | None = None,
        workspace_service: WorkspaceService | None = None,
    ) -> None:
        self.package_handler = package_handler
        self.model_source_handler = model_source_handler
        self.local_package_handler = local_package_handler
        self.unpublished_placement_handler = unpublished_placement_handler
        self.checkpoint_pager = checkpoint_pager
        self.checkpoint_transfer = checkpoint_transfer
        self.weights_checkpoint_validator = weights_checkpoint_validator
        self.model_source_releaser = model_source_releaser
        self.derived_retainer = derived_retainer
        self.derived_result_releaser = derived_result_releaser
        self.numerical_environment = numerical_environment
        self.store_collector = store_collector
        self.workspace_service = workspace_service

    def ProtocolInfo(
        self, _request: pb.ProtocolInfoRequest, _context: ServicerContext
    ) -> pb.ProtocolInfoResult:
        return pb.ProtocolInfoResult(
            wire_minor=WIRE_MINOR,
            minimum_wire_minor=MIN_COMPATIBLE_WIRE_MINOR,
        )

    def NumericalEnvironment(
        self, _request: pb.NumericalEnvironmentRequest, context: ServicerContext
    ) -> pb.NumericalEnvironmentResult:
        if self.numerical_environment is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "numerical environment is unavailable")
        try:
            digest = self.numerical_environment()
            if len(digest) != 32:
                raise NumericalEnvironmentRefusal("invalid numerical identity")
            return pb.NumericalEnvironmentResult(digest=digest)
        except Exception as exc:
            _custody_failure(
                context, exc, "numerical environment identity", NumericalEnvironmentRefusal
            )

    def ValidateWeightsCheckpoint(
        self, request: pb.ValidateWeightsCheckpointRequest, context: ServicerContext
    ) -> pb.ValidateWeightsCheckpointResult:
        if self.weights_checkpoint_validator is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "weights checkpoint validation unavailable"
            )
        return self.weights_checkpoint_validator(request)

    def CheckpointPage(
        self, request: pb.CheckpointPageRequest, context: ServicerContext
    ) -> pb.CheckpointPageResult:
        if self.checkpoint_pager is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "source checkpoint page unavailable")
        return self.checkpoint_pager(request)

    def CheckpointTransfer(
        self, request: pb.CheckpointTransferRequest, context: ServicerContext
    ) -> pb.CheckpointTransferStatus:
        if self.checkpoint_transfer is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "source checkpoint transfer unavailable")
        return self.checkpoint_transfer(request)

    def PreparePackageSet(
        self, request: pb.PreparePackageSetRequest, context: ServicerContext
    ) -> pb.PreparePackageSetResult:
        return _prepared(context, self.package_handler, request, "package preparation")

    def PrepareModelSource(
        self, request: pb.PrepareModelSourceRequest, context: ServicerContext
    ) -> pb.PrepareModelSourceResult:
        return _prepared(context, self.model_source_handler, request, "model source preparation")

    def ReleaseModelSource(
        self, request: pb.ReleaseModelSourceRequest, context: ServicerContext
    ) -> pb.ReleaseModelSourceResult:
        if self.model_source_releaser is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "source release is unavailable")
        try:
            self.model_source_releaser(request.operation_id)
        except Exception as exc:
            _custody_failure(context, exc, "source release", ModelSourceRefusal)
        return pb.ReleaseModelSourceResult(operation_id=request.operation_id, released=True)

    def RetainDerivedResult(
        self, request: pb.DerivedRetentionRequest, context: ServicerContext
    ) -> pb.DerivedRetentionResult:
        return self._derived_retention(request, context, False)

    def CollectStoreGarbage(
        self, _request: pb.CollectStoreGarbageRequest, context: ServicerContext
    ) -> pb.CollectStoreGarbageResult:
        if self.store_collector is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "store collection is unavailable")
        try:
            return self.store_collector()
        except Exception as exc:
            _custody_failure(context, exc, "store collection")

    def ReleaseDerivedResult(
        self, request: pb.DerivedResultReleaseRequest, context: ServicerContext
    ) -> pb.DerivedResultReleaseResult:
        if self.derived_result_releaser is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "derived result release is unavailable")
        try:
            return self.derived_result_releaser(request)
        except Exception as exc:
            _custody_failure(context, exc, "derived result release", RetentionRefusal)

    def ReleaseDerivedRetention(
        self, request: pb.DerivedRetentionRequest, context: ServicerContext
    ) -> pb.DerivedRetentionResult:
        return self._derived_retention(request, context, True)

    def _derived_retention(
        self, request: pb.DerivedRetentionRequest, context: ServicerContext, release: bool
    ) -> pb.DerivedRetentionResult:
        if self.derived_retainer is None:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "derived retention is unavailable")
        try:
            return self.derived_retainer(request, release)
        except Exception as exc:
            _custody_failure(context, exc, "derived retention", RetentionRefusal)

    def PrepareLocalPackage(
        self, request: pb.PrepareLocalPackageRequest, context: ServicerContext
    ) -> pb.PreparePackageSetResult:
        return _prepared(context, self.local_package_handler, request, "local package preparation")

    def PreparePrivatePlacement(
        self, request: pb.PreparePrivatePlacementRequest, context: ServicerContext
    ) -> pb.PreparePackageSetResult:
        return _prepared(
            context,
            self.unpublished_placement_handler,
            request,
            "unpublished package placement preparation",
        )


def _prepared[Q, R](
    context: ServicerContext, handler: Callable[[Q], R] | None, request: Q, what: str
) -> R:
    if handler is None:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, f"{what} is unavailable")
    try:
        return handler(request)
    except Exception as exc:
        _refuse(context, exc)


def _loopback_preparation_listener(listen: str) -> bool:
    return listen.rsplit(":", 1)[0] in {"127.0.0.1", "localhost", "::1", "[::1]"}


# --------------------------------------------------------------------------- in-memory


class InMemoryControlHost:
    """§8's local control adapter: ONE in-process stream driven by the local owner.

    `deliver` feeds OwnerFrames; `on_frame` observes every WorkerFrame; `close` half-closes
    the stream, which — this host being non-persistent — ends the run, the exact analogue
    of an in-memory owner going away. Progress frames reach `on_progress` directly: the
    lossy lane's separation is a queue-and-thread fact in the gRPC host, and here the
    driver IS the only consumer.
    """

    persistent = False

    def __init__(self) -> None:
        self.inbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
        self.sent: list[pb.WorkerFrame] = []
        self.on_frame: Callable[[pb.WorkerFrame], None] | None = None
        self.on_progress: Callable[[pb.AttemptProgress], None] | None = None
        self.closed = threading.Event()
        self._plane: ControlPlane | None = None

    def serve(self, plane: ControlPlane) -> None:
        self._plane = plane
        plane.serve_stream(self._iter(), self._send)

    def stop(self) -> None:
        self.close()

    # ---------------------------------------------------------------- driving

    def deliver(self, frame: pb.RecordOwnerFrame) -> None:
        if not self.closed.is_set():
            self.inbound.put(frame)

    def close(self) -> None:
        if not self.closed.is_set():
            self.closed.set()
            self.inbound.put(None)

    def publish_progress(self, frame: pb.AttemptProgress) -> None:
        if self.on_progress is not None:
            self.on_progress(frame)

    def _send(self, frame: pb.WorkerFrame) -> None:
        self.sent.append(frame)
        if self.on_frame is not None:
            self.on_frame(frame)

    def _iter(self) -> Iterator[pb.RecordOwnerFrame]:
        while True:
            frame = self.inbound.get()
            if frame is None:
                return
            yield frame


# --------------------------------------------------------------------------- watch fanout


class WatchFanout:
    """The lossy lane's registry: per-watch bounded rings, oldest shed first.

    Publishing never blocks — a slow watcher sheds ITS OWN oldest frames and nobody
    else's; a watcher bound to a fenced control epoch ends. The counters are the
    worker's own accounting (`progress_sent` / `progress_shed`), read at shutdown.
    """

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.lock = threading.Lock()
        self.watchers: list[_Watch] = []
        self.sent = 0
        self.shed = 0
        #: an in-process observer for the local door (§8): the one-shot driver IS the only
        #: watcher there, and opening a gRPC watch on itself would be ceremony.
        self.tap: Callable[[pb.AttemptProgress], None] | None = None

    def publish(self, frame: pb.AttemptProgress) -> None:
        if self.tap is not None:
            self.tap(frame)
            self.sent += 1
        with self.lock:
            live = [w for w in self.watchers if not w.ended.is_set()]
            self.watchers = live
            for watch in live:
                if watch.request_id and watch.request_id != frame.request_id:
                    continue
                with watch.cond:
                    if len(watch.ring) == self.depth:
                        self.shed += 1
                    watch.ring.append(frame)
                    watch.cond.notify()

    def open(self, request_id: str, epoch: int) -> _Watch:
        watch = _Watch(request_id=request_id, epoch=epoch, depth=self.depth)
        with self.lock:
            self.watchers.append(watch)
        return watch

    def close(self, watch: _Watch) -> None:
        watch.end()
        with self.lock:
            if watch in self.watchers:
                self.watchers.remove(watch)

    def end_epoch(self, epoch: int) -> None:
        """Fence: every watch bound to a superseded control stream ends."""
        with self.lock:
            for watch in self.watchers:
                if watch.epoch <= epoch:
                    watch.end()

    def end_all(self) -> None:
        with self.lock:
            for watch in self.watchers:
                watch.end()


class _Watch:
    def __init__(self, *, request_id: str, epoch: int, depth: int) -> None:
        self.request_id = request_id
        self.epoch = epoch
        self.ring: collections.deque[pb.AttemptProgress] = collections.deque(maxlen=depth)
        self.cond = threading.Condition()
        self.ended = threading.Event()

    def end(self) -> None:
        self.ended.set()
        with self.cond:
            self.cond.notify_all()

    def frames(self, counted: WatchFanout) -> Iterator[pb.AttemptProgress]:
        try:
            while True:
                with self.cond:
                    while not self.ring and not self.ended.is_set():
                        self.cond.wait()
                    if self.ring:
                        frame = self.ring.popleft()
                    elif self.ended.is_set():
                        return
                    else:  # pragma: no cover - the loop above re-checks
                        continue
                counted.sent += 1
                yield frame
        finally:
            counted.close(self)
