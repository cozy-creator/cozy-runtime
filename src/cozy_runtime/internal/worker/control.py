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
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc


class ControlPlane(Protocol):
    """What the worker exposes to a host: one conversation per accepted stream, and the
    lossy watch. The worker's fence (owner_epoch / control_generation / worker_boot_id)
    lives behind `serve_stream`; hosts route bytes and decide nothing."""

    def serve_stream(
        self, inbound: Iterator[pb.RecordOwnerFrame], send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """Run one control stream to its end. Returns when the stream closes or is fenced."""

    def watch(self, opened: pb.ProgressOpen) -> Iterator[pb.AttemptProgress]:
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


class _Servicer:
    # Deliberately not a subclass: the generated base is untyped (Any), and registration
    # only needs the two method names.
    def __init__(self, plane: ControlPlane) -> None:
        self.plane = plane

    def Control(self, request_iterator: Any, context: Any) -> Any:
        out: queue.Queue[pb.WorkerFrame | None] = queue.Queue()

        def conversation() -> None:
            try:
                self.plane.serve_stream(request_iterator, out.put)
            finally:
                out.put(None)

        threading.Thread(target=conversation, daemon=True, name="control-stream").start()
        while True:
            frame = out.get()
            if frame is None:
                return
            yield frame

    def WatchProgress(self, request: pb.ProgressOpen, context: Any) -> Any:
        yield from self.plane.watch(request)


class GrpcControlHost:
    """The production host: one gRPC server on the worker's own local binding.

    `listen` is `unix:<path>` where the OS has a Unix socket worth having, or a loopback
    `host:port` on win32 (cr-020 item 4; `host:0` binds an ephemeral port). Either way the
    REAL bound address is published to `addr_file` after bind, which is the whole discovery
    contract — the Windows ephemeral port is only knowable here.
    """

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
    ) -> None:
        self.listen = listen
        self.addr_file = addr_file
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.on_bound = on_bound
        self.preparer = preparer
        self.model_source_preparer = model_source_preparer
        self.server: Any = None
        self.stopped = threading.Event()

    def serve(self, plane: ControlPlane) -> None:
        # The worker is a serial machine behind one device lease; a small thread pool is
        # about stream lifetimes (control + watches), never about parallel dispatch.
        from concurrent import futures

        import grpc

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        pb_grpc.add_WorkerControlServicer_to_server(_Servicer(plane), self.server)
        if self.preparer is not None or self.model_source_preparer is not None:
            if not _loopback_preparation_listener(self.listen):
                raise RuntimeError("RuntimePreparation can only bind a loopback listener")
            pb_grpc.add_RuntimePreparationServicer_to_server(
                _PreparationServicer(self.preparer, self.model_source_preparer), self.server
            )
        if self.listen.startswith("unix:"):
            self.server.add_insecure_port(self.listen)
            address = self.listen
        elif self.tls_cert is not None and self.tls_key is not None:
            # THE REMOTE LEG (cl-015 / #445): one-way TLS with the worker's own cert —
            # self-signed on a pod, the owner pins it from the provision record — plus the
            # bearer owner token verified at Claim. Never mTLS (#445).
            credentials = grpc.ssl_server_credentials(
                [(self.tls_key.read_bytes(), self.tls_cert.read_bytes())]
            )
            port = self.server.add_secure_port(self.listen, credentials)
            host = self.listen.rsplit(":", 1)[0]
            address = f"{host}:{port}"
        else:
            port = self.server.add_insecure_port(self.listen)
            host = self.listen.rsplit(":", 1)[0]
            address = f"{host}:{port}"
        self.server.start()
        write_addr(self.addr_file, address)
        if self.on_bound is not None:
            try:
                self.on_bound(address)
            except Exception:
                self.server.stop(grace=0)
                self.stopped.set()
                raise
        self.stopped.wait()

    def stop(self) -> None:
        if self.server is not None:
            # The grace is zero because the worker's own shutdown already ended every
            # conversation; what remains is transport teardown, which has no work to drain.
            self.server.stop(grace=0)
        self.stopped.set()


class _PreparationServicer:
    def __init__(
        self,
        package_handler: (
            Callable[[pb.PreparePackageSetRequest], pb.PreparePackageSetResult] | None
        ),
        model_source_handler: (
            Callable[[pb.PrepareModelSourceRequest], pb.PrepareModelSourceResult] | None
        ),
    ) -> None:
        self.package_handler = package_handler
        self.model_source_handler = model_source_handler

    def PreparePackageSet(self, request: pb.PreparePackageSetRequest, context: Any) -> Any:
        handler = self.package_handler
        if handler is None:
            import grpc

            return context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "package preparation is unavailable"
            )
        try:
            return handler(request)
        except Exception as exc:
            import grpc

            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc)[:1024])

    def PrepareModelSource(self, request: pb.PrepareModelSourceRequest, context: Any) -> Any:
        handler = self.model_source_handler
        if handler is None:
            import grpc

            return context.abort(
                grpc.StatusCode.UNIMPLEMENTED, "model source preparation is unavailable"
            )
        try:
            return handler(request)
        except Exception as exc:
            import grpc

            context.abort(grpc.StatusCode.INTERNAL, type(exc).__name__)


def _loopback_preparation_listener(listen: str) -> bool:
    if listen.startswith("unix:"):
        return True
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
    else's; a watcher bound to a fenced control generation ends. The counters are the
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

    def open(self, request_id: str, generation: int) -> _Watch:
        watch = _Watch(request_id=request_id, generation=generation, depth=self.depth)
        with self.lock:
            self.watchers.append(watch)
        return watch

    def end_generation(self, generation: int) -> None:
        """Fence: every watch bound to a superseded control stream ends."""
        with self.lock:
            for watch in self.watchers:
                if watch.generation <= generation:
                    watch.end()

    def end_all(self) -> None:
        with self.lock:
            for watch in self.watchers:
                watch.end()


class _Watch:
    def __init__(self, *, request_id: str, generation: int, depth: int) -> None:
        self.request_id = request_id
        self.generation = generation
        self.ring: collections.deque[pb.AttemptProgress] = collections.deque(maxlen=depth)
        self.cond = threading.Condition()
        self.ended = threading.Event()

    def end(self) -> None:
        self.ended.set()
        with self.cond:
            self.cond.notify_all()

    def frames(self, counted: WatchFanout) -> Iterator[pb.AttemptProgress]:
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
