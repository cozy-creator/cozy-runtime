"""The coordinator LINK: the one seam between the supervisor and whoever drives it.

The supervisor speaks `cozy.worker.v1` and nothing else. WHO carries those messages —
a gRPC channel to a real coordinator, or a pair of in-process queues — is a transport
detail, and this module is the whole of it. That matters because of §8's one-kernel rule:
`run` and `job` are one-shot user experiences, never a second execution implementation.
They replace the network coordinator with an in-memory control adapter and change NOTHING
else — the same validation, admission, cancellation, output and cleanup state machine, the
same supervisor/executor process boundary, the same durable journal.

Two durability classes, kept apart here exactly as they are on the wire:

* **control** — the durable bidi stream. Messages are never shed; terminal authority lives
  only here.
* **progress** — bounded and LOSSY, on its own lane. Saturating it can never block control,
  which is true for the in-memory link for the same reason it is true for gRPC: a different
  queue, drained by a different thread, with no shared buffer for backpressure to cross.

`reconnects` is the one behavioural difference between the two coordinators, and it is a
fact about them rather than a switch: a network peer that drops the stream may come back, so
the worker redials the SAME session; an in-memory coordinator that ends the stream has ended
the run, and redialling a queue nobody is reading is a hang, not a recovery.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Iterator
from typing import Any, Protocol

from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc


class LinkError(Exception):
    """The transport failed. Transport faults never reach the session as themselves — a
    supervisor that catches `grpc.RpcError` has a gRPC dependency in its control flow."""


class Link(Protocol):
    """One coordinator connection."""

    def control(self, outbound: Iterator[pb.WorkerMessage]) -> Iterator[pb.CoordinatorMessage]:
        """Open the durable bidi stream. Raises `LinkError` when the transport fails."""

    def progress(self, frames: Iterator[pb.AttemptProgress]) -> None:
        """Drain the lossy lane. Blocks; its failure is invisible to control."""

    def cancel(self) -> None:
        """End the call from this side, so a parked peer cannot hold shutdown open."""


class Coordinator(Protocol):
    """A source of links, and whether losing one is worth redialling."""

    reconnects: bool

    def target(self) -> str: ...

    def link(self) -> Link: ...


# --------------------------------------------------------------------------- gRPC


class GrpcLink:
    """The production transport: one channel, `Control` plus `StreamProgress`."""

    def __init__(self, target: str) -> None:
        import grpc

        self._grpc = grpc
        self.channel = grpc.insecure_channel(target)
        self.stub = pb_grpc.WorkerStub(self.channel)
        self.call: Any = None

    def control(self, outbound: Iterator[pb.WorkerMessage]) -> Iterator[pb.CoordinatorMessage]:
        try:
            self.call = self.stub.Control(outbound)
        except self._grpc.RpcError as exc:
            raise LinkError(f"Control failed: {exc}") from exc
        return _translate(self.call, self._grpc.RpcError)

    def progress(self, frames: Iterator[pb.AttemptProgress]) -> None:
        with contextlib.suppress(self._grpc.RpcError):
            self.stub.StreamProgress(frames)

    def cancel(self) -> None:
        if self.call is not None:
            # Half-closing the request stream is not enough: the peer's response generator
            # may be parked waiting to send. Cancelling ends the call from this side.
            self.call.cancel()


def _translate(
    responses: Any, transport_error: type[BaseException]
) -> Iterator[pb.CoordinatorMessage]:
    try:
        yield from responses
    except transport_error as exc:
        raise LinkError(f"the control stream ended: {exc}") from exc


class GrpcCoordinator:
    """A real coordinator at a gRPC target. The session outlives the connection."""

    reconnects = True

    def __init__(self, target: str) -> None:
        self._target = target

    def target(self) -> str:
        return self._target

    def link(self) -> Link:
        return GrpcLink(self._target)


# --------------------------------------------------------------------------- in-memory


class InMemoryLink:
    """The transport half of the local control adapter: two queues and two lanes.

    `inbound` carries coordinator->worker messages and is CLOSED by a `None` sentinel, which
    is how an in-memory coordinator says "the run is over" — the exact analogue of a peer
    half-closing the stream.
    """

    def __init__(self) -> None:
        self.inbound: queue.Queue[pb.CoordinatorMessage | None] = queue.Queue()
        self.sent: list[pb.WorkerMessage] = []
        self.progress_frames: list[pb.AttemptProgress] = []
        self.on_message: Any = None  # set by the coordinator that owns this link
        self.on_progress: Any = None
        self.closed = threading.Event()

    # ---------------------------------------------------------------- Link

    def control(self, outbound: Iterator[pb.WorkerMessage]) -> Iterator[pb.CoordinatorMessage]:
        threading.Thread(
            target=self._pump, args=(outbound,), daemon=True, name="local-control"
        ).start()
        return self._inbound()

    def progress(self, frames: Iterator[pb.AttemptProgress]) -> None:
        for frame in frames:
            self.progress_frames.append(frame)
            if self.on_progress is not None:
                self.on_progress(frame)

    def cancel(self) -> None:
        self.close()

    # ---------------------------------------------------------------- driving

    def deliver(self, message: pb.CoordinatorMessage) -> None:
        if not self.closed.is_set():
            self.inbound.put(message)

    def close(self) -> None:
        if not self.closed.is_set():
            self.closed.set()
            self.inbound.put(None)

    def _pump(self, outbound: Iterator[pb.WorkerMessage]) -> None:
        for message in outbound:
            self.sent.append(message)
            if self.on_message is not None:
                self.on_message(message)

    def _inbound(self) -> Iterator[pb.CoordinatorMessage]:
        while True:
            message = self.inbound.get()
            if message is None:
                return
            yield message
