"""One executor session, driven as TensorD drives every function: Start, Load and Activate
once, then PrepareRequest and Invoke per attempt, over a real seam whose peer answers the
executor's durable requests with `handler`."""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._executor_requests import Handler, respond
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Invoke, Load, PrepareRequest, Start
from cozy_runtime.internal.seam import Channel


class Session:
    def __init__(self, executor: Executor) -> None:
        self.executor = executor
        #: Every frame the executor sent its owner, in order.
        self.frames: list[dict[str, Any]] = []
        self.handler: Handler | None = None

    def run(
        self,
        entrypoint: str,
        payload: object,
        spool: Path,
        *,
        input_metadata: dict[str, Any] | None = None,
        **invoke: Any,
    ) -> dict[str, Any]:
        """One attempt: prepared, then invoked with `invoke`'s fields (`sources`, `weights`)."""
        prepared = self.executor.serve_request(
            msgspec.convert(
                {
                    "request_id": spool.name,
                    "entrypoint": entrypoint,
                    "payload": payload,
                    "input_metadata": input_metadata or {},
                },
                PrepareRequest,
            )
        )
        if not prepared["ok"]:
            return prepared
        spool.mkdir(parents=True, exist_ok=True)
        command = {"request_id": spool.name, "entrypoint": entrypoint, "spool": str(spool)}
        return self.executor.serve_invoke(
            msgspec.convert({"deadline_s": 60, **command, **invoke}, Invoke)
        )


@contextlib.contextmanager
def session(root: Path, found: Discovered, *, devices: str = "", **start: Any) -> Iterator[Session]:
    """`found`'s release started on `devices` (and `start`'s other fields) with one
    construction of no constructed model."""
    interface = root / "package-interface.json"
    interface.write_bytes(package_interface.canonical_bytes(package_interface.build(found)))
    release = {"application": found.application, "package_interface": str(interface)}
    left, right = socket.socketpair()
    with left, right:
        peer = Channel(right)
        current = Session(Executor(Channel(left), root))

        def serve() -> None:
            while (frame := peer.recv()) is not None:
                current.frames.append(frame)
                if frame.get("event") != "request":
                    continue
                answer, handoff = respond(frame, current.handler)
                peer.send(answer)
                if isinstance(handoff, socket.socket):
                    with handoff:
                        peer.send_descriptor(handoff)

        owner = threading.Thread(target=serve, daemon=True)
        owner.start()
        executor = current.executor
        started = executor.start(msgspec.convert({**release, **start, "devices": devices}, Start))
        assert started["ok"], started
        load = {"devices": devices, "binding": release, "construction": "plan"}
        assert executor.load(msgspec.convert(load, Load))["ok"]
        assert executor.activate(Activate(construction="plan"))["ok"]
        try:
            yield current
        finally:
            left.shutdown(socket.SHUT_WR)
            owner.join(timeout=5)
            assert not owner.is_alive()
