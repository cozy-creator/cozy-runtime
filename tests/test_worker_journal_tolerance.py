"""Executor frames and journal rows from another Runtime version stay usable."""

from __future__ import annotations

import os
import socket
import threading

from cozy_runtime import canonical_json
from cozy_runtime.internal import proctree
from cozy_runtime.internal.executor_commands import Probe
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker import workspace_recovery
from cozy_runtime.internal.worker.child import Executor


def test_executor_event_this_worker_does_not_know_is_skipped() -> None:
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    peer = Channel(theirs)

    def newer_executor() -> None:
        assert peer.recv(5) == {"cmd": "probe"}
        peer.send({"event": "heartbeat", "load": 3})
        peer.send({"event": "progress", "position": 1})
        peer.send({"reply": "probe", "ok": True})

    thread = threading.Thread(target=newer_executor)
    thread.start()
    executor = Executor(
        epoch=1,
        process=proctree.ProcessIdentity(os.getpid(), 0),
        scope=proctree.ProcessTreeScope("token", os.getuid()),
        channel=Channel(ours),
        seal_digest="",
    )
    progressed: list[int] = []
    reply = executor.call(
        Probe(), timeout=5, on_progress=lambda frame: progressed.append(frame["position"])
    )
    thread.join()
    ours.close()
    theirs.close()
    assert reply == {"reply": "probe", "ok": True} and progressed == [1]


def test_process_identity_with_additive_fields_is_read() -> None:
    identity = canonical_json.decode(workspace_recovery.process_identity())
    assert (
        workspace_recovery.process_ended(
            canonical_json.encode({**identity, "runtime": "a newer writer"})
        )
        is False
    )
