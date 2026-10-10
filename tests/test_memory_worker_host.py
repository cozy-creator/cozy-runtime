"""Worker host credits belong to the originating real child incarnation, not its slot."""

import os
import socket
import subprocess
import sys
from pathlib import Path

from cozy_runtime.author._executor_requests import HostRoom
from cozy_runtime.internal import proctree
from cozy_runtime.internal.memory import Machine
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.worker.child import Executor
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from test_device_lanes import _config


def test_generation_change_releases_pending_and_rejects_old_handler(tmp_path: Path) -> None:
    worker = Worker(
        _config(tmp_path / "home"), WorkerOptions(root=tmp_path / "worker"), InMemoryControlHost()
    )
    worker.memory = Machine({}, lambda _: None, lambda _: None, host_available=lambda: 100)
    ours, theirs = socket.socketpair()
    peer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,sys; socket.socket(fileno=int(sys.argv[1])).recv(1)",
            str(theirs.fileno()),
        ],
        pass_fds=(theirs.fileno(),),
    )
    theirs.close()
    executor = Executor(
        epoch=1,
        process=proctree.process_identity(peer.pid),
        scope=proctree.ProcessTreeScope("host-credit-proof", os.getuid()),
        channel=Channel(ours),
        seal_digest="",
    )
    row = worker._primary_lane().row(worker._primary_slot())
    row.supervision = worker.supervision
    try:
        worker.supervision.current = executor
        worker.executor_changed(executor)
        answer = worker._room(worker._primary_lane(), worker._primary_slot())
        assert answer(HostRoom(reservation="copy", nbytes=70)).ok
        assert not answer(HostRoom(reservation="other", nbytes=40)).ok
        assert answer(HostRoom(reservation="copy", release=True)).ok
        assert not worker.memory.host_reserved
        assert answer(HostRoom(reservation="interrupted", nbytes=70)).ok
        executor.channel.close()
        peer.wait(timeout=5)
        worker.supervision.current = None
        worker.executor_changed(None)  # supervision publishes this only after confirmed reap
        assert not worker.memory.host_reserved
        assert not answer(HostRoom(reservation="stale", nbytes=1)).ok
        assert not worker.memory.host_reserved
    finally:
        worker.supervision.current = None
        executor.channel.close()
        peer.wait(timeout=5)
        worker.shutdown()
