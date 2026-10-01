"""Isolated installed-wheel worker with a real descriptor-four storage transport."""

from __future__ import annotations

import array
import base64
import json
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cozy_runtime.internal import builtin_operations, package_installation, package_interface
from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions

root = Path(sys.argv[1])
root.mkdir()
host, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
copy = host.dup()
host.close()
host = copy
fd = child.detach()
if fd != 4:
    os.dup2(fd, 4)
    os.close(fd)
storage.inherit(4)
leases: list[list[dict[str, Any]]] = []


def supervise() -> None:
    while True:
        body, ancillary, flags, _ = host.recvmsg(4096, socket.CMSG_SPACE(9 * 4))
        if not body:
            return
        assert flags == 0
        fds = array.array("i")
        fds.frombytes(ancillary[0][2])
        for fd in fds[1:]:
            os.close(fd)
        rows = json.loads(body)
        assert rows and all(row["bytes"] > 0 and row["inodes"] > 0 for row in rows)
        leases.append(rows)
        with socket.socket(fileno=fds[0]) as lease:
            lease.sendall(b'{"ok":true}\n')
            while lease.recv(1024):
                pass


thread = threading.Thread(target=supervise, daemon=True)
thread.start()


def boot() -> Worker:
    return Worker(
        RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(sorted(os.environ.items())),
        ),
        WorkerOptions(
            root=root / "worker",
            tensorfs_root=root / "store",
            install_root=root / "installs",
            artifact_cache=root / "artifacts",
        ),
        InMemoryControlHost(),
    )


worker = boot()
try:
    assert storage.enabled() and worker.machine_calls is not None
    builtins = worker.machine_calls.builtins
    captured = builtins.capture()
    assert builtins.capture() is captured
    installed = package_installation.open_installation(root / "installs", captured.installation_id)
    assert installed.python == Path(sys.executable)
    assert installed.release == captured.runtime_version
    placement = captured.placement
    assert placement["installation_id"] == captured.installation_id
    interface = package_interface.read_bytes(base64.b64decode(placement["package_interface"]))
    assert set(builtin_operations.EXPORTS) <= {row["name"] for row in interface["jobs"]}
    actual = subprocess.check_output(
        [
            str(installed.python),
            "-I",
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('cozy-runtime'))",
        ],
        text=True,
    )
    assert actual.strip() == captured.runtime_version
    # A restarted Runtime reopens its own operations and their interface: no describe.
    worker.shutdown()
    worker = boot()
    descriptions = 0

    def counted[**P, R](function: Callable[P, R]) -> Callable[P, R]:
        def observe(*args: P.args, **kwargs: P.kwargs) -> R:
            global descriptions
            descriptions += 1
            return function(*args, **kwargs)

        return observe

    worker._describe_in_slot = counted(worker._describe_in_slot)  # type: ignore[method-assign]
    assert worker.machine_calls is not None
    again = worker.machine_calls.builtins.capture()
    assert (again.installation_id, again.placement) == (captured.installation_id, placement)
    assert descriptions == 0, descriptions
    held = [
        path.name
        for path in (root / "installs" / "installations").iterdir()
        if json.loads((path / "installation.json").read_bytes())["package"] == "runtime/operations"
    ]
    assert held == [captured.installation_id], held
    print("RENTED BUILTIN PASS", captured.runtime_version, "leases", len(leases))
finally:
    worker.shutdown()
    assert storage._channel is not None
    storage._channel.close()
    thread.join(5)
    host.close()
