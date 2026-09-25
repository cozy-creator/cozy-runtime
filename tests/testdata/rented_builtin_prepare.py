"""Isolated installed-wheel worker with a real descriptor-four storage transport."""

from __future__ import annotations

import array
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from cozy_runtime.internal import package_environment as environments
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
worker = Worker(
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
try:
    assert storage.enabled() and worker.machine_calls is not None
    builtins = worker.machine_calls.builtins
    frozen = builtins.capture()
    assert frozen is not None
    capsule = root / "installs/runtime-builtins/carriers" / frozen["carrier"]["filename"]
    base = environments.observe_base()
    wheel, measured = environments.measure_wheel(capsule, "project", base)
    source = environments.Input(
        "sha256:" + "a" * 64,
        wheel.digest,
        (wheel,),
        (environments.BlobGrant(wheel.digest, wheel.length, capsule, measured),),
    )
    try:
        environments.materialize(source, root / "forbidden", dependency_python=Path(sys.executable))
    except environments.EnvironmentRefusal as exc:
        assert exc.code == "package_environment_dependency_snapshot_local_only"
    else:
        raise AssertionError("generic rented snapshot was allowed")
    # The mutable controller SDK is no longer the implementation owner.
    original = Path(environments.__file__).parents[1] / "derive/operations.py"
    original.unlink()  # Do not mutate uv's shared cache inode.
    original.write_text("raise AssertionError('mutable controller SDK used')\n")
    prepared = builtins._prepare(frozen)
    assert prepared["runtime_builtin"] == frozen
    assert prepared["placement"]["development"]["local_revision_digest"]
    assert prepared["receipt_digest"] and leases
    installed = environments.open_generation(
        root / "installs",
        receipt_digest=prepared["receipt_digest"],
        environment_digest=prepared["placement"]["environment_digest"],
    )
    actual = subprocess.check_output(
        [
            str(installed.python),
            "-I",
            "-c",
            "import json; from cozy_runtime.internal.builtin_environment import sdk_identity; "
            "print(json.dumps(sdk_identity()))",
        ],
        text=True,
    )
    assert json.loads(actual) == frozen["sdk"]
    print("RENTED BUILTIN PASS", frozen["sdk"], "leases", len(leases))
finally:
    worker.shutdown()
    assert storage._channel is not None
    storage._channel.close()
    thread.join(5)
    host.close()
