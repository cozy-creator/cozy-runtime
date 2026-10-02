"""Owned, single-threaded CPU process fixture; no authored package or device imports."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src"
BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv.pop(1)); "
    "from cozy_runtime.internal import trampoline; trampoline.main(sys.argv[1:])"
)
REPORT = """
import ctypes, json, os, signal, sys
libc = ctypes.CDLL(None, use_errno=True)
guard = ctypes.c_int()
assert libc.prctl(2, ctypes.byref(guard), 0, 0, 0) == 0
print(json.dumps({'pid': os.getpid(), 'uid': os.geteuid(), 'gid': os.getegid(),
                  'guard': guard.value, 'no_new_privs': libc.prctl(39, 0, 0, 0, 0)}), flush=True)
if sys.argv[1] == 'wait':
    signal.pause()
"""


def command(uid: int, gid: int, mode: str, parent: int) -> list[str]:
    return [
        sys.executable,
        "-c",
        BOOTSTRAP,
        str(SOURCE),
        "--expect-parent",
        str(parent),
        "--oom-adj",
        Path("/proc/self/oom_score_adj").read_text().strip(),
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        "--scope-backend",
        "inherit",
        "--",
        sys.executable,
        "-c",
        REPORT,
        mode,
    ]


def death(uid: int, gid: int) -> None:
    sys.path.insert(0, str(SOURCE))
    from cozy_runtime.internal import proctree

    proctree.arm_subreaper()
    reader, writer = os.pipe()
    creator = os.fork()
    if creator == 0:
        os.close(writer)
        subprocess.Popen(command(uid, gid, "wait", os.getpid()))
        os.read(reader, 1)
        os._exit(0)
    os.close(reader)
    action = sys.stdin.read(1)
    os.close(writer)
    _, creator_status = os.waitpid(creator, 0)
    assert os.waitstatus_to_exitcode(creator_status) == 0
    if action != "r":
        # An assertion failure requests explicit cleanup of this fixture's owned child.
        for child in Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split():
            os.kill(int(child), signal.SIGKILL)
    _, status = os.waitpid(-1, 0)
    print(json.dumps({"returncode": os.waitstatus_to_exitcode(status)}), flush=True)


def filesystem_identity() -> None:
    import tensorfs

    libc = ctypes.CDLL(None, use_errno=True)
    with tempfile.TemporaryDirectory(prefix="cozy-pdeath-store.") as directory:
        root = Path(directory) / "store"
        tensorfs.Store.init(root)
        for current, directories, files in os.walk(root):
            os.chown(current, 65532, 65532)
            for name in (*directories, *files):
                os.chown(Path(current) / name, 65532, 65532)
        assert libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0
        before = ctypes.c_int()
        assert libc.prctl(2, ctypes.byref(before), 0, 0, 0) == 0
        tensorfs.Store.open(root)
        after = ctypes.c_int()
        assert libc.prctl(2, ctypes.byref(after), 0, 0, 0) == 0
        print(
            json.dumps({"before": before.value, "after": after.value, "uid": os.geteuid()}),
            flush=True,
        )


if __name__ == "__main__":
    match sys.argv[1]:
        case "report":
            raise SystemExit(
                subprocess.call(command(int(sys.argv[2]), int(sys.argv[3]), "report", os.getpid()))
            )
        case "death":
            death(int(sys.argv[2]), int(sys.argv[3]))
        case "orphan":
            raise SystemExit(subprocess.call(command(-1, -1, "report", os.getpid() + 1)))
        case "filesystem_identity":
            filesystem_identity()
