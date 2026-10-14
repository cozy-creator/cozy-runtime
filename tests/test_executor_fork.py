"""`fork/1`: real executor processes, a package imported once and forked into executors.

The parent is an import-only executor; each child dials its own seam with the imports done,
belongs to its own process group, carries the machine's environment, and dies with the parent.
CUDA is hidden (`CUDA_VISIBLE_DEVICES=""`); the GPU half is proven on a rental.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import package_interface
from cozy_runtime.internal import zygote as fork_parent
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.seam import Channel

SOURCE = Path(__file__).resolve().parents[1] / "src"
APP = """
import msgspec
from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct):
    n: int


class Result(msgspec.Struct):
    value: int


@app.entrypoint
def generate(ctx: Context, payload: Request) -> Result:
    return Result(payload.n + 1)
"""


def _environment(tmp_path: Path, threads: str = "1") -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{tmp_path}:{SOURCE}",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": threads,
        "COZY_HOME": str(tmp_path / "home"),
    }


def _accept(listener: socket.socket) -> tuple[Channel, int]:
    connection, _ = listener.accept()
    peer = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, _uid, _gid = struct.unpack("3i", peer)
    return Channel(connection), pid


def _call(channel: Channel, frame: dict[str, Any]) -> dict[str, Any]:
    channel.send(frame)
    while True:
        reply = channel.recv(timeout=300)
        assert reply is not None, f"executor ended during {frame['cmd']}"
        if "event" not in reply:
            return reply


def _wait_zombie(pid: int) -> None:
    deadline = time.monotonic() + 30
    while Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
        assert time.monotonic() < deadline, f"{pid} did not exit"
        time.sleep(0.02)


def _listener(path: Path) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    return listener


@pytest.fixture
def zygote(tmp_path: Path) -> Iterator[tuple[subprocess.Popen[bytes], Channel, dict[str, Any]]]:
    (tmp_path / "forkapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("forkapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
    (tmp_path / "interface.json").write_text(json.dumps(interface))
    root = tmp_path / "zygote"
    root.mkdir()
    listener = _listener(tmp_path / "z.sock")
    process = subprocess.Popen(
        [
            sys.executable,
            *("-m", "cozy_runtime.internal.executor"),
            *("--socket", str(tmp_path / "z.sock"), "--root", str(root)),
        ],
        env=_environment(tmp_path),
    )
    channel, _ = _accept(listener)
    start = {
        "cmd": "start",
        "devices": "",
        "application": "forkapp:app",
        "package_interface": str(tmp_path / "interface.json"),
    }
    try:
        yield process, channel, start
    finally:
        process.kill()
        process.wait()


def test_a_segfaulting_executor_leaves_its_python_stack(tmp_path: Path) -> None:
    (tmp_path / "segvapp.py").write_text(APP)
    sys.path.insert(0, str(tmp_path))
    try:
        interface = package_interface.build(discover_installed("segvapp:app"))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("segvapp", None)
    (tmp_path / "interface.json").write_text(json.dumps(interface))
    (tmp_path / "segvapp.py").write_text("import ctypes\nctypes.string_at(0)\n" + APP)
    (tmp_path / "root").mkdir()
    listener = _listener(tmp_path / "s.sock")
    process = subprocess.Popen(
        [
            sys.executable,
            *("-m", "cozy_runtime.internal.executor"),
            *("--socket", str(tmp_path / "s.sock"), "--root", str(tmp_path / "root")),
        ],
        env=_environment(tmp_path),
        stderr=subprocess.PIPE,
    )
    try:
        channel, _ = _accept(listener)
        channel.send(
            {
                "cmd": "start",
                "devices": "",
                "application": "segvapp:app",
                "package_interface": str(tmp_path / "interface.json"),
            }
        )
        _, stderr = process.communicate(timeout=120)
    finally:
        process.kill()
        process.wait()
    assert process.returncode == -signal.SIGSEGV
    assert b"Fatal Python error: Segmentation fault" in stderr and b"Current thread" in stderr


def _fork(tmp_path: Path, channel: Channel, name: str, threads: str = "1") -> dict[str, Any]:
    root = tmp_path / name
    root.mkdir()
    return _call(
        channel,
        {
            "cmd": "fork",
            "socket": str(tmp_path / f"{name}.sock"),
            "root": str(root),
            "environment": _environment(tmp_path, threads),
        },
    )


def test_a_forked_child_is_a_sealed_executor_with_the_imports_done(
    zygote: tuple[subprocess.Popen[bytes], Channel, dict[str, Any]], tmp_path: Path
) -> None:
    process, channel, start = zygote
    hello = _call(channel, {"cmd": "hello"})
    assert "fork/1" in hello["memory"]
    assert _call(channel, {**start, "import_only": True})["ok"]
    listener = _listener(tmp_path / "a.sock")
    began = time.monotonic()
    forked = _fork(tmp_path, channel, "a")
    assert forked["ok"], forked
    child, peer = _accept(listener)
    ready_s = time.monotonic() - began
    assert peer == forked["pid"] != process.pid
    hello = _call(child, {"cmd": "hello"})
    assert hello["pid"] == peer and hello["ppid"] == process.pid and hello["pgid"] == peer
    assert (
        hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "" and hello["sealed"]["OMP_NUM_THREADS"] == "1"
    )
    started = _call(child, start)
    assert started["ok"], started
    assert "package_import" not in dict(started["stages"]), "the child imported again"
    assert ready_s < 5, f"a forked executor dialed after {ready_s:.1f} s"
    assert (tmp_path / "a" / "stderr.log").exists(), "the child logs in its own root"
    assert _call(child, {"cmd": "shutdown"})["ok"]
    _wait_zombie(peer)
    stat = Path(f"/proc/{peer}/stat").read_text().rsplit(")", 1)[1].split()
    assert int(stat[49]) == 0, "its exit status waits for the machine until the next fork"
    listener = _listener(tmp_path / "c.sock")
    assert _fork(tmp_path, channel, "c")["ok"]
    _accept(listener)
    assert not Path(f"/proc/{peer}").exists(), "the next fork reaped it"


def test_a_fork_refuses_what_it_cannot_keep(
    zygote: tuple[subprocess.Popen[bytes], Channel, dict[str, Any]], tmp_path: Path
) -> None:
    _process, channel, start = zygote
    unready = _fork(tmp_path, channel, "early")
    assert not unready["ok"] and unready["code"] == "fork_unready"
    assert _call(channel, {**start, "import_only": True})["ok"]
    moved = _fork(tmp_path, channel, "threads", threads="4")
    assert not moved["ok"] and moved["code"] == "fork_environment", moved
    assert _call(channel, start)["ok"]
    started = _fork(tmp_path, channel, "late")
    assert not started["ok"] and started["code"] == "fork_unready"


@pytest.mark.parametrize("many", [False, True])
def test_model_load_requires_sources_and_leaves_weightless_execution_usable(
    zygote: tuple[subprocess.Popen[bytes], Channel, dict[str, Any]], tmp_path: Path, many: bool
) -> None:
    """A real control peer cannot turn an installed full Store into model authority."""
    import tensorfs

    store = tensorfs.Store.init(tmp_path / "store")
    _process, channel, start = zygote
    assert _call(channel, start)["ok"]
    binding = {"model_class": "UnimportableModel", "store": str(store.root)}
    command: dict[str, Any] = {"cmd": "load", "construction": "ungranted", "devices": ""}
    command.update(
        {
            "models": [
                {"binding": {**binding, "model_binding_path": name, "model_parameter_name": name}}
                for name in ("first", "second")
            ]
        }
        if many
        else {"binding": binding}
    )
    refused = _call(channel, command)
    assert refused["code"] == "model_sources_ungranted", refused
    assert not refused["poisoned"]
    loaded = _call(
        channel,
        {
            "cmd": "load",
            "construction": "weightless",
            "devices": "",
            "binding": {
                "application": start["application"],
                "package_interface": start["package_interface"],
            },
        },
    )
    assert loaded["ok"], loaded
    # Grant presence advances to the next operation's own admission check. It does not
    # require a particular wheel flavor or fabricate a device budget for this peer.
    admitted = _call(channel, {**command, "model_sources": True})
    assert admitted["code"] == "authorization_absent", admitted


def test_children_die_with_their_parent(
    zygote: tuple[subprocess.Popen[bytes], Channel, dict[str, Any]], tmp_path: Path
) -> None:
    process, channel, start = zygote
    assert _call(channel, {**start, "import_only": True})["ok"]
    listener = _listener(tmp_path / "b.sock")
    forked = _fork(tmp_path, channel, "b")
    child, peer = _accept(listener)
    assert _call(child, {"cmd": "hello"})["pid"] == peer
    process.send_signal(signal.SIGKILL)
    process.wait()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(peer, 0)
        except ProcessLookupError:
            break
        if Path(f"/proc/{peer}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"forked child {forked['pid']} outlived its parent")


def test_an_import_only_kernel_compile_leaves_no_thread_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVML leaves a driver thread (driver 580) and a compile has a watcher thread: an
    import-only start runs both in a child, so the parent stays forkable while it compiles.
    CUDA is hidden here, so the child finds no card and ends; the GPU half is a rental's."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONPATH", str(SOURCE))
    before = set(os.listdir("/proc/self/task"))
    line = fork_parent.expect_kernels(["0"], [])
    assert line.startswith("kernel compiles started by child")
    assert set(os.listdir("/proc/self/task")) == before
    (child,) = fork_parent._COMPILERS
    assert child.wait(timeout=120) == 0
    fork_parent.reap()
    assert fork_parent._COMPILERS == []
