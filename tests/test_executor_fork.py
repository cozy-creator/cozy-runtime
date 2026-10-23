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
    # A leader is a zombie before its other threads end; its exit is reapable once they have.
    while (
        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
        or len(os.listdir(f"/proc/{pid}/task")) > 1
    ):
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


def test_memory_protocol_is_advertised_before_start_and_torch_import(zygote: Any) -> None:
    process, channel, _start = zygote
    reply = _call(channel, {"cmd": "hello"})
    assert reply["ok"] and reply["pid"] == process.pid
    assert reply["memory_protocol"] == 2
    assert reply["torch_loaded"] is False


@pytest.mark.parametrize("memory_protocol", [None, 0])
def test_an_older_tensord_s_ledger_closes_and_varena_runs_standalone(
    zygote: Any, memory_protocol: int | None
) -> None:
    """Skew degrades: a TensorD from before the memory protocol hands each GPU's ledger as a
    memfd and an eventfd. Runtime takes and closes them, says so, and starts on its own table."""
    _process, channel, start = zygote
    # 0.30's Start: its `memory_mode`, no `memory_protocol`, a memfd and an eventfd per GPU.
    command = {**start, "ledgers": 1, "import_only": True, "memory_mode": "shared"}
    if memory_protocol is not None:
        command["memory_protocol"] = memory_protocol
    memfd, eventfd = os.pipe()  # stand-ins: the descriptors' kinds are never read
    try:
        channel.send(command)
        channel._send_fd(memfd, "old ledger memfd")
        channel._send_fd(eventfd, "old ledger eventfd")
        reply = channel.recv(timeout=60, total_timeout=60)
    finally:
        os.close(memfd)
        os.close(eventfd)
    assert reply is not None and reply["ok"], reply
    assert "standalone" in reply["memory_standalone"]
    assert _call(channel, {"cmd": "hello"})["torch_loaded"] is False


def test_gpu_start_refuses_a_newer_memory_protocol_before_reading_descriptors(
    zygote: Any,
) -> None:
    _process, channel, start = zygote
    command = {**start, "devices": "0", "ledgers": 1, "memory_protocol": 3}
    channel.send(command)
    # No GPU or descriptor is supplied. Receiving this frame proves the real executor
    # refused before either the legacy first FD or the incompatible second FD could block.
    reply = channel.recv(timeout=5, total_timeout=5)
    assert reply is not None and not reply["ok"]
    assert reply["code"] == "memory_protocol_incompatible"
    assert _call(channel, {"cmd": "hello"})["torch_loaded"] is False


def test_cpu_start_needs_no_gpu_memory_protocol(zygote: Any) -> None:
    _process, channel, start = zygote
    assert _call(channel, {**start, "import_only": True})["ok"]


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
    assert "fork/1" in hello["capabilities"]
    assert _call(channel, {**start, "import_only": True})["ok"]
    assert os.listdir(f"/proc/{process.pid}/task") == [str(process.pid)], "a thread blocks forks"
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


def test_a_forked_child_loads_and_invokes_a_weightless_package(
    zygote: tuple[subprocess.Popen[bytes], Channel, dict[str, Any]], tmp_path: Path
) -> None:
    _process, channel, start = zygote
    assert _call(channel, {**start, "import_only": True})["ok"]
    listener = _listener(tmp_path / "run.sock")
    assert _fork(tmp_path, channel, "run")["ok"]
    child, pid = _accept(listener)
    assert _call(child, start)["ok"]
    loaded = _call(
        child,
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
    assert _call(child, {"cmd": "activate", "construction": "weightless"})["ok"]
    attempt = {"request_id": "r", "entrypoint": "generate"}
    assert _call(child, {"cmd": "prepare_request", **attempt, "payload": {"n": 41}})["ok"]
    spool = tmp_path / "run" / "r"
    spool.mkdir()
    reply = _call(child, {"cmd": "invoke", **attempt, "spool": str(spool), "deadline_s": 60})
    assert reply["outcome"]["terminal"] == "succeeded", reply
    assert json.loads((spool / "result.canonical").read_bytes()) == {"value": 42}
    assert not _call(child, {"cmd": "hello"})["torch_loaded"]
    assert _call(child, {"cmd": "shutdown"})["ok"]
    _wait_zombie(pid)


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
