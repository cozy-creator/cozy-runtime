"""The executor's Windows path on real processes: its import, the seam over a named pipe
(frames, handles duplicated into the peer, a writable file refused, timeouts and a torn frame
typed), the file lock, and parent death. The machine's end is the same `PipeChannel` on a pipe
this test serves, named as TensorD names it."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Windows's alone; the type checker and other platforms skip the block.
if sys.platform == "win32":
    import _winapi

    from cozy_runtime.internal import filelock, seam, winpipe

    #: A hang bound for a child that never dials or never dies; the product waits without one.
    HANG_MS = 60_000

    def _serve(path: str) -> int:
        return _winapi.CreateNamedPipe(
            winpipe.pipe_name(path),
            _winapi.PIPE_ACCESS_DUPLEX
            | _winapi.FILE_FLAG_OVERLAPPED
            | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE,
            0,  # byte mode, blocking
            1,
            1 << 16,
            1 << 16,
            0,
            _winapi.NULL,
        )

    def _accept(server: int, child: subprocess.Popen[bytes] | None = None) -> seam.Channel:
        connecting = _winapi.ConnectNamedPipe(server, overlapped=True)
        if _winapi.WaitForMultipleObjects([connecting.event], False, HANG_MS) != 0:
            connecting.cancel()
            detail = child.communicate(timeout=5)[1].decode() if child else ""
            pytest.fail(f"nobody dialed the pipe: {detail}")
        _, error = connecting.GetOverlappedResult(True)
        assert error == 0
        return seam.PipeChannel(winpipe.Pipe(server))

    EXECUTOR = """
import os, sys
from cozy_runtime.internal import seam
channel = seam.connect(sys.argv[1])
frame = channel.recv()
fd = channel.recv_file()
data = os.read(fd, 100)
os.close(fd)
cell = seam.anonymous("cozy-test")
os.write(cell, b"from the executor")
channel.send({"got": frame["n"], "file": data.decode()})
channel.send_memfd(cell)
os.close(cell)
try:
    channel.recv_file()
except seam.SeamError as exc:
    channel.send({"refused": exc.code})
channel.close()
"""

    def test_the_executor_imports_as_the_machine_starts_it() -> None:
        started = subprocess.run(
            [sys.executable, "-I", "-c", "import cozy_runtime.internal.executor"],
            capture_output=True,
            text=True,
        )
        assert started.returncode == 0, started.stderr

    def test_frames_and_files_cross_a_named_pipe_seam(tmp_path: Path) -> None:
        path = str(tmp_path / "e.sock")
        server = _serve(path)
        child = subprocess.Popen([sys.executable, "-c", EXECUTOR, path], stderr=subprocess.PIPE)
        machine = _accept(server, child)
        try:
            machine.send({"n": 7})
            objects = tmp_path / "object"
            objects.write_bytes(b"object bytes")
            read_only = os.open(objects, os.O_RDONLY | os.O_BINARY)
            machine._send_fd(read_only, "object file")
            os.close(read_only)
            assert machine.recv() == {"got": 7, "file": "object bytes"}
            cell = machine.recv_memfd()
            os.lseek(cell, 0, os.SEEK_SET)
            assert os.read(cell, 100) == b"from the executor"
            os.close(cell)
            writable = os.open(objects, os.O_RDWR | os.O_BINARY)
            machine._send_fd(writable, "object file")
            os.close(writable)
            assert machine.recv() == {"refused": "seam_descriptor"}
            # The executor's close: its normal end, at a frame boundary.
            assert machine.recv() is None
        finally:
            machine.close()
            assert child.wait(timeout=HANG_MS / 1000) == 0, (
                child.stderr.read() if child.stderr else ""
            )

    def test_a_stalled_peer_times_out_and_a_torn_frame_is_typed(tmp_path: Path) -> None:
        path = str(tmp_path / "e.sock")
        server = _serve(path)
        connecting = _winapi.ConnectNamedPipe(server, overlapped=True)
        executor = seam.connect(path)
        assert _winapi.WaitForMultipleObjects([connecting.event], False, HANG_MS) == 0
        machine = seam.PipeChannel(winpipe.Pipe(server))
        try:
            with pytest.raises(seam.SeamError) as stalled:
                executor.recv(timeout=0.2)
            assert stalled.value.code == "seam_timeout"
            machine.send({"after": "a stall"})
            assert executor.recv(timeout=5) == {"after": "a stall"}
            machine.sock.sendall(seam.HEADER.pack(10) + b"half")
            machine.close()
            with pytest.raises(seam.SeamError) as torn:
                executor.recv()
            assert torn.value.code == "seam_eof_mid_frame"
        finally:
            executor.close()
            machine.close()

    def test_a_held_file_lock_refuses_another_holder_until_released(tmp_path: Path) -> None:
        path = tmp_path / "lock"
        path.touch()
        first, second = os.open(path, os.O_RDWR), os.open(path, os.O_RDWR)
        try:
            filelock.lock(first)
            with pytest.raises(BlockingIOError):
                filelock.lock(second, wait=False)
            with pytest.raises(BlockingIOError):
                filelock.lock(second, shared=True, wait=False)
            filelock.unlock(first)
            filelock.lock(second, shared=True, wait=False)
            filelock.lock(first, shared=True, wait=False)
            # The lock covers no byte of the file: its holder's readers and writers go on.
            os.write(first, b"written while locked")
        finally:
            os.close(first)
            os.close(second)

    #: The interpreter itself, as the environment's: a venv's python.exe is a launcher that
    #: starts it as a child, which would make the launcher the watched parent.
    DIRECT = {**os.environ, "__PYVENV_LAUNCHER__": sys.executable}

    PARENT = """
import os, subprocess, sys, time
watcher = subprocess.Popen(
    [sys._base_executable, "-c", "import time\\nfrom cozy_runtime.internal import proctree\\n"
     "proctree.arm_parent_death()\\nprint('armed', flush=True)\\ntime.sleep(600)"],
    env={**os.environ, "__PYVENV_LAUNCHER__": sys.executable}, stdout=subprocess.PIPE, text=True,
)
assert watcher.stdout.readline().strip() == "armed"
print(watcher.pid, flush=True)
time.sleep(600)
"""

    def test_an_executor_armed_against_parent_death_dies_with_its_parent() -> None:
        parent = subprocess.Popen(
            [sys._base_executable, "-c", PARENT], env=DIRECT, stdout=subprocess.PIPE, text=True
        )
        assert parent.stdout is not None
        watcher = _winapi.OpenProcess(
            0x00100000 | 0x0001,  # SYNCHRONIZE | PROCESS_TERMINATE
            False,
            int(parent.stdout.readline()),
        )
        try:
            parent.kill()
            parent.wait()
            if _winapi.WaitForSingleObject(watcher, HANG_MS) != 0:
                _winapi.TerminateProcess(watcher, 1)
                pytest.fail("the armed process outlived its parent")
        finally:
            _winapi.CloseHandle(watcher)
