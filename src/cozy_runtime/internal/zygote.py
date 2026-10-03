"""Executors forked from one imported, device-free process (`fork/1`).

An import-only executor holds torch, the Runtime and the package with no device, no other
thread and no NVIDIA descriptor. Its fork is an executor whose imports are done: it seals
itself as the trampoline does (own process group, parent death, the machine's environment)
and dials its own seam. A child of a process that initialized the CUDA driver cannot use
CUDA, so the parent measures that it did not, and refuses rather than forks.
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from cozy_runtime.internal import proctree
from cozy_runtime.internal.exits import Exit

CAPABILITY = "fork/1"

#: Names their libraries read once at load, during the parent's imports: a child keeps them.
LOADED = ("OMP_NUM_THREADS",)


@contextlib.contextmanager
def driverless(torch: Any, devices: int) -> Iterator[None]:
    """Imports that ask whether CUDA exists (Diffusers does, at import) get the seal's answer.
    Torch asks the CUDA driver (`cuInit`) or NVML, and on current drivers either opens the
    device and starts a driver thread that outlives it: no child of that process can fork."""
    saved = torch.cuda.is_available, torch.cuda.device_count
    torch.cuda.is_available = lambda: devices > 0
    torch.cuda.device_count = lambda: devices
    try:
        yield
    finally:
        torch.cuda.is_available, torch.cuda.device_count = saved


#: Fused-glue compiles this process started in a child, reaped when they end.
_COMPILERS: list[subprocess.Popen[bytes]] = []


def expect_fusion(entries: Sequence[str]) -> str:
    """Start the fused glue's compile for the sealed cards in a child that lives until the
    compile ends: NVML (which leaves a driver thread on current drivers) and the compile's
    watcher thread live there, so this process stays forkable. The compile and its key are
    the ones `fusion.expect` starts in-process; construction takes the build once ready."""
    reap()
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _EXPECT, *entries], stdin=subprocess.DEVNULL
    )
    _COMPILERS.append(child)
    return f"fusion compile started by child {child.pid}"


def reap() -> None:
    """Reap fused-glue compile children that have ended."""
    _COMPILERS[:] = [child for child in _COMPILERS if child.poll() is None]


_EXPECT = """
import os, sys, threading
from cozy_runtime.internal import accel, fusion, kernel_cache
kernel_cache.configure(kernel_cache.from_sealed(os.environ))
archs = sorted({accel.device_capability(entry) for entry in sys.argv[1:]} - {0})
for arch in archs:
    print(f"[fusion] sm{arch} {fusion.expect(arch)}", file=sys.stderr, flush=True)
for thread in threading.enumerate():
    if thread is not threading.main_thread():
        thread.join()
"""


def _threads() -> list[str]:
    names = []
    for task in os.listdir("/proc/self/task"):
        with contextlib.suppress(OSError), open(f"/proc/self/task/{task}/comm") as comm:
            names.append(comm.read().strip())
    return sorted(names)


def _device_files() -> list[str]:
    found = []
    for fd in os.listdir("/proc/self/fd"):
        with contextlib.suppress(OSError):
            target = os.readlink(f"/proc/self/fd/{fd}")
            if target.startswith("/dev/nvidia"):
                found.append(target)
    return sorted(found)


def unsafe(torch: Any) -> str:
    """Why this process must not fork, or "". Every fact is read, none assumed."""
    if len(_threads()) > 1:
        # A throwaway fork runs every pthread_atfork prepare handler; OpenBLAS stops its pool.
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
    if torch is not None and torch.cuda.is_initialized():
        return "torch initialized CUDA"
    threads = _threads()
    if len(threads) > 1 or threading.active_count() > 1:
        return f"threads alive: {threads}"
    files = _device_files()
    if files:
        return f"NVIDIA device files open: {files}"
    return ""


def collect(children: set[int]) -> set[int]:
    """Reap the exited ones of `children`; the reaped pids."""
    reaped = set()
    for pid in children:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                reaped.add(pid)
        except ChildProcessError:
            reaped.add(pid)
    return reaped


def fork(environment: Mapping[str, str], child: Callable[[], int]) -> int:
    """The child's pid. The child seals itself, runs `child` and exits; it never returns."""
    gc.collect()
    gc.freeze()  # imported objects stay shared: the collector never writes their pages
    parent = os.getpid()
    pid = os.fork()
    if pid:
        return pid
    code = int(Exit.internal)
    try:
        os.setpgid(0, 0)
        proctree.arm_parent_death()
        if os.getppid() != parent:
            os._exit(int(Exit.structural))
        os.environ.clear()
        os.environ.update(environment)
        code = child()
    except BaseException:
        traceback.print_exc()
    finally:
        os._exit(code)
