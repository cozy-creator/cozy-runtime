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
import json
import os
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from cozy_runtime.internal import config, proctree
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


#: Kernel compiles this process started in a child, reaped when they end.
_COMPILERS: list[subprocess.Popen[bytes]] = []


def expect_kernels(entries: Sequence[str], modules: Sequence[str]) -> str:
    """Start the package's kernels' compiles for the sealed cards in a child that lives until
    they end: NVML (which leaves a driver thread on current drivers) and the compiles' watcher
    threads live there, so this process stays forkable. It imports the package's kernel
    `modules`; the compiles and keys are the ones `triton_kernels.expect` starts in-process;
    construction takes each build once ready."""
    reap()
    plan = json.dumps({"entries": list(entries), "modules": list(modules)})
    child = subprocess.Popen([sys.executable, "-I", "-c", _EXPECT, plan], stdin=subprocess.DEVNULL)
    _COMPILERS.append(child)
    return f"kernel compiles started by child {child.pid}"


def reap() -> None:
    """Reap kernel compile children that have ended."""
    _COMPILERS[:] = [child for child in _COMPILERS if child.poll() is None]


_EXPECT = """
import json, sys, threading
from importlib import import_module
from cozy_runtime.author.kernels import _SETS
from cozy_runtime.internal import accel, config, kernel_cache, triton_kernels
kernel_cache.configure(kernel_cache.from_sealed(config.inherited_environment()))
plan = json.loads(sys.argv[1])
for module in plan["modules"]:
    import_module(module)
archs = sorted({accel.device_capability(entry) for entry in plan["entries"]} - {0})
for arch in archs:
    for name, kernels in sorted(_SETS.items()):
        line = triton_kernels.expect(kernels, arch)
        print(f"[kernels] {name} sm{arch} {line}", file=sys.stderr, flush=True)
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


def fork(environment: Mapping[str, str], child: Callable[[], int], *, group: bool = True) -> int:
    """The child's pid. The child seals itself, runs `child` and exits; it never returns.
    `group`: it leads its own process group; a group's follower stays in its leader's."""
    gc.collect()
    gc.freeze()  # imported objects stay shared: the collector never writes their pages
    parent = os.getpid()
    pid = os.fork()
    if pid:
        return pid
    code = int(Exit.internal)
    try:
        if group:
            os.setpgid(0, 0)
        proctree.arm_parent_death()
        if os.getppid() != parent:
            os._exit(int(Exit.structural))
        config.replace_environment(environment)
        code = child()
    except BaseException:
        traceback.print_exc()
    finally:
        os._exit(code)
