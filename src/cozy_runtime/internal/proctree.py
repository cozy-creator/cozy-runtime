"""ProcessTree: the process-containment boundary, one of exactly two platform seams (#447).

Amends #422's "no abstraction layer": the runtime's process story leaned on libc `prctl`,
PDEATHSIG, `/proc` and Linux OOM scoring in-line, which read as portable code and was not.
This module and `accel.py` are the ONLY two platform boundaries — enumerated
implementations (linux today; darwin is cr-021's lane, win32 is cr-020's), selected by
`sys.platform`, with no registration mechanism and no plugin system. A platform whose
implementation does not exist yet refuses TYPED here, naming its lane, instead of failing
somewhere inside a lifecycle path.

The surface is the codex-named containment set: contained spawn support (the trampoline's
seal steps), parent-death/orphan handling, forced tree kill, birth identity, and CPU/RSS
metrics. Cooperative stop stays where it is — it is wire/journal semantics, not a platform
primitive. `spawn.py`'s `posix_spawn` call is POSIX-portable and is not wrapped.
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
from pathlib import Path

PLATFORM = sys.platform

_PR_SET_PDEATHSIG = 1
_PR_SET_NO_NEW_PRIVS = 38


class ProcessTreeUnsupported(Exception):
    """This platform's ProcessTree implementation does not exist yet — typed, with a lane."""

    def __init__(self, operation: str) -> None:
        lane = {"darwin": "cr-021", "win32": "cr-020"}.get(PLATFORM, "unplanned")
        super().__init__(
            f"proctree.{operation} has no {PLATFORM} implementation yet — {lane} owns it; "
            "serving on this platform before that lane lands would be pretending"
        )
        self.operation = operation
        self.lane = lane


def _linux_only(operation: str) -> None:
    if PLATFORM != "linux":
        raise ProcessTreeUnsupported(operation)


def _prctl(option: int, arg: int = 0) -> int:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    result = int(libc.prctl(option, arg, 0, 0, 0))
    if result < 0:
        raise OSError(ctypes.get_errno(), f"prctl({option}, {arg})")
    return result


# ------------------------------------------------------------------ child-side sealing
# Called by the trampoline, in a fresh single-threaded process, before the executor exec
# (#455 naming: the machine's control process is the WORKER; 'supervision' is its verb).


def arm_parent_death() -> None:
    """SIGKILL this process when its parent dies. Covers FUTURE deaths only — the caller
    must re-check the parent afterwards (trampoline step 2)."""
    _linux_only("arm_parent_death")
    _prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)


def seal_no_new_privs() -> None:
    """After this, exec can never gain privileges; setuid bits are inert."""
    _linux_only("seal_no_new_privs")
    _prctl(_PR_SET_NO_NEW_PRIVS, 1)


def raise_own_oom_score(adj: int) -> None:
    """The EXECUTOR raises its OWN score so the kernel's victim ordering picks the
    disposable process; the worker control process stays at the default (§3.1)."""
    _linux_only("raise_own_oom_score")
    with open("/proc/self/oom_score_adj", "w") as handle:
        handle.write(str(adj))


def own_process_group() -> None:
    """A fresh process group, so one forced kill takes the executor AND every grandchild."""
    _linux_only("own_process_group")
    os.setpgid(0, 0)


# ------------------------------------------------------------------ worker-side ops


def tree_kill(pid: int, sig: int) -> None:
    """Forced kill of `pid`'s whole process group (the executor's own, set at seal time).

    The worker control process calls this; the executor never kills anything."""
    _linux_only("tree_kill")
    os.killpg(os.getpgid(pid), sig)


def birth_identity(pid: int) -> str:
    """A pid plus its kernel start time: the identity that survives pid reuse.

    Two processes can share a pid across time; none share (pid, starttime). Empty string
    means the process is gone — never a guess.
    """
    _linux_only("birth_identity")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[-1].split()
        return f"{pid}:{fields[19]}"  # starttime is field 22; 19 after the (comm) split
    except (OSError, IndexError):
        return ""


def cpu_seconds(pid: int) -> float:
    """utime+stime for one pid, in seconds — the progress-observation clock (#434), or
    -1.0 when the process is gone (UNREADABLE, never 0)."""
    _linux_only("cpu_seconds")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[-1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return -1.0


def rss_bytes(pid: int) -> int:
    """Current resident bytes for one pid, read BY THE PARENT from the kernel — never the
    child's claim about itself. -1 when the process is gone (UNREADABLE)."""
    _linux_only("rss_bytes")
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return -1
    return -1


def available_host_bytes() -> int:
    """The LIVE host budget, from the cgroup when there is one, else MemAvailable.

    §3.2's host-RAM rule: re-read at every grant, because a budget fixed at boot describes
    a machine nobody else is on. -1 means UNREADABLE, never 0.
    """
    _linux_only("available_host_bytes")
    try:
        relative = Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split("::")[-1]
        cgroup = Path("/sys/fs/cgroup") / relative.lstrip("/")
        limit = (cgroup / "memory.max").read_text().strip()
        if limit != "max":
            current = int((cgroup / "memory.current").read_text().strip())
            return max(int(limit) - current, 0)
    except (OSError, IndexError, ValueError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return -1
    return -1
