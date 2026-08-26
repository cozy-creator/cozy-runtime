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

import contextlib
import ctypes
import errno
import hashlib
import os
import secrets
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PLATFORM = sys.platform

_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
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


class CgroupUnavailable(Exception):
    """Linux cannot provide the executor subtree boundary it promised."""


@dataclass(frozen=True, slots=True)
class CgroupScope:
    """One exact cgroup-v2 directory, protected against path reuse by its inode.

    This is lifecycle containment for ordinary endpoint process trees: sessions, process
    groups, double-forks, and nested cgroups stay below it and ``cgroup.kill`` reaches them.
    It is not a hostile-code sandbox. Same-UID code that can write a delegated ancestor's
    ``cgroup.procs`` may deliberately migrate out; privilege isolation owns that threat.
    """

    relative_path: str
    inode: int


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """One Linux process birth, not merely a reusable PID."""

    pid: int
    started_ticks: int


@dataclass(slots=True)
class WorkerRootLease:
    """One process-held lease over a worker root."""

    descriptor: int


def _linux_only(operation: str) -> None:
    if PLATFORM != "linux":
        raise ProcessTreeUnsupported(operation)


def _posix_only(operation: str) -> None:
    # Process groups are POSIX: linux and darwin share these; only win32 needs its own
    # implementation (Job Objects, #449).
    if PLATFORM == "win32":
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
    """Give executor-directed signals a group boundary separate from the worker."""
    _posix_only("own_process_group")
    os.setpgid(0, 0)


def arm_subreaper() -> None:
    """Make executor descendants orphan to this worker instead of init.

    The executor may create helper processes. Once its cgroup subtree has been killed, those
    helpers must be waitable here so the worker can prove their PIDs absent before it
    publishes a successor.
    """
    _linux_only("arm_subreaper")
    _prctl(_PR_SET_CHILD_SUBREAPER, 1)


def _cgroup_root() -> Path:
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        raise CgroupUnavailable("a writable unified cgroup-v2 hierarchy is required")
    return root


def _own_cgroup_relative() -> str:
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split("::", 1)[1] for line in entries if line.startswith("0::"))
    except (OSError, StopIteration, IndexError) as exc:
        raise CgroupUnavailable("cannot resolve this worker's cgroup-v2 delegation") from exc
    return "/" + relative.strip("/") if relative.strip("/") else "/"


def _scope_path(scope: CgroupScope) -> Path:
    if not scope.relative_path.startswith("/") or ".." in Path(scope.relative_path).parts:
        raise CgroupUnavailable(f"invalid executor cgroup path {scope.relative_path!r}")
    path = _cgroup_root() / scope.relative_path.lstrip("/")
    try:
        inode = path.stat().st_ino
    except FileNotFoundError as exc:
        raise CgroupUnavailable(f"executor cgroup {scope.relative_path} is absent") from exc
    except OSError as exc:
        raise CgroupUnavailable(
            f"cannot inspect executor cgroup {scope.relative_path}: {exc}"
        ) from exc
    if inode != scope.inode:
        raise CgroupUnavailable(
            f"executor cgroup {scope.relative_path} was replaced: "
            f"expected inode {scope.inode}, found {inode}"
        )
    return path


def executor_cgroup_namespace(worker_root: Path) -> str:
    """Stable, opaque cgroup namespace for one worker root."""
    return hashlib.sha256(os.fsencode(str(worker_root.resolve()))).hexdigest()[:16]


def validate_executor_cgroup(scope: CgroupScope, namespace: str) -> None:
    """Refuse a durable record that could name another worker root's scope."""
    name = Path(scope.relative_path).name
    if not namespace or not name.startswith(f"cozy-executor-{namespace}-"):
        raise CgroupUnavailable(
            f"executor cgroup {scope.relative_path!r} is outside namespace {namespace!r}"
        )


def create_executor_cgroup(namespace: str) -> CgroupScope:
    """Create and validate a fresh child scope before an executor can allocate."""
    _linux_only("create_executor_cgroup")
    root = _cgroup_root()
    parent = root / _own_cgroup_relative().lstrip("/")
    if not namespace or any(char not in "0123456789abcdef" for char in namespace):
        raise CgroupUnavailable(f"invalid executor cgroup namespace {namespace!r}")
    path = parent / f"cozy-executor-{namespace}-{secrets.token_hex(8)}"
    try:
        path.mkdir(mode=0o700)
        required = ("cgroup.procs", "cgroup.events", "cgroup.freeze", "cgroup.kill")
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            raise CgroupUnavailable(f"executor cgroup {path} lacks required kernel files {missing}")
        # Opening the two control files catches a non-delegated hierarchy now, before the
        # trampoline or any executor import exists.
        for name in ("cgroup.procs", "cgroup.freeze", "cgroup.kill"):
            descriptor = os.open(path / name, os.O_WRONLY)
            os.close(descriptor)
        relative = "/" + str(path.relative_to(root))
        return CgroupScope(relative_path=relative, inode=path.stat().st_ino)
    except BaseException as exc:
        with contextlib.suppress(OSError):
            path.rmdir()
        if isinstance(exc, (CgroupUnavailable, ProcessTreeUnsupported)):
            raise
        raise CgroupUnavailable(
            f"cannot create delegated executor cgroup below {parent}: {exc}"
        ) from exc


def join_executor_cgroup(scope: CgroupScope) -> None:
    """Move this trampoline into its exact scope before importing the executor."""
    _linux_only("join_executor_cgroup")
    path = _scope_path(scope)
    try:
        (path / "cgroup.procs").write_text("0\n")
    except OSError as exc:
        raise CgroupUnavailable(
            f"cannot join executor cgroup {scope.relative_path}: {exc}"
        ) from exc
    if _own_cgroup_relative() != scope.relative_path or _scope_path(scope) != path:
        raise CgroupUnavailable(
            f"joining executor cgroup {scope.relative_path} did not take effect"
        )


def _cgroup_event(path: Path, name: str) -> int:
    try:
        events = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
        return int(events[name])
    except (OSError, KeyError, ValueError) as exc:
        raise CgroupUnavailable(f"cannot read {name!r} from {path / 'cgroup.events'}") from exc


def _cgroup_processes(path: Path) -> set[ProcessIdentity]:
    found: set[ProcessIdentity] = set()
    try:
        directories = [path, *(item for item in path.rglob("*") if item.is_dir())]
        for directory in directories:
            for text in (directory / "cgroup.procs").read_text().splitlines():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    found.add(process_identity(int(text)))
    except OSError as exc:
        raise CgroupUnavailable(f"cannot enumerate executor cgroup {path}: {exc}") from exc
    return found


def cgroup_contains(scope: CgroupScope, process: ProcessIdentity) -> bool:
    """Whether one exact process birth is currently inside this exact cgroup subtree."""
    return process in _cgroup_processes(_scope_path(scope))


def executor_cgroup_exists(scope: CgroupScope) -> bool:
    """Whether the exact scope still exists; path reuse remains a typed failure."""
    if not scope.relative_path.startswith("/") or ".." in Path(scope.relative_path).parts:
        raise CgroupUnavailable(f"invalid executor cgroup path {scope.relative_path!r}")
    path = _cgroup_root() / scope.relative_path.lstrip("/")
    try:
        inode = path.stat().st_ino
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CgroupUnavailable(
            f"cannot inspect executor cgroup {scope.relative_path}: {exc}"
        ) from exc
    if inode != scope.inode:
        raise CgroupUnavailable(
            f"executor cgroup {scope.relative_path} was replaced: "
            f"expected inode {scope.inode}, found {inode}"
        )
    return True


def freeze_executor_cgroup(scope: CgroupScope, *, seconds: float = 10.0) -> set[ProcessIdentity]:
    """Freeze and census an executor subtree before its durable kill boundary."""
    path = _scope_path(scope)
    deadline = time.monotonic() + seconds
    try:
        (path / "cgroup.freeze").write_text("1\n")
        while _cgroup_event(path, "populated") and not _cgroup_event(path, "frozen"):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not freeze executor cgroup {scope.relative_path}")
            time.sleep(0.01)
        return _cgroup_processes(path)
    except OSError as exc:
        raise CgroupUnavailable(
            f"cannot freeze executor cgroup {scope.relative_path}: {exc}"
        ) from exc


def kill_frozen_executor_cgroup(scope: CgroupScope, *, seconds: float = 10.0) -> None:
    """Kill a frozen subtree whose exact process census is already durable."""
    path = _scope_path(scope)
    deadline = time.monotonic() + seconds
    try:
        (path / "cgroup.kill").write_text("1\n")
        while _cgroup_event(path, "populated"):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"executor cgroup {scope.relative_path} remained populated")
            time.sleep(0.01)
    except OSError as exc:
        raise CgroupUnavailable(
            f"cannot reclaim executor cgroup {scope.relative_path}: {exc}"
        ) from exc


def remove_executor_cgroup(scope: CgroupScope) -> None:
    """Remove the exact, proven-empty scope and any empty descendant cgroups."""
    path = _scope_path(scope)
    if _cgroup_event(path, "populated"):
        raise CgroupUnavailable(f"executor cgroup {scope.relative_path} is still populated")
    try:
        descendants = sorted(
            (item for item in path.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for descendant in descendants:
            descendant.rmdir()
        path.rmdir()
    except OSError as exc:
        raise CgroupUnavailable(
            f"cannot remove executor cgroup {scope.relative_path}: {exc}"
        ) from exc


def acquire_worker_root_lease(path: Path) -> WorkerRootLease:
    """Acquire the process-lifetime single-supervisor lease for one worker root."""
    _posix_only("acquire_worker_root_lease")
    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return WorkerRootLease(descriptor)


def release_worker_root_lease(lease: WorkerRootLease) -> None:
    _posix_only("release_worker_root_lease")
    import fcntl

    if lease.descriptor < 0:
        return
    fcntl.flock(lease.descriptor, fcntl.LOCK_UN)
    os.close(lease.descriptor)
    lease.descriptor = -1


# ------------------------------------------------------------------ worker-side ops


def process_identity(pid: int) -> ProcessIdentity:
    """Read the kernel birth marker for `pid`.

    `/proc/<pid>/stat` field 22 is the process start time in clock ticks since boot. A PID
    may be reused; the pair cannot name a later process by accident.
    """
    _linux_only("process_identity")
    try:
        fields = _linux_stat(pid)
        return ProcessIdentity(pid=pid, started_ticks=int(fields[19]))
    except (IndexError, ValueError) as exc:
        raise ProcessLookupError(pid) from exc


def _linux_stat(pid: int) -> list[str]:
    """The fields after ``comm`` in Linux ``/proc/<pid>/stat``."""
    _linux_only("process_identity")
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[-1].split()
    except OSError as exc:
        raise ProcessLookupError(pid) from exc


def same_process(process: ProcessIdentity) -> bool:
    """Whether the PID still names the exact process birth captured by `process`."""
    try:
        return process_identity(process.pid) == process
    except ProcessLookupError:
        return False


def process_state(process: ProcessIdentity) -> str | None:
    """The exact birth's Linux state, or ``None`` once that birth is absent.

    A zombie remains present.  That distinction lets the worker keep its direct child
    unreaped while driver reclaim is observed, preventing the PID from being reused under
    a raw per-process driver query.
    """
    try:
        fields = _linux_stat(process.pid)
        if int(fields[19]) != process.started_ticks:
            return None
        return fields[0]
    except (ProcessLookupError, IndexError, ValueError):
        return None


def process_group_members(pgid: int) -> list[tuple[ProcessIdentity, str]]:
    """Every observable birth in process group ``pgid``, including zombies."""
    _linux_only("process_group_members")
    found: list[tuple[ProcessIdentity, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            fields = _linux_stat(pid)
            if int(fields[2]) != pgid:
                continue
            found.append((ProcessIdentity(pid=pid, started_ticks=int(fields[19])), fields[0]))
        except (ProcessLookupError, IndexError, ValueError):
            continue
    return found


def _open_pidfd(process: ProcessIdentity) -> int:
    """Open a stable kernel handle for exactly ``process``.

    This Python build does not expose :func:`os.pidfd_open`, but glibc does.  Opening the
    handle before re-reading the birth marker closes the usual check-then-``kill`` PID
    reuse race: the handle keeps naming the birth it was opened against.
    """
    _linux_only("pidfd_open")
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        pidfd_open = libc.pidfd_open
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "libc has no pidfd_open") from exc
    pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    pidfd_open.restype = ctypes.c_int
    fd = int(pidfd_open(process.pid, 0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), process.pid)
    if not same_process(process):
        os.close(fd)
        raise ProcessLookupError(process.pid)
    return fd


def signal_process(process: ProcessIdentity, sig: int) -> None:
    """Signal one exact process birth through a pidfd, never a reusable numeric PID."""
    fd = _open_pidfd(process)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            pidfd_send_signal = libc.pidfd_send_signal
        except AttributeError as exc:
            raise OSError(errno.ENOSYS, "libc has no pidfd_send_signal") from exc
        pidfd_send_signal.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        pidfd_send_signal.restype = ctypes.c_int
        if int(pidfd_send_signal(fd, int(sig), None, 0)) < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), process.pid)
    finally:
        os.close(fd)


def freeze_process_group(leader: ProcessIdentity, *, seconds: float = 10.0) -> set[ProcessIdentity]:
    """Stop and return every exact birth in an executor's process group.

    There is deliberately no ``killpg`` here.  The leader is stopped first, then every
    observed member is stopped through a pidfd.  Two equal all-stopped censuses prove that
    no member can still fork while the caller proceeds to exact per-birth signals.

    The exact leader must still exist.  Its stopped birth pins the numeric group ID while
    members are enumerated.  An absent or reused leader leaves only a reusable number, so
    nothing is touched; restart recovery may observe that number but must not signal it.
    """
    _linux_only("freeze_process_group")
    try:
        current = process_identity(leader.pid)
    except ProcessLookupError:
        current = None
    if current != leader:
        return set()

    deadline = time.monotonic() + seconds
    known: set[ProcessIdentity] = set()
    stable: frozenset[ProcessIdentity] | None = None
    while True:
        observed = process_group_members(leader.pid)
        if same_process(leader) and all(process != leader for process, _ in observed):
            # A provisional child may not have installed its own group yet.  It is still
            # safe to stop by exact birth; its inherited worker group is never signalled.
            state = process_state(leader)
            if state is not None:
                observed.append((leader, state))
        for process, state in observed:
            known.add(process)
            if state not in {"T", "t", "Z"}:
                with contextlib.suppress(ProcessLookupError):
                    signal_process(process, signal.SIGSTOP)

        census = process_group_members(leader.pid)
        if same_process(leader) and all(process != leader for process, _ in census):
            state = process_state(leader)
            if state is not None:
                census.append((leader, state))
        known.update(process for process, _ in census)
        stopped = all(state in {"T", "t", "Z"} for _, state in census)
        identities = frozenset(process for process, _ in census)
        if stopped and identities == stable:
            return known
        stable = identities if stopped else None
        if time.monotonic() >= deadline:
            states = {process.pid: state for process, state in census}
            raise TimeoutError(f"could not freeze process group {leader.pid}: states={states}")
        time.sleep(0.01)


def kill_process_group(leader: ProcessIdentity, *, seconds: float = 10.0) -> set[ProcessIdentity]:
    """Freeze one owned group, then SIGKILL each captured birth through a pidfd."""
    known = freeze_process_group(leader, seconds=seconds)
    for process in known:
        if process_state(process) != "Z":
            with contextlib.suppress(ProcessLookupError):
                signal_process(process, signal.SIGKILL)
    return known


def reap_child(process: ProcessIdentity) -> int | None:
    """Reap one exact child zombie if it belongs to this worker."""
    if process_state(process) != "Z":
        return None
    try:
        pid, status = os.waitpid(process.pid, os.WNOHANG)
    except ChildProcessError:
        return None
    return status if pid else None


def progress_burn(pid: int) -> int:
    """The two-handed progress meter: CPU nanoseconds PLUS bytes read from disk — a child
    can be making progress by computing or by waiting on a cold page cache, and a meter
    with only the first hand kills the second. THIS process, threads included, unreaped
    children excluded. -1 is UNREADABLE, never a zero (an unreadable meter decides
    nothing)."""
    _linux_only("progress_burn")
    if not pid:
        return -1
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[-1].split()
        ticks = int(fields[11]) + int(fields[12])  # utime + stime, after `state`
    except (OSError, IndexError, ValueError):
        return -1
    burned = ticks * (1_000_000_000 // os.sysconf("SC_CLK_TCK"))
    try:
        io = dict(
            line.split(": ", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines() if line
        )
        return burned + int(io["read_bytes"])
    except (OSError, KeyError, ValueError):
        return burned  # no `io` for this process; the CPU hand still decides


def processes_with_cmdline_markers(
    markers: tuple[bytes, ...], *, exclude: int = 0
) -> list[ProcessIdentity]:
    """Every live process birth whose argv contains all `markers`."""
    _linux_only("processes_with_cmdline_markers")
    found: list[ProcessIdentity] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == exclude:
            continue
        try:
            text = b" ".join((entry / "cmdline").read_bytes().split(b"\0"))
        except OSError:
            continue
        if all(marker in text for marker in markers):
            with contextlib.suppress(ProcessLookupError):
                found.append(process_identity(int(entry.name)))
    return found


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
