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
metrics. Cooperative stop stays where it is — it is wire/control semantics, not a platform
primitive. `spawn.py`'s `posix_spawn` call is POSIX-portable and is not wrapped.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hashlib
import os
import secrets
import selectors
import signal
import stat
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from cozy_runtime.internal.child_env import EXECUTOR_SCOPE_ENV

PLATFORM = sys.platform

_PR_SET_PDEATHSIG = 1
_PR_GET_DUMPABLE = 3
_PR_SET_DUMPABLE = 4
_PR_SET_CHILD_SUBREAPER = 36
_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_NO_NEW_PRIVS = 39

# Linux assigns these through asm-generic and x86_64 alike. The process backend is selected
# only on Linux; Python builds that expose the typed wrappers use those instead.
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434


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


class ContainmentUnavailable(Exception):
    """Linux cannot provide or prove an executor containment boundary."""


@dataclass(frozen=True, slots=True)
class CgroupScope:
    """One exact cgroup-v2 directory, protected against path reuse by its inode.

    This is lifecycle containment for ordinary package process trees: sessions, process
    groups, double-forks, and nested cgroups stay below it and ``cgroup.kill`` reaches them.
    It is not a hostile-code sandbox. Same-UID code that can write a delegated ancestor's
    ``cgroup.procs`` may deliberately migrate out; privilege isolation owns that threat.
    """

    relative_path: str
    inode: int


@dataclass(frozen=True, slots=True)
class ProcessTreeScope:
    """One first-party executor family in a positively proven container.

    ``token`` is non-secret process identity.  It is one complete inherited environment
    entry, never a substring marker. In a same-UID seat this is lifecycle containment for
    our own executor, not a hostile-code sandbox: code can evade it by scrubbing the token
    and escaping both ancestry and process group, or cause collateral by replaying the
    token. The pod seat instead gives the executor a dedicated UID, so the census owns that
    whole UID domain even after token scrubbing. Unlike a delegated child cgroup, its
    repeated SIGSTOP census is convergent rather than kernel-atomic; resource limits come
    from the positively verified read-only provider cgroup.
    """

    token: str
    uid: int


ExecutorScope = CgroupScope | ProcessTreeScope

_PROCESS_SCOPE_TOKEN_BYTES = 32
_CONTAINER_MARKERS = (Path("/.dockerenv"), Path("/run/.containerenv"))


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """One Linux process birth, not merely a reusable PID."""

    pid: int
    started_ticks: int


@dataclass(slots=True)
class ProcessExitObserver:
    """A pollable kernel handle for one exact process birth.

    Waiting on the pidfd reports exit without reaping the child.  The unreaped zombie keeps
    its numeric PID pinned until executor retirement also proves driver absence, which is the
    lifecycle ordering :mod:`cozy_runtime.internal.worker.child` requires.
    """

    process: ProcessIdentity
    descriptor: int

    def wait(self) -> None:
        # Model preparation can leave descriptors above select's FD_SETSIZE. The kernel
        # selector has no such limit and still observes exit without reaping the child.
        with selectors.DefaultSelector() as selector:
            selector.register(self.descriptor, selectors.EVENT_READ)
            while True:
                try:
                    if selector.select():
                        return
                except InterruptedError:
                    continue

    def close(self) -> None:
        if self.descriptor < 0:
            return
        os.close(self.descriptor)
        self.descriptor = -1


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


def deny_process_inspection() -> None:
    """Make the trusted worker non-dumpable before a package child exists.

    The pod executor runs as a different uid, which already closes ordinary same-uid
    ptrace.  Keeping the worker non-dumpable makes the intended boundary explicit and
    prevents a future credential transition from silently reopening ``/proc/<pid>/mem``.
    """

    _linux_only("deny_process_inspection")
    _prctl(_PR_SET_DUMPABLE, 0)


def own_protection() -> dict[str, int | None]:
    """Observe this process's credentials/protection without changing its boundary."""
    facts: dict[str, int | None] = {
        name: getattr(os, method)() if hasattr(os, method) else None
        for name, method in (
            ("uid", "getuid"),
            ("gid", "getgid"),
            ("euid", "geteuid"),
            ("egid", "getegid"),
        )
    }
    for name, option in (("no_new_privs", _PR_GET_NO_NEW_PRIVS), ("dumpable", _PR_GET_DUMPABLE)):
        facts[name] = None
        if PLATFORM == "linux":
            with contextlib.suppress(OSError):
                facts[name] = _prctl(option)
    return facts


@dataclass(frozen=True, slots=True)
class _CgroupMount:
    root: Path
    target: Path
    controllers: frozenset[str]


def _mount_path(value: str) -> Path:
    """Decode mountinfo's four octal escapes; cgroup paths otherwise stay opaque."""
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return Path(value)


def _cgroup_v1_mounts() -> tuple[_CgroupMount, ...]:
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError as exc:
        raise ContainmentUnavailable("cannot read cgroup mount facts") from exc
    mounts: list[_CgroupMount] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 3 >= len(fields) or fields[separator + 1] != "cgroup":
            continue
        mounts.append(
            _CgroupMount(
                root=_mount_path(fields[3]),
                target=_mount_path(fields[4]),
                controllers=frozenset(fields[separator + 3].split(",")),
            )
        )
    return tuple(mounts)


def _inside_mount(mount: _CgroupMount, relative: Path) -> Path | None:
    try:
        suffix = relative.relative_to(mount.root)
    except ValueError:
        return None
    return mount.target / suffix


def _cgroup_v1_controller_path(controller: str) -> Path:
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError as exc:
        raise ContainmentUnavailable("cannot read cgroup-v1 memberships") from exc
    relative: Path | None = None
    for line in memberships:
        try:
            _, names, raw_path = line.split(":", 2)
        except ValueError as exc:
            raise ContainmentUnavailable("malformed cgroup-v1 membership") from exc
        if controller in names.split(","):
            relative = Path(raw_path)
            break
    if relative is None or not str(relative).startswith("/"):
        raise ContainmentUnavailable(f"no cgroup-v1 {controller} membership exists")
    candidates = [
        resolved
        for mount in _cgroup_v1_mounts()
        if controller in mount.controllers
        if (resolved := _inside_mount(mount, relative)) is not None and resolved.is_dir()
    ]
    if len(candidates) != 1:
        raise ContainmentUnavailable(
            f"cgroup-v1 {controller} membership resolves to {len(candidates)} paths"
        )
    return candidates[0]


def _root_owned_container_marker() -> bool:
    for marker in _CONTAINER_MARKERS:
        try:
            found = marker.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISREG(found.st_mode) and found.st_uid == 0 and found.st_mode & 0o022 == 0:
            return True
    return False


def _full_proc_visibility(process_uid: int, *, require_environment: bool) -> bool:
    """Require every target-UID process identity, plus environments when same-UID."""
    try:
        entries = [entry for entry in Path("/proc").iterdir() if entry.name.isdigit()]
        if not entries or not Path("/proc/1/stat").is_file():
            return False
    except OSError:
        return False
    for entry in entries:
        try:
            fields = _linux_stat(int(entry.name))
            status = (entry / "status").read_text().splitlines()
            uid_row = next(line for line in status if line.startswith("Uid:"))
            if require_environment and int(uid_row.split()[1]) == process_uid and fields[0] != "Z":
                (entry / "environ").read_bytes()
        except FileNotFoundError:
            continue  # one birth ended during the census
        except ProcessLookupError as exc:
            # _linux_stat wraps filesystem errors. Only a confirmed vanished
            # birth is harmless; permission and unknown I/O failures still refuse.
            cause = exc.__cause__
            if isinstance(cause, OSError) and cause.errno in (errno.ENOENT, errno.ESRCH):
                continue
            return False
        except (OSError, StopIteration, ValueError):
            return False
    return True


def _prove_zero_cgroup_capabilities() -> None:
    try:
        rows: dict[str, str] = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            name, separator, value = line.partition(":")
            if name in {"CapEff", "CapPrm"}:
                if not separator or name in rows:
                    raise ValueError("capability fields differ")
                rows[name] = value.strip()
        if set(rows) != {"CapEff", "CapPrm"}:
            raise ValueError("capability fields differ")
        capabilities = {name: int(value, 16) for name, value in rows.items()}
    except (OSError, ValueError) as exc:
        raise ContainmentUnavailable("cannot prove the worker capability set") from exc
    nonzero = [name for name, value in capabilities.items() if value]
    if nonzero:
        raise ContainmentUnavailable(
            f"process-tree backend requires zero effective and permitted capabilities; "
            f"nonzero fields={sorted(nonzero)}"
        )


def _effective_writable(path: Path) -> bool:
    if os.access not in os.supports_effective_ids:
        raise ContainmentUnavailable("effective-credential access checks are unavailable")
    try:
        return os.access(path, os.W_OK, effective_ids=True)
    except (NotImplementedError, OSError) as exc:
        raise ContainmentUnavailable(f"cannot check effective write authority for {path}") from exc


def _require_cgroup_controls_unwritable(
    mount_target: Path,
    resolved: Path,
    *,
    version: str,
    controls: tuple[str, ...],
) -> None:
    try:
        suffix = resolved.relative_to(mount_target)
    except ValueError as exc:
        raise ContainmentUnavailable(
            f"{version} membership {resolved} is outside mount {mount_target}"
        ) from exc
    lineage = [mount_target]
    for part in suffix.parts:
        lineage.append(lineage[-1] / part)
    for directory in lineage:
        try:
            if not stat.S_ISDIR(directory.stat().st_mode):
                raise ContainmentUnavailable(f"{version} lineage {directory} is not a directory")
        except OSError as exc:
            raise ContainmentUnavailable(f"cannot inspect {version} lineage {directory}") from exc
        if _effective_writable(directory):
            raise ContainmentUnavailable(
                f"worker credentials can write {version} directory {directory}"
            )
        for name in controls:
            control = directory / name
            try:
                if not stat.S_ISREG(control.stat().st_mode):
                    raise ContainmentUnavailable(
                        f"{version} migration control {control} is not a regular file"
                    )
            except OSError as exc:
                raise ContainmentUnavailable(
                    f"cannot inspect {version} migration control {control}"
                ) from exc
            if _effective_writable(control):
                raise ContainmentUnavailable(
                    f"worker credentials can write {version} migration control {control}"
                )


def _require_cgroup_lineage_unwritable(mount_target: Path, resolved: Path) -> None:
    _require_cgroup_controls_unwritable(
        mount_target,
        resolved,
        version="cgroup-v1",
        controls=("tasks", "cgroup.procs"),
    )


def _prove_unwritable_container_cgroup_v1(process_uid: int) -> None:
    """Require every fact behind the narrow first-party process-tree backend.

    A marker alone proves little and an inaccessible cgroup alone can be an ordinary host.
    Selection requires their conjunction, non-root execution, zero effective and permitted
    capabilities, a genuine v1 membership for every controller, no effective write access
    to its cgroup lineage or migration controls, full same-UID proc visibility, and working
    pidfds. Unified v2 is checked by the caller first and never reaches this path.
    """

    euid = os.geteuid()
    if process_uid < 0 or (euid != 0 and process_uid != euid):
        raise ContainmentUnavailable(
            f"process-tree backend cannot supervise uid {process_uid} from euid {euid}"
        )
    if not _root_owned_container_marker():
        raise ContainmentUnavailable(
            "process-tree backend found no regular uid-0 non-writable container marker"
        )
    # An ordinary same-UID worker must have no ambient authority. The pod worker is
    # deliberately trusted root only long enough to launch a sealed uid-split child;
    # setuid clears the child's capabilities after no_new_privs is armed.
    if process_uid == euid:
        _prove_zero_cgroup_capabilities()
    try:
        memberships: list[tuple[frozenset[str], Path]] = []
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controller_text, relative_text = line.split(":", 2)
            if hierarchy == "0" and not controller_text:
                # A hybrid kernel can report an unused unified membership even though no
                # cgroup2 hierarchy is mounted. The caller already proved that
                # cgroup.controllers is absent; only v1 controller memberships are paths
                # this backend must account for.
                continue
            names = frozenset(item for item in controller_text.split(",") if item)
            if not hierarchy.isdigit() or not names or not relative_text.startswith("/"):
                raise ContainmentUnavailable(f"invalid cgroup-v1 membership line {line!r}")
            memberships.append((names, Path(relative_text)))
    except OSError as exc:
        raise ContainmentUnavailable("cannot read cgroup-v1 memberships") from exc
    except ValueError as exc:
        raise ContainmentUnavailable("malformed cgroup-v1 membership") from exc
    if not memberships:
        raise ContainmentUnavailable("no cgroup-v1 controller membership exists")
    mounts = _cgroup_v1_mounts()
    for controllers, relative in memberships:
        candidates = [
            (mount, path)
            for mount in mounts
            if controllers.issubset(mount.controllers)
            if (path := _inside_mount(mount, relative)) is not None and path.is_dir()
        ]
        if len(candidates) != 1:
            raise ContainmentUnavailable(
                f"cgroup-v1 controllers {sorted(controllers)} resolve to "
                f"{len(candidates)} containing mount paths"
            )
        mount, path = candidates[0]
        _require_cgroup_lineage_unwritable(mount.target, path)
    if not _full_proc_visibility(process_uid, require_environment=process_uid == euid):
        raise ContainmentUnavailable("full target-UID /proc visibility is unavailable")
    try:
        signal_process(process_identity(os.getpid()), 0)
    except (OSError, ProcessLookupError) as exc:
        raise ContainmentUnavailable("pidfd open/send proof failed") from exc


def _bounded_cgroup_limit(path: Path, name: str) -> int:
    """Read one inherited container limit, refusing an absent or unbounded value."""

    try:
        value = (path / name).read_text().strip()
        limit = int(value)
    except (OSError, ValueError) as exc:
        raise ContainmentUnavailable(
            f"read-only cgroup-v2 container has no finite {name} limit"
        ) from exc
    if limit <= 0:
        raise ContainmentUnavailable(
            f"read-only cgroup-v2 container has invalid {name} limit {limit}"
        )
    return limit


def _prove_unwritable_container_cgroup_v2(process_uid: int) -> None:
    """Qualify the exact uid-split, read-only unified container topology.

    A read-only cgroup mount cannot provide a child cgroup, but it also prevents the
    executor from migrating out of the provider-owned container scope. The process-tree
    backend is admitted only when that inherited scope has finite memory and process limits
    and the executor owns a UID which the root worker uses for nothing else.
    """

    root = _cgroup_root()
    euid = os.geteuid()
    if euid != 0 or process_uid <= 0 or process_uid == euid:
        raise ContainmentUnavailable(
            "read-only cgroup-v2 containment requires a root worker and a distinct "
            f"non-root executor uid; euid={euid}, executor_uid={process_uid}"
        )
    if not _root_owned_container_marker():
        raise ContainmentUnavailable(
            "process-tree backend found no regular uid-0 non-writable container marker"
        )
    try:
        read_only = bool(os.statvfs(root).f_flag & os.ST_RDONLY)
    except OSError as exc:
        raise ContainmentUnavailable("cannot inspect the cgroup-v2 mount flags") from exc
    if not read_only:
        raise ContainmentUnavailable("cgroup-v2 mount is not read-only")

    current = root / _own_cgroup_relative().lstrip("/")
    _require_cgroup_controls_unwritable(
        root,
        current,
        version="cgroup-v2",
        controls=("cgroup.procs", "cgroup.threads", "cgroup.subtree_control"),
    )

    _bounded_cgroup_limit(current, "memory.max")
    _bounded_cgroup_limit(current, "pids.max")
    if not _full_proc_visibility(process_uid, require_environment=False):
        raise ContainmentUnavailable("full target-UID /proc visibility is unavailable")
    try:
        signal_process(process_identity(os.getpid()), 0)
    except (OSError, ProcessLookupError) as exc:
        raise ContainmentUnavailable("pidfd open/send proof failed") from exc


def validate_process_tree_host(process_uid: int | None = None) -> None:
    """Revalidate every point-in-time fact required by the process-tree backend."""
    _linux_only("validate_process_tree_host")
    uid = os.geteuid() if process_uid is None else process_uid
    if (Path("/sys/fs/cgroup") / "cgroup.controllers").is_file():
        _prove_unwritable_container_cgroup_v2(uid)
    else:
        _prove_unwritable_container_cgroup_v1(uid)


def create_executor_scope(namespace: str, process_uid: int | None = None) -> ExecutorScope:
    """Select and create the strongest proven executor scope on this Linux host."""
    _linux_only("create_executor_scope")
    uid = os.geteuid() if process_uid is None else process_uid
    if (Path("/sys/fs/cgroup") / "cgroup.controllers").is_file():
        # A read-only cgroup2 mount is a positive container topology, not a fallback after
        # arbitrary cgroup creation failure. Writable unified hosts keep the stronger kernel
        # child scope and still fail closed when that delegation is incomplete.
        root = _cgroup_root()
        try:
            read_only = bool(os.statvfs(root).f_flag & os.ST_RDONLY)
        except OSError as exc:
            raise ContainmentUnavailable("cannot inspect the cgroup-v2 mount flags") from exc
        if not read_only:
            return create_executor_cgroup(namespace)
        validate_process_tree_host(uid)
        return ProcessTreeScope(secrets.token_hex(_PROCESS_SCOPE_TOKEN_BYTES), uid)
    validate_process_tree_host(uid)
    return ProcessTreeScope(secrets.token_hex(_PROCESS_SCOPE_TOKEN_BYTES), uid)


def _cgroup_root() -> Path:
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        raise ContainmentUnavailable("a writable unified cgroup-v2 hierarchy is required")
    return root


def _own_cgroup_relative() -> str:
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
        relative = next(line.split("::", 1)[1] for line in entries if line.startswith("0::"))
    except (OSError, StopIteration, IndexError) as exc:
        raise ContainmentUnavailable("cannot resolve this worker's cgroup-v2 delegation") from exc
    return "/" + relative.strip("/") if relative.strip("/") else "/"


def _scope_path(scope: CgroupScope) -> Path:
    if not scope.relative_path.startswith("/") or ".." in Path(scope.relative_path).parts:
        raise ContainmentUnavailable(f"invalid executor cgroup path {scope.relative_path!r}")
    path = _cgroup_root() / scope.relative_path.lstrip("/")
    try:
        inode = path.stat().st_ino
    except FileNotFoundError as exc:
        raise ContainmentUnavailable(f"executor cgroup {scope.relative_path} is absent") from exc
    except OSError as exc:
        raise ContainmentUnavailable(
            f"cannot inspect executor cgroup {scope.relative_path}: {exc}"
        ) from exc
    if inode != scope.inode:
        raise ContainmentUnavailable(
            f"executor cgroup {scope.relative_path} was replaced: "
            f"expected inode {scope.inode}, found {inode}"
        )
    return path


def executor_scope_namespace(worker_root: Path) -> str:
    """Stable, opaque kernel-scope namespace for one worker root."""
    return hashlib.sha256(os.fsencode(str(worker_root.resolve()))).hexdigest()[:16]


def validate_executor_cgroup(scope: CgroupScope, namespace: str) -> None:
    """Refuse a durable record that could name another worker root's scope."""
    name = Path(scope.relative_path).name
    if not namespace or not name.startswith(f"cozy-executor-{namespace}-"):
        raise ContainmentUnavailable(
            f"executor cgroup {scope.relative_path!r} is outside namespace {namespace!r}"
        )


def validate_executor_scope(scope: ExecutorScope, namespace: str) -> None:
    if isinstance(scope, CgroupScope):
        validate_executor_cgroup(scope, namespace)
        return
    if (
        len(scope.token) != _PROCESS_SCOPE_TOKEN_BYTES * 2
        or scope.token != scope.token.lower()
        or any(char not in "0123456789abcdef" for char in scope.token)
    ):
        raise ContainmentUnavailable("process-tree scope token is not 64 lowercase hex")
    if scope.uid < 0:
        raise ContainmentUnavailable("process-tree scope uid is negative")


def create_executor_cgroup(namespace: str) -> CgroupScope:
    """Create and validate a fresh child scope before an executor can allocate."""
    _linux_only("create_executor_cgroup")
    root = _cgroup_root()
    parent = root / _own_cgroup_relative().lstrip("/")
    if not namespace or any(char not in "0123456789abcdef" for char in namespace):
        raise ContainmentUnavailable(f"invalid executor cgroup namespace {namespace!r}")
    path = parent / f"cozy-executor-{namespace}-{secrets.token_hex(8)}"
    try:
        path.mkdir(mode=0o700)
        required = ("cgroup.procs", "cgroup.events", "cgroup.freeze", "cgroup.kill")
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            raise ContainmentUnavailable(
                f"executor cgroup {path} lacks required kernel files {missing}"
            )
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
        if isinstance(exc, (ContainmentUnavailable, ProcessTreeUnsupported)):
            raise
        raise ContainmentUnavailable(
            f"cannot create delegated executor cgroup below {parent}: {exc}"
        ) from exc


def join_executor_cgroup(scope: CgroupScope) -> None:
    """Move this trampoline into its exact scope before importing the executor."""
    _linux_only("join_executor_cgroup")
    path = _scope_path(scope)
    try:
        (path / "cgroup.procs").write_text("0\n")
    except OSError as exc:
        raise ContainmentUnavailable(
            f"cannot join executor cgroup {scope.relative_path}: {exc}"
        ) from exc
    if _own_cgroup_relative() != scope.relative_path or _scope_path(scope) != path:
        raise ContainmentUnavailable(
            f"joining executor cgroup {scope.relative_path} did not take effect"
        )


def _cgroup_event(path: Path, name: str) -> int:
    try:
        events = dict(line.split() for line in (path / "cgroup.events").read_text().splitlines())
        return int(events[name])
    except (OSError, KeyError, ValueError) as exc:
        raise ContainmentUnavailable(f"cannot read {name!r} from {path / 'cgroup.events'}") from exc


def _cgroup_processes(path: Path) -> set[ProcessIdentity]:
    found: set[ProcessIdentity] = set()
    try:
        directories = [path, *(item for item in path.rglob("*") if item.is_dir())]
        for directory in directories:
            for text in (directory / "cgroup.procs").read_text().splitlines():
                with contextlib.suppress(ProcessLookupError, ValueError):
                    found.add(process_identity(int(text)))
    except OSError as exc:
        raise ContainmentUnavailable(f"cannot enumerate executor cgroup {path}: {exc}") from exc
    return found


def adopted_zombies(scope: CgroupScope) -> set[ProcessIdentity]:
    """This process's zombie children that died inside `scope`. cgroup v2 lists no zombie in
    `cgroup.procs`, so a follower that exited before its leader was killed, and was then
    adopted by this subreaper, is found only here, by its `/proc/<pid>/cgroup`."""
    _linux_only("adopted_zombies")
    me, found = os.getpid(), set()
    inside = scope.relative_path.rstrip("/")
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        with contextlib.suppress(OSError, ValueError, IndexError, ProcessLookupError):
            fields = _linux_stat(pid)
            if fields[0] != "Z" or int(fields[1]) != me:
                continue
            path = Path(f"/proc/{pid}/cgroup").read_text().split("::", 1)[1].strip()
            if path == inside or path.startswith(inside + "/"):
                found.add(process_identity(pid))
    return found


def cgroup_contains(scope: CgroupScope, process: ProcessIdentity) -> bool:
    """Whether one exact process birth is currently inside this exact cgroup subtree."""
    return process in _cgroup_processes(_scope_path(scope))


def executor_cgroup_exists(scope: CgroupScope) -> bool:
    """Whether the exact scope still exists; path reuse remains a typed failure."""
    if not scope.relative_path.startswith("/") or ".." in Path(scope.relative_path).parts:
        raise ContainmentUnavailable(f"invalid executor cgroup path {scope.relative_path!r}")
    path = _cgroup_root() / scope.relative_path.lstrip("/")
    try:
        inode = path.stat().st_ino
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ContainmentUnavailable(
            f"cannot inspect executor cgroup {scope.relative_path}: {exc}"
        ) from exc
    if inode != scope.inode:
        raise ContainmentUnavailable(
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
        raise ContainmentUnavailable(
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
        raise ContainmentUnavailable(
            f"cannot reclaim executor cgroup {scope.relative_path}: {exc}"
        ) from exc


def remove_executor_cgroup(scope: CgroupScope) -> None:
    """Remove the exact, proven-empty scope and any empty descendant cgroups."""
    path = _scope_path(scope)
    if _cgroup_event(path, "populated"):
        raise ContainmentUnavailable(f"executor cgroup {scope.relative_path} is still populated")
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
        raise ContainmentUnavailable(
            f"cannot remove executor cgroup {scope.relative_path}: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class _ProcessSnapshot:
    identity: ProcessIdentity
    state: str
    parent_pid: int
    process_group: int
    scope_token: bool


def _open_pidfd(process: ProcessIdentity) -> int:
    """Open a kernel handle for one exact birth and recheck after the open."""
    _linux_only("pidfd_open")
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is not None:
        descriptor = int(pidfd_open(process.pid))
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        descriptor = int(libc.syscall(_SYS_PIDFD_OPEN, process.pid, 0))
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OSError(error or errno.ENOSYS, os.strerror(error or errno.ENOSYS))
    if not same_process(process):
        os.close(descriptor)
        raise ProcessLookupError(process.pid)
    return descriptor


def observe_process_exit(process: ProcessIdentity) -> ProcessExitObserver:
    """Open the exact-birth, non-reaping exit observer used by executor supervision."""

    return ProcessExitObserver(process, _open_pidfd(process))


def signal_process(process: ProcessIdentity, sig: int) -> None:
    """Signal one exact process birth through a pidfd, never a reusable PID."""
    descriptor = _open_pidfd(process)
    try:
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd_send_signal is not None:
            pidfd_send_signal(descriptor, sig)
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            result = int(libc.syscall(_SYS_PIDFD_SEND_SIGNAL, descriptor, sig, 0, 0))
            if result < 0:
                error = ctypes.get_errno()
                raise OSError(error or errno.ENOSYS, os.strerror(error or errno.ENOSYS))
    finally:
        os.close(descriptor)


def _same_uid_processes(scope: ProcessTreeScope) -> dict[int, _ProcessSnapshot]:
    """One full proc census, matching the scope as one complete environment entry."""
    _linux_only("process_tree_census")
    expected = f"{EXECUTOR_SCOPE_ENV}={scope.token}".encode()
    process_uid = scope.uid
    uid_domain = process_uid != os.geteuid()
    snapshots: dict[int, _ProcessSnapshot] = {}
    try:
        entries = [entry for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    except OSError as exc:
        raise ContainmentUnavailable("cannot enumerate /proc for executor reclaim") from exc
    for entry in entries:
        pid = int(entry.name)
        fields: list[str] | None = None
        try:
            fields = _linux_stat(pid)
            status = (entry / "status").read_text().splitlines()
            uid = next(line for line in status if line.startswith("Uid:"))
            if int(uid.split()[1]) != process_uid:
                continue
            state = fields[0]
            # Linux exposes no useful environment for a zombie. It cannot fork or hold a
            # device context; an adopted same-UID zombie is included below by parentage so
            # the subreaper can still reap it.
            if state in {"X", "x", "Z"}:
                environment = []
            else:
                try:
                    environment = (entry / "environ").read_bytes().split(b"\0")
                except PermissionError:
                    if uid_domain:
                        environment = []
                        snapshots[pid] = _ProcessSnapshot(
                            identity=ProcessIdentity(pid, int(fields[19])),
                            state=state,
                            parent_pid=int(fields[1]),
                            process_group=int(fields[2]),
                            scope_token=False,
                        )
                        continue
                    # A SIGKILLed task can shed its mm (and therefore environ) before its
                    # proc state reaches Z/X. Empty cmdline proves that exit transition;
                    # a live non-dumpable task remains unreadable and fails closed.
                    if (entry / "cmdline").read_bytes():
                        raise
                    environment = []
            snapshots[pid] = _ProcessSnapshot(
                identity=ProcessIdentity(pid, int(fields[19])),
                state=state,
                parent_pid=int(fields[1]),
                process_group=int(fields[2]),
                scope_token=expected in environment,
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, StopIteration, IndexError, ValueError) as exc:
            observed = (
                f" state={fields[0]} ppid={fields[1]} pgrp={fields[2]}"
                if fields is not None and len(fields) >= 3
                else ""
            )
            raise ContainmentUnavailable(
                f"cannot read complete same-UID proc identity for pid {pid}{observed}"
            ) from exc
    return snapshots


def _process_tree_census(
    scope: ProcessTreeScope,
    leader: ProcessIdentity | None,
    known: set[ProcessIdentity],
    *,
    worker_pid: int,
) -> dict[ProcessIdentity, str]:
    snapshots = _same_uid_processes(scope)
    uid_domain = scope.uid != os.geteuid()
    owned: set[ProcessIdentity] = {
        row.identity for row in snapshots.values() if uid_domain or row.scope_token
    }
    for process in known:
        row = snapshots.get(process.pid)
        if row is not None and row.identity == process:
            owned.add(process)
    if leader is not None:
        row = snapshots.get(leader.pid)
        if row is not None and row.identity == leader:
            owned.add(leader)

    while True:
        pids = {process.pid for process in owned}
        expanded = {
            row.identity
            for row in snapshots.values()
            if row.parent_pid in pids
            or (row.parent_pid == worker_pid and (row.scope_token or row.state == "Z"))
        }
        group_id = leader.pid if leader is not None else 0
        group_is_pinned = group_id > 0 and any(
            snapshots.get(process.pid) is not None
            and snapshots[process.pid].process_group == group_id
            for process in owned
        )
        if group_is_pinned:
            expanded.update(
                row.identity for row in snapshots.values() if row.process_group == group_id
            )
        before = len(owned)
        owned.update(expanded)
        if len(owned) == before:
            break
    return {
        process: snapshots[process.pid].state
        for process in owned
        if snapshots.get(process.pid) is not None and snapshots[process.pid].identity == process
    }


def process_scope_contains(scope: ProcessTreeScope, process: ProcessIdentity) -> bool:
    row = _same_uid_processes(scope).get(process.pid)
    return (
        row is not None
        and row.identity == process
        and (scope.uid != os.geteuid() or row.scope_token)
        and row.process_group == process.pid
    )


def wait_process_stopped(process: ProcessIdentity, *, seconds: float = 10.0) -> None:
    """Wait for the trampoline's pre-import self-stop without reaping the child."""
    deadline = time.monotonic() + seconds
    while True:
        if not same_process(process):
            raise ProcessLookupError(process.pid)
        waited, status = os.waitpid(process.pid, os.WNOHANG | os.WUNTRACED)
        if waited:
            if os.WIFSTOPPED(status):
                return
            raise ProcessLookupError(process.pid)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"process-tree trampoline {process.pid} did not stop before import")
        time.sleep(0.01)


def freeze_process_tree(
    scope: ProcessTreeScope,
    leader: ProcessIdentity | None,
    known: set[ProcessIdentity],
    *,
    worker_pid: int,
    seconds: float = 10.0,
) -> set[ProcessIdentity]:
    """Converge on two equal all-stopped first-party descendant censuses."""
    deadline = time.monotonic() + seconds
    stable: frozenset[ProcessIdentity] | None = None
    captured = set(known)
    while True:
        observed = _process_tree_census(scope, leader, captured, worker_pid=worker_pid)
        captured.update(observed)
        for process, state in observed.items():
            if state not in {"T", "t", "Z"}:
                with contextlib.suppress(ProcessLookupError):
                    signal_process(process, signal.SIGSTOP)
        census = _process_tree_census(scope, leader, captured, worker_pid=worker_pid)
        captured.update(census)
        identities = frozenset(census)
        stopped = all(state in {"T", "t", "Z"} for state in census.values())
        if stopped and identities == stable:
            return captured
        stable = identities if stopped else None
        if time.monotonic() >= deadline:
            states = {process.pid: state for process, state in census.items()}
            raise TimeoutError(f"could not freeze first-party process tree: states={states}")
        time.sleep(0.01)


def kill_process_tree(processes: set[ProcessIdentity]) -> None:
    """Kill each frozen exact birth through its pidfd."""
    for process in processes:
        if process_state(process) != "Z":
            with contextlib.suppress(ProcessLookupError):
                signal_process(process, signal.SIGKILL)


def process_scope_processes(scope: ProcessTreeScope) -> set[ProcessIdentity]:
    uid_domain = scope.uid != os.geteuid()
    return {
        row.identity
        for row in _same_uid_processes(scope).values()
        if (uid_domain and row.state not in {"X", "x", "Z"}) or row.scope_token
    }


def describe_scope_census(scope: ExecutorScope) -> str:
    """One bounded diagnostic line for a reclaim verdict: every live same-UID row with its
    state, parentage, and token attribution, plus the live-scope registry."""
    if not isinstance(scope, ProcessTreeScope):
        return "cgroup"
    try:
        rows = [
            f"{pid}:{row.state},ppid={row.parent_pid},own={int(row.scope_token)}"
            for pid, row in sorted(_same_uid_processes(scope).items())
        ]
    except ContainmentUnavailable as exc:
        rows = [f"unavailable:{exc}"]
    return f"scope={scope.token[:8]} uid={scope.uid} rows=[{'; '.join(rows)}]"[:600]


def executor_scope_exists(scope: ExecutorScope) -> bool:
    if isinstance(scope, CgroupScope):
        return executor_cgroup_exists(scope)
    return bool(process_scope_processes(scope))


def remove_executor_scope(scope: ExecutorScope) -> None:
    if isinstance(scope, CgroupScope):
        remove_executor_cgroup(scope)
        return
    leftovers = process_scope_processes(scope)
    if leftovers:
        births = sorted(leftovers, key=lambda process: process.pid)
        raise ContainmentUnavailable(
            f"cannot clear process-tree scope with live token births {births}"
        )


def acquire_worker_root_lease(path: Path, *, wait: bool = False) -> WorkerRootLease:
    """Acquire the process-lifetime single-worker lease for one worker root; with `wait`,
    block until its holder releases it."""
    _posix_only("acquire_worker_root_lease")
    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except BaseException:
        os.close(descriptor)
        raise
    return WorkerRootLease(descriptor)


def lease_holder(path: Path) -> int | None:
    """The pid holding a flock on `path`, from /proc/locks, else None."""
    try:
        inode = os.stat(path).st_ino
        with open("/proc/locks", encoding="ascii") as locks:
            for line in locks:
                fields = line.split()
                if (
                    len(fields) >= 6
                    and fields[1] == "FLOCK"
                    and fields[5].rsplit(":", 1)[-1] == str(inode)
                ):
                    return int(fields[4])
    except (OSError, ValueError):
        return None
    return None


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


def process_state_of(pid: int) -> str | None:
    """A NUMERIC pid's Linux state, or ``None`` when no such process exists. Only for a
    direct child whose unreaped wait slot pins the number; everywhere else use the birth."""
    try:
        return _linux_stat(pid)[0]
    except (ProcessLookupError, IndexError):
        return None


def reap_child(process: ProcessIdentity) -> int | None:
    """Reap one exact child zombie if it belongs to this worker."""
    if process_state(process) != "Z":
        return None
    try:
        pid, status = os.waitpid(process.pid, os.WNOHANG)
    except ChildProcessError:
        return None
    return status if pid else None


def dying(process: ProcessIdentity) -> bool:
    """Whether this birth has let go of its memory. The kernel does that first, then closes
    the process's descriptors, then reports its exit: such a process will exit."""
    try:
        return same_process(process) and not Path(f"/proc/{process.pid}/cmdline").read_bytes()
    except OSError:
        return False


def peek_child_exit(process: ProcessIdentity) -> int | None:
    """One exact child zombie's wait status, read WITHOUT reaping: its PID stays pinned."""
    if process_state(process) != "Z":
        return None
    try:
        info = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
    except ChildProcessError:
        return None
    if info is None:
        return None
    # wait(2)'s encoding, so this reads exactly like the status `reap_child` returns later.
    if info.si_code == os.CLD_EXITED:
        return info.si_status << 8
    return info.si_status | (0x80 if info.si_code == os.CLD_DUMPED else 0)


def progress_burn(pid: int, tid: int = 0) -> int:
    """The progress meter: CPU nanoseconds PLUS every byte moved — read from and written to
    storage, and read from or written through any descriptor (`rchar`/`wchar`, which is how
    a socket or a network filesystem shows up). A child can be making progress by computing,
    by waiting on a cold page cache, by draining a socket or by writing its result, and a
    meter missing any one hand kills that one. THIS process, threads included and exited
    threads folded in, unreaped children excluded — or ONE thread of it when `tid` names
    one, which is how a lane is judged apart from the pollers beside it. -1 is UNREADABLE,
    never a zero (an unreadable meter decides nothing)."""
    _linux_only("progress_burn")
    if not pid:
        return -1
    base = f"/proc/{pid}/task/{tid}" if tid else f"/proc/{pid}"
    try:
        fields = Path(f"{base}/stat").read_text().rsplit(") ", 1)[-1].split()
        ticks = int(fields[11]) + int(fields[12])  # utime + stime, after `state`
    except (OSError, IndexError, ValueError):
        return -1
    burned = ticks * (1_000_000_000 // os.sysconf("SC_CLK_TCK"))
    try:
        io = dict(
            line.split(": ", 1) for line in Path(f"{base}/io").read_text().splitlines() if line
        )
        return burned + sum(
            int(io[hand]) for hand in ("rchar", "wchar", "read_bytes", "write_bytes")
        )
    except (OSError, KeyError, ValueError):
        return burned  # no `io` for this process; the CPU hand still decides


def descendants(pid: int) -> list[int]:
    """Every live process under `pid`, from ONE /proc census — an executor, a `uv` install,
    a native-operator probe. Zombies included: an exited child still holds its counters
    until it is reaped."""
    _linux_only("descendants")
    parents: dict[int, int] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            tail = Path(entry.path, "stat").read_text().rsplit(") ", 1)[-1].split()
            parents[int(entry.name)] = int(tail[1])  # ppid, after `state`
        except (OSError, IndexError, ValueError):
            continue
    found: list[int] = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        children = [child for child, above in parents.items() if above == parent]
        found.extend(children)
        frontier.extend(children)
    return sorted(found)


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


class HostMemory(NamedTuple):
    """What the host can still give this process, and the shared memory it already counts."""

    #: bytes left before the tightest limit; -1 is UNREADABLE, never 0
    available: int
    #: shared (tmpfs/memfd) bytes charged where that limit is: the pinned weight tiers
    shmem: int


def host_memory() -> HostMemory:
    """The LIVE host budget: the tightest of every cgroup on this process's path (its own and
    each parent's `memory.high` and `memory.max`, less what that cgroup uses and cannot give
    back) and the host's `MemAvailable`.

    `memory.high` counts: past it the kernel throttles every allocation, and pinned memory
    cannot be reclaimed to get back under it. A cgroup's page cache is reclaimable, so it is
    available, as `MemAvailable` counts it. §3.2's host-RAM rule: re-read at every grant,
    because a budget fixed at boot describes a machine nobody else is on.
    """
    _linux_only("host_memory")
    meminfo = _meminfo()
    best = HostMemory(meminfo.get("MemAvailable", -1), meminfo.get("Shmem", 0))
    for cgroup, limits, usage in _memory_cgroups():
        try:
            bounds = [int(text) for name in limits if (text := _cgroup_text(cgroup, name)) != "max"]
            # cgroup v1 spells "unlimited" as a page-rounded value near LONG_MAX
            bounds = [bound for bound in bounds if 0 < bound < (1 << 60)]
            if not bounds:
                continue
            stat = _cgroup_stat(cgroup)
            used = int(_cgroup_text(cgroup, usage)) - stat["inactive_file"] - stat["active_file"]
            room = max(min(bounds) - max(used, 0), 0)
        except (OSError, ValueError):
            continue
        if best.available < 0 or room < best.available:
            best = HostMemory(room, stat["shmem"])
    return best


def available_host_bytes() -> int:
    """`host_memory().available`: -1 means UNREADABLE, never 0."""
    return host_memory().available


def _memory_cgroups() -> Iterator[tuple[Path, tuple[str, ...], str]]:
    """This process's memory cgroup and every parent of it, innermost first: its directory,
    its limit files and the file holding its usage."""
    root = Path("/sys/fs/cgroup")
    limits: tuple[str, ...]
    try:
        if (root / "cgroup.controllers").is_file():
            relative = next(
                line.split("::", 1)[1]
                for line in Path("/proc/self/cgroup").read_text().splitlines()
                if line.startswith("0::")
            )
            leaf, top = root / relative.lstrip("/"), root
            limits, usage = ("memory.high", "memory.max"), "memory.current"
        else:
            leaf = _cgroup_v1_controller_path("memory")
            # the controller's mount; a container's own cgroup is mounted there itself
            top = next((p for p in (leaf, *leaf.parents) if p.name == "memory"), leaf)
            limits, usage = ("memory.limit_in_bytes",), "memory.usage_in_bytes"
    except (ContainmentUnavailable, OSError, StopIteration):
        return
    for cgroup in (leaf, *leaf.parents):
        yield cgroup, limits, usage
        if cgroup == top:
            return


def _cgroup_text(cgroup: Path, name: str) -> str:
    return (cgroup / name).read_text().strip()


def _cgroup_stat(cgroup: Path) -> dict[str, int]:
    """A cgroup's page cache and shared memory, in bytes (0 where it does not say). cgroup v1
    states its subtree's under `total_`."""
    stat = {"inactive_file": 0, "active_file": 0, "shmem": 0}
    rows = dict(
        line.split(None, 1) for line in (cgroup / "memory.stat").read_text().splitlines() if line
    )
    for name in stat:
        stat[name] = int(rows.get(f"total_{name}", rows.get(name, "0")))
    return stat


def _meminfo() -> dict[str, int]:
    """`/proc/meminfo` fields in bytes (empty when unreadable)."""
    fields: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, rest = line.partition(":")
            fields[name] = int(rest.split()[0]) * 1024
    except (OSError, IndexError, ValueError):
        return {}
    return fields
