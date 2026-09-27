"""Executor spawn: the env SEAL and the exec-first launch.

Class B of tracker-v2/spawn-allowlists.md (#616.d); `internal/child_env.py` holds the
erase/impose tables that must stay equal to that row.

The seal is ERASE -> IMPOSE -> SEAL and it completes BEFORE the child's CUDA init, because
the allocator reads its environment exactly once and a late imposition is a silent no-op
(pgw#1640). Here the three steps are three different places, on purpose:

* **ERASE** happened in `internal/config.py`, at the process's one environment read:
  `RuntimeConfig.child_base_env` is already stripped of every `child_env.ERASED_PREFIXES`
  name, credentials included.
* **IMPOSE** is `project_child_env`, which refuses any name outside the reviewed allowlist
  and any allowlisted name the erase pass would not have cleared first.
* **SEAL** is this module: the composed mapping is handed to `posix_spawn` as the child's
  COMPLETE environment. The child inherits nothing implicitly, and `seal_digest` lets the
  executor prove at CUDA-init time that the environment it is initializing under is exactly
  the one that was sealed — the late-imposition red arm is a mismatch of that digest, not a
  hope that nobody mutated the process environment afterwards.

`posix_spawn` is used with FILE ACTIONS ONLY: no Python runs between fork and exec in this
threaded process. The program it execs is `internal/trampoline.py`, which is a fresh process
and does the privilege/parent-death work before exec'ing the executor.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

from cozy_runtime.internal import canonical, proctree
from cozy_runtime.internal.child_env import project_child_env

#: The executor raises its own OOM score so the kernel picks the disposable process. The
#: worker never touches its own (it stays a bare interpreter for victim ordering, §3.1).
EXECUTOR_OOM_SCORE_ADJ = 500


@dataclass(frozen=True, slots=True)
class Spawned:
    pid: int
    env: Mapping[str, str]
    seal_digest: str
    argv: tuple[str, ...]


def seal(base: tuple[tuple[str, str], ...], imposed: Mapping[str, str]) -> dict[str, str]:
    """ERASE (already applied to `base`) -> IMPOSE (allowlisted) -> the sealed mapping."""
    sealed = dict(base)
    sealed.update(project_child_env(imposed))
    return sealed


def seal_digest(env: Mapping[str, str]) -> str:
    """The identity of a sealed environment: the imposed, allowlist-relevant names only.

    A digest over the WHOLE environment would move whenever an unrelated inherited name
    changed and would prove nothing about the seal. This one covers exactly the names the
    seal is responsible for.
    """
    from cozy_runtime.internal.child_env import ALLOWLIST

    return canonical.digest({name: env.get(name, "") for name in sorted(ALLOWLIST)})


def spawn_executor(
    *,
    python: str,
    module_argv: list[str],
    base_env: tuple[tuple[str, str], ...],
    imposed: Mapping[str, str],
    cwd: str,
    scope: proctree.ExecutorScope,
    uid: int = -1,
    gid: int = -1,
) -> Spawned:
    """EXEC-FIRST launch of one device executor. Returns before the child has run anything."""
    env = seal(base_env, imposed)
    containment = (
        [
            "--scope-backend",
            "cgroup_v2",
            "--cgroup",
            scope.relative_path,
            "--cgroup-inode",
            str(scope.inode),
        ]
        if isinstance(scope, proctree.CgroupScope)
        else ["--scope-backend", "process_tree"]
    )
    argv = [
        python,
        "-I",
        "-c",
        "import sys; from cozy_runtime.internal.trampoline import main; main(sys.argv[1:])",
        "--expect-parent",
        str(os.getpid()),
        "--oom-adj",
        str(EXECUTOR_OOM_SCORE_ADJ),
        "--uid",
        str(uid),
        "--gid",
        str(gid),
        *containment,
        "--",
        python,
        "-I",
        *module_argv,
    ]
    pid = os.posix_spawn(
        python,
        argv,
        env,
        file_actions=[(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0o600)],
        setsigdef=[],
        setsigmask=[],
    )
    del cwd  # the executor's cwd is the worker's; the spool is passed as a path
    return Spawned(pid, env, seal_digest(env), tuple(argv))


@dataclass(slots=True)
class Child:
    """One spawned follower as its rank-0 parent holds it: the pid and its exit status,
    read through `waitpid` exactly once (the same shape `subprocess.Popen` gives).

    As in `Popen`, one lock serializes the reap: the group's seam reader polls while rank 0
    may be waiting, and the thread that loses a reap race must not record -1 over the
    status the winner read."""

    pid: int
    returncode: int | None = None
    _reap: threading.Lock = field(default_factory=threading.Lock)

    def poll(self) -> int | None:
        if self.returncode is None and self._reap.acquire(blocking=False):
            try:
                if self.returncode is None:
                    try:
                        pid, status = os.waitpid(self.pid, os.WNOHANG)
                    except ChildProcessError:
                        return self.returncode
                    if pid == self.pid:
                        self.returncode = os.waitstatus_to_exitcode(status)
            finally:
                self._reap.release()
        return self.returncode

    def wait(self) -> int:
        with self._reap:
            if self.returncode is None:
                try:
                    _pid, status = os.waitpid(self.pid, 0)
                    self.returncode = os.waitstatus_to_exitcode(status)
                except ChildProcessError:
                    self.returncode = -1
        return self.returncode

    def send_signal(self, number: int) -> None:
        os.kill(self.pid, number)


def spawn_follower(
    *, python: str, module_argv: list[str], env: Mapping[str, str], inherit_fd: int
) -> Child:
    """EXEC-FIRST launch of ONE follower rank by rank 0 (cr-068).

    The same trampoline the worker execs, under its `inherit` backend: the parent check,
    pdeathsig, no_new_privs and the executor's OOM score are set in a fresh process, and
    the follower keeps rank 0's cgroup and PGID exactly as inherited. `posix_spawn` with
    FILE ACTIONS ONLY, like the worker's own launch: rank 0 is threaded by now (its
    follower readers) and runs no Python between fork and exec. The environment is rank
    0's own sealed one, verbatim; `inherit_fd` is the follower's seam and the only
    descriptor made inheritable.
    """
    argv = [
        python,
        "-I",
        "-c",
        "import sys; from cozy_runtime.internal.trampoline import main; main(sys.argv[1:])",
        "--expect-parent",
        str(os.getpid()),
        "--oom-adj",
        str(EXECUTOR_OOM_SCORE_ADJ),
        "--uid",
        "-1",
        "--gid",
        "-1",
        "--scope-backend",
        "inherit",
        "--",
        python,
        "-I",
        *module_argv,
    ]
    os.set_inheritable(inherit_fd, True)
    pid = os.posix_spawn(
        python,
        argv,
        dict(env),
        file_actions=[(os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0o600)],
        setsigdef=[],
        setsigmask=[],
    )
    return Child(pid)
