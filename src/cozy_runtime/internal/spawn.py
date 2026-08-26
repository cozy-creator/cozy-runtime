"""Executor spawn: the env SEAL and the exec-first launch.

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
from collections.abc import Mapping
from dataclasses import dataclass

from cozy_runtime.internal import canonical
from cozy_runtime.internal.child_env import project_child_env

#: The executor raises its own OOM score so the kernel picks the disposable process. The
#: supervisor never touches its own (it stays a bare interpreter for victim ordering, §3.1).
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

    return canonical.digest({name: env[name] for name in sorted(ALLOWLIST) if name in env})


def spawn_executor(
    *,
    python: str,
    module_argv: list[str],
    base_env: tuple[tuple[str, str], ...],
    imposed: Mapping[str, str],
    cwd: str,
    uid: int = -1,
    gid: int = -1,
) -> Spawned:
    """EXEC-FIRST launch of one device executor. Returns before the child has run anything."""
    env = seal(base_env, imposed)
    argv = [
        python,
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
        "--",
        python,
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
    del cwd  # the executor's cwd is the supervisor's; the spool is passed as a path
    return Spawned(pid, env, seal_digest(env), tuple(argv))
