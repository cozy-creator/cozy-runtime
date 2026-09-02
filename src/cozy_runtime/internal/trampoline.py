"""The exec-first trampoline: the only code that runs between the worker and CUDA.

§3.1 — a threaded worker never runs Python between fork and exec (v1: a `preexec_fn`
running in a fork of a process that already has gRPC threads corrupts epoll state; naive
`posix_spawn` loses the privilege and parent-death setup). So the worker uses
`posix_spawn` with FILE ACTIONS ONLY and execs THIS module, which is a fresh process with no
inherited threads, no gRPC state and no torch. It sets the process properties that must
exist before CUDA, then `execv`s the executor — so the interpreter that initializes CUDA is
the third exec, and it starts with the seal already applied.

Order matters and is the point:

1. **expected parent** — `getppid()` must be the pid the worker named. A child that was
   reparented before it got here is an orphan and exits rather than serving.
2. **pdeathsig** — SIGKILL on parent death, then the parent check RUNS AGAIN: between (1)
   and (2) the parent could have died, and pdeathsig only fires on future deaths. The
   re-check is what closes that race.
3. **containment** — join the worker-created cgroup-v2 scope, or prove the exact inherited
   process-tree token in a positively qualified credential-unwritable container — or, for
   a follower rank of a group lane (cr-068, `inherit`), keep rank 0's scope and PGID exactly
   as inherited: a follower is a child of the executor, never of the worker.
4. **no_new_privs** — after this, exec can never gain privileges; setuid bits are inert.
5. **oom_score_adj** — the EXECUTOR raises its OWN score. The worker stays a bare
   interpreter at the default score so the kernel's OOM victim ordering picks the disposable
   process, and nothing in the worker needs privilege to arrange it.
6. **uid/gid** — dropped only when this process is privileged enough to drop them; on an
   unprivileged box there is nothing to drop and the step is a recorded no-op, never a
   pretended success.
7. **signals** — default dispositions and an empty mask, so an inherited mask cannot make a
   child unkillable by the cancellation path.

Exit codes are the runtime's shared matrix so a trampoline refusal is distinguishable from
an executor failure in `waitpid` alone.
"""

from __future__ import annotations

import os
import signal
import sys

from cozy_runtime.internal import proctree
from cozy_runtime.internal.exits import Exit


def _fail(message: str, code: Exit) -> None:
    sys.stderr.write(f"trampoline: {message}\n")
    sys.stderr.flush()
    os._exit(int(code))


def main(argv: list[str]) -> None:
    """Seal one child and enter its selected containment scope before executor import."""
    expect_parent = 0
    oom_adj = 0
    uid = gid = -1
    cgroup_path = ""
    cgroup_inode = 0
    scope_backend = ""
    rest: list[str] = []
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag == "--":
            rest = argv[index + 1 :]
            break
        value = argv[index + 1]
        if flag == "--expect-parent":
            expect_parent = int(value)
        elif flag == "--oom-adj":
            oom_adj = int(value)
        elif flag == "--uid":
            uid = int(value)
        elif flag == "--gid":
            gid = int(value)
        elif flag == "--cgroup":
            cgroup_path = value
        elif flag == "--cgroup-inode":
            cgroup_inode = int(value)
        elif flag == "--scope-backend":
            scope_backend = value
        else:
            _fail(f"unknown flag {flag!r}", Exit.usage)
        index += 2
    if not rest:
        _fail("no program to exec", Exit.usage)
    if scope_backend == "cgroup_v2":
        if not cgroup_path or cgroup_inode <= 0:
            _fail("an exact cgroup-v2 executor scope is required", Exit.structural)
    elif scope_backend in ("process_tree", "inherit"):
        if cgroup_path or cgroup_inode:
            _fail(f"a {scope_backend} scope cannot also name a cgroup", Exit.structural)
    else:
        _fail(f"unknown containment backend {scope_backend!r}", Exit.structural)

    if expect_parent and os.getppid() != expect_parent:
        _fail(
            f"expected parent {expect_parent}, found {os.getppid()} — an already-orphaned "
            "child never becomes an executor",
            Exit.structural,
        )
    try:
        proctree.arm_parent_death()
    except proctree.ProcessTreeUnsupported as exc:
        _fail(str(exc), Exit.structural)
    except OSError as exc:
        _fail(f"PR_SET_PDEATHSIG: {exc}", Exit.internal)
    if expect_parent and os.getppid() != expect_parent:
        # The parent died in the window above; pdeathsig only covers FUTURE deaths.
        _fail("parent died before pdeathsig was armed", Exit.structural)
    if scope_backend == "cgroup_v2":
        try:
            proctree.join_executor_cgroup(proctree.CgroupScope(cgroup_path, cgroup_inode))
        except (proctree.ContainmentUnavailable, proctree.ProcessTreeUnsupported) as exc:
            _fail(str(exc), Exit.structural)
    try:
        proctree.seal_no_new_privs()
    except OSError as exc:
        _fail(f"PR_SET_NO_NEW_PRIVS: {exc}", Exit.internal)
    try:
        proctree.raise_own_oom_score(oom_adj)
    except OSError as exc:
        _fail(f"oom_score_adj={oom_adj}: {exc}", Exit.internal)

    if gid >= 0 and os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
    if uid >= 0:
        if os.geteuid() == 0:
            os.setuid(uid)
        elif uid != os.geteuid():
            _fail(
                f"--uid {uid} requested at euid {os.geteuid()}: an unprivileged trampoline "
                "cannot become another user, and pretending it did is worse than refusing",
                Exit.structural,
            )

    # Keep executor-directed signals out of the worker's group. Under process-tree
    # containment, stop before package import so the parent can fsync this exact birth and
    # PGID before allowing any first-party descendant to exist. A FOLLOWER RANK (cr-068,
    # `inherit`) is spawned by rank 0 inside rank 0's scope: it keeps rank 0's cgroup and
    # PGID by inheritance, so it dies with the group under either backend, and joins nothing.
    if scope_backend != "inherit":
        proctree.own_process_group()
    if scope_backend == "process_tree":
        os.kill(os.getpid(), signal.SIGSTOP)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGXFSZ, signal.SIGPIPE):
        signal.signal(number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, set())
    os.execv(rest[0], rest)


if __name__ == "__main__":
    main(sys.argv[1:])
