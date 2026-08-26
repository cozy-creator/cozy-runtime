"""The exec-first trampoline: the only code that runs between the supervisor and CUDA.

§3.1 — a threaded supervisor never runs Python between fork and exec (v1: a `preexec_fn`
running in a fork of a process that already has gRPC threads corrupts epoll state; naive
`posix_spawn` loses the privilege and parent-death setup). So the supervisor uses
`posix_spawn` with FILE ACTIONS ONLY and execs THIS module, which is a fresh process with no
inherited threads, no gRPC state and no torch. It sets the process properties that must
exist before CUDA, then `execv`s the executor — so the interpreter that initializes CUDA is
the third exec, and it starts with the seal already applied.

Order matters and is the point:

1. **expected parent** — `getppid()` must be the pid the supervisor named. A child that was
   reparented before it got here is an orphan and exits rather than serving.
2. **pdeathsig** — SIGKILL on parent death, then the parent check RUNS AGAIN: between (1)
   and (2) the parent could have died, and pdeathsig only fires on future deaths. The
   re-check is what closes that race.
3. **no_new_privs** — after this, exec can never gain privileges; setuid bits are inert.
4. **oom_score_adj** — the EXECUTOR raises its OWN score. The supervisor stays a bare
   interpreter at the default score so the kernel's OOM victim ordering picks the disposable
   process, and nothing in the supervisor needs privilege to arrange it.
5. **uid/gid** — dropped only when this process is privileged enough to drop them; on an
   unprivileged box there is nothing to drop and the step is a recorded no-op, never a
   pretended success.
6. **signals** — default dispositions and an empty mask, so an inherited mask cannot make a
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
    """`trampoline --expect-parent PID [--oom-adj N] [--uid U] [--gid G] -- ARGV...`"""
    expect_parent = 0
    oom_adj = 0
    uid = gid = -1
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
        else:
            _fail(f"unknown flag {flag!r}", Exit.usage)
        index += 2
    if not rest:
        _fail("no program to exec", Exit.usage)

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

    # Its own process group, so the supervisor's forceful path kills the executor AND every
    # grandchild it spawned (dataloader workers, compile subprocesses) with one `killpg`
    # instead of leaving orphans behind a dead parent. pdeathsig survives this.
    proctree.own_process_group()
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGXFSZ, signal.SIGPIPE):
        signal.signal(number, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_SETMASK, set())
    os.execv(rest[0], rest)


if __name__ == "__main__":
    main(sys.argv[1:])
