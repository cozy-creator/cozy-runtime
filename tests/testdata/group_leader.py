"""Rank 0 as the machine starts it (`executor.main`). Its followers run isolated (`-I`), so the
test's package, outside this environment, is put on their path as an installed one is.
argv: <package dir> <executor arguments>."""

import sys

from cozy_runtime.internal import executor
from cozy_runtime.internal.parallel import group

PACKAGE = sys.argv.pop(1)
sys.path.insert(0, PACKAGE)
_argv = group.executor_argv


def executor_argv(*, rank: int, world: int, rank_fd: int, root: str) -> list[str]:
    argv = _argv(rank=rank, world=world, rank_fd=rank_fd, root=root)
    argv[1] = f"import sys; sys.path.insert(0, {PACKAGE!r}); {argv[1]}"
    return argv


group.executor_argv = executor_argv
raise SystemExit(executor.main())
