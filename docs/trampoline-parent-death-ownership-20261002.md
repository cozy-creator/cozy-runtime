# Trampoline parent-death guard ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: restore SIGKILL-on-parent-death after the trampoline's UID/GID transitions,
  and check the expected parent again before executor import.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/trampoline-rearm-parent-death-20261002`.
- Branch: `fix/trampoline-rearm-parent-death-20261002`.
- Fetched origin/master base: `2745f3c72a2d36882ae25222cfe1375eb4d74373`.
- Owned scope: trampoline, narrow actual-process CPU tests and this qualification record.
- Existing quota/native comparison wheels and environments remain immutable.
- CPU tests only; no GPU/NVML/rentals. Privileged credential fixtures run only as root.
- The independently owned persistent-spawner fix is also required: Linux binds
  PDEATHSIG to the creating thread, not the lifetime of the entire parent process.
- Later filesystem credential changes may clear the guard again. This narrow fix
  does not claim protection across a privileged TensorFS store-owner adoption.
- Draft PR: https://github.com/cozy-creator/cozy-runtime/pull/1117.
- Local actual-process gates: nine passed, three root-only cases skipped. The tests
  observe the kernel guard after exec, actual creator death and wrong-parent refusal;
  process-exit observer gates also pass. Fixture pipes control termination; no timer
  kills or GPU/device imports are used.
- Privileged gates still required: foreign UID/GID exec and creator death, plus the
  native Store-owner interaction. The latter intentionally documents the existing
  loss of the guard; it is not a claim that this narrow trampoline fix repairs it.
- Current native 0.3.93 source: `filesystem_identity.rs:59-60` changes fsGID/fsUID,
  `:78-79` restores them, and `:41-42` skips adoption for nonroot callers. Neither
  transition re-arms PDEATHSIG. The fixture opens the actual native Store after
  changing the fixture Store's owner, rather than simulating native behavior.
- Ruff passes on all changed Python files. A scoped mypy invocation reports only
  146 existing generated `worker_pb2.pyi` errors, with none in the three owned files.
- The FD observer test duplicates a real pidfd above 2048; its test process raises
  its own soft FD limit to 4096. This is unrelated to descriptor consumer bounds.
