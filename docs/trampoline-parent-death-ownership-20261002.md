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
