# Machine descriptor source ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: optional source-provider seam through the existing Checkpoint/ModelRegistry/
  Loader/PlaneBackend path for unchanged model packages; CPU metadata/source gates first.
- Branch: `feat/machine-descriptor-source-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/machine-descriptor-source-20261002`.
- Base: fetched `origin/master` `96942c19a31c600e265009e137054ba0bda09c40`.
- Dependency candidate: TensorFS descriptor-source PR #300, implementation `2df6f51`.
- Owned scope: typed source requests/replies, optional Checkpoint/backend source provider,
  negotiated device-executor hookup, CPU source tests and documentation.
- Primary Runtime untracked fusion work and all primary repositories remain untouched.
- Root coordinates GPU use/rentals; this agent does not independently run GPU work.
- Legacy provider and peers remain supported; no global SDK version floors.
