# Matched diagnostic instrumentation ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: aggregate source/registration timing in an immutable diagnostic SDK cohort;
  actual GPU runs remain root-owned.
- Branch/worktree: `feat/machine-descriptor-matched-profile-20261002` under
  `/home/fidika/cozy/.worktrees/cozy-runtime/machine-descriptor-matched-profile-20261002`.
- Fetched origin/master: `c3964503e59f34fa1e9339c6cb6f2e4a9bb1e323`.
- Matched base: verified owned candidate `c723baf59b9388a1748c37af5f21188b90785f6a`.
  Root explicitly requested the existing Runtime100/TFS candidate cohort; current upstream
  has unrelated weights/loader behavior changes and is retained separately, not merged.
- Owned scope: trusted internal model-source and weight-registration diagnostic counters.
- Integrity checks, model/worker behavior and legacy envs are unchanged. No model code
  imports or GPU operations are used to validate this diagnostic.
