# Source verification CPU entitlement ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: measured loader optimization; reuse shared cgroup path resolution for v1/v2
  CPU quotas/affinity, preserve full native integrity and bounded fd/byte verification.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/machine-source-cpu-entitlement-20261002`.
- Branch: `feat/machine-source-cpu-entitlement-20261002`.
- Fetched origin/master base: `b3416e34` (includes current stage-end timing fix).
- Carried safety descriptor baseline: `9eb29a1c`, clean merge `8e1acba01e46d8e5acd98e5a8c13327ac1f3bf44`.
- Both matched arms must use the same new base/stage timing, launch/cache/allocator/resource settings.
- Owned files: shared proctree CPU entitlement helper, verifier call site, narrow existing
  CPU consumers and CPU quota/native integrity tests/docs. No new control plane or D2 surface.
- Legacy loading stays default. Descriptor promotion requires matched parity or better;
  full byte verification and inference precision/inputs remain unchanged.
- No GPU/NVML/rental calls; root owns hardware checks.
- Draft PR: https://github.com/cozy-creator/cozy-runtime/pull/1114
- CPU gates include actual v1/v2 filesystem fixtures, native full verification and corrupt
  byte rejection, strict fd cleanup, compiler/weight-policy and real cross-version inference.
- Existing quantization SIM115 lint at spill TemporaryFile predates this task; unchanged.
