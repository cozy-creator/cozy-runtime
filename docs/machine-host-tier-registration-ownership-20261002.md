# Host tier SDK registration ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: optional typed manifest/traversal/partition registration for machine-created
  TensorFS host tiers. Preserve released baseline and latest Runtime safety fences.
- Branch: `feat/machine-host-tier-registration-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/machine-host-tier-registration-20261002`.
- Fetched origin/master base: `0d67384aa84aaef7b8b169a1a315eb96e99f0b39`.
- Carried safe descriptor checkpoint: `9eb29a1c` (PR1111); task delta follows it.
- Owned files: new `internal/host_tiers.py`, narrow request/Load/HostTiers prepare additions
  and CPU source/transport tests/docs. No worker port or model implementation.
- No GPU/NVML/rental/process management; root owns hardware qualification.
- Draft PR: https://github.com/cozy-creator/cozy-runtime/pull/1112
- Paired owner: cozy-machine PR18/native host plane; wire/header prerequisite e9c520a.
- 46 actual SDK/source/native CPU gates pass; checksum and readonly integrity unchanged.
- Preserve original local-safe env/wheel, old candidate/profiler envs and primary checkouts.
