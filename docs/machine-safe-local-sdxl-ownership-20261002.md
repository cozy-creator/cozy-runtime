# Safe local descriptor candidate ownership

- Owner: Codex agent `runtime_package_bridge`.
- Purpose: preserve latest Runtime context/retry/compute-fence safety, then integrate
  negotiated machine descriptor sources and bounded full verification for root's local SDXL gate.
- Branch: `feat/machine-safe-local-sdxl-20261002`.
- Worktree: `/home/fidika/cozy/.worktrees/cozy-runtime/machine-safe-local-sdxl-20261002`.
- Fetched base: origin/master `0d67384aa84aaef7b8b169a1a315eb96e99f0b39`.
- Safety ancestry to retain: `722913ee` compute-stream fence/resident seal, `40eab632`
  measured-memory-progress retry guard, `027a73ed` context admission, `1c7554a2` bind recovery.
- Owned scope: source records/provider/seam/Checkpoint/backend integration and CPU tests/docs.
- No primary edits, model implementation copies or authored execution for describe. Existing
  candidate worktrees/environments remain retained. No arbitrary peer floor/environment switches.
- Root alone operates local GPU/inference/NVML/processes/rentals. DCE owns coherent native
  TensorFS safety integration/CPU packaging; this task performs CPU source checks/packaging only.
