Owner: /root/astra_cohort_audit
Purpose: distribute managed single-device child inference over the granted GPU envelope, then safely regroup for sequence-parallel inference.
Branch: feat/managed-child-gpu-fanout-20260925
Base: 8a4e7101

The existing machine-owned call journal retains each child. Runtime selects a deterministic placement replica and device group from that journal and the actual worker envelope. Convergence preserves disjoint placements and uses existing process retirement before overlapping group replacement. CPU orchestration parents hold no GPU lane. No package-specific scheduling, GPU-count input, CLI subprocess, rental, image activation, or live daemon changes.

Qualification must show concurrent independent lane execution, queued excess children, stable replica reuse, cancellation, and empty/reclaimed lanes before group fusion. CPU process evidence is not GPU inference evidence; live multi-GPU qualification belongs to the root task.

Tracking: cozy-runtime issue #609, draft PR #607.
Implementation: machine_lanes owns durable replica selection and complete desired-set composition. Serving submissions retain chosen pins in the existing execution preparation; no protocol or package public API changes.
Validation checkpoint: durable 2/4-device reservation and ownership tests passed; real two-process disjoint-seal test adds a replica during a blocked invocation and transitions to a fused lane only after both terminal acknowledgments. These are CPU executor/process proofs, not GPU inference. Focused regression set running; mypy passes all six changed source/test modules.
Final focused checkpoint: 22 tests passed across durable reservations, preparation failures and real executor repartition, with one redundant case subsequently removed. Follow-up cold-start and external-GPU-owner cases passed (three focused cases), and changed-module mypy/ruff and all 21 architecture fences passed. The additional existing managed-child serving end-to-end fixture fails during initial preparation: prepare-nested cannot see cozy-runtime>=0.16.8,<1. The same failure was reproduced on clean primary master 8a4e7101 before scheduler execution. CI jobs did not start because the GitHub Actions budget prevents use. Actual multi-GPU model inference remains a separate live gate.
