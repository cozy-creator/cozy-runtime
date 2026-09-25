Owner: /root/astra_cohort_audit
Purpose: distribute managed single-device child inference over the granted GPU envelope, then safely regroup for sequence-parallel inference.
Branch: feat/managed-child-gpu-fanout-20260925
Base: 8a4e7101

The existing machine-owned call journal retains each child. Runtime selects a deterministic placement replica and device group from that journal and the actual worker envelope. Convergence preserves disjoint placements and uses existing process retirement before overlapping group replacement. CPU orchestration parents hold no GPU lane. No package-specific scheduling, GPU-count input, CLI subprocess, rental, image activation, or live daemon changes.

Qualification must show concurrent independent lane execution, queued excess children, stable replica reuse, cancellation, and empty/reclaimed lanes before group fusion. CPU process evidence is not GPU inference evidence; live multi-GPU qualification belongs to the root task.
