# Catalog checkpoint root inputs

Owner: Codex `/root/consumer_delivery_finish`.
Worktree: `~/cozy/.worktrees/cozy-runtime/catalog-checkpoint-root-inputs`.
Branch: `fix/catalog-checkpoint-root-inputs`; fetched base `f0808eeb`.
Integrated parents: root `f204aed9`, native byte input owner `d55725ac`.
Protocol: wire55 source grant from `a494e5c` (worker-protocol PR86).
Native capability: TensorFS PR173 / tracker tfs-117, public TensorFS 0.3.43 from tag 30fe20c, whose actual wheel/source and native
controls were verified after official publication.

An explicit catalog repository grant chooses this path. TensorFS validates its
native repository checkpoint membership and verified runtime closure, then retains
an independent CheckpointRoot without creating a derived writer/producer/receipt.
Absent catalog provenance keeps the existing strict owner-derived Model handling.

Schema11 journals the execution's input obligation in execution_checkpoint_inputs.
Native roots, not these rows, own bytes. All source inputs are preflighted before
creating recipients. Failed unaccepted intake releases its own catalog pins; a
retry uses a new native owner generation because released native roots are
permanent. An accepted submission replay does not reacquire collected inputs.
The execution insert checks catalog custody atomically with byte-intake admission.
Collection/abandonment release catalog pins independently of other input kinds.

Public TensorFS 0.3.42 and Runtime 0.18.0 remain immutable. This draft must not be
released with the old TensorFS floor. The dependency and committed lock now use
genuine TensorFS 0.3.43 after source/wheel and ordinary standalone CLI proof;
Runtime 0.18.1 version/publication remain coordinated by root.
