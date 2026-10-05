# Current block replay red arm

Tracker #327 (parent #320/#319). Owner: Codex Runtime memory lane.
Base: current origin/master 50af76dd216c56c8fabe978a049db8aa29a896f7.

At base 50af76dd, the actual WeightResidency._block path learned only whether a forward wrote its
input tensor storage. Once region.mutates is false, it retries the whole forward
when OOM occurs and input versions are unchanged. Module/Python state and RNG
are outside that observation.

The first proof uses real Torch modules and the imported production wrapper,
not a reproduction of the retry algorithm. CPU-only acquisition/accounting
boundaries may supply controlled leases and room receipts. The successful path,
first-pass mutation learning, real dispatch, OOM catch, lease release and replay
must execute their actual current source. A counter, a module buffer and RNG
advance before the deliberate one-shot OOM; input tensors stay unchanged.
Compare that call with the no-OOM control and retain the state/output difference.

This branch initially changed tests/documentation only. The reviewed production
change and its current proof are described in `recovery-contract.md`. No GPU fault,
rental, release or smaller substitute inference request is part of this proof.
