# Replay contract proposal for root review

Tracker #327; original red arm uses 50af76dd. Root reviewed this proposal and the
decoder leaf guard on October 5. The implementation/proof is in `recovery-contract.md`.

**CONFIRMED CPU:** three expected contract failures and one successful-path pass.
The actual production wrapper learned `region.mutates=False` in every case, then
replayed a complete real Module forward after a one-shot OOM raised outside its
Torch operators. Inputs stayed bit-identical and all three acquisitions released.
Counter/buffer outputs became [3.25,2.5] instead of [2.25,1.5]; RNG consumed a third
sample instead of the second. The normal successful wrapper matched its control.
Device leases and allocator observations were CPU I/O fixtures; this does not
claim a physical CUDA OOM, full model or ordinary-CLI qualification.

## Proposed narrow behavior

1. Successful forwards, lease acquisition/rebinding and before-forward capacity
   recovery keep their current behavior.
2. Input-write observation remains an observation. It grants no whole-forward
   replay authority. Generic Module forwards recover at eligible Torch operator
   boundaries, rather than running their Python frame again.
3. A whole forward/body can replay only through an explicit trusted contract.
   The initial internal contract is the existing reviewed image decoder pass:
   immutable input map, separate output map, scratch/statistics reset on entry,
   no RNG or caller state, and output committed only after the step succeeds.
   Its body is SDK-owned. A model's declaration that its tensor inputs were not
   written is insufficient; no generic introspection-based state snapshot is added.
4. An OOM escaping an uncontracted forward ends that attempt with a distinct
   recovery-contract failure and marks the residency/generation unsafe. Use the
   existing poisoned-generation propagation (`weights.py:3766`,
   `executor.py:3103`) so a later operation cannot reuse changed buffers/RNG/state.
   Explicit user retry remains a new attempt; do not resubmit a started run or
   relabel contract absence as a measured physical-capacity limit.
5. Preserve the known-safe functional Torch allocation/operator recovery path.
   Mutable, RNG and opaque/custom operations need their own defensible operator
   contract if an OOM escapes them; this proposal does not infer it from schema
   mutability alone. The first production slice is whole-forward authority and
   retirement. Root should review the exact eligible operator boundary before
   broadening that slice, using current code rather than historical draft1163.

## Required next proof

After root review, assert the real counter/buffer/RNG cases stop without a second
forward and expose unsafe/retired state; keep the successful control exact.
Exercise a known-safe functional Torch operator retry and the existing trusted
image decoder pass replay. Then prove the executor's actual failure-to-replacement
consumer path. Existing paging/spare-idle paths must keep working on unchanged
requests; physical low-memory/ordinary-CLI qualification remains a separate gate.
