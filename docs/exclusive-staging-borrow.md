# Exclusive borrowing of existing staging

StreamingFillBackend.borrow_staging(required_bytes) returns an exclusive HostStaging-compatible lease or None. It uses only the already-owned, disjoint, fully pinned ring buffers and reports their actual storage capacity. It neither allocates backing nor grows a ring, reclaims another tenant, makes an RPC, or enables a Planner transfer policy. Insufficient capacity, missing/pageable buffers, an active native transaction, or another borrow declines.

The borrower must keep the lease in executor-lifetime custody and call release only after proving all DMA using the buffers complete. Release is idempotent after a valid return; an old returned lease cannot disarm a subsequent owner. Releasing an unowned/poisoned token refuses rather than guessing that storage is safe. Ending a borrow is not RSS relief.

A short lock serializes ownership changes and trusted host-quarantine bookkeeping, including the drained callback. It is never held across CUDA work, native reading or RPC. Native transactions claim their busy state before allocation/dependency setup, closing the earlier check-to-use gap. A borrow marks the actual existing ring live, so construction-finalization and unload keep their established failed-drain protection.

Idle release declines while borrowed. Native fill, prepare or close attempted during a borrow is a typed device_fault and latches poison. The borrow and host quarantine remain held even if the borrower subsequently attempts release. The caller must cancel/drain a forecast before such operations; this defensive fault is not an alternative cancellation protocol. Existing native-fill drain behavior remains synchronous.

Guarded CPU tests cover real buffer geometry, delayed simulated copy completion through HostTransfer, atomic release/reborrow callbacks, native-claim exclusion, stale tokens, poison retention, successful-construction finalizers and unload/GC. No GPU/DMA performance qualification or Planner enablement is included.
