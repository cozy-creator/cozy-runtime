# P3 owner-controlled cache reclamation

Status: source/design audit; no implementation or new GPU qualification yet.
Owner: /root/retention_restart. Root owns builds, SDK installation, rentals and GPU runs.

## Existing mechanism and missing link

The native working allocator already excludes a graph pool while it is prepared,
referenced, capturing, contains live outputs, or has unfinished replay/consumer events
(`working.rs::trim_ready_capped`, `pool_record_stream`, `pool_release`). This remains
the physical safety authority. It must not infer graph retirement from an idle timer.

Runtime already has owner-side retirement: `residency.Region.pin/discard`,
`Memory._retire` and `Model.retire_graphs`; denoise loops register `program.close`,
and LM registers `_drop` while retaining a separate pin for live KV rows. These
callbacks currently run only from synchronous `Memory.reclaim` preparation paths.
Native `working::alloc_inner`, Region preparation and cooperative donor service see
no requestable Python-owned cache. Native ledger `Physical.caches` is presently zero.
There is also no page-addressed KV reservation with interior release/recommit.

## Small API and ownership seam

1. Add a native `Cache` reservation backed by the existing per-device Working owner.
   `new(device, reserve_bytes)`, `base`, `granularity`, `commit(offset, bytes)`,
   `release(offset, bytes, stream)`, `stats`, and explicit `close` are sufficient.
   A zero-reservation registration represents an external owner such as a graph cache;
   it does not invent backing bytes. Public C ABI versioning remains P4 work.
2. A committed page is live owner storage. An owner release first records the supplied
   stream completion, then changes those exact pages to pending release. Only completed
   releases are unmapped/released. Recommit requires the exact range to be entirely
   unmapped. Commit is all-or-nothing, including rollback after real driver exhaustion.
   A neighboring live range retains the same VA and bytes. The owner must join every
   consumer before its release stream; no arbitrary Torch alias is silently remapped.
3. One narrowly scoped cache-owner registration carries the shrink callback and
   lifecycle, not a byte estimator. The allocator holds host-only request metadata,
   never the last callback/graph/Driver owner. A separate callback worker owns callbacks;
   cooperation's native worker must remain independent of Python/GIL progress.
4. Native local pressure queues `shrink(requested_bytes)` after completed working pages
   and native region reclaim. Donor service queues the same intent after its native
   pass. Callbacks run after allocator/arena/ledger mutexes are released and cannot be
   invoked on malloc/free stacks. The allocating thread never waits for callback entry,
   return, GIL acquisition or future preemption. Actual returned pages may help its real
   retry; otherwise it can return Exhausted and a later request can use those pages.
   Submitted release fences are the only pending-release evidence, never callback asks.
5. Retirement closes admission before draining queued/running callbacks. Explicit Python
   close detaches the GIL while draining; close from the callback itself refuses.
   Finalizers enqueue retirement rather than blocking. Callback leases and backing
   owners are released only outside native locks and capture/replay exclusion. A private
   fork rejects inherited handles before inherited mutexes. The callback worker and
   close paths must preserve owners if a capture or completion failure prevents release.

## Native storage and accounting

Keep cache pages in a separate range table in Working's device state, sharing its
Driver, primary context, capture exclusion and actual physical ceiling. Do not present
live cache pages as working free blocks or distort the phase working-memory watermark.
Ledger physical.caches reports live/pending physical storage without double-counting
working or graph-pool backing. Owner requests and virtual reservations contribute zero
reclaimable bytes. Every completed unmap updates the same managed total used by actual
allocation admission; no external allocator, byte target or reservation grant appears.

Initial cache callbacks do not add a priority model that regions lack. The current
one-class native region order and existing graph pool safety are preserved; full
per-priority/global multi-arena policy remains separately bounded in the design.

## Runtime adoption

Bridge existing `Region` pins and nonblocking discard to the native registration.
Use weak owner links to avoid native-Python cycles. A pinned or busy owner declines by
releasing nothing. Successful discard marks the owner fresh before a subsequent use;
a failed retirement must refuse subsequent replay/use rather than publish fresh state.
Register model graph retirement separately through the existing `_retire` exclusion;
never call broad `Memory.reclaim` from the native callback, which would recreate a
Python byte policy. Keep graph executable/output/stream completion lifetimes intact.
Existing LM live-row pins continue to protect live KV; migrating Transformers KV tensor
layout to page-addressed storage is a separate consumer adoption, not inferred here.

## Required proof before qualification

- Real KV-style three-range reservation: release/recommit the middle, stable VA and
  unchanged neighbors; overlap/misalignment refusal; real commit failure rollback.
- Real stream-delayed release: only submitted events count pending, no early unmap;
  capture/replay refuse mapping/retirement without driver work.
- Owner callback declines, raises, closes concurrently and is blocked on GIL held by
  a genuine allocating native call. That allocation returns without waiting on Python;
  later callback progress remains separate. No mocks or fabricated OOM.
- Genuine Runtime captured graph cache: native pressure requests retirement; replay
  eligibility closes before references drop, an escaped output remains valid, and
  physical backing becomes reclaimable only after all actual owners/work retire.
- Cooperative donor accounting reports only actual completed cache release; request
  generation and callback coalescing never multiply byte credit. Same-PID cleanup and
  fork/close tests retain exact owners on refusal/error.

Foreign-pressure sampling/reserve policy and complete TensorD byte-policy retirement
remain separate P3 work. This seam does not claim either, nor full-model performance,
public API completion, release or deployment.
