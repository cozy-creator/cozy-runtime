# Deferred prefetch eligibility

Queue hints still authorize only optional immutable Store reads. GPU grants, executor
startup, package imports and normal demanded loading are unchanged.

A queue observation previously scheduled one supervised pass. If that pass found a
foreground Store demand or an attempt preparing/entering/accepted, it returned without
recording a reason. Ending that demand or moving the attempt to running did not schedule
another pass when the queue was unchanged.

The existing Machine gate now publishes an empty-demand edge outside its condition.
AttemptEngine publishes its running edge after releasing the attempt transition lock.
Worker retains only deferred optional interest, rechecks current eligibility, and asks the
existing Supervisor to spawn/poke the same unit. It performs a fresh check after arming
interest, closing the release-before-arm race. There is no timer or new scheduling loop.
Callbacks are best effort: an optional scheduling failure cannot fail a StoreRead ACK or
an admitted execution. Queue removal, shutdown, owner/generation checks, per-batch native
reader cancellation, bounded host allowance and final drain before demand ACK remain intact.

Diagnostic reason changes are bounded to live frontier/activation identities. They and
actual byte/time reports go through Worker.note: stdout plus the durable Worker Report's
activity tail. The in-memory ring retains 64 events; each Report carries the last 8. These
are not automatically per-run receipt events. Foreground reads that actually wait for an
optional reader now report the observed drain-wait wall time without emitting per-read
messages when no waiting occurred.

The R17 grouped 1 GiB host log already proves successful first-use work: 897,870,336 delivered
bytes over 57 batches in 2,894.8 ms, with 515,624,960 B of process block-layer reads, stopped by
optional unreclaimed-headroom exhaustion. The feature was not globally disabled. Earlier
warmup receipts show a queued 58.344 s opportunity, but missing historical notes do not prove
which eligibility condition skipped it. The event gap is established by source and guarded
CPU regressions, not by inventing a historical cause.
