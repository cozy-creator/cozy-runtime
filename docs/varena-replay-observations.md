# Native replay observations

This followup connects graph-specific capture metadata to the existing native
continuation boundary. It does not change production split or prefetch decisions.
The preceding eager-observation checkpoint remains separately buildable.

## Capture and definition lifetime

Runtime creates an inert `vt.capture_footprint(device, pool)` recorder and passes
it to `vt.graph(..., footprint=recorder)`. TensorFS starts the recorder after pool
preparation and synchronization, immediately before actual capture begins. A
clean end seals the immutable allocation-range trace; an incomplete capture
aborts it. Recording alone does not publish a successful execution sample.

The token describes allocations made by this graph, including temporary storage
logically freed during capture. Native code unions the touched physical pages;
the whole pool reservation and other graphs' idle backing are not its demand.
The token adds no physical pin, mapping, synchronization or capture-time driver
operation. TensorFS binds retirement to the graph's own reset/destruction callback,
so another graph keeping the shared pool alive cannot validate a stale token.

`PreparedRepeat` retains the prefix/body/suffix tokens with its existing graph
owners. Closing a definition drops its tokens after checked native-owner closure.
Other definitions sharing an actual graph remain responsible for that graph's
lifetime; a Runtime token reference does not replace the graph's existing owner.

## Semantic execution key

Each recorded role has the declared loop key and tensor geometry, immutable
model/fill identity, phase/range, role/callable identity, current attention/backend
configuration, and the actual cut/layout fingerprint observed during capture.
A streamed role additionally names its component-relative block and ordered stage
position. A repeat key combines its ordered prefix/body/suffix identities and
repeat count. It excludes addresses, Python object IDs and refill generations.
Unknown or changed configurations do not publish reusable observations.

Opaque cuDNN/cuBLAS engine identity is fixed by the captured graph but is not
fully represented by Runtime's high-level backend metadata. Native replay records
must therefore also be scoped to the ordered immutable graph-definition tokens.
A fresh capture with the same declared shape/key cannot borrow another definition's
record merely because it shares a pool. This still permits repeated observations
of one definition; cross-definition reuse needs a separately qualified actual
dispatch signature. Definition retirement removes that native record scope.

The existing weight-preference key remains separate; it is allowed to identify a
particular submitted residency generation. It cannot double as the execution key.

## Native completion

`PhaseController.wrap` receives `footprints` and the separate `execution_key`.
Its native start/observe/rearm boundary owns stage observations. Only checked,
successful replay completion can qualify the recorded footprint. Cancellation,
failed preparation, replay faults and incomplete stages preserve the last
successful sample. Shared-pool pages are counted once, with outside live inputs,
outputs and other holdings charged separately.

Stages without weight bindings use `key=None, regions=()` to observe through the
sequence's unique memory owner without changing preferences. An explicit string
with an empty region list keeps the existing intentional-clear meaning. The
Runtime's weight-policy cache is invalidated only if a controller actually changed
preferences.

Progress callbacks are not part of the declared execution key. A bounded native
`external_callback` scope surrounds each callback. Entering or leaving this scope
does not invalidate anything; an actual allocation/free while it is active
invalidates the current observation before mutating allocator state. This covers
a callback interval spanning native stage completion/rearm, without a late rollback
of already published samples. Allocation-free CPU telemetry remains measurable.
The guard is metadata-only and adds no synchronization or Python allocator hook.
Concurrent native allocation during the interval conservatively invalidates that
sample too. Work launched without joining after the callback returns remains
outside this bounded callback contract.

The former outer Python replay scope would overlap these native scopes, so it is
removed. There is no new per-step Python scheduler or continuation ABI. Native
observations enter the request's normal record once after checked completion and
readout; unproven cleanup retains the original owners. Model ownership retires
semantic records through the existing phase-forget path.

## Qualification boundary

Before merge, exercise real capture/replay with the matching TensorFS candidate:
different footprints sharing one pool; prefix/body/suffix page union; graph reset
while another graph retains the pool; changed capture cuts; a failed capture;
cancellation and incomplete replay; reused stages; and a weightless inter-block
stage whose observation does not clear the preference plan. Require no allocator
driver/event work during capture and no stale token acceptance.
Also cross a native observe/rearm boundary while one callback scope is active:
every allocation/free must invalidate its current observation before publication,
while a callback interval with no allocator transitions leaves it publishable.

Existing full-model/stream/alias checks remain necessary. H3's explicit input key
does not permit cross-call reuse of its request-local derived caches. Production
prefetch still needs the later qualified policy cutover.

This branch starts from Runtime `99fb012ab` / 0.36.16, including the later INT4
kernel source. Unlike the earlier checkpoint, it cannot reuse the older 585-member
native artifact cohort without a new exact-source artifact audit. No native build
or GPU proof is claimed by the source checks.
