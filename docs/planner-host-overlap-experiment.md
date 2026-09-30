# Observed next-unit HOST overlap experiment

This candidate adds an automatic, optional Planner path for a single-rank managed
operation. It is not a measured performance improvement or a release qualification.
The normal demand path remains the fallback. No numerical mode, call, model setting,
allocator grant, or resident keep decision changes.

A successful exact operation key records the actual unique block order. A later
matching operation with a positive measured working-memory bound can prepare one
observed successor while the current block runs. Repeated blocks, unknown contracts,
groups, compiled/whole components, gradient mode, mutable state, pending casts and
non-HOST candidates decline. History is local to the held construction, checked
against hook/forward identities, and cleared on scope close, conversion, and failure.
A failed replay is not a new source of lookahead authority.

Preparation borrows only the fill backend's already materialized, entirely pinned
ring. The whole next unit must fit that exact existing capacity. There is no new pool,
resizing, host admission request, or eviction to fund speculative work. The source
backing and the ring already have physical ownership; the destination must fit fresh
allocator/driver room while preserving the entire measured working obligation. This
conservative quote can decline otherwise profitable overlap. An optional allocator
OOM permits another attempt only after room exceeds both pre-attempt and cleaned
post-failure room by at least the failed allocation request; unknown request size
blocks further optional attempts in that executor. Other errors propagate.

The existing positive readonly/pure or covered retry-state contract supplies exclusive
immutable source custody. Version checks cannot detect arbitrary `.data` or inference
tensor writes; such writes are not newly authorized. Preparation executes no future
Python. At actual demand, a CPU event join and source revalidation precede atomic
binding. Every participating current block drains the device before homing: purity is
not a promise that authored code uses only the current CUDA stream. This can reduce
the benefit relative to a stronger stream contract.

Wrong-order hints and pending borrowers are drained before a normal fill or reclaim.
A nested operation cancels unbound hints but retains the outer consumer's custody.
Failed completion poisons the executor and retains strong ownership; no cleanup claim
is made until the existing executor retirement/reap boundary. Ordinary alias/view
storage geometry is preserved by the reviewed HostTransfer mechanism.

`residency.host_overlap` reports actual CPU staged and H2D submitted bytes, prepared
bytes, hits, waste, decline reasons, optional OOMs, preparation time and publication
wait time. Publication time includes its CPU join and binding work. Submitted bytes
are enqueue evidence, not a claim of physical bus traffic or successful completion.
Pending destinations are excluded from activation sampling and become loaded weights
once at publication. The allocator's absolute peaks remain visible. The existing
moved-byte counter records successful demanded publication; speculative waste appears
in the separate overlap counters.

CPU qualification uses real Torch CPU storage/views and simulated CUDA events through
actual Planner and installed unit hooks, under the device-open-denying launcher.
It cannot establish CUDA allocator behavior, real overlap, image parity, throughput,
or thermal comparability. Those require a separately recorded immutable GPU candidate.

History deliberately does not cross Model scope closure. A public pipeline that opens
a new scope around every denoise step may therefore remain cold; a broader scope with
repeated matching component operations can learn and reuse order. Eligibility is
structural, not model-specific, and the experiment promises no uniform pipeline gain.

## Production wrapper correction

The first isolated candidate07ad721e passed its CPU tests but those tests entered
Planner.run directly. They missed that production tiers.wrap adapts both Module's
_call_impl and forward: treating the second entry as authored recursion disabled
learning on every ordinary root(...) call. A real wrapped-root regression reproduced
that failure before the follow-up correction. Only the exact canonical Module
_call_impl → current bound root.forward gateway may now join once without invalidating
lookahead. Deeper recursion and unknown same-component encode/decode joins disable it;
the join flag is restored even when the body raises. Both root(...) and direct
root.forward() are tested through the actual adapters, with positive second-call hits.
