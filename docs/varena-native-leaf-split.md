# First native split consumer

This checkpoint migrates one closed allocation contract: the Runtime's already
served fused channel norm on a noncontiguous CUDA activation with at least two
batch rows. The full batch sample must have a 16-byte-aligned byte extent. Other
layouts keep their existing path. A contiguous norm is output-only and is not
claimed as a useful memory-saving split.

## Declaration and execution

Runtime enumerates exact divisors of the batch in descending order. Each opaque
`batch:<rows>` ID names one geometry. Every choice declares the new full output
and, when needed, one contiguous input copy of that chunk. Equal chunks avoid a
different tail allocation class; the copy is released before the next same-stream
chunk. Native code applies its actual size classes and slab/backing rules, chooses
one entry, and revalidates that exact choice at begin. No Python Room, byte budget,
fit or capacity fallback participates in the migrated path.

Select and begin receive the same saved CUstream handle used for every chunk.
Native admission matches unique exact-class reusable blocks, including safe
in-order pending reuse on that stream. It does not credit global pending bytes,
other-stream work, live/unjoined storage or graph pools, and begin preserves the
compatible cache it matched. Learned choices retain a per-class measured envelope;
scalar peak bytes alone cannot determine which cached geometry can satisfy them.
For example, a pending default-stream 8 MiB block plus 32 MiB of free capacity can
serve the declared 32 MiB output and 8 MiB scratch without a false refusal.

For BF16 `[4,64,256,256]` in channels-last storage, the output is 32 MiB and batch
choices 4/2/1 have 32/16/8 MiB copies: declared simultaneous backing is 64/48/40 MiB
before any existing allocator reuse. This is a real shrinking-copy menu, not an
output-only example. Input storage and bound weights remain separate existing
physical charges.

The actual compiled Runtime launcher and its artifact/entry/source/options are
part of the key, together with full tensor geometry/strides/dtypes/alignment,
model source identity, scale/epsilon, output contract and exact choice menu.
Lazy gamma/bias operands are bound and held before the leaf begins, so source or
decode allocations cannot hide inside its incremental requirement. The owned
Triton kernel has no private allocation/workspace callback. Its arithmetic and
reduction order are unchanged; each chunk writes directly to an aligned contiguous
view of one private full output.

## Scope and failure

The native metadata-only eligibility query excludes capture, replay and another
active leaf; it is not a capacity promise. Select/begin revalidate. Capture keeps
its existing whole path until a choice is explicitly committed before recording.
Multi-rank execution and an unqualified legacy outer split are not migrated.
Otherwise a leaf minimum could reject a geometry that the outer legacy retry
would still change. General `author.split`, convolution workspace, generic FFNs
and Ulysses remain explicit gaps.

Once selected, the operation executes once. Native pre-execution exhaustion and
an unexpected OOM after begin carry `varena_no_retry`, so the old cut catcher does
not replay an operation that may have written earlier chunks. The original error
propagates; the private output is not published. Unrelated legacy recovery keeps
its current behavior. Context ineligibility is checked before selection and is
never implemented by catching a capacity failure and trying the old path.

## Completion and record ownership

`split_end(submitted=True)` means that the operation was enqueued, not completed.
The leaf passes a completion-owner ID only when it runs on the active eager
phase's recorded stream. That stream object remains owned for the observation.
The phase's existing synchronization is the proof; a changed final stream, fault,
cancel, retry or callback allocation invalidates publication. There is no new
per-norm or per-chunk synchronization/event.

An unknown overall phase key still cannot publish a phase watermark, but a valid
same-stream completion may qualify its fully specified leaf record. Outside a
completion-owning eager phase, the native menu can select and execute from its
complete declaration but does not claim a learned successful sample.

Keys belong to participating live model owners. A componentless request owns its
keys only until request cleanup, which forgets them; there is no unbounded global
shape cache or age-based eviction rule. Runtime reports selected geometry and
submitted status separately from terminal native split observations attached to
the completed phase. Neither Python return nor a pending row is a learning pass.

## Required real proof

Use the actual installed fused kernel with whole and at least two divisor chunks
at different native limits, then verify an exact-choice sample only after its
owning phase completes. Include changed geometry/dtype/dispatch, unknown parent
phase identity, side-stream non-learning, aborted parent, callback allocation and
request/model retirement. An impossible minimum must launch the kernel zero times.
A deliberate under-declaration must fail without operation replay or partial
output publication. Nested legacy scopes must retain their existing path.

Static checks are not GPU qualification. No convolution workspace estimate, old
two-point fit or generic author closure is moved into Rust by this checkpoint.
