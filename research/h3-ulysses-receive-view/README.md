# cr-298: Ulysses receive views

Owner: Codex `/root/optimize_h3_attention`. Purpose: eliminate one receive
materialization only for unquantized batch-one Q/K/V, after proving storage and
stream custody. Worktree: `~/cozy/.worktrees/cozy-runtime/cr-298-h3-ulysses-receive-view-20261115`.
Branch: `codex/cr-298-h3-ulysses-receive-view-20261115`.
Base: fetched `origin/master`, `af2a4753cb70598b75405942901e4f8f3b24eda6`.
Target: `rc/0.26.0`. The baseline `parallel/ulysses.py` is identical on both refs.

Proposed internal change: `_Arriving.operands: Tensor | None`. `None` records
that `_send` omitted separate operand storage for `batch == 1 and not narrow`.
`_unpack` must still perform the transport wait on the inbound stream and order
the calling stream after it, then return views retaining the received storage.
There is no public API change. Batch >1 and quantized transforms retain their
existing materialization.

Each `_send` allocates fresh receive tensors on the calling stream. The copy
engines' reusable inboxes are copied into those tensors before the completion
event. They are not the view's backing storage. The returned views retain the
receive allocation after `_Arriving` is dropped; attention runs after the same
inbound dependency on the allocation's calling stream. This source argument is
not proof of Varena or multi-GPU CUDA execution.

The allocation avoided per eligible head group is
`3 * batch * total_rows * local_heads * head_dim * itemsize`, with twice that
many copy bytes. No time saving has been measured. Preserve the existing pipeline,
uneven rows, head tails, dtype, ordering and failure propagation. Memory sizing
must retain the current group's received storage while attention consumes it,
and count the next in-flight group's send/receive tensors.

Status: preparing an actual `_send`/`_unpack` Gloo regression. No serving change
is merged or installed. Native multi-GPU correctness/performance remains pending.
