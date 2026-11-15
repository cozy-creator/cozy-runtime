# cr-298: Ulysses receive views

Owner: Codex `/root/optimize_h3_attention`. Purpose: eliminate one receive
materialization only for unquantized batch-one Q/K/V, after proving storage and
stream custody. Worktree: `~/cozy/.worktrees/cozy-runtime/cr-298-h3-ulysses-receive-view-20261115`.
Branch: `codex/cr-298-h3-ulysses-receive-view-20261115`.
Base: fetched `origin/master`, `af2a4753cb70598b75405942901e4f8f3b24eda6`.
Target: `rc/0.26.0`. The baseline `parallel/ulysses.py` is identical on both refs.
Draft: [Runtime #2187](https://github.com/cozy-creator/cozy-runtime/pull/2187).
The file remains unchanged upstream at RC `399773af601dd7b9696ea91e840b894f61aa6a07`.

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

## Prototype and evidence

Implemented only in this draft:

- Eligible `_send` calls omit the separate operand allocation. `_unpack` retains
  the same wait/stream dependency and returns receive views; the noneligible paths
  still copy or widen exactly as before.
- `_need` retains three current operand-sized buffers (the received storage),
  but omits the next eligible group's extra operand scratch. It does not subtract
  both groups' received storage from the pipeline estimate.
- `_exchange` transfers its pending receive object to `current` and clears the
  pending slot on the last group. Previously the last `arriving` reference kept
  its send/receive tensors alive through the return exchange, contrary to the
  modeled phases. The existing `del current`, returned views, and stream waits
  now define the intended lifetime without that duplicate reference.

| Check | Result | Boundary |
|---|---|---|
| Before implementation | Eligible receive-alias assertion fails on rank 0, batch 1, float32, non-narrow | Actual baseline `_send`/`_unpack` path; correct red signal |
| Two-rank Gloo | 48 cases pass, matching output bytes from actual baseline functions loaded at `af2a4753` | Uneven 3/5-row shards, one-head edges/two-head middle, random FP32/BF16 inputs, batch 1/2, narrow/non-narrow |
| Retained view custody | Pass | Receive storage survives `_Arriving` destruction; later receives leave earlier views unchanged; noncanonical singleton strides recorded |
| Failed wait | Pass on each rank | No operand returned when the transport wait raises; cleanup invoked |
| Whole/pipelined Gloo allocation trace | 8 cases pass | Actual transport-owned allocations fit `_need` with the larger rank's row count; identity attention, no native allocator |
| Local CUDA events | 12 cases pass | FP16/BF16, 65/4103 rows, delayed write on another stream, consumer clones before synchronize, allocator churn after dropping views |
| Static checks | Ruff, Python compilation and focused mypy pass | Changed source file and probe syntax; not the complete cross-repository CI suite |

Raw [CPU/Gloo receipt](evidence/cpu.json), [CUDA receipt](evidence/cuda.json), and
[environment](evidence/environment.json) are retained. No timing benchmark was
performed. The CUDA check uses stock Torch storage and a synthetic transport
completion event: it does **not** prove NCCL, actual copy-engine exchange, Varena,
or graph replay. There is no multi-GPU CUDA or H3 output/performance qualification.

The allocation trace initially failed at 167,936 observed bytes versus 164,000
estimated bytes in a small whole-exchange case. That exposed the last `arriving`
reference described above. After clearing that redundant reference, the same rank
observed 131,072 bytes against the unchanged 164,000 estimate. The assertion was
not relaxed. These tiny transport values are not predicted H3 peaks.

An early mathematical control for the narrow FP32 path differed by `2.38e-7`
already on the baseline because it batched the CPU matrix products differently.
The final regression compares the real baseline and candidate send/unpack paths
on identical random input bytes, avoiding that confound without loosening byte
equality. The quantized computation itself is unchanged.

## Reproduce and remaining gate

Use Python with compatible Torch and TensorFS/Varena dependencies, from this root:

```sh
/path/to/python research/h3-ulysses-receive-view/probe.py \
  --expect-view --output research/h3-ulysses-receive-view/build/cpu.json
/path/to/python research/h3-ulysses-receive-view/cuda_probe.py \
  --output research/h3-ulysses-receive-view/build/cuda.json
```

The CUDA probe refuses an already-busy compute device. The CPU probe imports the
candidate normally and loads only the exact baseline module into a separate
namespace for its differential control; neither changes installed packages.

No serving change is merged or installed. Before merging, exercise the actual
native executor and transport on at least two GPUs, including repeated pipelines,
unequal shards, cancellation/failure and memory pressure. Confirm the reduced
estimate does not change the intended precision or attention recipe; compare
matched full-length H3 output and timing before claiming a benefit. Parent owns
the GPU/rental authorization. This draft introduces no public API or tuning knob.
