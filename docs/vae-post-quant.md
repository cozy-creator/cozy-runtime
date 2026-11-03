# H3 post-quant: prepared pointwise GEMM alternative

This additive Runtime slice prepares the actual biased 24-channel, 1x1x1 H3
post-quant Conv3d operation. It explicitly uses a **pointwise GEMM alternative**.
Matched actual Conv3d numerical and model-quality qualification is required before
selection in the outer decoder; cuDNN algorithm/rounding equivalence is not claimed.

The existing prepared dense provider (`_dense` and `internal/dense_linear.py`) is
a dependency from Runtime `5498a8f09e25672952abeb67ddccfc28205c3181`; it is not copied
into this fresh-master branch. Shared exports, build wiring, outer sequence admission
and package integration remain coordinator-owned. The existing retryable-parent
scope guard remains required, and this module creates no native split of its own.

## Call contract

1. `describe(module, input)` reads metadata for inference Conv3d with bias,
   channels 24->24, kernel/stride/dilation 1, zero padding, groups 1 and contiguous
   NCDHW input. Input/matrix/bias may be FP16 or FP32. Unsupported geometry, user
   hooks, overrides and layouts decline. Runtime source slots reuse `native_vae`'s
   reviewed raw-source identity/binding contract and retain the actual owners.
2. `prepare(description, context=shared_context, stream=explicit_stream)` requires
   the pinned dense provider's FP16 CUDA autocast/math policy. It allocates a real
   retained 48-byte FP16 bias destination, then queries the device-owned Context for
   rows `B*D*H*W`, K=24, N=24. `facts()` reports actual algorithm, library identity,
   workspace and separate preparation/execution allocations. Query has no numerical
   writes or source hooks. It is outside capture and retryable parent scopes.
3. The caller admits `prepared.allocations`, counting the already-live bias once,
   then calls `Buffers.allocate`. The list includes input rows, dense result, NCDHW
   result, optional FP16 matrix cast, and actual queried workspace. The caller owns
   source-hook invocation and explicit ordering of ready input/source tensors.
4. `prepared.bind(input, buffers)` is one-shot. It checks current module/hook
   configuration, sources, dtype/math policy, every pointer/layout/capacity and
   independent destination storage before writes. A final validation runs immediately
   before execution, including source identity and actual raw Source bindings.
5. `bound.execute()` packs/casts input into owned rows, copies a matrix when needed,
   casts bias into its retained destination, executes the selected Lt plan once,
   and copies rows into the owned contiguous NCDHW result. All tensors are already
   allocated. There is no query, source hook, alternate kernel or retry in the body.
6. Keep `bound` through `bound.wait()` or successful `bound.poll()`. These prove
   completion on its explicit stream, not the whole device. Input/sources remain
   immutable through completion. Unproven submissions retain exact source, buffer,
   plan, bias and Context custody; failures are marked non-retryable. A result handed
   to another stream needs that caller's normal stream ordering/storage lifetime.

A shared Context outlives all its Plans; this operation never closes a caller's
Context. Its library-owned resources are distinct from the declared tensor payloads.
Their actual allocation closure still requires independent GPU/driver evidence.

## Evidence and pending gates

The real CPU client in Creator's `release-gate/suites/varena/vae-post-quant/host.py`
passes 24 actual Conv3d comparisons: batches 1/3/7, tiny and actual 7x16x16 geometry,
FP16/FP32 input and matrix storage, FP32 bias. It checks the complete output,
packing/copy dispatch and exact tensor payload inventory, plus 15 unsupported
metadata/layout cases. Max absolute difference was 0.000244140625 with pinned
`rtol=1e-3, atol=1e-3`. This is CPU evidence only; no CUDA initialization occurred.

Creator also supplies the ordinary-child `post_quant` GPU fixture and strict
wheel-derived payload freezer. It extracts the actual Diffusers H3 `post_quant_conv`
module and checks 12 full/ragged GPU output cases, actual allocation census,
alias refusal before writes, native minimum refusal, exactly-once post-write
allocator refusal, retained Context refusal and checked stream cleanup. Those
GPU cells are **unrun**. Asynchronous device-error custody, raw Source retirement,
whole outer-sequence adoption and model quality remain additional operator gates.
The full H3 acceptance remains 15 seconds, 362 frames, 1344x768, 30 steps and audio.
