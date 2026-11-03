# H3 post-quant: prepared pointwise GEMM alternative

This additive Runtime slice prepares the actual biased 24-channel, 1x1x1 H3
post-quant Conv3d operation. It explicitly uses a **pointwise GEMM alternative**,
not cuDNN algorithm equivalence. Matched actual Conv3d numerical and model-quality
qualification is required before selection in the outer decoder.

The existing prepared dense provider (`_dense` and `internal/dense_linear.py`) is
a dependency from Runtime `5498a8f09e25672952abeb67ddccfc28205c3181`; it is not copied
or silently imported into this fresh-master branch. Shared exports, build wiring,
outer sequence admission and package integration remain coordinator-owned.

## Contract

- Only ordinary inference Conv3d with bias, channels 24->24, kernel/stride/dilation
  1, zero padding, groups 1, contiguous NCDHW input and contiguous stored matrices
  is eligible. FP16 CUDA autocast is explicit; source input/matrix may be FP16 or
  FP32, and bias is FP32 or FP16. Unsupported geometry declines before execution.
- Prepare a retained real FP16 bias destination and query the shared device-owned
  Lt Context for rows `B*D*H*W`, K=24, N=24. Record the actual algorithm, library
  identity and required workspace. No warm trace, guessed scratch or byte policy.
- Bind current sources and validate every actual pointer, layout, capacity, policy
  and alias restriction before numerical writes. Ordinary source hooks are the
  outer caller's responsibility; binding retains their actual tensor owners.
- Pack/cast NCDHW input into an owned contiguous FP16 `[B*D*H*W,24]` matrix;
  prepare an FP16 weight copy when needed; cast bias into the retained destination.
  Execute the selected plan once, then copy its FP16 rows into the owned NCDHW
  result. Copies use preallocated destinations; execution allocates no tensors.
- Keep source, destination, plan, shared Context and workspace owners through
  checked stream completion. Failed or unproven completion retains exact custody.
  No Python retry, fallback kernel, context-wide synchronization or parent-guard
  change is part of this operation.

The complete allocation list contains the actual bias, input pack, optional matrix
cast, dense result, NCDHW result and **queried** workspace. Preparation holdings
are distinguished from remaining execution allocations and never counted twice.

## Qualification

Implementation and real host clients are in progress. GPU cells are unrun. The
fixture must compare full/ragged batches and both stored matrix dtypes against
actual `nn.Conv3d` under the same autocast/math policy, check all output values,
capture actual allocations, refuse unsupported layouts before writes, and check
failure/lifetime ownership. No mocks qualify the provider. The full H3 gate remains
15 seconds, 362 frames, 1344x768, 30 denoising steps and audio.
