# Actual H3 input projection and rotary prefix

`native_vae_prefix.py` is the declared `h3-prefix-lt-fused-bias/1` alternative.
Input is post-quant contiguous FP16 `[B,24,7,16,16]`; `proj_in` is actual biased
24->2048, with FP16/FP32 matrix and FP32 bias. FP32 registers are `[1,4,2048]`;
the actual nonpersistent FP32 `rope.inv_freq` buffer is `[8]`. Output is contiguous
FP16 `[B,1797,2048]` and two shared FP32 rotary tensors `[1,1797,1,48]`.
The rotary leading dimension is 1, matching package `_decode_tiles`, not B.

## Explicit projection difference

The stock NCDHW-to-token permute/reshape is an alias with noncontiguous strides.
Pinned [Torch Linear.cpp](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/Linear.cpp)
therefore uses matmul followed by a separate FP16 bias addition by default.
`TORCH_LINEAR_FLATTEN_3D` can change that process-cached path. The existing Runtime
prepared provider always selects the Lt bias epilogue, so packing rows and using it
removes the stock intermediate FP16 matmul-output rounding. This is intentional,
explicitly authorized as an alternative, and recorded in the semantic key/facts.
No stock round-point, algorithm or bit-identity claim applies to the projection.
There is no zero-bias workaround or new custom provider.

The real shared Context queries rows B*1792, K=24, N=2048 with its actual retained
FP16 bias destination. Algorithm, library and workspace come from that query.
A legally shaped query that is unsupported declines before writes; no fallback
kernel or invented workspace policy is supplied.

Component acceptance is fixed before the comparison: rtol=0.002, atol=0.0001 and
relative L2<=0.001 on the bounded-input fixture (input scale 0.125, matrix scale 0.02,
bias range +/-0.05). The relative term covers roughly two FP16 ulps; the absolute
term covers near-zero cancellation. This is a component gate, not evidence that
arbitrary model activations or final video quality meet a bound. Full H3 quality
and performance remain mandatory before cutover. The earlier failed exact-identity
probe of packed mm + separate bias is preserved; even that changes accumulation.

## Exact packing and arithmetic

The full 1792 projected rows copy into the hidden prefix. Broadcast/cast FP32
registers copy directly into its four destination rows; its last row is zeroed.
This removes the separate register-cast/zero/cat tensors while preserving their
FP16 values. Input packing and all slices/reshape/unsqueeze operations are views
plus writes to declared destinations. No module source hook executes in the body.

Each of the three coordinate buffers is filled with `arange(0.5,size)` and separately
divided by size, multiplied by 2, then reduced by 1 in FP32. Broadcast views write
coordinate components into the owned positions buffer; the five suffix rows are
zeroed there. This replaces meshgrid views plus stack/suffix/cat allocations with
explicit copies. It preserves t/h/w coordinate ordering and each arithmetic round.

RoPE first multiplies positions by 2*pi into a separate FP32 buffer, then multiplies
that by the actual inv_freq into `[1,1797,3,8]`. Its flattened 24 angles copy twice
into 48 channels. Separate out cosine/sine calls produce the returned tables.
Do not combine the two multiplies or compute a different frequency table.

Pinned CUDA [arange](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cuda/RangeFactories.cu)
uses the supplied correctly-sized contiguous destination. Same-device CUDA
[copies](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cuda/Copy.cu)
do not require temporary tensors. Native structured
[unary](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/UnaryOps.cpp)
and [binary](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/BinaryOps.cpp)
out kernels reuse correctly shaped destinations. Metadata validation is required
before writes so resizing, device transfers and implicit dtype/output allocations
cannot select a different path. Actual managed GPU census remains an unrun gate.

## Owned payloads and lifetimes

| Buffer | Bytes |
|---|---:|
| Query's actual FP16 bias | 4096, already live preparation holding |
| Packed FP16 input | 2*B*1792*24 |
| Optional FP16 matrix cast | 2*2048*24, FP32 source only |
| Full FP16 projection result | 2*B*1792*2048 |
| Hidden tokens including registers/zero | 2*B*1797*2048 |
| Queried workspace | Actual provider result |
| Three FP32 coordinate grids | 4*7, 4*16, 4*16 |
| FP32 positions and scaled positions | 4*1797*3 each |
| FP32 angles | 4*1797*3*8 |
| FP32 duplicated angles | 4*1797*48 |
| FP32 cosine and sine | 4*1797*48 each |

Existing input, matrix, bias, registers and frequency owners are borrowed holdings,
not new allocations. The bias preparation holding is counted once. Every remaining
buffer is preallocated after the caller's complete admission, and every exact
binding/configuration/source identity is validated before one numerical execution.
All intermediates, sources, Plan and shared Context survive checked completion on
the explicit stream. After `wait`/successful `poll`, dropping Bound/Prepared releases
scratch/Plan custody; returned hidden/cos/sin owners remain live for subsequent
blocks and their streams. Unproven submissions retain actual available custody.

The caller owns source hooks, stream ordering and whole outer admission. This module
has no selector, nested split, guessed reserve or Python retry policy. Shared
`native_vae`, catch/phase, the package caller and old chooser remain unchanged.
The initial ABI is pinned Torch 08187d9e; unsupported builds decline before writes.
