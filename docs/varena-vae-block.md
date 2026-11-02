# Closed fused VAE block

Target the actual observed `_BlockRows` route: 792 wrapper calls and 794 selected
Kitchen INT8 attention calls in full H3 25499. Preserve the 36-block fused Runtime
body, both FP16/FP32 residual arms, FP16 autocast, FP32 normalization/gates and all
existing round points. Integration61fe and every earlier cohort remain frozen.

Before implementing admission, establish the pinned Torch dense-linear/autocast and
cuBLAS/cuBLASLt workspace ownership contract. No warm trace supplies a workspace
estimate. The current research lane is read-only implementation/source research;
any native shim or build needs an explicit frozen proposal and root scheduling.

The eventual fixture must invoke the real fused block and selected Kitchen attention,
cover actual weight dtypes, full/smaller batch choices, complete allocation census,
minimum refusal before writes and failure without post-write application replay.
No generic framework redesign or whole-H3 prefetch opt-in is part of this lane.

## First implementation boundary

The prepared dense provider is deliberately narrower than a generic `linear` replacement.
It accepts the six contiguous, biased FP16 execution matrices of this declared VAE,
with source weights/biases stored as FP16 or FP32. It reproduces the pinned Torch
column-major transpose recipe and math/reduction policy. Missing bias, unsupported
layout, tuning, opaque hooks or an unestablished effective Lt route decline before
acceptance. The actual fused block remains the numerical reference.

One `_dense.Context` belongs to the Runtime device owner. It owns one cuBLASLt handle;
individual plans retain that context and their operation/layout/algorithm metadata.
Creating a context, enumerating algorithms and final destruction are preparation or
owner-retirement work, never the admitted compute interval. The context does not borrow
Torch's stream workspace maps. Library internal/module resources remain observable
external usage under the design's existing library boundary; a warm run is not a proof
that those resources cannot grow.

The native source interface is:

```text
Context(device, preparation_stream)
Context.query(rows, inputs, outputs, actual_bias_address, policy, preparation_stream)
    -> Plan(required_workspace, checked_algorithm, library_version, geometry)
Plan.validate(x, weight, output, workspace, capacities, explicit_stream)
Plan.execute(x, weight, output, workspace, capacities, explicit_stream)
```

The caller first admits and retains the actual 32 KiB FP16 bias destination. Its real
256-byte-aligned pointer is installed in each plan, including every full/tail geometry.
A/B/C/D minimum alignment is fixed at256, then actual addresses/capacities are validated
before writes. This avoids allocating matrix variants merely to query a shape. The
maximum **queried** workspace among the selected window's six projections and its
ragged tail is one shared, explicitly admitted allocation. Every planned full/tail binding is validated before the first norm/cast/output write.
`execute` repeats its local check to catch a changed invocation. The provider never chooses
again there; non-success preserves the vendor domain/status and propagates through the
existing typed CUDA/OOM boundary without another kernel/backend attempt.
The caller retains the context, descriptors, bias and workspace through completion or
all capturing graph borrowers. Releasing one plan cannot retire another graph's owner.

The default first consumer is eager and single GPU. It does not enable graph capture
or whole-H3 committed prefetch merely because its tensor menu is closed. Those require
separate owner/initialization evidence. Preparation refuses active capture.

## Complete window storage schedule

Let `B` be the full tile batch, `b` the selected batch window, `T=1797`, `W=2048`,
`H=32`, `D=64`, and `R=b*T*W` elements. Only batch is cut: each tile retains all tokens,
all heads and the same attention/reduction domain. A ragged last window has its own
queried dense geometry and packed Kitchen views. The whole returned output remains
FP32 for either incoming residual dtype.

| Actual allocation | Bytes | Lifetime |
| --- | ---: | --- |
| Full result `[B,T,W]`, FP32 | `4*B*T*W` | Returned owner; each window is also its FP32 intermediate residual |
| Packed projection slot A `[b*T,8*W]`, FP16 | `16*R` | Q/K/V and their normed forms, attention output/copy, then packed up projection, then down output |
| Expanded slot B `[b*T,4*W]`, FP16 | `8*R` | First/second normalized rows in its prefix, then full SwiGLU output |
| Matrix-cast destination, FP16 | Largest actual FP32 matrix payload, at most67,108,864 | Refilled sequentially before each dense call; zero when every source is FP16 |
| Shared bias destination, FP16 | 32,768 | Admitted before query, then refilled before each dense call |
| cuBLASLt workspace | Actual maximum selected full/tail query requirement | Shared by sequential linears only |
| Quantized Q/K `[b,H,T,D]`, INT8, each | `b*H*T*D` | Kitchen quantization through attention |
| Packed V `[b*H*D,ceil(T/64)*64]`, INT8 | `b*H*D*ceil(T/64)*64` | Same |
| Q scales `[b,H,ceil(T/128)*32]`, FP32 | `4*b*H*ceil(T/128)*32` | Same |
| K scales `[b,H,ceil(T/64)*4]`, FP32 | `16*b*H*ceil(T/64)` | Same |
| V scales `[b*H*D]`, FP32 | `4*b*H*D` | Same |
| Anchor indices `[b,H]`, INT32 | `4*b*H` | Same |
| Modulation row index `[1]`, INT64 | 8 | Preparation-owned actual zero index, retained through all windows |

This is tensor payload, **not** allocator capacity. Varena rounds and admits the actual
allocation list; its measured successful key replaces declared demand. The bias and
index preparation holdings participate in total demand and must not be counted again
as missing compute allocations. No workspace constant or percentage of free memory is
used. RoPE operands must already be matching contiguous FP32 tables with the declared
48-channel split-half layout; otherwise preparation declines or explicitly declares a
separate conversion before the compute sample. No `.float().contiguous()` may hide
inside the accepted operation.

Within slot A, the first five `R`-element contiguous regions hold raw Q, raw K, V,
normed Q and normed K respectively. All six original dense outputs have contiguous
row pitches. After Kitchen consumes float Q/K/V, A's first region is reinterpreted as
contiguous BHND output; a same-device transpose copy writes A's second region as NHD.
The out projection writes A's third region. The existing second norm/residual kernel
writes the full-result window and B's normalized prefix. The packed up projection then
consumes that prefix and writes all of A. SwiGLU reads its value-first halves and writes
all of B; down consumes B and writes A's first region. The existing residual kernel
reads the FP32 result window and down branch and writes back to that same result window.
Its programs load their complete row before storing; rows are disjoint. No RoPE kernel
is changed to in-place operation, and no attention kernel writes over unconsumed codes.
Every reuse is ordered on the same explicit stream. There is no synchronization between
individual operations or windows; completion belongs to the existing outer phase.

The two large scratch slots use `24*R` bytes. At one tile this is88,326,144 bytes; at28
it is2,473,132,032 bytes. The full FP32 output at28 tiles is412,188,672 bytes and remains
when `b` shrinks. Thus batch cuts have a real scratch reduction; no output-copy window
is manufactured to make a chooser useful. Actual queried Lt scratch and allocator
rounding are additional, so these figures alone never establish fit.

## Invocation and refusal boundary

The package owns the declared `_BlockRows` operation. Metadata inventory identifies the
installed fused block, its exact layout/kernel identity, the six plain dense modules,
selected unmasked Kitchen INT8 route, dimensions/dtypes and legal batch windows. It
must not run a warm forward or bind every VAE block at once. Only the current block's
ordinary source preparation and actual read aliases precede admission. Their binding
identity is revalidated before the first output/seed write.

Preserve the current block's module/source hook invocation once. A narrow interception
at the installed fused forward can enter the admitted executor after ordinary source
hooks, without executing the original numerical body first. Unknown hooks or a changed
binding decline during structural preflight or fail before writes, respectively. No
accepted error returns to `_BlockRows`, generic `split`, an inner Kitchen chooser, or
Torch's GEMM fallback. Refusal at the minimum leaves caller input and prior successful
sample unchanged. A post-write kernel/allocator failure reports the failed operation
and preserves completed output diagnostics; it does not rerun it.

## Qualification still required

First qualify the new dense provider against actual FP16 autocast `nn.Linear` for all
six real shapes and source dtypes, including full/ragged batch windows, exact effective
math policy and library version. Record algorithm configuration, actual caller-buffer
allocator census, library resource observations and lifetime/cross-stream teardown.
Then qualify the actual installed fused block and selected Kitchen recipe: incoming
FP16 and FP32 residuals, nontrivial rotary/gates, whole/split/ragged batches, all buffer
classes/counts, minimum refusal before writes, and a real failure after acceptance
with exactly one invocation and no application replay. Preserve prior cohorts and
compare complete outputs under the established numerical budget. Source and meta-layout
checks do not qualify CUDA execution or full-model performance.


The source now implements the prepared provider, status bridge and standalone complete
fused consumer. The real meta-device block/layout check passes with CUDA uninitialized
(docs/evidence/vae-block-source). Dense and fused CUDA execution remain unqualified.


The complete consumer continues on `codex/xs138-p2-vae-composite` from fresh mastercb6
plus the frozen provider5948028be. Provider source and the31-case dense fixture remain
immutable for root-owned build/qualification. The implementation uses an explicit package
`_BlockRows` intent intercepted at the installed fused forward after its ordinary source
hooks, with only the current block bound, one invocation, and all actual buffers checked
before the first write. Existing row/RoPE/SwiGLU kernel entries accept caller outputs
without changing their numerical bodies or cache identities.


## Remaining outer decoder policy

The actual package wraps `_decode_tiles` in legacy `split(..., Cut(0))` before reaching
any `_BlockRows`. `catch.uncommitted_scope()` is true for every active legacy Scope,
so preserving that guard prevents the new closed block from adopting under this parent.
The observed outer wrapper had21 calls (20 whole,1 split); replacing it with a fixed
whole call would discard an actually used capacity fallback.

The bounded block proof remains useful but is not full-H3 adoption. The outer operation
contains post-quant Conv3d, proj-in/register assembly,36 blocks, final norm/proj-out and
layout copies. Its actual allocation/choice contract must be migrated separately, or
an explicit irrevocable parent-choice transition must be reviewed with its changed
failure semantics. Do not clear the scope flag, infer closure from a successful trace,
or opt in the whole decoder through the block declaration.

The declaration now names the two actual direct layer-scale parameter slots; no
opaque modulation callback runs inside the admitted body. Optional row/RoPE/SwiGLU
outputs preserve every original JIT body, definition line and the imported VAE kernel
identity. The native block executor and real package declaration are implemented;
full-H3 adoption remains blocked by the outer decoder policy described above.


## Lazy vendor loading after the first native build

Root built exact5948028be successfully, but its CPU-only import before Torch failed
because a direct `libcuda.so.1` dependency requires a driver at import. The prototype
also carried a `/cublas/lib` build-tree RUNPATH. That artifact and failure remain
preserved under outputs/xs138-p2-vae-block/native-5948028be.

This successor follows Varena's existing dynamic CUDA loader: import creates no
driver/Lt handle and resolves no vendor symbol. GPU preparation loads the real installed
CUDA driver and cuBLASLt13 by their runtime library names; all calls retain their exact
typed vendor statuses. Libraries stay loaded for their registered exit handlers and
opaque borrowers. Context/descriptor/workspace ownership is unchanged. No driver stub
is bundled and the native target has no direct vendor-library link or build-tree RUNPATH.
Root built exact2a78 and verified581 native members (580 preserved plus `_dense`),
Python3.12/3.13/3.14 CPU-only imports and controlled no-driver errors, with no vendor
DT_NEEDED and no build RUNPATH. Those checks do not automatically qualify a later
composite binary; GPU/numerical/library-allocation proof remains separate.


The pre-launch consumer requires the current declaration/prepared-dense API rather
than an old-Varena compatibility fallback. Unsupported live execution routes still
remain migration work. Queries use only actual slot metadata and the retained bias
destination before the sole module invocation; a genuine unsupported Lt query can
decline before source hooks or numerical work. The body then reads and revalidates
only its current block, chooses the native menu and validates all full/tail buffers
before writes. It uses the actual framework stream explicitly and adds no per-operation
synchronization. Shared context destruction is final-owner teardown.


## Actual rotary presentation

The real `MiniMaxH3VideoRotaryPosEmbed(48)` returns `[1,T,1,48]` cosine/sine tables.
The first CPU file-consumer attempt exposed a fixture using only `[T,48]`, which the
unfused reference cannot broadcast across its32 heads. The fixture now calls the actual
rotary module. Native preflight accepts that actual broadcast shape as well as the
existing compact prepared table, with contiguous FP32 storage, exactly`T*48` elements
and token stride48. Kernel math, buffers and the existing flatten ABI are unchanged.
The a09 source/artifact is preserved; this is a Python-only correction.
