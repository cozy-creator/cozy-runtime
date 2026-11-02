# Native refiner SwiGLU FFN

The audited H3 refiner is Diffusers 0.41 `MiniMaxH3TokenRefinerBlock`:
`hidden + ff(norm2(hidden))`, where `ff` is a bias-free `FeedForward` with a
value-first SwiGLU up projection, zero dropout, and down projection. Its official
geometry is D=5376, F=14336. Attention, norm and residual addition remain outside
this operation. The separate `ResidualFFN` contract owns AdaLN/norm/residual gates
and does not describe this refiner.

`cuts._ffn` now recognizes this complete stock FeedForward structure at install.
Before entering its legacy chooser, it tries `native_refiner_ffn.split_ffn`.
Acceptance requires contiguous CUDA BF16/FP16 `[1,M,D]`, actual bias-free
`NativeLinear` int8.convrot leaves, matching dtype and shape, aligned contiguous
int8 weights and FP32 scales, the actual serving cozy-int8 mma.sync library, and
supported `int8_plan` queries for every full/tail projection. Subclasses, custom
forwards/hooks, nonzero dropout, autocast, capture, multi-rank, shared operands,
other providers and unsupported shapes decline before native selection. Only
existing baked projection LoRA hooks are accepted, through the original normal
module invocation and supplied-output interceptor. The declaration is rechecked
on every invocation, including exact Runtime cut/naming wrapper identities.

This applies to H3 refiner FFNs when their **actual installed projections** use
int8.convrot/mma.sync. It does not infer that provider from a package precision
flag. Dense, FP8, W4A8, other kernel families and multi-rank configurations retain
their prior route. CPU/MPS retain the original callable and sizing behavior. No
claim that every refiner configuration is migrated is made.

The accepted operation allocates a fresh full output. Each legal row choice owns:

| Allocation | Exact shape/type |
| --- | --- |
| Output | `[1,M,D]`, input dtype |
| Packed up projection | `[1,m,2F]`, input dtype |
| SiLU output | `[1,m,F]`, input dtype |
| Product output | `[1,m,F]`, input dtype |
| Shared quantized operand | maximum `dispatch.rows * input_width` across selected GEMM calls, int8 |
| Shared row scales | maximum selected GEMM row count, FP32 |
| Shared workspace | maximum exact `int8_plan.workspace_bytes` across up/down full/tail |

All buffers coexist and are declared as separate allocations. The up and down
projections reuse the quantized operand/scales/workspace sequentially on the
saved stream. GEMV requires neither row quantization storage nor a guessed
workspace. Both projections keep their exact selected kind/configuration.
`aten.silu.out` and `aten.mul.out` use their pointwise out kernels with supplied
matching-shape/type outputs; they require no device scratch. Separate output
buffers preserve the stock SiLU and multiply round points. No activation fusion,
dense provider fallback, hidden contiguous copy or bias cast enters the body.

The native selector owns the row choice and allocator recovery. Every buffer is
allocated before the first projection launch; subsequent chunks reuse storage.
The accepted body invokes neither the outer FeedForward nor nested SwiGLU
wrapper, so only this invocation bypasses the old Python chooser. Cancellation
or a launch/allocation failure ends the operation; OOM carries `varena_no_retry`.
No failed body is retried, and no failed observation replaces a successful one.
The model/request owns the semantic key and normal request retirement cleans it.

## Evidence and remaining proof

`research/varena-refiner-ffn/host_check.py` uses real Diffusers refiner modules,
real Runtime NativeLinear leaves, and real CPU ATen operators. On Torch2.14.1+cpu
and Diffusers0.41 it verifies unchanged CPU output after full `execution.install`,
exact FP16/BF16/F32 activation outputs for odd rows, provider/bias/stride refusals,
and custom-forward/hook/dropout refusals. Official H3 geometry is constructed on
meta. CUDA remains uninitialized and libcuda unmapped. Targeted mypy and Ruff
checks are source evidence only.

`research/varena-refiner-ffn/fixture` is a genuine Cozy consumer app. It constructs
the actual H3-width refiner FFN with real int8 data/scales, compares its stock
forward to the accepted composite for BF16 and FP16 at 5/17/513 rows, profiles one
composite and zero old cuts, records exact selected plans/allocation inventories,
and checks unsupported bias before any selection. A same-PID cleanup entry checks
request-owned key retirement. `prepare.py` freezes the fixture and derives strict
Runtime/native/GPU identity input from supplied wheels. No model download or
mocked native backend is used.

The fixture has not run on a GPU. Required follow-up includes actual full/tail
workspace execution, forced legal split choices under real pressure, impossible
minimum/prewrite refusal, aborted submission and previous-sample preservation,
baked-hook execution, allocation census/opaque growth and cleanup. Then run the
unchanged 15 s / 362 frame / 1344x768 / 30 step / audio full H3 gate, preserving the
original performance failure. This change does not close P2 or qualify a release.
