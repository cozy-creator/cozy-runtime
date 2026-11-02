# One declared residual-FFN tail

This P2 consumer targets current H3's complete in-place block tail. Package master
28ff0e75/H3 0.2.6 already wraps residual/norm, FFN and final gate-add in `_Tail.run`;
this change preserves that topology. It does not nest a native FFN beneath an
uncommitted outer legacy cut or revert upstream memory-lifetime work.

## Author intent and execution

`author.kernels.ResidualFFN` is one frozen operation declaration. It names the
up/down projections, three existing compiled Triton entries/launcher, readonly
residual/norm/gate/index tensors, epsilon and the author's exclusive-output
promise. No byte budget, split count or allocating callback is supplied. Public
`split(..., out=..., intent=...)` attempts this complete operation before entering
legacy cut/retry policy. Unsupported contracts fall back only before native
selection/execution; an accepted operation never retries partially written rows.

The first scope is one CUDA device, contiguous BF16 `[1,M,D]` input/output,
int8-convrot mma.sync projections, fixed existing package kernels, and built-in
baked LoRA hooks. Hooked down projections preserve the actual materialized BF16
SwiGLU route; unhooked fused-prologue, FP8, unbaked/custom hooks, global hooks,
autocast, shared operands, capture/replay and multi-rank paths remain excluded.
Runtime imports no private H3 class. Normal projection module calls still run
through the existing `_LINEAR_SPLIT` supplied-output seam, retaining hooks.

The exact operations are:

```text
hidden = BF16(residual + BF16(gate_msa[index] * attention_out))  # into caller out
normed = existing BF16 RMSNorm + modulation kernel
packed = up(normed)                                           # [value | gate]
activated = existing BF16(value * BF16(silu(gate))) kernel
ffn = down(activated)
caller_out = existing BF16(hidden + BF16(gate_mlp[index] * ffn)) kernel
```

Kernel bodies/round points/options remain the package's existing ones. Runtime
checks each supplied entry against the actual loaded launcher's registered set,
argument order, signature, constexprs and source identity. It prebinds readonly
weights and rejects output overlap with any declared readonly storage.

## Complete native menu

Full output already exists and is counted as outside-live memory. Each choice
allocates exactly four BF16 buffers—normed `[m,D]`, packed `[m,2F]`, activation
`[m,F]`, FFN result `[m,D]`—plus one shared flat int8 buffer, FP32 row scales and
the maximum exact queried workspace across both projections' full/tail kernels.
All allocations precede the first in-place output write. Every chunk reuses them
on the saved stream; no per-chunk synchronize or nested native observation occurs.

Existing `int8_plan` supplies actual automatic GEMM/GEMV choice and workspace,
without a new C++ API or changed dispatch. The old standalone projection's native
call/argument/error-check AST is pinned and unchanged after extraction into the
shared allocation-free `launch_projection` helper. Its GPU regressions remain a
required paired gate.

The package kernels require a 16-byte-aligned Int64 index pointer. Whole calls can
have odd M because they start at zero; subsequent chunks have even row extents,
with legal odd final lengths. Halving rounds to even so odd sizes do not lose all
useful intermediate choices (109123→54562→27280…→2). M=1 runs whole; M>1 has a
legal minimum of 2 rows. No index copy or weaker alignment is hidden in the menu.

Native selection/begin own capacity and existing reusable same-stream storage.
Minimum exhaustion precedes output/scratch allocation. A later allocation,
launch or cancellation failure ends the composite, marks it unsubmitted and
preserves the prior successful measurement. Parent completion publishes one
whole-tail observation; the selected ID enters its phase fingerprint. Model or
Request ownership retires semantic keys as for the qualified projection adapter.

## Controls and qualification

Frozen API-only control `1aca1d6cd` accepts the descriptor/keyword but has no native
tail backend, so it executes the original callable. The same new package source
can therefore serve matched control/candidate arms without a user toggle. Both
are unreleased dev cohorts; no published Runtime version floor is invented.

The new package branch is fresh master 28ff0e75 plus the existing 93861 explicit
phase-input/key declaration. Original `_Tail.run`, `_FFN.__call__`, GLUE kernel
bodies and entry metadata remain unchanged. The older frozen 938 retention pair
is not modified and is not evidence for this new consumer.

Required gates: exact full/tail workspace versus automatic native calls;
original-current `_Tail.run` versus composite at whole/half/quarter, odd rows and
minimum; strict unchanged output before refusal; actual baked-hook behavior and
pre-execution exclusions; lifetime/previous-sample/cleanup checks; standalone
projection regression; then full 15 s / 362 frame / 1344×768 / 30 step / audio H3 with the same
new package in both arms. No GPU result or performance acceptance is claimed by
this source checkpoint.

Source checks pass: Ruff, AST parsing, no new mypy findings versus API-only
control, preserved standalone launch-call AST, and CPU legal-index menus. Both
package GLUE sets (15 entries) retain exact Triton identities. Creator #2382's
real Diffusers/SwiGLU/NativeLinear/norm construction passes with CUDA uninitialized.
The existing inherited `_RECENT.last` optional-value type finding remains in
both arms; these checks do not establish GPU acceptance.
