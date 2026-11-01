# Closed int8 projection menu

This slice targets an actual H3 caller: `H3AttnProcessor.__call__` calls
`author.split(attn.to_out[0], attended.flatten(2,3), Cut(1))`. The dispatcher tries
this closed NativeLinear adapter before entering generic split estimation/retry.
It preserves ordinary module invocation and hooks. It does not migrate an inner
leaf under an uncommitted legacy outer cut, nor the whole H3 FFN/attention block.

The first path supports one CUDA device, mma.sync int8.convrot, contiguous aligned
BF16/F16 input, matching output dtype, one row axis (2-D or batch-one 3-D), bound
aligned weights/scales/bias, and the already-serving native op. Autocast, explicit
PreQuantized input, quantize_once scope, current/learned sibling sharing, arbitrary
hooks, unbaked LoRA, custom forward, capture/replay and outer cuts remain outside
this adapter. A normal tensor qualifies after the leaf's existing sibling owner
has learned single use; inference tensors do not use that implicit cache.

Exact built-in baked `lora.Hook` returns before its factor getter: the bound
weight already carries the delta and the fill still calls lora_factors. Current
H3, Flux2 and Qwen factor getters only look up armed factors; they do no required
validation or state transition. The hook stays installed and its ordinary module
call occurs. The graph eligibility guard is unchanged.

## One native choice, one execution

The menu uses whole rows then successively halved window maxima down to one row.
Ragged tails have their own exact queried dispatch. Before choosing, the new
`_gemm.int8_plan` selects the same automatic GEMV predicate or SM80-family config
as normal execution. GEMM query and launch share argument construction,
can_implement, occupancy and `Kernel::Params`; the reported scratch is exactly
`Params.get_workspace_size()`. Query launches no kernel, initializes no workspace,
and allocates no tensor/device scratch. It may initialize the existing host-side
kernel occupancy/attribute metadata, outside capture.

Every choice declares these actual simultaneous allocations:

- one contiguous full output;
- one reusable int8 quantization buffer for the largest GEMM chunk;
- one reusable FP32 row-scale buffer for that chunk;
- one reusable workspace buffer of the maximum exact full/tail requirement.

GEMV-only choices need only the output. Existing input and bound weights are
outside demand, charged by their existing owners. Each chunk uses the saved
stream and its pinned auto-selected config, writes its own full-output rows, and
reuses the same temporary buffers. No tail class or pending chunk allocation is
hidden. The bounded workspace callback can only return the already allocated
buffer; an undeclared size is an error, not a new allocation.

Native select/begin receive the exact stream and complete menu. The key binds
input/weight/bias layouts, dtype/alignment, module provider/encoding, native module
artifact, full/tail geometry/config/workspace, output contract and participating
model identities. The selected native choice also enters the enclosing eager phase's
actual dispatch fingerprint, including the previously introduced norm adapter.
One request owns componentless records. Matching eager-phase
completion publishes submitted samples; without that owner, execution is only
declaration-qualified. Failure after writes aborts the sample and propagates the
original error with the no-retry OOM marker. Unsupported capability returns to its
existing path only before native selection/execution.

## Qualification boundary

Source, native formation and the closed SM89 operator GPU checks passed.
The new GEMM was built from `ebc25419a` with the pinned manylinux/CUDA/CUTLASS
inputs. Its SHA256 is
`9523c5070bd1646f43b505e0ce0bb05670d7b39f8d7e4be3a52846e2b7a81d3e`.
The full unchanged module checker passes with all six modules present, exact
cubin/PTX coverage, dependency/export checks, Python 3.12/3.13/3.14 no-GPU
loads/calls, imports before Torch, and ABI3 audit at baseline 3.12. Maximum required
glibc is 2.17. Exactly the GEMM changed; 579 other native/Triton files remain
byte-identical to the previously qualified prefix. The first target build's Ninja
command span was 664.386 seconds on one CPU; its CMake object cache is durable.

Focused AST/Ruff and adapter mypy pass. Paired broader mypy reports the same four
inherited findings in both the 977 foundation and this source, with zero new.
A CPU smoke check imports the actual Runtime/fixture, verifies unbaked Hook adds
its delta, baked Hook skips the getter, fill factor reads remain active, and leaves
CUDA uninitialized. These checks do not substitute for GPU operation evidence.

Creator's actual `linear-consumer` passed runs **24925** and **24926** in
PID **1694** on an RTX 4090. Tested Runtime was `b2225f7f0`, TensorFS `0e05a2539`,
and fixture `55759abf0`. [Creator PR2365](https://github.com/cozy-creator/cozy-creator/pull/2365)
contains `linear-consumer/evidence/2026-11-01.json` with the complete records and
exact artifact hashes.

| Query rows | Actual automatic dispatch | Query and actual workspace request |
| --- | --- | ---: |
| 8193 | GEMM config 5 (Stream-K) | 33,554,944 bytes |
| 4097 | GEMM config 1 | 0 |
| 3 | GEMV | 0 |

The real public split selected **8193 / 4097 / 2049** rows, including tails
**8193 / 4096 / 2046**. Each exact leaf published after the existing eager stream
completion, and each parent recorded one native choice. Observed leaf backing
was **180 / 118 / 104 MiB**. The half-size selection needed 116 MiB of new backing,
reusing a 24,576-byte slot in an already-backed 2 MiB slab. Outputs all matched
the ordinary whole native result exactly, with relative L2 zero; the existing
prebuilt acceptance budget remains 0.01 rather than a required byte-identity gate.

At one byte below the legal minimum, the operation refused before any output or
scratch allocation: **90,177,536 required / 90,177,535 available** bytes. The
no-retry error was caught by the fixture, but the parent was correctly invalidated
with `split_select_error` and no sample published. A real baked zero-delta LoRA
hook performed no factor lookup and used the closed adapter; unbaked execution
looked up factors three times, performed its update through the existing path,
and published no closed leaf sample. Both output checks had relative L2 zero.

Live allocation bytes returned exactly to the **17,039,360-byte** executor baseline,
with two library owners, no pending bytes and no graph pools. The next request in
the same PID confirmed that both prior request keys were already retired.

Failed run **24923** remains preserved. The fixture called `phase_forget` inside
an active unknown-key parent, which correctly refused retirement. Moving only
fixture capacity-menu cleanup after `denoise_loop` closed the phase fixed the
rerun; neither product binary nor any numerical/lifetime assertion changed.

This is the closed operator proof. The parent was componentless, unknown and
unpublished; it does not establish reusable named-model parent identity or full
H3 quality/performance. Those remain real-model gates before broader migration.
The qualified 977 cohort and b222 binaries/receipts remain immutable. Only the new
GEMM was rebuilt; other modules and Triton artifacts retained their checked bytes.
