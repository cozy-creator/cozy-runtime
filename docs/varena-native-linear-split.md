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

Source and native formation checks passed; this checkpoint has no GPU result.
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

Creator's `linear-consumer` fixture (draft PR2365) supplies the real caller proof. Qualification must use a normal GPU app with an actual
NativeLinear plus public `author.split` inside a real Request/eager phase, not a
mocked context. Require multiple selected chunks including a ragged tail, exact
query-versus-execution scratch, baked/unbaked hook behavior, minimum refusal before
output allocation, dtype-appropriate numerical agreement, and same-process
request/key cleanup. Full H3 is a later performance/quality gate.

The changed `_gemm` binary requires a new targeted build. Its old binary cannot be
reused. Other native modules and Triton artifacts may be reused only after exact
source/toolchain/catalog/checker identity checks; all qualified 977 artifacts stay
immutable. A fresh durable CMake cache is a first partial rebuild, not a cache hit.
