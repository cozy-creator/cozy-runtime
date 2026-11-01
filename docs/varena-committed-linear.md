# Explicit closed linear phases

The first committed eager cohort is one author-selected native INT8 linear operation.
Runtime owns its execution sequence; it accepts no arbitrary Python compute callback.
It is a component proof, not a claim that an H3 denoising loop is closed. Existing opaque
loops, graph capture, concurrent independent phases, unsupported leaves and adaptive
package calls remain outside this contract.

## Inventory before binding

The producer supplies the actual leaf and input. Runtime makes an immutable inventory
from the current input layout, declared placeholder weight layouts and encoding,
source/fill/representation identities, loaded kernel identity, legal row menus and exact
native dispatch/workspace queries. Inventory contains metadata only: no `read`, Region
prepare, Tensor allocation, pin, fill, or forward call. An incomplete representation is
an explicit refusal before compute entry; a successful earlier trace cannot establish
the declaration.

The ordinary leaf invocation then enters its existing model/source hooks once, outside
the committed compute phase. Source reads, transformations, binding and readiness stay
outside the measured compute scope. Only this operation's operands are materialized;
preparing an inventory never pins every model block. The native interceptor validates
the actual bound layouts, source identities and stream against the current inventory.
A changed binding ends this preparation before any compute write.

## Commit and execute

After binding validation, native commit copies the current descriptor and legal menu,
chooses once against current holdings/phase-region intent, and returns an owned token
with the fixed choice and execution fingerprint. The fingerprint covers the current
configuration and choice; it is never recovered from a previous observation. Native
entry consumes that token's key. The operation consumes its fixed choice and does not
call `split_select` again. Capacity is revalidated before writes; a failure cannot choose
another dispatch or replay partially executed package code.

The initial runner supports exactly one site, which makes closure structural. Its
inventory excludes callbacks and its body invokes only the declared projection. A later
multi-site producer must explicitly own an ordered typed program and all participating
operations; a generic callback registry or an observed call list is not an extension.

The runner owns actual same-stream completion and phase leave. An abort or incomplete
submission does not replace the last successful sample. External callback accounting
continues to invalidate observations. Tokens cannot nest, overlap, enter capture, be
reused, or outlive their execution ownership. Region Plan generation remains distinct
from the working execution token and must still match.

## Measured prefetch boundary

Cold source/refill preparation occurs before compute entry. Once the committed phase
is active, the existing raw-source prefetch adapter may offer the declared next weight
block. Unknown current fingerprints decline speculation. An identical later current
configuration can use its own successful measured sample; a new shape, route, source,
dispatch or selected choice starts unknown. Warm compute demand must not silently cover
different cold-fill work. Required fills still use ordinary demand admission.

The first real proof uses actual TensorFS-backed native linear blocks and Requests:
unknown committed execution with no prefetch, then an identical successful committed
configuration with admitted/finalized raw bytes; changed configuration declining;
aborted work preserving the prior sample; source preparation outside the working scope;
no inventory weight materialization; no nested/captured execution; actual stream and
request/model retirement. Full H3 retains its demand-only path until all participating
operations have producer-owned inventories and no legacy adaptive cut remains.

## Remaining Python decisions (source audit)

| Site | Current behavior | Required boundary |
|---|---|---|
| `execution/sizing.py:choose/descend` | Room capacity, 1/64 knee, retry descent | Retire only after actual consumers declare complete menus |
| `execution/catch.py:cut` | Pre-reclaim, retry `run(level)`, then finer replay | Forbidden inside accepted closed execution |
| `execution/cuts.py:split` | Two measured windows, fitted unseen sizes | No closure or current key may be inferred from these samples |
| FFN/norm/conv/attention wrappers | Python workspace formulas and generic fallback | Each needs a real consumer contract or remains demand-only |
| `parallel/ulysses.py:_level` | Cross-rank byte proposal and collective-safe agreement | Preserve agreement; a later multi-rank declaration is separate |
| `Memory.room/_room_floor` | CUDA free headroom and legacy byte advice | Still used by unmigrated callers; not the new native chooser |
| `Memory._prepare_native` | Python free-byte gate for nonreclaim preparation | Forward native no-reclaim admission directly |
| Native consumer prepare methods | `read`/bind before inventory is complete | Separate metadata description from per-call binding |
| `execution/phase.py:observing` | Enters with `None`, seals choices only on leave | Explicit committed producer alone supplies a current native key |

Full run 25405 still records 900 legacy sparse Heads calls, 150 final norms, 50 output
linear fallback calls, VAE band/tile/block-attention calls and refiner norms/FFNs/attention.
The frozen sparse candidate is GPU-pending, so its 900 calls are not yet retired in
qualification evidence. Host-tier cache displacement is host-source policy, not a CUDA
workspace selector; fresh master se-091 changes are preserved separately.
