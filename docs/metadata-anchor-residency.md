# Scope metadata anchors

Draft implementation: split a component's common weights only when the package's existing
`encoded_leaves="accept"` contract promises forward-only component access, the current
forward/state contract is covered, registered backing is readonly/nonmutable, and the
store can refill the exact censused units. Unknown/dynamic access retains ordinary common
residency; no new public annotation is introduced.

The anchor contains the real first-parameter owner and its alias closure. The remaining
common owners are loaded before actual managed operations and may be reclaimed as cache
between them. Whole owners remain indivisible for dtype-conversion replay. `read_whole`,
compiled state, mutable tensors and active operations retain their stronger pins. No fake
tensor, property override, GPU lease change or plan-order change is introduced.

The supported metadata invariant is the existing standard first-parameter/root-device
probe, including its real dtype. It is not a promise about arbitrary user-defined property
code. A positively established split construction is checked for changed registered owner,
slot geometry or cross-unit aliases before further movement and after its operation;
ordinary Python retry caches are not tensor topology. Unknown contracts never get split.

The former deep fallback could evict the real metadata common and return to authored code
without restoring it. The metadata pin is now unconditional. The redundant shallow/deep pass is removed;
richest-to-smallest placement and inexact-forecast floor fallback remain. Only legal
remainder/block cache may be reclaimed; cost-ranking alternatives remain separate work.

CPU/meta tests cover owner/census identity, alias closures, mutation between quote and
move, retry-cache versus tensor-graph changes, metadata-preserving replay and unsupported
initial alias refills. Meta sizes use full unique backing; detection never claims the store
can reconstruct alias/view geometry.

`scope_backing` telemetry separates metadata anchor, operation-common remainder, current
remainder device bytes and scope minima. `anchor_validation` reports owner/alias validation
calls and CPU nanoseconds so its overhead is visible. No plan-cost changes are included.

Static safetensors/source inspection suggests Anima transformer siblings retain at least
376978816B in real first owners out of632851750B common: at most255872934B (~244MiB)
remains before additional aliases/buffers. This is a lower-bound owner census, not an
instantiated partition or measured speedup. The large text embeddings cannot be replaced
with small metadata stand-ins.

The separately reviewed accounting and pending-DISK dtype commits are included as
explicit dependencies. A real TensorFS CPU-store test uses production Planner.adopt
interception and tiers.from_disk: an8B anchor settles at16B, a48B remainder at96B,
and the common load sequence quotes160B transient peak before either arrives. The
real anchor tensor remains unchanged while the remainder fills. No GPU qualification
or performance gain is claimed; this remains a draft for independent review.
