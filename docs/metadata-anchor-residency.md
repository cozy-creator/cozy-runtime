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
without restoring it. The metadata pin now survives both planning passes. Only legal
remainder/block cache may be reclaimed; cost-ranking alternatives remains separate work.

CPU/meta tests and static component-size evidence are being added. No GPU qualification or
performance gain is claimed by this draft.
