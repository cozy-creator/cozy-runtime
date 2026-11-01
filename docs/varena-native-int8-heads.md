# Exact Kitchen INT8 projected heads

Extend the complete ProjectedHeads operation using the actual selected Kitchen INT8 dense/exact route. The pinned existing quantize and prequantized APIs keep all buffers caller-owned. Reuse consumed internal Q float scratch as contiguous BHND output, then copy its transpose into the declared full NHD output on the same stream. No C++ kernel or numerical-tiling change is planned.

Use the real pinned artifact identity and shared dense/exact/prefix-heavy policy. Validate the fused reference against split ABI execution, actual allocation classes/counts, zero-allocating final copy, both CTA sizes, seed/complete-processor behavior and refusal/retirement. Sparse producer remains a separate migration. The qualified INT4 cohort and its running full-model pair stay frozen.

The descriptor reads only an already-served Kitchen INT8 kernel, retaining its ready artifact and exact loaded extension hash. Direct Kitchen INT8 and the selected INT4QK Sol route's exact/prefix-heavy calls are eligible. Sparse calls, other backends and multi-rank calls still decline before native acceptance. The ordinary Sol path and the closed consumer share the same route resolution and exact-site receipt.

The branch includes qualified Runtime phase-region Plan commit `190fe844d`. Heads forwards the native selection's opaque generation through `phase.begin_split`, as the other native consumers do. The next coherent cohort requires TensorFS `4dcf26183`; the fixture preserves unplanned calls and adds a complete processor under a real active native Plan with a READY Region. This checks generation forwarding and Region preservation; it does not claim that the fixture's manually constructed projection weights came from Runtime's model Region inventory.

The matched control retains these APIs, the qualified INT4 consumer, Plan integration and the same fresh master. It declines only INT8 routes before their closed descriptor is built, so the ordinary selected INT8 path serves them.

CPU construction, real encoding eligibility and unchanged GLUE identities pass. The only current mypy diagnostic is the existing `author/kernels.py` optional `_RECENT.last._version` access; no new type error remains. No INT8 GPU qualification is claimed. Primary source inventory: `outputs/xs138-attention-closure-review` in the shared workspace, pinned Kitchen commit `f8b7e97a1418d2e529bd5405fbf86312be198ae6`.
