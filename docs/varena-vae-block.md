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
