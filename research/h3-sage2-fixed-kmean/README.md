# H100 Sage2 fixed K-mean research

The completed fixed-input H100 probe5232 found the first recorded head-partition difference in BF16 K smoothing means. At seven/eight partitions it propagated to live quantized K codes and dense output; no collectives were involved.

This isolated package will compare the actual public per-thread Sage2 SM90 adapter, an exact replay through its exposed quantizers/native attention call, and that same replay with a fixed token-reduction tree for K mean. It changes no Runtime default, sparse path, V/P conversion, or attention accumulation.

CPU checks and a later explicitly authorized ordinary-CLI GPU diagnostic must establish public-replay equality, input custody, intermediate and output repeat/partition identity, and complete-attention speed. No GPU is authorized by this document. Existing quality runs continue unchanged.

A partition-stable kernel may differ from the original floating-point reduction order. Synthetic identity does not establish full-video identity or visual quality. A candidate must not impose a speed penalty before any serving integration is considered.

The fixture uses B1/NHD/D128 BF16 operands at N129 and N109225, cloned contiguously for full56 and head groups28/14/8/7. The fixed mean uses 256-token by32-channel partial sums and a second fixed FP32 tree, then one BF16 rounding. Token/head addressing widens to int64 before stride products; this research wrapper currently refuses noncontiguous inputs.

The per-thread SM90 replay keeps upstream BLKQ64/WARPQ16/BLKK128/WARPK128, V padding to128, per-channelFP8 conversion, and the same bufferedFP32 native attention call. A byte-equality control against the actual Runtime adapter must pass before interpreting the replay. Live means, codes, scales, FP8V and outputs are compared; unused Q/K tail scales are excluded by their token ownership.

Five warmed CUDA-event samples alternate actual-public/replay/fixed order; input creation, copies and metrics are excluded. The mean-only row is diagnostic; the complete actual public callable is the speed baseline. Full cases require28GiB free/reusable memory and record actual peak; no shape reduction. The owner should use the existing eight-minute request bound and preserve the accepted identity. No model weights or LoRA are involved.

CPU interpreter tests check actual reduction kernels across partitions, input custody, live-token denominator and int64 address arithmetic. Four SM90 kernels compile offline with TTIR retained; no CUDA context is opened. Ordinary CLI describe validates the accelerator job surface. Those are preparation checks, not GPU qualification.
