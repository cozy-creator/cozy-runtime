Blackwell attention kernels, each pinnable and context-parallel through the Ulysses wrapper:
`flash-attn4-fp8` (FA4's SM100 e4m3 forward, per-head descales), `sageattention3-fp4`
(SageAttention3 NVFP4, SM120), `sageattention` on SM89/SM120 (upstream's SM89 kernel,
FP32+FP16), and `kitchen-int8` on SM120/121. `flashinfer-bf16-fp8` now quantises V per head
and folds the descale into the output, so it shards; its kernel is JIT-compiled locally instead
of downloaded. A site kernel's device code is bounded by the image manifest's arches, and a
budget not yet measured on a card is reported `"budget": "uncalibrated"`.
