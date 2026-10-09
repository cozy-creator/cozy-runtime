# Explicit H3 attention precision matrix

Owner: Paul Fidika; implementation agent: linear_resume.
Branch: bench/h3-attention-matrix-20261009.
Base: origin/master c6fea990497a43d7f31bf016a8a06ae57cb14324.
Purpose: request-selectable full dense and four-dense/four-Sol attention comparisons on SM120, without changing defaults.

The running private cohort is allfast 6a429905054893bcac6514a7ba39fddc5f28f5a5d57cf885565c043392a709ba. Candidate wheels overlay only reviewed attention modules; existing bounded linear GEMM, LoRA bake, fusion, native executable and dependencies are preserved. Root owns GPU qualification and installation.

The existing FP8 Sol selectors use Sage or NVFP4 dense calls and do not represent pure FP8 dense attention. This task adds explicit dense FP8/F16 and BF16/F32 controls, with matching mixed selectors. Full-dense controls must never invoke sparse routing. All numerical changes remain opt-in pending the user's video review.

The prerequisite attention sources are carried byte-for-byte from the running wheel (the prior opt-in producer, shared-QKV, FP8 sparse and low-memory Sage3 work through 072ed5e2). No default ranking changes are introduced. They are a separate prerequisite commit; this PR is not evidence that those unmerged experiments are qualified defaults.

| Requested row | Full dense selector (all 8 steps) | Mixed selector (4 dense + 4 Sol) | Actual arithmetic |
|---|---|---|---|
| Kitchen INT8 | `kitchen-int8` | `kitchen-sol-producer-fast-shared-qkv` | Exact QK signed INT8 and PV unsigned INT8/signed INT8 to INT32; FP32 state; sparse pooled tail BF16 to FP32 |
| Sage3 NVFP4 | `sageattention3-fp4-global-lowmem` | `kitchen-sol-producer-fp4-lowmem-shared-qkv` | Dense NVFP4 to FP32; sparse and its protected prefix use Kitchen INT8, explicitly a hybrid |
| FP8/F16 | `fp8-fp16-dense` | `sol-fp8-fp16-pure` | Dense and exact sparse QK/PV E4M3 to FP16 tiles; FP32 online state; sparse routing and approximate tail BF16 to FP32 |
| BF16/F32 | `bf16-fp32-dense` | `sol-bf16-fp32` | Forced Torch Flash dense and upstream BF16 Sol, both BF16 to FP32 |

Both Kitchen mixed controls retain previous-step statistic reuse and shared QKV activation codes. The FP8 and BF16 controls materialize QKV; results compare the actual implementations, not just an isolated MMA dtype. Dense model paths/refiners remain dense in mixed mode. Full-dense selectors do not enter Sol policy or preprocessing at any denoising step. The fixed allfast linear path is identical across all rows.

Source-only validation: 50 CPU tests passed, 4 GPU-only tests skipped. Four broader Gloo tests hit the current master fixture's ProcessGroup-versus-Comms mismatch in `parallel/cp.py`; these are outside the new one-GPU controls. Focused Ruff and mypy checks passed. The ordinary `research/attention-matrix-gate/verify` job exercises each new selector through the real Diffusers registry, odd tails, unequal prefixes, dense/sparse policy, repeatability and unchanged inputs, CUDA-only kernel traces, and dense FP8 PTX MMA identity. It is not a full-video quality qualification. Candidate installation and GPU execution belong to the root agent.
