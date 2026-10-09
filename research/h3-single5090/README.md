# Single RTX 5090 H3 optimization

This study targets full 15-second videos at 1344×768 and 24 fps with audio. Regular generation uses 30 evaluations (10 dense then 20 Sol); Turbo uses eight (4 dense then 4 Sol). Protected blocks and prefixes are explicitly recorded. No all-dense timing substitutes for this workload.

Compare current ComfyUI and native candidates on the same rented RTX 5090, with matching action prompts and seeds. Count setup/download/compilation separately. Record actual attention implementations, request duration, phase timings, memory/transfer evidence, and warm output hashes. Candidate settings remain opt-in until the user watches and approves their videos.

Evidence: `/home/fidika/cozy_v2/outputs/h3-single5090-optimization-20261008/`.

Current status (2026-10-09): optimized warm Turbo completed in 280.875 and 277.306 seconds across sparring and rally clips (mean 279.0905 seconds). The same-host Comfy controls are running; the updated regular 30-step result remains pending. Morlock (`pr-df6942e0bdbda053d874`, Vast.ai) was ended through Cozy CLI after CUDA initialization began failing in both fresh Comfy and native processes. Its completed evidence remains retained: RTX5090,575W,driver595.84,73.84GiB container RAM,$0.649167/hour including321GB disk.

The replacement, Chibisuke (`pr-3fafabe2d3b1c634d3d9`, RunPod), was rented through the same CLI and configured Hub. It has an RTX5090 at600W, driver580.126.20,85.68GiB container RAM and321GB disk, at$1.234619/hour including storage. The old Comfy325.362-second result is a target, not a same-host control for Chibisuke. Local GPU inference is excluded.

## Completed measurements

The first table is retained historical evidence from Morlock. All rows below generate 362 frames at 1344×768 and 24 fps, plus stereo 32 kHz audio. Native times are reported handler execution, and Comfy times are server execution start through success; client transfer and observer reconnect delays are excluded. Setup and compilation are separate. These are individual completed requests, not replicated means.

| Mode | Implementation | Request | Execution | Denoise | Video decode |
|---|---|---:|---:|---:|---:|
| Turbo, 4 dense + 4 Sol | ComfyUI 0.39.0, Kitchen INT8 dense/Sol | warm sparring | 325.362 s | 262.443 s | 36.530 s |
| Turbo, 4 dense + 4 Sol | Cozy Kitchen INT8 dense + chunked INT8 Sol producer | 4987, warm sparring | 370.790 s | 317.685 s | 40.332 s |
| Turbo, 4 dense + 4 Sol | Cozy Sage3 global-mean FP4 dense + chunked INT8 Sol producer | 4990, warm sparring | 347.820 s | 293.335 s | 40.251 s |
| Turbo, 4 dense + 4 Sol | Cozy shorter-lived FP4 buffers + shared Q/K/V quantization | 5004, warm sparring | **333.560 s** | 283.834 s | 40.369 s |
| Turbo, 4 dense + 4 Sol | Cozy Sage3 global-mean FP4 dense + materialized FP8/F16 Sol | 4994, warm sparring | 357.682 s | 306.991 s | 40.616 s |
| Regular, 10 dense + 20 Sol | Cozy Sage3 global-mean FP4 dense + chunked INT8 Sol producer | 4993, first sparring | **969.867 s** | 907.559 s | 40.755 s |

On Morlock, the best completed Turbo request was 8.198 seconds above Comfy, improved by 14.260 seconds from request4990. Its denoising gap is 21.391 seconds. Text conditioning was 5.772 seconds versus 2.305 seconds, while video decode differed by 3.838 seconds. Native video encoding overlaps streamed decode; Comfy reports a separate 24.038-second mux/save node. These spans cannot be added as though their boundaries and overlap were identical. Request5004's output SHA256 is identical to4990 (`0a78d463ecfc7b6512f26213e9a7a19954ea7489e74996e07f0bbc0701a21c8b`), including encoded video/audio bytes.

The regular target of 939.1355 seconds is historical Comfy evidence on a different 600 W RTX 5090 with a lighthouse prompt. The new 969.867-second regular request is 30.7315 seconds above that target, but is not a matched fresh comparison. The user stopped additional clean Comfy baseline runs; no fresh regular baseline is claimed. The second fresh Comfy Turbo warm request failed during decode and is excluded. Container memory pressure is consistent with that failure, but no pre-run OOM-counter delta was retained to establish causality.

All completed native rows above have retained original videos and passed full audio/video decoding. Those checks do not establish subjective quality. Precision, statistics reuse, and global-mean FP4 remain explicit opt-in choices pending the user's video review.

On Chibisuke, request5015 completed the first full Turbo video in405.565seconds including initial kernel compilation; it is not a warm benchmark. Request5016, using the matching sparring input/seed, completed in**280.875seconds (4m40.875s)**: denoise232.311s, text6.406s, video decode37.940s. Its23527874-byte video (`cd54bfc0ed35039c6cc78c826b5159ab9cf510719eed6ccd23d11476c6e8d692`) passed full AV checks. It uses restored DiT fusion plus exact LoRA preparation cleanup, shorter-lived FP4 buffers and shared Q/K/V quantization; the experimental alternative GEMM is excluded. The observer lost authorization mid-run, but a read-only `cozy run show` recovered the original completed request without resubmission; the383.024-second CLI wall includes that observation delay.

Request 5017 completed the second warm clip, rally seed 581205, in **277.306 seconds (4m37.306s)**: denoise 227.858 s, text 8.263 s, video decode 37.889 s. Its 20,583,484-byte video (`4f50e45aa0aa034071dd685df07384e900477780e061caab1e38c8aa8ac2f020`) also passed complete audio/video decoding. These two different scenes average 279.0905 seconds; they are not repeated trials of one scene. Both exclude the alternative GEMM.

| Chibisuke Turbo, 4 dense + 4 Sol | Seed | Execution | Denoise | Video decode |
|---|---:|---:|---:|---:|
| Cozy optimized, sparring, run 5016 | 581204 | **280.875 s** | 232.311 s | 37.940 s |
| Cozy optimized, rally, run 5017 | 581205 | **277.306 s** | 227.858 s | 37.889 s |
| ComfyUI, matching controls | matching | Running | Pending | Pending |

The prior Comfy result is 46.272 seconds above this two-scene average, but hardware/driver differences and the mixed-scene mean prevent treating that subtraction as a matched speedup. Current Comfy has exclusive GPU use for setup, matching clean controls, and a fresh-process CUDA trace. Regular 30-step timing follows the GPU handoff.

## Kernel investigation

Source comparison identifies duplicated Q/K/V activation quantization in native execution. The chunked producer invokes the original Q, K, and V modules separately, and each encoded leaf quantizes the same BF16 input. Comfy's fused producer also combines the projections. PR1245 shares quantized activation while retaining original module calls, BF16 LoRA input, and weight-custody wrappers. Actual RTX 5090 admission job4997 passed inactive, baked, and unbaked LoRA cases with bit-identical outputs and quantization counts reduced from six to two. Requests5003/5004 confirm the full model uses shared quantization (54 chunks on the initial statistics replay, then27). It does not yet fuse three weight GEMMs into one.

PR1247 releases Sage3-owned BF16 work arrays after quantization and avoids an intermediate output copy. GPU admission5002 passed bitwise agreement, input preservation, ragged lengths, and mixed dense/Sol policies. Combined with shared quantization, it produced the333.560-second Turbo result above. The tests do not prove each change's independent speed contribution.

A late source audit found that the existing DiT fusion gate rejects the exact Diffusers0.41 wheel because `SwiGLU.forward` gained only a return-type annotation. The method's arithmetic is unchanged; four other checked functions match. With `fusion="accept"`, that source mismatch causes an eager fallback before any fusion substitution. Earlier statements about available native fusion were not proof of active fusion. Trace5006 confirms zero launches of `_norm_mod_kernel`, `_gate_add_kernel`, and `_swiglu_kernel`. PR1250 replaces source fingerprints with callable/topology checks and refuses incompatible semantic hooks before substitution. Actual GPU gate5011 passes all five BF16/encoded/LoRA cases with zero output error and preserved hooks/custody.

GPU microbenchmark5000 compared identical rowwise-quantized operands through the native CUTLASS path versus scalar-unit cuBLAS FP32 output followed by a fused row/column-scale epilogue. Four tested shapes took6.6–16.9% less time for the complete alternative path and produced bit-identical outputs for those random inputs. This excludes paging, checkpoint/video quality, and full-sequence chunking overhead. SM89 and SM90 retain their existing architecture-specific FP8 paths; a5090 microbenchmark is not evidence of gains on those GPUs.

The bounded integration's strict GPU gate5013 **failed exact byte equality** on rare partial-tile values (one to four differing elements in affected cases), while finite/repeat/input/hook/scratch checks passed. Full-sequence benchmark5014 measured complete109192-row operations, including allocations and the chunk loop:5.8–12.8% faster across four shapes, with identical outputs in those particular samples. This does not erase5013's differences. GEMM remains a separate quality variant. PR1248's LoRA preparation cleanup passed5012's ten exhaustive decode-layout cases and eight complete bake cases with exact payload/scales/statistics/RNG state; it removes unused diagnostic work, uses native FP8 decoding with the original reserved-code saturation behavior, and reuses the encoding table within each bake.

The best candidate's dense phase uses SageAttention3 NVFP4 with a global head mean. The ordinary per-block correction workspace would be roughly 19.475 GiB at this sequence length, before conversion temporaries; the global-mean version changes that computation and is a quality variant. FP8/F16 sparse attention was slower than the INT8 chunked producer in the measured Turbo requests; arithmetic peak alone does not predict full-run speed.

The user subsequently authorized full Comfy and native execution traces. Capture covers text encoding, every dense/Sol evaluation, scheduler updates, both VAEs, and output handling. CPU launch/synchronization, GPU kernels, copies, and union GPU busy intervals must be compared without double-counting concurrent streams. External FFmpeg instructions are outside PyTorch's instruction trace, so stream/finalization wall spans are recorded separately.

Native diagnostic request4996 rendered all eight evaluations and decoded the video, but failed exporting a compressed trace above a1GiB diagnostic bound. That bound is fixed: continuous file-backed gzip/tree export retains and verifies a1,190,659,470-byte compressed CPU trace, and actual-schema capacity plumbing passes ordinary CLI job and nested-serving exports above1GiB. Native4998 exhausted RAM while processing the full CPU/CUDA trace after rendering; the Comfy full CPU/CUDA capture also hit the pod's memory limit during video decode. Neither failed trace is a speed measurement or a retained kernel artifact.

CUDA-only full native capture5006 succeeded:10,693,764,397raw bytes compressed to845,906,302bytes,24,516,776events,5,467,152GPU kernels. All kernels correlate to host ranges;3339 external pinned-H2D events lack matching CPU APIs and remain unassigned. The trace directly identifies SM120 FP8 Tensor Core GEMMs (80.350s aggregate), FP4 dense attention(80.403s) and Sol exact kernels(32.848s). Those are overlapping diagnostic sums, not independently additive speed budgets.

Steady later steps transfer12.439GiB of weight data from RAM to VRAM each, with no physical disk reads; the full request's198,199,973,216H2D bytes match Runtime counters exactly and take5.227s summed GPU-copy time. Denoising has only tiny D2H transfers; video decoding owns almost all1.160s pageable D2H time. All193545 lookup-table-construction kernels are in denoising;18513 per steady step equal three table builds for6171 bake chunks. Most tiny eager kernels fall in block setup/weight work outside attention and FFN scopes. Dedicated bake ranges are added to profiler0.0.10 for stronger attribution after optimization.

The full diagnostic device busy union is290.745s across392.734s, with101.989s in traced gaps. Cold setup and profiler overhead contribute to these values; they are **not**102seconds of demonstrated clean-run savings. Clean warm5004's interior282.031seconds had zero physical disk reads, swap, major faults or memory-pressure stalls. Its charged73.8GiB RAM was mostly file cache, not live tensors. PCIe samples are20ms snapshots despite2s polling and are not valid transfer-byte totals.

No complete Comfy kernel comparison is retained yet: after the failed CPU/CUDA capture, an unprofiled warmup failed AudioVAE allocation with `cuMemSetAccess` CUDA999. Fresh Comfy and native processes then also failed CUDA initialization. The rental was ended after artifact retention and replaced. A new same-host control and CUDA trace are required before claiming a matched win.

Actual GPU probe4999 retained kernels, copies, and CUDA runtime/driver API events in CUDA-only mode. Probe5005 confirms that Chrome CUDA thread IDs use the absolute signed low32 pthread identity; the attribution helper matches all12 device events to their manual launch ranges. Disk/RAM/process-fault/PCIe sampling remains active, independently of PyTorch tracing. Partial native4998 evidence shows substantial initial disk reads but near-zero physical disk reads in later dense steps; this does not exclude RAM-to-VRAM transfers or quantify their critical-path cost.

`attribute_trace.py` uses the common PR1243 streaming parser to associate GPU correlation IDs with CPU CUDA API calls and their enclosing annotation ranges. It retains ambiguous and unassociated activities in the totals. Seven CPU regression tests cover asynchronous execution beyond a CPU scope, thread boundaries/aliases, duplicate IDs, nested ranges, invalid events, clock conversion, and transfer-byte preservation. These are diagnostic duration sums; use the common parser's interval unions for GPU busy time.

The earliest historical denoising gap was approximately 119 s. Those progress-based estimates preceded the retained trace and subsequent optimizations; they are not the current gap. The trace now directly measures weight transfers, while their overlap with useful computation still prevents interpreting summed copy duration as critical-path savings.

## Evidence and candidate boundaries

`results.json` retains request IDs, exact attention observations, phase/step times, and output hashes extracted from the original CLI receipts. Raw receipts, media checks, candidate wheels, and Comfy provenance remain in the evidence directory above. Comfy commit is `b0b743566f65daafc423b4fea8a2fbda94b3384a`, Kitchen 0.2.37, Torch 2.14.0+cu130, with its fixed cuBLAS 13.2.0.9 overlay.

Candidate work is split across draft PR1240 (chunked producer), PR1242 (global-mean FP4 dense), PR1244 (FP8/F16 Sol hybrid), and PR1245 (shared Q/K/V quantization). PR1243 owns profiling. None of these measurements promotes a new default or proves another GPU/workload wins. The active rental must be ended through Cozy CLI after all required artifacts are retained.
