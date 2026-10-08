# Single RTX 5090 H3 optimization

This study targets full 15-second videos at 1344×768 and 24 fps with audio. Regular generation uses 30 evaluations (10 dense then 20 Sol); Turbo uses eight (4 dense then 4 Sol). Protected blocks and prefixes are explicitly recorded. No all-dense timing substitutes for this workload.

Compare current ComfyUI and native candidates on the same rented RTX 5090, with matching action prompts and seeds. Count setup/download/compilation separately. Record actual attention implementations, request duration, phase timings, memory/transfer evidence, and warm output hashes. Candidate settings remain opt-in until the user watches and approves their videos.

Evidence: `/home/fidika/cozy_v2/outputs/h3-single5090-optimization-20261008/`.

Current status (2026-10-08): neither speed target has been met. The active rental is Morlock (`pr-df6942e0bdbda053d874`), obtained and maintained exclusively through ordinary Cozy CLI on Vast.ai. It has one RTX 5090 with a 575 W cap, driver 595.84, and a 73.84 GiB container RAM limit. Its total quoted rate including 321 GB disk is $0.649167/hour. Local GPU inference is excluded.

## Completed measurements

All rows below generate 362 frames at 1344×768 and 24 fps, plus stereo 32 kHz audio. Native times are reported handler execution, and Comfy times are server execution start through success; client transfer and observer reconnect delays are excluded. Setup and compilation are separate. These are individual completed requests, not replicated means.

| Mode | Implementation | Request | Execution | Denoise | Video decode |
|---|---|---:|---:|---:|---:|
| Turbo, 4 dense + 4 Sol | ComfyUI 0.39.0, Kitchen INT8 dense/Sol | warm sparring | 325.362 s | 262.443 s | 36.530 s |
| Turbo, 4 dense + 4 Sol | Cozy Kitchen INT8 dense + chunked INT8 Sol producer | 4987, warm sparring | 370.790 s | 317.685 s | 40.332 s |
| Turbo, 4 dense + 4 Sol | Cozy Sage3 global-mean FP4 dense + chunked INT8 Sol producer | 4990, warm sparring | 347.820 s | 293.335 s | 40.251 s |
| Turbo, 4 dense + 4 Sol | Cozy shorter-lived FP4 buffers + shared Q/K/V quantization | 5004, warm sparring | **333.560 s** | 283.834 s | 40.369 s |
| Turbo, 4 dense + 4 Sol | Cozy Sage3 global-mean FP4 dense + materialized FP8/F16 Sol | 4994, warm sparring | 357.682 s | 306.991 s | 40.616 s |
| Regular, 10 dense + 20 Sol | Cozy Sage3 global-mean FP4 dense + chunked INT8 Sol producer | 4993, first sparring | **969.867 s** | 907.559 s | 40.755 s |

The best completed Turbo request remains 8.198 seconds above Comfy, improved by 14.260 seconds from request4990. Its denoising gap is 21.391 seconds. Text conditioning was 5.772 seconds versus 2.305 seconds, while video decode differed by 3.838 seconds. Native video encoding overlaps streamed decode; Comfy reports a separate 24.038-second mux/save node. These spans cannot be added as though their boundaries and overlap were identical. Request5004's output SHA256 is identical to4990 (`0a78d463ecfc7b6512f26213e9a7a19954ea7489e74996e07f0bbc0701a21c8b`), including encoded video/audio bytes.

The regular target of 939.1355 seconds is historical Comfy evidence on a different 600 W RTX 5090 with a lighthouse prompt. The new 969.867-second regular request is 30.7315 seconds above that target, but is not a matched fresh comparison. The user stopped additional clean Comfy baseline runs; no fresh regular baseline is claimed. The second fresh Comfy Turbo warm request failed during decode and is excluded. Container memory pressure is consistent with that failure, but no pre-run OOM-counter delta was retained to establish causality.

All completed native rows above have retained original videos and passed full audio/video decoding. Those checks do not establish subjective quality. Precision, statistics reuse, and global-mean FP4 remain explicit opt-in choices pending the user's video review.

## Kernel investigation

Source comparison identifies duplicated Q/K/V activation quantization in native execution. The chunked producer invokes the original Q, K, and V modules separately, and each encoded leaf quantizes the same BF16 input. Comfy's fused producer also combines the projections. PR1245 shares quantized activation while retaining original module calls, BF16 LoRA input, and weight-custody wrappers. Actual RTX 5090 admission job4997 passed inactive, baked, and unbaked LoRA cases with bit-identical outputs and quantization counts reduced from six to two. Requests5003/5004 confirm the full model uses shared quantization (54 chunks on the initial statistics replay, then27). It does not yet fuse three weight GEMMs into one.

PR1247 releases Sage3-owned BF16 work arrays after quantization and avoids an intermediate output copy. GPU admission5002 passed bitwise agreement, input preservation, ragged lengths, and mixed dense/Sol policies. Combined with shared quantization, it produced the333.560-second Turbo result above. The tests do not prove each change's independent speed contribution.

A late source audit found that the existing DiT fusion gate rejects the exact Diffusers0.41 wheel because `SwiGLU.forward` gained only a return-type annotation. The method's arithmetic is unchanged; four other checked functions match. With `fusion="accept"`, that source mismatch causes an eager fallback before any fusion substitution. Earlier statements about available native fusion were not proof of active fusion. Runtime class census and a compatibility fix are pending; no fusion speed contribution is yet measured.

GPU microbenchmark5000 compared identical rowwise-quantized operands through the native CUTLASS path versus scalar-unit cuBLAS FP32 output followed by a fused row/column-scale epilogue. Four tested shapes took6.6–16.9% less time for the complete alternative path and produced bit-identical outputs for those random inputs. This excludes paging, checkpoint/video quality, and full-sequence chunking overhead. SM89 and SM90 retain their existing architecture-specific FP8 paths; a5090 microbenchmark is not evidence of gains on those GPUs.

The best candidate's dense phase uses SageAttention3 NVFP4 with a global head mean. The ordinary per-block correction workspace would be roughly 19.475 GiB at this sequence length, before conversion temporaries; the global-mean version changes that computation and is a quality variant. FP8/F16 sparse attention was slower than the INT8 chunked producer in the measured Turbo requests; arithmetic peak alone does not predict full-run speed.

The user subsequently authorized full Comfy and native execution traces. Capture covers text encoding, every dense/Sol evaluation, scheduler updates, both VAEs, and output handling. CPU launch/synchronization, GPU kernels, copies, and union GPU busy intervals must be compared without double-counting concurrent streams. External FFmpeg instructions are outside PyTorch's instruction trace, so stream/finalization wall spans are recorded separately.

Native diagnostic request4996 rendered all eight evaluations and decoded the video, but failed exporting a compressed trace above a1GiB diagnostic bound. That bound is fixed: continuous file-backed gzip/tree export retains and verifies a1,190,659,470-byte compressed CPU trace, and actual-schema capacity plumbing passes ordinary CLI job and nested-serving exports above1GiB. However, native4998 exhausted RAM while processing the full CPU/CUDA trace after rendering; the Comfy full CPU/CUDA capture also hit the pod's memory limit during video decode. Neither failed trace is a speed measurement or a retained kernel artifact. The next capture uses continuous CUDA-only activity plus manual host ranges, avoiding ATen CPU-event accumulation and key-averages reconstruction.

Actual GPU probe4999 retained kernels, copies, and CUDA runtime/driver API events in CUDA-only mode. Probe5005 confirms that Chrome CUDA thread IDs use the absolute signed low32 pthread identity; the attribution helper matches all12 device events to their manual launch ranges. Disk/RAM/process-fault/PCIe sampling remains active, independently of PyTorch tracing. Partial native4998 evidence shows substantial initial disk reads but near-zero physical disk reads in later dense steps; this does not exclude RAM-to-VRAM transfers or quantify their critical-path cost.

`attribute_trace.py` uses the common PR1243 streaming parser to associate GPU correlation IDs with CPU CUDA API calls and their enclosing annotation ranges. It retains ambiguous and unassociated activities in the totals. Six CPU regression tests cover asynchronous execution beyond a CPU scope, thread boundaries/aliases, duplicate IDs, nested ranges, invalid events, and clock conversion. These are diagnostic duration sums; use the common parser's interval unions for GPU busy time.

The historical gap is approximately119s in denoising. Progress-based estimates suggest43s in dense evaluations and79s in sparse evaluations; those rounded values are not a CUDA profile. The fused producer targets Q/K/V materialization and quantization work; weight-paging attribution remains unmeasured.

## Evidence and candidate boundaries

`results.json` retains request IDs, exact attention observations, phase/step times, and output hashes extracted from the original CLI receipts. Raw receipts, media checks, candidate wheels, and Comfy provenance remain in the evidence directory above. Comfy commit is `b0b743566f65daafc423b4fea8a2fbda94b3384a`, Kitchen 0.2.37, Torch 2.14.0+cu130, with its fixed cuBLAS 13.2.0.9 overlay.

Candidate work is split across draft PR1240 (chunked producer), PR1242 (global-mean FP4 dense), PR1244 (FP8/F16 Sol hybrid), and PR1245 (shared Q/K/V quantization). PR1243 owns profiling. None of these measurements promotes a new default or proves another GPU/workload wins. The active rental must be ended through Cozy CLI after all required artifacts are retained.
