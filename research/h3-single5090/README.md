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
| Turbo, 4 dense + 4 Sol | Cozy Sage3 global-mean FP4 dense + chunked INT8 Sol producer | 4990, warm sparring | **347.820 s** | 293.335 s | 40.251 s |
| Turbo, 4 dense + 4 Sol | Cozy Sage3 global-mean FP4 dense + materialized FP8/F16 Sol | 4994, warm sparring | 357.682 s | 306.991 s | 40.616 s |
| Regular, 10 dense + 20 Sol | Cozy Sage3 global-mean FP4 dense + chunked INT8 Sol producer | 4993, first sparring | **969.867 s** | 907.559 s | 40.755 s |

The best completed Turbo request remains 22.458 seconds above Comfy. Its denoising gap is 30.892 seconds. Text conditioning was 9.955 seconds versus 2.305 seconds, while video decode differed by 3.721 seconds. Native video encoding overlaps streamed decode; Comfy reports a separate 24.038-second mux/save node. These spans cannot be added as though their boundaries and overlap were identical.

The regular target of 939.1355 seconds is historical Comfy evidence on a different 600 W RTX 5090 with a lighthouse prompt. The new 969.867-second regular request is 30.7315 seconds above that target, but is not a matched fresh comparison. The user stopped additional clean Comfy baseline runs; no fresh regular baseline is claimed. The second fresh Comfy Turbo warm request failed during decode and is excluded. Container memory pressure is consistent with that failure, but no pre-run OOM-counter delta was retained to establish causality.

All completed native rows above have retained original videos and passed full audio/video decoding. Those checks do not establish subjective quality. Precision, statistics reuse, and global-mean FP4 remain explicit opt-in choices pending the user's video review.

## Kernel investigation

Source comparison identifies duplicated Q/K/V activation quantization in native execution. Turbo LoRA hooks prevent the existing fused activation handoff from being installed during construction. The chunked producer invokes the original Q, K, and V modules separately, and each encoded leaf quantizes the same BF16 input. Comfy's fused producer also combines the projections. PR1245 first shares quantized activation while retaining original module calls, BF16 LoRA input, and weight-custody wrappers. Its CPU tests include baked/unbaked LoRA equality and quantization counts; real GPU admission and full-model usage counters remain required. It does not yet fuse three weight GEMMs into one.

The best candidate's dense phase uses SageAttention3 NVFP4 with a global head mean. The ordinary per-block correction workspace would be roughly 19.475 GiB at this sequence length, before conversion temporaries; the global-mean version changes that computation and is a quality variant. FP8/F16 sparse attention was slower than the INT8 chunked producer in the measured Turbo requests; arithmetic peak alone does not predict full-run speed.

The user subsequently authorized full Comfy and native execution traces. Capture covers text encoding, every dense/Sol evaluation, scheduler updates, both VAEs, and output handling. CPU launch/synchronization, GPU kernels, copies, and union GPU busy intervals must be compared without double-counting concurrent streams. External FFmpeg instructions are outside PyTorch's instruction trace, so stream/finalization wall spans are recorded separately.

Native diagnostic request 4996 rendered all eight evaluations and decoded the video, but failed exporting a compressed trace above a 1 GiB diagnostic bound. The temporary raw trace was deleted during exception unwinding. Its 729.031-second execution includes profiling and failed export, is not a benchmark, and provides no retained kernel trace. Continuous file-backed gzip/tree export and actual-schema output-capacity plumbing are being fixed and tested above 1 GiB before another capture. Neither a complete retained native trace nor a matched Comfy trace is available yet.

The historical gap is approximately119s in denoising. Progress-based estimates suggest43s in dense evaluations and79s in sparse evaluations; those rounded values are not a CUDA profile. The fused producer targets Q/K/V materialization and quantization work; weight-paging attribution remains unmeasured.

## Evidence and candidate boundaries

`results.json` retains request IDs, exact attention observations, phase/step times, and output hashes extracted from the original CLI receipts. Raw receipts, media checks, candidate wheels, and Comfy provenance remain in the evidence directory above. Comfy commit is `b0b743566f65daafc423b4fea8a2fbda94b3384a`, Kitchen 0.2.37, Torch 2.14.0+cu130, with its fixed cuBLAS 13.2.0.9 overlay.

Candidate work is split across draft PR1240 (chunked producer), PR1242 (global-mean FP4 dense), PR1244 (FP8/F16 Sol hybrid), and PR1245 (shared Q/K/V quantization). PR1243 owns profiling. None of these measurements promotes a new default or proves another GPU/workload wins. The active rental must be ended through Cozy CLI after all required artifacts are retained.
