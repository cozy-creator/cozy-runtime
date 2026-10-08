# Single RTX 5090 H3 optimization

This study targets full 15-second videos at 1344×768 and 24 fps with audio. Regular generation uses 30 evaluations (10 dense then 20 Sol); Turbo uses eight (4 dense then 4 Sol). Protected blocks and prefixes are explicitly recorded. No all-dense timing substitutes for this workload.

Compare current ComfyUI and native candidates on the same rented RTX 5090, with matching action prompts and seeds. Count setup/download/compilation separately. Record actual attention implementations, request duration, phase timings, memory/transfer evidence, and warm output hashes. Candidate settings remain opt-in until the user watches and approves their videos.

Evidence: `/home/fidika/cozy_v2/outputs/h3-single5090-optimization-20261008/`.

Current status: rental requested exclusively through ordinary Cozy CLI; native producer draft PR1240 has CPU tests and an independently checked private wheel, but no GPU admission or performance result yet. The prior 17m37s versus15m39s single-card result is historical context, not a new measurement.

The historical gap is approximately119s in denoising. Progress-based estimates suggest43s in dense evaluations and79s in sparse evaluations; those rounded values are not a CUDA profile. The fused producer targets Q/K/V materialization and quantization work; weight-paging attribution remains unmeasured.
