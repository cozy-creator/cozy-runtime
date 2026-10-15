Runtime 0.24.4 adds the int8 lane and fused DiT blocks, bundles TensorD 0.5.33 and requires `tensorfs>=0.9.2,<0.10`.

- `int8-convrot256/1`: int8 W8A8 weights rotated per 256 channels (ComfyUI's ConvRot), served by the reviewed comfy-kitchen CUTLASS int8 GEMM with a fused rotate-and-quantize; the eager `torch._int_mm` path serves until the kernel is built. A spec the installed TensorFS lacks is skipped with one warning.
- Fused DiT blocks a package declares (`cozy_runtime.fusion`): qwen-image-2 at 1 MP on a 4090 matches ComfyUI int8 (9.70 s vs 9.66 s).
- #1409: mirrored calls cross by the group broadcast (213 → 10 ms per H3 DiT call), followers fork from an imported rank 0, spill notes ops only under pressure.
- TensorD 0.5.33: output custody reads each output once; a caller's GPU count caps its children. TensorFS 0.9.2: the int8-convrot256/1 seed.
