# LoRA bake GPU equivalence gate

This ordinary accelerator job compares the immutable original bde helper with the
helper installed in its worker SDK. The bundled reference file is byte-identical
to the old wheel's `cozy_runtime/internal/lora_bake.py`; it keeps its existing
namespace imports and is test reference code, never a serving fallback.

The gate uses identical synthetic stored weights, LoRA factors and independently
seeded CUDA generators for each side. Plain BF16, plain FP32, rowwise FP8 and MXFP8
are tested with statistics disabled and enabled. Each weight uses H3's K5376,
rank64, and781 output rows: two full default2M-element bake chunks plus a one-row
tail. This deliberately tests an operator's chunk boundaries, not a full H3 layer
or video-performance workload.

Every payload/scale is compared bitwise, the RNG state is compared, immutable
factors are checked, finite outputs are checked, and all statistics except timing
must match exactly. Unsupported CUDA operations fail visibly. The result includes
an immediate `passed` flag and a JSON artifact with per-case evidence. TF32 is
disabled identically for both helpers and restored at exit. No timing advantage is
claimed by this gate.

Root can run it on the exclusively handed-off rental with ordinary Cozy:

```
cozy run /absolute/path/to/gate/verify --rental=morlock --input=/absolute/path/to/gate-input.json --await
```

The package is prepared through ordinary `cozy run .../verify --describe`. No GPU
launch is performed while preparing the artifact.
