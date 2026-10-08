# FP8 GEMM and scale epilogue microbenchmark

This ordinary accelerator job compares the existing rowwise `torch._scaled_mm` call
with a scalar-unity-scale FP32 `torch._scaled_mm` followed by a fused Triton
`(acc * row_scale) * column_scale` FP32 epilogue that stores BF16. Both paths receive
the exact same resident FP8 E4M3 A/B tensors. There is no bias, calibration, model
checkpoint, paging test or video inference here.

The explicit SM120 study uses the actual H3 matrix geometries: Q/K/V N7168 K5376
with M4096, attention output N5376 K7168 with M1024, fc1 N28672 K5376 with M1024,
and fc2 N5376 K14336 with M1024. Turbo's FP32 heads are excluded.

The native path preserves `use_fast_accum=True`; the scalar FP32 candidate uses
`False`. Those accumulator choices are reported, not treated as bitwise equivalent.
The inspected Torch SM120 rowwise CUTLASS body does not use its FastAccum template,
so these API flags do not by themselves form an accumulation-mode A/B. The study
changes the GEMM implementation and adds FP32 output/epilogue staging.
The numerical reference is an FP32 dot of the same quantized operands, sequential
FP32 scaling and BF16 rounding, with TF32 disabled. Errors against the unquantized
FP32 dot and between implementations are also retained. A/B/scales are checked
byte-for-byte after timings, and activation quantization is separately timed and
verified repeatable.

CUDA events measure warmed quantization, native GEMM, scalar GEMM, epilogue, and the
combined scalar-GEMM/epilogue path. The combined path is measured directly, not
estimated by adding independent medians. Inputs are already resident; output
allocation is retained in each GEMM call. References, comparisons and compilation
are outside measured repetitions. Matrix errors do not establish video quality or
full-request speedup. Scalar/rowwise API contracts select the intended routes;
actual kernel names still require a profiler observation.

Prepare the package using `cozy run /absolute/path/to/package/compare --describe`.
A remote run, only after exclusive GPU handoff, uses the ordinary form:

```
cozy run /absolute/path/to/package/compare --rental=morlock --input=/absolute/path/to/input.json --await
```

Example input: `{"case":"all","seed":20261008,"warmup":10,"repeats":30}`.
The local artifact copy can point `tool.uv.sources.cozy-runtime` at the immutable
study wheel for metadata preparation. The selected worker SDK remains execution
authority. No local GPU run is part of preparation.

The separate `trace_cuda_only` job is a tiny profiler-capability probe. It records
CUDA-only activities around labeled H2D, matmul and D2H operations, then exports
the ChromeTrace plus an actual category/correlation/annotation census. Missing
user annotations or CUDA-runtime events are reported rather than assumed present.
Its default512x512 matrices and3repeats do not measure H3 performance.
