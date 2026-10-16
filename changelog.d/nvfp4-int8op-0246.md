Runtime 0.24.6 adds the nvfp4 leaf and a native int8 op, and bundles TensorD 0.5.36.

- `nvfp4-w4a4/1`: W4A4 through cuBLASLt on sm100+ above 16 rows, a Triton W4A16 GEMV (or dequant + bf16 GEMM) below; `QuantizationPlan.encodings` mixes nvfp4 and int8 in one lane (Klein nvfp4 lanes; the upsampler on 8 GiB in 24–32 s instead of 61–86 s).
- `torch.ops.cozy.int8_linear`, compiled at boot (recipe `cozy-int8`): rotate + quantize + CUTLASS int8 GEMM in one call (wgmma on Hopper, tcgen05 on B200); comfy-kitchen then eager serve until it lands.
- After an attempt, the restore maps back only components a stage placed whole (#1433: 8 GB cards stop re-paging streamed prefixes).
- A plan document is as large as its model: `DOC_MAX_BYTES` 256 MiB (#1435; TensorD 0.5.36 #292).
