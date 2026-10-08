# Bounded rowwise FP8 GEMM gate

An ordinary Cozy accelerator job exercises the installed candidate on four H3 matrix
geometries, including a partial final tile and multiple tiles. Every comparison shares
the quantized operands and scales with the unchanged native rowwise reference.
The gate checks byte equality, finite outputs, untouched inputs, request-local scalar
scales, original module hooks, bias, and `PreQuantized` reuse. It saves a small actual
CUDA kernel trace of both implementations after compilation warmup.

Run with the private candidate installed on the rental. Passing synthetic matrices
does not establish full-video performance or visual quality. Timing includes candidate
FP32 scratch allocation and rescaling; all operands are already GPU resident.

The final output tensor is necessary for the layer and is outside the temporary FP32
scratch bound. Vendor GEMM workspace is also outside that explicit tensor bound.
