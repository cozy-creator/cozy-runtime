// SPDX-License-Identifier: Apache-2.0
// svdq.nvfp4's linear on RTX 50's block-scaled FP4 MMA (`svdq.cu`): raw pointers and a stream,
// no torch. The caller allocates every operand; `x`, `bias` and `out` are bf16 (`half == false`)
// or fp16, the low-rank factors and `la` bf16. Each returns its launch's error.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cozy_svdq {

// The GEMM's row tile: the activation operand, its scales and `la` hold M rounded up to it.
constexpr int kRows = 256;
// N is a multiple of the GEMM's column tile; K of its k-step.
constexpr int kCols = 128;
constexpr int kDepth = 64;

// Whether the kernels take N output columns, K input columns and rank R.
inline bool fits(int N, int K, int R) {
    return N >= kCols && N % kCols == 0 && N / kCols <= 65535 && K > 0 && K % kDepth == 0 &&
           (R == 32 || R == 64 || R == 128);
}

// `x` [M, K] -> the GEMM's activation operand: nvfp4 codes of bf16(x * p) at second level g[0]
// (`act`, kRows-row tiles in fragment order, Rows(M) * K / 2 bytes) with their E4M3 block scales
// (`ascales`, Rows(M) * K / 16 bytes), and la = bf16(x · downᵀ) [Rows(M), R] row-major (`down`
// [R, K] bf16). Rows past M are zeros. R is 32, 64 or 128.
cudaError_t quantize(const void* x, bool x_half, const float* p, const float* g, const void* down,
                     void* act, void* ascales, void* la, int M, int K, int R,
                     cudaStream_t stream);

// out[M, N] = (A · Wᵀ) * g_x[0] * g_w[0] * c[n] + la · upᵀ + bias[n], rounded once. `w` and
// `wscales` are the weight's codes and E4M3 scales in the GEMM's tiles (`svdq_linear.tiles`);
// `up` [N, R] bf16; `c` (f32 [N]) and `bias` may be null.
cudaError_t gemm(const void* act, const void* ascales, const void* w, const void* wscales,
                 const float* gx, const float* gw, const float* c, const void* la, const void* up,
                 const void* bias, void* out, bool out_half, int M, int N, int K, int R,
                 cudaStream_t stream);

}  // namespace cozy_svdq
