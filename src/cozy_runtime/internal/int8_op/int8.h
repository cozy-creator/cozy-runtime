// SPDX-License-Identifier: Apache-2.0
// The int8-convrot256 linear's launchers: raw pointers and a stream, no torch, so the CUTLASS
// units compile without torch's headers. `out` and `bias` are bf16 (`half == false`) or fp16.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace cozy_int8 {

// Scratch for a GEMM that needs it (stream-K partials), from the caller's allocator.
struct Workspace {
    void* (*allocate)(size_t bytes, void* context);
    void* context;
};

// The largest M the one-launch GEMV takes: rotate, quantize and multiply in one kernel.
constexpr int kGemvRows = 4;
// The widest K a row's rotated values stay in registers for (512 threads x 4 x 16).
constexpr int kMaxK = 32768;

// Each 256-group of every row of `x` [M, K] times the regular Hadamard / 16, then per-row
// int8: `q` [M, K], `scales` [M]. K % 256 == 0, K <= kMaxK.
void rotate_quantize(const void* x, bool x_half, int8_t* q, float* scales, int M, int K,
                     cudaStream_t stream);

// The whole linear for M <= kGemvRows, in one launch. False when K is too wide for it.
bool gemv(const void* x, bool x_half, const int8_t* w, const float* w_scale, const void* bias,
          void* out, bool out_half, int M, int N, int K, int sms, cudaStream_t stream);

// D[M, N] = (q @ w.T) * scales[m] * w_scale[n] + bias[n] on this build's card; `config` < 0
// picks the tile by shape, else names one (a sweep). False when that kernel cannot run.
bool gemm(const int8_t* q, const int8_t* w, const float* scales, const float* w_scale,
          const void* bias, void* out, bool out_half, int M, int N, int K, int sms, int config,
          Workspace workspace, cudaStream_t stream);

}  // namespace cozy_int8
