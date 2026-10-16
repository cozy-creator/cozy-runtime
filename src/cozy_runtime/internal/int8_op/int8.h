// SPDX-License-Identifier: Apache-2.0
// The 8-bit linears' launchers (int8-convrot256, fp8-rowwise): raw pointers and a stream, no
// torch, so the CUTLASS units compile without torch's headers. `out` and `bias` are bf16
// (`half == false`) or fp16.
// Each returns its launch's error: a module loaded lazily into a full GPU fails with
// cudaErrorMemoryAllocation, which the op raises as an out-of-memory error Runtime recovers.
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

enum { kNormNone = 0, kNormLayer = 1, kNormRms = 2 };
enum { kActNone = 0, kActGelu = 1, kActGeluTanh = 2 };

// What the producer of `x` would have done to it, done here instead: `activation(norm(x) *
// (1 + scale) + shift)`. The norm has no affine; `scale` and `shift` are tables in x's dtype
// whose rows are `*_stride` elements apart (16-byte aligned) and row r of x reads table row
// r / rows_per_entry. Each part is optional.
struct Prologue {
    const void* scale = nullptr;
    const void* shift = nullptr;
    int64_t scale_stride = 0, shift_stride = 0;
    int rows_per_entry = 1;
    int norm = kNormNone;
    float eps = 0.f;
    int activation = kActNone;
};

// Each 256-group of every row of `x` [M, K] (after `prologue`) times the regular Hadamard / 16,
// then per-row int8: `q` [M, K], `scales` [M]. K % 256 == 0, K <= kMaxK.
cudaError_t rotate_quantize(const void* x, bool x_half, int8_t* q, float* scales, int M, int K,
                            const Prologue& prologue, cudaStream_t stream);

// Each row of `x` [M, K] (after `prologue`) per-row E4M3, the fp8 leaf's law: `scales` [M] =
// max(amax * (1 / 448), 1e-12), `q` [M, K] = x / scale rounded to nearest even, saturated.
// K % 16 == 0, K <= kMaxK.
cudaError_t fp8_quantize(const void* x, bool x_half, void* q, float* scales, int M, int K,
                         const Prologue& prologue, cudaStream_t stream);

// The whole linear for M <= kGemvRows, in one launch; cudaErrorNotSupported when K is too
// wide for it.
cudaError_t gemv(const void* x, bool x_half, const int8_t* w, const float* w_scale, const void* bias,
          void* out, bool out_half, int M, int N, int K, int sms, cudaStream_t stream);

// D[M, N] = (q @ w.T) * scales[m] * w_scale[n] + bias[n] on this build's card; `config` < 0
// picks the tile by shape, else names one (a sweep); cudaErrorNotSupported when that tile
// cannot take the problem.
cudaError_t gemm(const int8_t* q, const int8_t* w, const float* scales, const float* w_scale,
          const void* bias, void* out, bool out_half, int M, int N, int K, int sms, int config,
          Workspace workspace, cudaStream_t stream);

// The same epilogue on E4M3 operands, f32 accumulation: wgmma on Hopper, tcgen05 on datacenter
// Blackwell, RTX 50's TMA mma. Only builds that define COZY_FP8_GEMM carry it.
cudaError_t fp8_gemm(const void* q, const void* w, const float* scales, const float* w_scale,
                     const void* bias, void* out, bool out_half, int M, int N, int K, int sms,
                     int config, cudaStream_t stream);

}  // namespace cozy_int8
