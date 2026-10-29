// SPDX-License-Identifier: Apache-2.0
// The 8-bit linears' launchers (int8-convrot256, fp8-rowwise): raw pointers and a stream, no
// torch. `out` and `bias` are bf16 (`half == false`) or fp16. Each returns its launch's error: a
// module loaded lazily into a full GPU fails with cudaErrorMemoryAllocation, which the caller
// raises as an out-of-memory error Runtime recovers.
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
enum { kActNone = 0, kActGelu = 1, kActGeluTanh = 2, kActSitu = 3 };

// What the producer of `x` would have done to it, done here instead: `activation(norm(x) *
// (1 + scale) + shift)`. The norm has no affine; `scale` and `shift` are tables in x's dtype
// whose rows are `*_stride` elements apart (16-byte aligned) and row r of x reads table row
// r / rows_per_entry. Each part is optional. kActSitu is gated and stands alone: `x` is
// [M, 2K], each row its [hidden | gate] halves, and the K-wide row quantized is
// `4 tanh(g / 4) sigmoid(g) * 25 tanh(h / 25)` (MOSS-Image's SiTU).
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

// D[M, N] = (q @ w.T) * scales[m] * w_scale[n] + bias[n]; `config` < 0 picks the tile by shape,
// else names one (a sweep); cudaErrorNotSupported when that tile cannot take the problem.
using Gemm = cudaError_t(const int8_t* q, const int8_t* w, const float* scales,
                         const float* w_scale, const void* bias, void* out, bool out_half, int M,
                         int N, int K, int sms, int config, Workspace workspace,
                         cudaStream_t stream);

// The same epilogue on E4M3 operands, f32 accumulation: wgmma on Hopper, tcgen05 on datacenter
// Blackwell, RTX 50's TMA mma.
using Fp8Gemm = cudaError_t(const void* q, const void* w, const float* scales,
                            const float* w_scale, const void* bias, void* out, bool out_half,
                            int M, int N, int K, int sms, int config, cudaStream_t stream);

// The w4a8 linear's GEMM (`w4a8.convrot`, group 16, rotation 256): `w` [N, K/2] 4-bit
// codes (even k in the low nibble), `s` [N, K/16] E4M3 group scales, `codebook` 16 floats or null
// (`q - 8`); `q`, `scales`, `w_scale`, `bias` and `out` as `Gemm`'s. mma.sync on every target from
// sm80 (`w4a8.cu`); `config` < 0 picks the tile by M.
cudaError_t w4a8_gemm(const int8_t* q, const uint8_t* w, const uint8_t* s, const float* codebook,
                      const float* scales, const float* w_scale, const void* bias, void* out,
                      bool out_half, int M, int N, int K, int config, cudaStream_t stream);

// The int8 levels the codes stand for, `out` [N, K]: the int8.convrot weight they decode to before
// its row scale, for a family GEMM to multiply.
cudaError_t w4a8_unpack(const uint8_t* w, const uint8_t* s, const float* codebook, int8_t* out,
                        int N, int K, int sms, cudaStream_t stream);

// Each tensor-core family's GEMMs, compiled for that family's targets only and kept apart by
// namespace, so one module links every family (`CMakeLists.txt`). A unit names its family by its
// define (`COZY_FAMILY`); `gemm_sm80.cu` is always mma.sync's int8.
namespace mma_sync {
Gemm gemm;
Fp8Gemm fp8_gemm;
}  // namespace mma_sync
namespace wgmma {
Gemm gemm;
Fp8Gemm fp8_gemm;
}  // namespace wgmma
namespace tcgen05 {
Gemm gemm;
Fp8Gemm fp8_gemm;
}  // namespace tcgen05

}  // namespace cozy_int8

#if defined(COZY_TCGEN05)
#define COZY_FAMILY tcgen05
#elif defined(COZY_WGMMA)
#define COZY_FAMILY wgmma
#else
#define COZY_FAMILY mma_sync
#endif
