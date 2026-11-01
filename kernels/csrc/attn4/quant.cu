// SPDX-License-Identifier: Apache-2.0
// INT4 Q and K for comfy-kitchen's attention kernel at `DataType::kInt4` (`fwd.cu`), head dimension
// 128, a lane holding four channels. Each row is turned by one fixed random-sign Hadamard (Q and K
// alike, so Q·Kᵀ is unchanged) after K loses its mean over the sequence (softmax-invariant), then
// takes symmetric int4 [-7, 7] at the kernel's per-thread scales: one per Q rows
// {32g + r + 8j : j < 4} and per K keys {8j + 2t + p : j < CTA_K / 8, p < 2} of each key block,
// the rows each thread's MMA accumulators hold. Two codes a byte, the even channel low, as s4
// mma.sync reads them.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "attn4.h"

namespace cozy::attn4 {
namespace {

constexpr int D = 128;
constexpr int SUM_ROWS = 1024;

__device__ __forceinline__ float value(__half x) { return __half2float(x); }
__device__ __forceinline__ float value(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ __forceinline__ void load4(const T *p, float *v) {
  const uint2 raw = __ldg(reinterpret_cast<const uint2 *>(p));
  const T *e = reinterpret_cast<const T *>(&raw);
#pragma unroll
  for (int c = 0; c < 4; ++c) v[c] = value(e[c]);
}

// v <- H128 S v / sqrt(128) over the warp's 128 channels: H4 within the lane, H32 across lanes.
__device__ __forceinline__ void rotate(float *v, int lane) {
  const uint32_t signs = lane < 8 ? 0x9e3779b9u : lane < 16 ? 0x7f4a7c15u
                       : lane < 24 ? 0xf39cc060u : 0x5ced9e6du;
#pragma unroll
  for (int c = 0; c < 4; ++c)
    v[c] = __uint_as_float(__float_as_uint(v[c]) ^ (((signs >> ((lane & 7) * 4 + c)) & 1u) << 31));
  const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2, v[1] = a1 + a3, v[2] = a0 - a2, v[3] = a1 - a3;
#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(0xffffffffu, v[c], bit);
      v[c] = (lane & bit) ? other - v[c] : v[c] + other;
    }
  }
#pragma unroll
  for (int c = 0; c < 4; ++c) v[c] *= 0.08838834764831845f;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}

__device__ __forceinline__ uint32_t code(float x, float inv) {
  return static_cast<uint32_t>(max(-7, min(7, __float2int_rn(x * inv)))) & 0xFu;
}

__device__ __forceinline__ uint16_t pack(const float *v, float inv) {
  return static_cast<uint16_t>(code(v[0], inv) | code(v[1], inv) << 4 | code(v[2], inv) << 8 |
                               code(v[3], inv) << 12);
}

// K's per-channel sum over its rows, a 1024-row chunk a block.
template <typename T>
__global__ __launch_bounds__(128) void k_sum(const T *k, float *sum, int L, int64_t sb,
                                             int64_t sh, int64_t sn) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const T *in = k + blockIdx.z * sb + blockIdx.y * sh + lane * 4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int end = min(L, static_cast<int>(blockIdx.x + 1) * SUM_ROWS);
  for (int n = blockIdx.x * SUM_ROWS + warp; n < end; n += 4) {
    float v[4];
    load4(in + n * sn, v);
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[c] += v[c];
  }
  __shared__ float part[4][D];
#pragma unroll
  for (int c = 0; c < 4; ++c) part[warp][lane * 4 + c] = acc[c];
  __syncthreads();
  const int c = threadIdx.x;
  atomicAdd(&sum[(static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y) * D + c],
            part[0][c] + part[1][c] + part[2][c] + part[3][c]);
}

// One 32-row group of Q a block; warp w quantizes its rows r = 2w, 2w + 1.
template <typename T>
__global__ __launch_bounds__(128) void quant_q(const T *q, uint8_t *out, float *scales, int L,
                                               int64_t sb, int64_t sh, int64_t sn) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int64_t bh = static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y;
  const T *in = q + blockIdx.z * sb + blockIdx.y * sh + lane * 4;
  uint8_t *dst = out + bh * L * (D / 2) + lane * 2;
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int r = warp * 2 + g, first = blockIdx.x * 32 + r;
    float v[4][4], mx = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = first + 8 * j;
      if (n < L) {
        load4(in + n * sn, v[j]);
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) v[j][c] = 0.f;
      }
      rotate(v[j], lane);
#pragma unroll
      for (int c = 0; c < 4; ++c) mx = fmaxf(mx, fabsf(v[j][c]));
    }
    const float s = warp_max(mx) / 7.f + 1e-7f;
    if (lane == 0) scales[bh * (gridDim.x * 8) + blockIdx.x * 8 + r] = s;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int n = first + 8 * j;
      if (n < L) *reinterpret_cast<uint16_t *>(dst + static_cast<int64_t>(n) * (D / 2)) = pack(v[j], 1.f / s);
    }
  }
}

// One CTA_K-key block of K a block; warp t quantizes its keys {8j + 2t + p}.
template <typename T, int CTA_K>
__global__ __launch_bounds__(128) void quant_k(const T *k, const float *sum, uint8_t *out,
                                               float *scales, int L, int64_t sb, int64_t sh,
                                               int64_t sn) {
  constexpr int J = CTA_K / 8;
  const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
  const int64_t bh = static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y;
  const T *in = k + blockIdx.z * sb + blockIdx.y * sh + lane * 4;
  uint8_t *dst = out + bh * L * (D / 2) + lane * 2;
  float mean[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) mean[c] = sum[bh * D + lane * 4 + c] / L;
  float v[J][2][4], mx = 0.f;
#pragma unroll
  for (int j = 0; j < J; ++j) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int n = blockIdx.x * CTA_K + 8 * j + 2 * t + p;
      if (n < L) {
        load4(in + n * sn, v[j][p]);
#pragma unroll
        for (int c = 0; c < 4; ++c) v[j][p][c] -= mean[c];
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) v[j][p][c] = 0.f;
      }
      rotate(v[j][p], lane);
#pragma unroll
      for (int c = 0; c < 4; ++c) mx = fmaxf(mx, fabsf(v[j][p][c]));
    }
  }
  const float s = warp_max(mx) / 7.f + 1e-7f;
  if (lane == 0) scales[bh * (gridDim.x * 4) + blockIdx.x * 4 + t] = s;
#pragma unroll
  for (int j = 0; j < J; ++j) {
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int n = blockIdx.x * CTA_K + 8 * j + 2 * t + p;
      if (n < L) *reinterpret_cast<uint16_t *>(dst + static_cast<int64_t>(n) * (D / 2)) = pack(v[j][p], 1.f / s);
    }
  }
}

template <typename T>
void launch(const Call &call, cudaStream_t stream) {
  const auto *q = static_cast<const T *>(call.q);
  const auto *k = static_cast<const T *>(call.k);
  const auto &qs = call.q_strides, &ks = call.k_strides;
  const int blocks_k = (call.len_k + call.cta_k - 1) / call.cta_k;
  cudaMemsetAsync(call.k_sum, 0, sizeof(float) * call.batch * call.heads_k * D, stream);
  k_sum<T><<<dim3((call.len_k + SUM_ROWS - 1) / SUM_ROWS, call.heads_k, call.batch), 128, 0,
             stream>>>(k, call.k_sum, call.len_k, ks[0], ks[1], ks[2]);
  quant_q<T><<<dim3((call.len_q + 127) / 128 * 4, call.heads, call.batch), 128, 0, stream>>>(
      q, static_cast<uint8_t *>(call.q4), call.q_scale, call.len_q, qs[0], qs[1], qs[2]);
  const dim3 grid(blocks_k, call.heads_k, call.batch);
  auto *k4 = static_cast<uint8_t *>(call.k4);
  if (call.cta_k == 128)
    quant_k<T, 128><<<grid, 128, 0, stream>>>(k, call.k_sum, k4, call.k_scale, call.len_k,
                                               ks[0], ks[1], ks[2]);
  else
    quant_k<T, 64><<<grid, 128, 0, stream>>>(k, call.k_sum, k4, call.k_scale, call.len_k,
                                              ks[0], ks[1], ks[2]);
}

}  // namespace

void quantize_qk(const Call &call, cudaStream_t stream) {
  if (call.bf16)
    launch<__nv_bfloat16>(call, stream);
  else
    launch<__half>(call, stream);
}

}  // namespace cozy::attn4
