// SPDX-License-Identifier: Apache-2.0
// The activation side of int8-convrot256: each 256-group of a row times the regular Hadamard
// H4 x H4 x H4 x H4 (`formats.hadamard256`) / 16, quantized per row to int8. With
// M <= kGemvRows, `gemv` does the same in every block and multiplies against the weight rows
// in the same launch.
//
// Register layout after SGLang's `convrot_int8` (Apache-2.0): a warp holds a 1024-value tile,
// 8 lanes per group and 32 values per lane. In-group index bits 0, 1, 2, 4, 6 are the register
// index and bits 3, 5, 7 the lane, so a lane loads four runs of 8 contiguous values, the
// first base-4 digit is in registers and each other digit has one bit in registers and one in
// the lane: one shuffle per value per digit, and no barrier until the row maximum.
#include "int8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

namespace cozy_int8 {
namespace {

constexpr int kTile = 1024;
constexpr int kLaneValues = 32;
constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float as_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float as_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }

// Group offset of register run j (registers 8j..8j+7) on lane q of its group.
__device__ __forceinline__ int run_offset(int q, int j) {
    return ((q & 1) << 3) | ((j & 1) << 4) | ((q & 2) << 4) | ((j >> 1) << 6) | ((q & 4) << 5);
}

// The regular H4 on a digit held in registers: y_d = S - 2 x_{3-d}.
template <int B0, int B1>
__device__ __forceinline__ void stage_registers(float (&v)[kLaneValues]) {
#pragma unroll
    for (int r = 0; r < kLaneValues; ++r) {
        if ((r & (B0 | B1)) == 0) {
            const float a = v[r], b = v[r | B0], c = v[r | B1], d = v[r | B0 | B1];
            const float s = (a + b) + (c + d);
            v[r] = s - 2.f * d, v[r | B0] = s - 2.f * c, v[r | B1] = s - 2.f * b,
            v[r | B0 | B1] = s - 2.f * a;
        }
    }
}

// The same on a digit whose low bit is register bit RB and high bit lane bit X. Both lanes of a
// pair form S as (own pair) + (partner's pair), the same fp32 sum, so the result does not
// depend on which lane holds a value.
template <int RB, int X>
__device__ __forceinline__ void stage_split(float (&v)[kLaneValues]) {
#pragma unroll
    for (int r = 0; r < kLaneValues; ++r) {
        if ((r & RB) == 0) {
            const float a = v[r], b = v[r | RB];
            const float ap = __shfl_xor_sync(kFull, a, X), bp = __shfl_xor_sync(kFull, b, X);
            const float s = (a + b) + (ap + bp);
            v[r] = s - 2.f * bp;
            v[r | RB] = s - 2.f * ap;
        }
    }
}

// Tile `tile` of `row` rotated and scaled by 1/16, its absolute maximum folded into `amax`.
// A group past K reads zeros and is never stored.
template <typename T>
__device__ __forceinline__ void rotate_tile(const T* row, int tile, int K,
                                            float (&v)[kLaneValues], float& amax) {
    const int lane = threadIdx.x & 31, q = lane & 7;
    const int group = tile * 4 + (lane >> 3);
    const T* base = row + group * 256;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (group * 256 < K) raw = __ldg(reinterpret_cast<const uint4*>(base + run_offset(q, j)));
        const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[8 * j + i] = as_float(e[i]);
    }
    stage_registers<1, 2>(v);
    stage_split<4, 1>(v);
    stage_split<8, 2>(v);
    stage_split<16, 4>(v);
#pragma unroll
    for (int r = 0; r < kLaneValues; ++r) {
        v[r] *= 0.0625f;
        amax = fmaxf(amax, fabsf(v[r]));
    }
}

__device__ __forceinline__ uint2 pack8(const float* v, float inverse) {
    uint32_t w[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        int q[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            q[e] = __float2int_rn(fmaxf(-127.f, fminf(127.f, v[4 * h + e] * inverse)));
        w[h] = __byte_perm(__byte_perm(q[0], q[1], 0x0040), __byte_perm(q[2], q[3], 0x0040),
                           0x5410);
    }
    return make_uint2(w[0], w[1]);
}

// The tile's int8 values into `row_out` (global or shared memory).
__device__ __forceinline__ void store_tile(int8_t* row_out, int tile, int K,
                                           const float (&v)[kLaneValues], float inverse) {
    const int lane = threadIdx.x & 31, q = lane & 7;
    const int group = tile * 4 + (lane >> 3);
    if (group * 256 >= K) return;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<uint2*>(row_out + group * 256 + run_offset(q, j)) =
            pack8(v + 8 * j, inverse);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
    return v;
}

__device__ __forceinline__ float row_scale(float amax) { return fmaxf(amax / 127.f, 1e-12f); }

// Hopper and B200 launch the GEMM as a programmatic dependent of this kernel: it starts its
// prologue as soon as every block here has begun, and waits for this grid before reading.
#if COZY_SM == 90 || COZY_SM == 100
constexpr bool kDependentGemm = true;
#else
constexpr bool kDependentGemm = false;
#endif

// blockDim (32 * tiles, rows): each warp holds one tile of its row in registers across the one
// barrier, where the row's warps exchange their maxima.
template <typename T>
__global__ void __launch_bounds__(1024) rotate_quantize_kernel(
    const T* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scales, int M, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (kDependentGemm) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    __shared__ float partial[32];
    const int tiles = blockDim.x >> 5, tile = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const bool live = row < M;  // a padding row rotates nothing and still meets the barrier
    float v[kLaneValues];
    float amax = 0.f;
    rotate_tile(x + int64_t(live ? row : 0) * K, tile, live ? K : 0, v, amax);
    amax = warp_max(amax);
    if (lane == 0) partial[threadIdx.y * tiles + tile] = amax;
    __syncthreads();
    const float scale = row_scale(warp_max(lane < tiles ? partial[threadIdx.y * tiles + lane] : 0.f));
    if (!live) return;
    if (threadIdx.x == 0) scales[row] = scale;
    store_tile(q + int64_t(row) * K, tile, K, v, 1.f / scale);
}

__device__ __forceinline__ int dot16(const int4& a, const int4& b, int acc) {
    acc = __dp4a(a.x, b.x, acc);
    acc = __dp4a(a.y, b.y, acc);
    acc = __dp4a(a.z, b.z, acc);
    return __dp4a(a.w, b.w, acc);
}

constexpr int kGemvThreads = 256;
constexpr int kGemvWarps = kGemvThreads / 32;

// M rows: every block rotates and quantizes them into shared memory (two passes, the first
// for the row maxima), then each warp takes whole weight rows, 16 bytes a lane.
template <typename T, typename Out, int M>
__global__ void __launch_bounds__(kGemvThreads) gemv_kernel(
    const T* __restrict__ x, const int8_t* __restrict__ w, const float* __restrict__ w_scale,
    const Out* __restrict__ bias, Out* __restrict__ out, int N, int K) {
    extern __shared__ int4 shared[];
    int8_t* xq = reinterpret_cast<int8_t*>(shared);
    __shared__ float partial[M][kGemvWarps];
    __shared__ float xs[M];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int tiles = (K + kTile - 1) / kTile;
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v[kLaneValues];
        float amax = 0.f;
        for (int t = warp; t < tiles; t += kGemvWarps) rotate_tile(x + int64_t(m) * K, t, K, v, amax);
        amax = warp_max(amax);
        if (lane == 0) partial[m][warp] = amax;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float scale = row_scale(warp_max(lane < kGemvWarps ? partial[m][lane] : 0.f));
        if (threadIdx.x == 0) xs[m] = scale;
        for (int t = warp; t < tiles; t += kGemvWarps) {
            float v[kLaneValues];
            float unused = 0.f;
            rotate_tile(x + int64_t(m) * K, t, K, v, unused);
            store_tile(xq + m * K, t, K, v, 1.f / scale);
        }
    }
    __syncthreads();
    for (int n = blockIdx.x * kGemvWarps + warp; n < N; n += gridDim.x * kGemvWarps) {
        const int8_t* wr = w + int64_t(n) * K;
        int acc[M];
#pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = 0;
#pragma unroll 4
        for (int k = lane * 16; k < K; k += 512) {
            const int4 wv = __ldg(reinterpret_cast<const int4*>(wr + k));
#pragma unroll
            for (int m = 0; m < M; ++m)
                acc[m] = dot16(wv, *reinterpret_cast<const int4*>(xq + m * K + k), acc[m]);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(kFull, acc[m], o);
        }
        if (lane == 0) {
            const float ws = w_scale[n];
            const float b = bias != nullptr ? as_float(bias[n]) : 0.f;
#pragma unroll
            for (int m = 0; m < M; ++m)
                out[int64_t(m) * N + n] = from_float<Out>(float(acc[m]) * xs[m] * ws + b);
        }
    }
}

template <typename T>
void rotate_quantize_as(const T* x, int8_t* q, float* scales, int M, int K, cudaStream_t s) {
    const int tiles = (K + kTile - 1) / kTile;
    const int rows = std::max(1, std::min(M, 8 / tiles));
    rotate_quantize_kernel<T>
        <<<(M + rows - 1) / rows, dim3(32 * tiles, rows), 0, s>>>(x, q, scales, M, K);
}

template <typename T, typename Out>
void gemv_as(const T* x, const int8_t* w, const float* ws, const Out* bias, Out* out, int M,
             int N, int K, int sms, cudaStream_t s) {
    const int blocks = std::min((N + kGemvWarps - 1) / kGemvWarps, sms * 4);
    const size_t smem = size_t(M) * K;
    switch (M) {
        case 1: gemv_kernel<T, Out, 1><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
        case 2: gemv_kernel<T, Out, 2><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
        case 3: gemv_kernel<T, Out, 3><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
        default: gemv_kernel<T, Out, 4><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
    }
}

}  // namespace

void rotate_quantize(const void* x, bool x_half, int8_t* q, float* scales, int M, int K,
                     cudaStream_t stream) {
    if (x_half)
        rotate_quantize_as(static_cast<const __half*>(x), q, scales, M, K, stream);
    else
        rotate_quantize_as(static_cast<const __nv_bfloat16*>(x), q, scales, M, K, stream);
}

bool gemv(const void* x, bool x_half, const int8_t* w, const float* w_scale, const void* bias,
          void* out, bool out_half, int M, int N, int K, int sms, cudaStream_t stream) {
    if (M < 1 || M > kGemvRows || size_t(M) * K > 48 * 1024) return false;
    if (x_half && out_half)
        gemv_as(static_cast<const __half*>(x), w, w_scale, static_cast<const __half*>(bias),
                static_cast<__half*>(out), M, N, K, sms, stream);
    else if (!x_half && !out_half)
        gemv_as(static_cast<const __nv_bfloat16*>(x), w, w_scale,
                static_cast<const __nv_bfloat16*>(bias), static_cast<__nv_bfloat16*>(out), M, N,
                K, sms, stream);
    else
        return false;
    return true;
}

}  // namespace cozy_int8
