// SPDX-License-Identifier: Apache-2.0
// The activation side of the 8-bit linears, each row in one pass, optionally after its
// producer's norm, modulation and activation (`Prologue`): int8-convrot256 multiplies each
// 256-group by the regular Hadamard H4 x H4 x H4 x H4 (`formats.hadamard256`) / 16 and
// quantizes the row to int8; fp8-rowwise quantizes it to E4M3. With M <= kGemvRows, `gemv`
// does the int8 pass in every block and multiplies against the weight rows in the same launch.
//
// Register layout after SGLang's `convrot_int8` (Apache-2.0): a warp holds a 1024-value tile,
// 8 lanes per group and 32 values per lane. In-group index bits 0, 1, 2, 4, 6 are the register
// index and bits 3, 5, 7 the lane, so a lane loads four runs of 8 contiguous values, the
// first base-4 digit is in registers and each other digit has one bit in registers and one in
// the lane: one shuffle per value per digit, and no barrier until the row maximum.
#include "int8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

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

// Where this lane's run j of tile `tile` starts in its row; the run is in the row when that is
// under K (K % 8 == 0).
__device__ __forceinline__ int run_start(int tile, int j) {
    const int lane = threadIdx.x & 31;
    return (tile * 4 + (lane >> 3)) * 256 + run_offset(lane & 7, j);
}

// Tile `tile` of `row` as f32 in the register layout; a run past K reads zeros.
template <typename T>
__device__ __forceinline__ void load_tile(const T* row, int tile, int K, float (&v)[kLaneValues]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint4 raw = make_uint4(0, 0, 0, 0);
        const int at = run_start(tile, j);
        if (at < K) raw = __ldg(reinterpret_cast<const uint4*>(row + at));
        const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[8 * j + i] = as_float(e[i]);
    }
}

// The registers rotated and scaled by 1/16, their absolute maximum folded into `amax`.
__device__ __forceinline__ void rotate_values(float (&v)[kLaneValues], float& amax) {
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

template <typename T>
__device__ __forceinline__ void rotate_tile(const T* row, int tile, int K,
                                            float (&v)[kLaneValues], float& amax) {
    load_tile(row, tile, K, v);
    rotate_values(v, amax);
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
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int at = run_start(tile, j);
        if (at < K) *reinterpret_cast<uint2*>(row_out + at) = pack8(v + 8 * j, inverse);
    }
}

// `v / scale` by its reciprocal `r` and one correction step: the rounded quotient, without the
// IEEE divide's cost per element (which bounded the pass at under half of B300's bandwidth).
__device__ __forceinline__ float quotient(float v, float scale, float r) {
    const float q = v * r;
    return fmaf(fmaf(-q, scale, v), r, q);
}

// Two values as E4M3 bytes, `v / scale` rounded to nearest even and saturated: the eager leaf's
// divide, clamp and cast.
__device__ __forceinline__ uint32_t fp8x2(float a, float b, float scale, float r) {
    return __nv_cvt_float2_to_fp8x2(make_float2(quotient(a, scale, r), quotient(b, scale, r)),
                                    __NV_SATFINITE, __NV_E4M3);
}

// The tile's E4M3 values into `row_out`.
__device__ __forceinline__ void store_fp8(uint8_t* row_out, int tile, int K,
                                          const float (&v)[kLaneValues], float scale) {
    const float r = 1.f / scale;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int at = run_start(tile, j);
        if (at >= K) continue;
        const float* e = v + 8 * j;
        uint32_t w[2];
#pragma unroll
        for (int h = 0; h < 2; ++h)
            w[h] = fp8x2(e[4 * h], e[4 * h + 1], scale, r) |
                   fp8x2(e[4 * h + 2], e[4 * h + 3], scale, r) << 16;
        *reinterpret_cast<uint2*>(row_out + at) = make_uint2(w[0], w[1]);
    }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
    return v;
}

__device__ __forceinline__ float row_scale(float amax) { return fmaxf(amax / 127.f, 1e-12f); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
    return v;
}

// The sum over the row of each lane's `v`, through `slots` (one per warp of the block).
__device__ __forceinline__ float row_sum(float v, float* slots, int tiles) {
    const int tile = threadIdx.x >> 5, lane = threadIdx.x & 31;
    v = warp_sum(v);
    if (lane == 0) slots[threadIdx.y * tiles + tile] = v;
    __syncthreads();
    return warp_sum(lane < tiles ? slots[threadIdx.y * tiles + lane] : 0.f);
}

// `activation(norm(x) * (1 + scale) + shift)` on the registers, before the rotation.
template <typename T>
__device__ __forceinline__ void apply_prologue(const Prologue& p, int row, int tile, int K,
                                               float (&v)[kLaneValues], float* slots) {
    const int tiles = blockDim.x >> 5;
    bool valid[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) valid[j] = run_start(tile, j) < K;
    if (p.norm != kNormNone) {
        float mean = 0.f;
        if (p.norm == kNormLayer) {
            float sum = 0.f;
#pragma unroll
            for (int r = 0; r < kLaneValues; ++r) sum += v[r];
            mean = row_sum(sum, slots, tiles) / float(K);
        }
        float squares = 0.f;
#pragma unroll
        for (int r = 0; r < kLaneValues; ++r) {
            const float d = valid[r >> 3] ? v[r] - mean : 0.f;
            squares += d * d;
        }
        const float rstd = rsqrtf(row_sum(squares, slots + 32, tiles) / float(K) + p.eps);
#pragma unroll
        for (int r = 0; r < kLaneValues; ++r) v[r] = valid[r >> 3] ? (v[r] - mean) * rstd : 0.f;
    }
    if (p.scale != nullptr || p.shift != nullptr) {
        const int64_t entry = row / p.rows_per_entry;
        float m[kLaneValues];
        if (p.scale != nullptr) {
            load_tile(static_cast<const T*>(p.scale) + entry * p.scale_stride, tile, K, m);
#pragma unroll
            for (int r = 0; r < kLaneValues; ++r) v[r] *= 1.f + m[r];
        }
        if (p.shift != nullptr) {
            load_tile(static_cast<const T*>(p.shift) + entry * p.shift_stride, tile, K, m);
#pragma unroll
            for (int r = 0; r < kLaneValues; ++r) v[r] += m[r];
        }
    }
    if (p.activation == kActGelu) {
#pragma unroll
        for (int r = 0; r < kLaneValues; ++r) v[r] = 0.5f * v[r] * (1.f + erff(v[r] * 0.70710678f));
    } else if (p.activation == kActGeluTanh) {
#pragma unroll
        for (int r = 0; r < kLaneValues; ++r) {
            const float x = v[r];
            v[r] = 0.5f * x * (1.f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
        }
    }
}

// The gated SiTU of a row's [hidden | gate] halves: `v` holds the hidden tile, `gate_row` the gate.
template <typename T>
__device__ __forceinline__ void apply_situ(const T* gate_row, int tile, int K,
                                           float (&v)[kLaneValues]) {
    float g[kLaneValues];
    load_tile(gate_row, tile, K, g);
#pragma unroll
    for (int r = 0; r < kLaneValues; ++r)
        v[r] = (4.f * tanhf(g[r] * 0.25f) / (1.f + expf(-g[r]))) * (25.f * tanhf(v[r] / 25.f));
}

// `silu(gate) * up` on the gate tile in the registers, rounded as the eager SwiGLU rounds it:
// torch's silu `g / (1 + exp(-g))` (IEEE divide), then the product, each to the row dtype.
template <typename T>
__device__ __forceinline__ void apply_swiglu(const T* up_row, int tile, int K,
                                             float (&v)[kLaneValues]) {
    float u[kLaneValues];
    load_tile(up_row, tile, K, u);
#pragma unroll
    for (int r = 0; r < kLaneValues; ++r) {
        const float silu = as_float(from_float<T>(__fdiv_rn(v[r], 1.f + expf(-v[r]))));
        v[r] = as_float(from_float<T>(silu * u[r]));
    }
}

// The TMA GEMMs (wgmma, tcgen05, block-scaled mma.sync's fp8) launch as programmatic dependents
// of this kernel: they start their prologue as soon as every block here has begun, and wait
// for this grid before reading. Before any other kernel the signal does nothing.

// blockDim (32 * tiles, rows): each warp holds one tile of its row in registers across the
// barriers, where the row's warps exchange their sums (a prologue's norm) and maxima.
template <typename T, bool kPrologue, bool kFp8>
__global__ void __launch_bounds__(1024) quantize_kernel(
    const T* __restrict__ x, void* __restrict__ q, float* __restrict__ scales, int M, int K,
    const Prologue prologue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    __shared__ float partial[kPrologue ? 96 : 32];
    const int tiles = blockDim.x >> 5, tile = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const bool live = row < M;  // a padding row rotates nothing and still meets the barriers
    float v[kLaneValues];
    float amax = 0.f;
    // A gated row is twice the quantized width: its hidden half, then its gate half.
    const bool gated = kPrologue && prologue.activation == kActSitu;
    const T* in = x + int64_t(live ? row : 0) * (gated ? 2 * int64_t(K) : K);
    load_tile(in, tile, live ? K : 0, v);
    if constexpr (kPrologue) {
        if (gated)
            apply_situ(in + K, tile, live ? K : 0, v);
        else if (prologue.activation == kActSwiglu)
            apply_swiglu(static_cast<const T*>(prologue.up) + int64_t(live ? row : 0) * K, tile,
                         live ? K : 0, v);
        else
            apply_prologue<T>(prologue, live ? row : 0, tile, K, v, partial + 32);
    }
    if constexpr (kFp8) {
#pragma unroll
        for (int r = 0; r < kLaneValues; ++r) amax = fmaxf(amax, fabsf(v[r]));
    } else {
        rotate_values(v, amax);
    }
    amax = warp_max(amax);
    if (lane == 0) partial[threadIdx.y * tiles + tile] = amax;
    __syncthreads();
    amax = warp_max(lane < tiles ? partial[threadIdx.y * tiles + lane] : 0.f);
    // fp8: eager's `amax / 448` as torch runs it on CUDA, a multiply by the f32 reciprocal.
    const float scale = kFp8 ? fmaxf(amax * (1.f / 448.f), 1e-12f) : row_scale(amax);
    if (!live) return;
    if (threadIdx.x == 0) scales[row] = scale;
    if constexpr (kFp8)
        store_fp8(static_cast<uint8_t*>(q) + int64_t(row) * K, tile, K, v, scale);
    else
        store_tile(static_cast<int8_t*>(q) + int64_t(row) * K, tile, K, v, 1.f / scale);
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

template <typename T, bool kFp8>
cudaError_t quantize_as(const void* x, void* q, float* scales, int M, int K,
                        const Prologue& prologue, cudaStream_t s) {
    const int tiles = (K + kTile - 1) / kTile;
    const int rows = std::max(1, std::min(M, 8 / tiles));
    const dim3 grid((M + rows - 1) / rows), block(32 * tiles, rows);
    const bool any = prologue.norm != kNormNone || prologue.scale != nullptr ||
                     prologue.shift != nullptr || prologue.activation != kActNone;
    const T* rows_in = static_cast<const T*>(x);
    if (any)
        quantize_kernel<T, true, kFp8><<<grid, block, 0, s>>>(rows_in, q, scales, M, K, prologue);
    else
        quantize_kernel<T, false, kFp8><<<grid, block, 0, s>>>(rows_in, q, scales, M, K, prologue);
    return cudaGetLastError();
}

template <typename T, typename Out>
cudaError_t gemv_as(const T* x, const int8_t* w, const float* ws, const Out* bias, Out* out, int M,
             int N, int K, int sms, cudaStream_t s) {
    const int blocks = std::min((N + kGemvWarps - 1) / kGemvWarps, sms * 4);
    const size_t smem = size_t(M) * K;
    switch (M) {
        case 1: gemv_kernel<T, Out, 1><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
        case 2: gemv_kernel<T, Out, 2><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
        case 3: gemv_kernel<T, Out, 3><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
        default: gemv_kernel<T, Out, 4><<<blocks, kGemvThreads, smem, s>>>(x, w, ws, bias, out, N, K); break;
    }
    return cudaGetLastError();
}

}  // namespace

cudaError_t rotate_quantize(const void* x, bool x_half, int8_t* q, float* scales, int M, int K,
                            const Prologue& prologue, cudaStream_t stream) {
    return (x_half ? quantize_as<__half, false> : quantize_as<__nv_bfloat16, false>)(
        x, q, scales, M, K, prologue, stream);
}

cudaError_t fp8_quantize(const void* x, bool x_half, void* q, float* scales, int M, int K,
                         const Prologue& prologue, cudaStream_t stream) {
    return (x_half ? quantize_as<__half, true> : quantize_as<__nv_bfloat16, true>)(
        x, q, scales, M, K, prologue, stream);
}

cudaError_t gemv(const void* x, bool x_half, const int8_t* w, const float* w_scale,
                 const void* bias, void* out, bool out_half, int M, int N, int K, int sms,
                 cudaStream_t stream) {
    // Its rows sit in shared memory beside the per-warp partial sums: within the 48 KiB a
    // launch has without opting in, or the rotation and GEMM serve the call.
    const size_t fixed = size_t(M) * (kGemvWarps + 1) * sizeof(float);
    if (M < 1 || M > kGemvRows || size_t(M) * K + fixed > 48 * 1024 || x_half != out_half)
        return cudaErrorNotSupported;
    if (x_half)
        return gemv_as(static_cast<const __half*>(x), w, w_scale,
                       static_cast<const __half*>(bias), static_cast<__half*>(out), M, N, K, sms,
                       stream);
    return gemv_as(static_cast<const __nv_bfloat16*>(x), w, w_scale,
                   static_cast<const __nv_bfloat16*>(bias), static_cast<__nv_bfloat16*>(out), M,
                   N, K, sms, stream);
}

}  // namespace cozy_int8
