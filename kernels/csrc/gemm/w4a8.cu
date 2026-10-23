// SPDX-License-Identifier: Apache-2.0
// The w4a8 linear (`w4a8.convrot`, cozytensors spec §6; design `four-bit-lanes.md` §5):
// 4-bit codes in int8.convrot's rotated basis, unpacked to int8 in registers and multiplied on the
// int8 tensor cores (mma.sync) with int8.convrot's activation. D = out(acc * scales[m] * w_scale[n]
// + bias[n]), the int8 GEMM's epilogue.
//
// Unpack: each CTA builds T[b][q] = clamp(rne(f32(codebook[q] * e4m3(b))), ±127) for every group
// scale byte b and code q, the spec's arithmetic done once; three byte permutes then turn four
// codes into their four int8 levels. The MMA runs transposed (D^T = W · X^T): the unpacked weight
// is its A operand, built in registers in the order the instruction reads it. K is permuted
// inside each 64-deep tile (the sum does not care which MMA slot a k fills, only that A and B
// agree): lane t takes k = 16t + 8s .. + 7 at step s, so its codes for a weight row are one 8-byte
// read and one group (one row of T), and its activation per token one 16-byte read for both steps.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

#include "int8.h"

namespace cozy_int8 {
namespace {

constexpr int kThreads = 128;
constexpr int kBK = 64;             // k per tile: two MMA steps, four 16-groups
constexpr int kTable = 256 * 16;    // T[b][q]
constexpr int kGroupM = 8;          // M tiles a wave walks N with, so both operands stay in L2

struct Args {
    const int8_t* q;
    const uint8_t* w;
    const uint8_t* s;
    const float* codebook;
    const float* xs;
    const float* ws;
    const void* bias;
    void* out;
    int M, N, K;
};

__device__ __forceinline__ uint32_t shared(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int Bytes>
__device__ __forceinline__ void copy(uint32_t to, const void* from, bool valid) {
    if constexpr (Bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(to), "l"(from),
                     "r"(valid ? 16 : 0));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(to), "l"(from),
                     "n"(Bytes), "r"(valid ? Bytes : 0));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mma(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The int8 levels of eight codes (code i in nibble i) from a group's 16 levels `L`, four per word:
// levels 0-7 and 8-15 permuted apart, then each byte taken by its code's top bit.
__device__ __forceinline__ uint2 levels(const uint4& L, uint32_t codes) {
    const uint32_t low = codes & 0x77777777u, pick = ((codes >> 1) & 0x44444444u) | 0x32103210u;
    const uint32_t first = __byte_perm(__byte_perm(L.x, L.y, low), __byte_perm(L.z, L.w, low), pick);
    const uint32_t lo = low >> 16, top = pick >> 16;
    return make_uint2(first, __byte_perm(__byte_perm(L.x, L.y, lo), __byte_perm(L.z, L.w, lo), top));
}

// The 8-bit GEMM's own epilogue arithmetic, so every config of either rounds alike: CUTLASS's sm80
// EVT (mma.sync cards) scales by the token, then fuses the weight scale and the bias (nvcc
// contracts its multiply and add); its TMA epilogue (sm90, sm100) fuses the token scale instead.
__device__ __forceinline__ float epilogue(int acc, float x, float w, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1200
    return __fmaf_rn(x, __fmul_rn(w, float(acc)), b);
#else
    return __fmaf_rn(__fmul_rn(float(acc), x), w, b);
#endif
}

__device__ __forceinline__ float e4m3(uint32_t bits) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(bits), __NV_E4M3);
    return __half2float(__half(h));
}

// T[b][q]: a NaN group scale (the spec refuses its decode) gives zeros.
__device__ __forceinline__ void build(int8_t* table, const float* codebook, int threads) {
    for (int i = threadIdx.x; i < kTable; i += threads) {
        const int q = i & 15;
        const float p = __fmul_rn(codebook ? __ldg(codebook + q) : float(q - 8), e4m3(i >> 4));
        table[i] = static_cast<int8_t>(p != p ? 0 : max(-127, min(127, __float2int_rn(p))));
    }
}

// Shared memory: T, then per stage the activation [BM, 64], the codes [BN, 32] and the group scales
// [BN, 4]. A quarter warp's 16-byte activation reads cover two whole rows: no bank meets twice.
template <int BM, int BN, int WM, int WN, int Stages, typename Out>
__global__ void __launch_bounds__(kThreads, 2) w4a8_kernel(const Args a) {
    static_assert((BM / WM) * (BN / WN) == kThreads / 32, "four warps");
    constexpr int MX = WM / 8, NW = WN / 16;  // token tiles of 8, weight tiles of 16
    constexpr int A_BYTES = BM * kBK, W_BYTES = BN * kBK / 2, STAGE = A_BYTES + W_BYTES + BN * 4;
    static_assert(BM * BN * sizeof(Out) <= kTable + Stages * STAGE, "the staged output fits");
    extern __shared__ __align__(16) uint8_t smem[];
    int8_t* table = reinterpret_cast<int8_t*>(smem);
    uint8_t* stages = smem + kTable;

    const int tiles_m = (a.M + BM - 1) / BM, tiles_n = (a.N + BN - 1) / BN;
    const int band = blockIdx.x / (kGroupM * tiles_n), in_band = blockIdx.x % (kGroupM * tiles_n);
    const int height = min(tiles_m - band * kGroupM, kGroupM);
    const int m0 = (band * kGroupM + in_band % height) * BM, n0 = (in_band / height) * BN;

    // Each thread's copies, fixed for the whole K loop: a k-tile moves every source by its width.
    constexpr int A_COPIES = (BM * 4 + kThreads - 1) / kThreads;
    constexpr int W_COPIES = (BN * 2 + kThreads - 1) / kThreads, S_COPIES = (BN + kThreads - 1) / kThreads;
    const int8_t* a_from[A_COPIES];
    const uint8_t *w_from[W_COPIES], *s_from[S_COPIES];
    uint32_t a_to[A_COPIES], w_to[W_COPIES], s_to[S_COPIES];
    bool a_in[A_COPIES], w_in[W_COPIES], s_in[S_COPIES];
    const uint32_t base = shared(stages);
#pragma unroll
    for (int i = 0; i < A_COPIES; ++i) {
        const int c = threadIdx.x + i * kThreads, r = (c >> 2) % BM, chunk = c & 3, m = m0 + r;
        a_in[i] = c < BM * 4 && m < a.M;
        a_from[i] = a.q + int64_t(a_in[i] ? m : 0) * a.K + chunk * 16;
        a_to[i] = base + r * 64 + chunk * 16;
    }
#pragma unroll
    for (int i = 0; i < W_COPIES; ++i) {
        const int c = threadIdx.x + i * kThreads, r = (c >> 1) % BN, chunk = c & 1, n = n0 + r;
        w_in[i] = c < BN * 2 && n < a.N;
        w_from[i] = a.w + int64_t(w_in[i] ? n : 0) * (a.K / 2) + chunk * 16;
        w_to[i] = base + A_BYTES + r * 32 + chunk * 16;
    }
#pragma unroll
    for (int i = 0; i < S_COPIES; ++i) {
        const int r = (threadIdx.x + i * kThreads) % BN, n = n0 + r;
        s_in[i] = threadIdx.x + i * kThreads < BN && n < a.N;
        s_from[i] = a.s + int64_t(s_in[i] ? n : 0) * (a.K / 16);
        s_to[i] = base + A_BYTES + W_BYTES + r * 4;
    }
    auto load = [&](int kt, int stage) {
        const uint32_t at = stage * STAGE;
#pragma unroll
        for (int i = 0; i < A_COPIES; ++i)
            if (BM * 4 % kThreads == 0 || threadIdx.x + i * kThreads < BM * 4)
                copy<16>(a_to[i] + at, a_from[i] + kt * kBK, a_in[i]);
#pragma unroll
        for (int i = 0; i < W_COPIES; ++i)
            if (BN * 2 % kThreads == 0 || threadIdx.x + i * kThreads < BN * 2)
                copy<16>(w_to[i] + at, w_from[i] + kt * (kBK / 2), w_in[i]);
#pragma unroll
        for (int i = 0; i < S_COPIES; ++i)
            if (BN % kThreads == 0 || threadIdx.x + i * kThreads < BN)
                copy<4>(s_to[i] + at, s_from[i] + kt * (kBK / 16), s_in[i]);
    };

    const int tiles_k = a.K / kBK;
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) {
        if (s < tiles_k) load(s, s);
        commit();
    }
    build(table, a.codebook, kThreads);

    // A lane reads weight rows wn + 16w + g (+8) and token rows wm + 8i + g.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm = (warp / (BN / WN)) * WM, wn = (warp % (BN / WN)) * WN;
    const int x_at = (wm + g) * 64 + t * 16, w_at = A_BYTES + (wn + g) * 32 + t * 8;
    const int s_at = A_BYTES + W_BYTES + (wn + g) * 4, group = 8 * t;
    int acc[NW][MX][4] = {};

    int stage = 0, next = Stages - 1;
    for (int kt = 0; kt < tiles_k; ++kt) {
        wait<Stages - 2>();
        __syncthreads();
        if (kt + Stages - 1 < tiles_k) load(kt + Stages - 1, next);
        commit();
        next = next == Stages - 1 ? 0 : next + 1;
        const uint8_t* tile = stages + stage * STAGE;
        stage = stage == Stages - 1 ? 0 : stage + 1;
        // Rows g and g + 8 of each weight tile: 16 codes each, one group, its 16 levels, both
        // steps' A fragments.
        uint32_t af[2][NW][4];
#pragma unroll
        for (int w = 0; w < NW; ++w)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = w * 16 + h * 8;
                const uint2 codes = *reinterpret_cast<const uint2*>(tile + w_at + row * 32);
                const uint32_t scales = *reinterpret_cast<const uint32_t*>(tile + s_at + row * 4);
                const uint4 lut = *reinterpret_cast<const uint4*>(table + ((scales >> group) & 0xFFu) * 16);
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const uint32_t word = s ? codes.y : codes.x;
                    const uint2 level = levels(lut, word);
                    af[s][w][h] = level.x;
                    af[s][w][2 + h] = level.y;
                }
            }
#pragma unroll
        for (int i = 0; i < MX; ++i) {
            const uint4 v = *reinterpret_cast<const uint4*>(tile + x_at + i * 8 * 64);
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
                for (int w = 0; w < NW; ++w) {
                    mma(acc[w][i], af[s][w], s ? v.z : v.x, s ? v.w : v.y);
                }
        }
    }
    wait<0>();
    __syncthreads();

    // acc[w][i] holds weight rows wn + 16w + g (c0, c1) and + 8 (c2, c3) at tokens wm + 8i + 2t, + 1.
    // Each value is scaled into shared memory as [BM, BN] Out, then rows leave as 16-byte stores.
    Out* staged = reinterpret_cast<Out*>(smem);
    const Out* bias = static_cast<const Out*>(a.bias);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = wn + w * 16 + h * 8 + g, n = n0 + col;
            const float scale = n < a.N ? a.ws[n] : 0.f;
            const float add = bias && n < a.N ? float(bias[n]) : 0.f;
#pragma unroll
            for (int i = 0; i < MX; ++i) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int row = wm + i * 8 + t * 2 + e, m = m0 + row;
                    const float x = m < a.M ? a.xs[m] : 0.f;
                    staged[row * BN + col] = Out(epilogue(acc[w][i][h * 2 + e], x, scale, add));
                }
            }
        }
    }
    __syncthreads();
    constexpr int VEC = 16 / sizeof(Out), ROW_VECS = BN / VEC;
    for (int v = threadIdx.x; v < BM * ROW_VECS; v += kThreads) {
        const int row = v / ROW_VECS, col = (v % ROW_VECS) * VEC, m = m0 + row, n = n0 + col;
        if (m < a.M && n < a.N)  // N % 8 == 0: a vector is whole or absent
            *reinterpret_cast<uint4*>(static_cast<Out*>(a.out) + int64_t(m) * a.N + n) =
                *reinterpret_cast<const uint4*>(staged + row * BN + col);
    }
}

template <int BM, int BN, int WM, int WN, int Stages, typename Out>
cudaError_t run(const Args& a, cudaStream_t stream) {
    constexpr int smem = kTable + Stages * (BM * kBK + BN * kBK / 2 + BN * 4);
    auto* kernel = &w4a8_kernel<BM, BN, WM, WN, Stages, Out>;
    if (smem >= (48 << 10)) {
        const cudaError_t e =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    const int64_t tiles = int64_t((a.M + BM - 1) / BM) * ((a.N + BN - 1) / BN);
    if (tiles > 0x7fffffff) return cudaErrorNotSupported;
    kernel<<<unsigned(tiles), kThreads, smem, stream>>>(a);
    return cudaGetLastError();
}

template <typename Out>
cudaError_t run_as(const Args& a, int config, cudaStream_t stream) {
    if (config < 0) config = a.M <= 16 ? 2 : a.M <= 64 ? 1 : 0;
    switch (config) {
        case 0: return run<128, 128, 128, 32, 3, Out>(a, stream);
        case 1: return run<64, 128, 64, 32, 4, Out>(a, stream);
        case 2: return run<16, 64, 16, 16, 4, Out>(a, stream);
        default: return cudaErrorNotSupported;
    }
}

// One thread per 16-group: its 8 code bytes to 16 int8 levels.
__global__ void unpack_kernel(const uint8_t* w, const uint8_t* s, const float* codebook,
                              int8_t* out, int64_t groups) {
    __shared__ __align__(16) int8_t table[kTable];
    build(table, codebook, blockDim.x);
    __syncthreads();
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < groups;
         i += int64_t(gridDim.x) * blockDim.x) {
        const uint4 L = *reinterpret_cast<const uint4*>(table + s[i] * 16);
        const uint2 codes = *reinterpret_cast<const uint2*>(w + i * 8);
        const uint2 first = levels(L, codes.x), second = levels(L, codes.y);
        *reinterpret_cast<uint4*>(out + i * 16) = make_uint4(first.x, first.y, second.x, second.y);
    }
}

}  // namespace

cudaError_t w4a8_gemm(const int8_t* q, const uint8_t* w, const uint8_t* s, const float* codebook,
                      const float* scales, const float* w_scale, const void* bias, void* out,
                      bool out_half, int M, int N, int K, int config, cudaStream_t stream) {
    const Args a{q, w, s, codebook, scales, w_scale, bias, out, M, N, K};
    return out_half ? run_as<__half>(a, config, stream) : run_as<__nv_bfloat16>(a, config, stream);
}

cudaError_t w4a8_unpack(const uint8_t* w, const uint8_t* s, const float* codebook, int8_t* out,
                        int N, int K, int sms, cudaStream_t stream) {
    const int64_t groups = int64_t(N) * (K / 16);
    const int64_t blocks = std::min<int64_t>((groups + 255) / 256, int64_t(sms) * 8);
    unpack_kernel<<<unsigned(blocks), 256, 0, stream>>>(w, s, codebook, out, groups);
    return cudaGetLastError();
}

}  // namespace cozy_int8
