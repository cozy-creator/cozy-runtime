// SPDX-License-Identifier: Apache-2.0
// svdq.nvfp4's linear (SVDQuant, xs-066) on RTX 50's block-scaled FP4 MMA (mma.sync
// kind::mxf4nvf4, sm_120a/sm_121a): out = round((A · Wᵀ) * g_x * g_w * c[n] + la · upᵀ + bias[n]).
//
// `quantize` reads x once. It takes bf16(x * p) to nvfp4 at the activation's second level g_x,
// bit for bit the leaf's law (`svdq_linear.quantize`), and stores the codes and E4M3 scales in the
// order the GEMM's MMA reads them. Its 16-bit MMAs make la = x · downᵀ from the same registers:
// the sum does not care which MMA slot a k fills, only that both operands agree, so lane t takes
// k = 8t + 4s .. + 3 at step s. Eight warps share a 32-row slice's k-steps (a few row slices
// still fill the card); their f32 partials add in a fixed order, so la is the same every run.
//
// `gemm` multiplies 256x128 tiles with 8 warps of 32 rows by 128 columns. Each warp loads its
// operands from global memory straight into registers, already in fragment order (no shared
// memory, no ldmatrix), one k-step ahead of the MMAs. The epilogue scales the f32 accumulators,
// adds the low-rank term with 16-bit MMAs into those same accumulators (the C layouts match),
// adds the bias and stores once.
//
// Adapted from Nunchaku's W4A4 GEMM (https://github.com/nunchux-ai/nunchaku, Apache-2.0,
// src/kernels/zgemm/gemm_w4a4.cuh and gemm_base.cuh at 302e0e9): the tile shape, the register
// pipeline, the operand and scale fragment layouts and the block-scaled MMA call. Changed here:
// the activation takes nvfp4's second-level scale (Nunchaku's has one level); the accumulators
// stay f32 through the epilogue (Nunchaku rounds them to 16 bits before its low-rank term); the
// low-rank factors and la are read in their natural layouts; the down-projection is fused into
// this quantize; no torch.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "svdq.h"

namespace cozy_svdq {
namespace {

constexpr int kWarps = 8;       // GEMM warps: kRows / kSlice
constexpr int kSlice = 32;      // rows per GEMM warp, and per quantize block
constexpr int kQuantWarps = 8;  // quantize warps sharing a slice's k-steps
constexpr float kScaleFloor = 0.001953125f;  // E4M3's smallest subnormal, 2^-9

template <bool Half>
struct Types;
template <>
struct Types<false> {
    using T = __nv_bfloat16;
};
template <>
struct Types<true> {
    using T = __half;
};

template <bool Half>
__device__ __forceinline__ float2 to_float2(uint32_t v) {
    if constexpr (Half) {
        return __half22float2(*reinterpret_cast<const __half2*>(&v));
    } else {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    }
}

template <bool Half>
__device__ __forceinline__ uint32_t from_float2(float a, float b) {
    if constexpr (Half) {
        const __half2 h = __floats2half2_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&h);
    } else {
        const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&h);
    }
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint4 load16(const void* p, bool ok) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (ok) v = __ldg(static_cast<const uint4*>(p));
    return v;
}

// D += A · B on the 16-bit tensor cores, f32 accumulation (m16n8k16).
template <bool Half>
__device__ __forceinline__ void mma16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                      uint32_t a3, uint32_t b0, uint32_t b1) {
    if constexpr (Half) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}

#if defined(COZY_SVDQ_EMULATE)
// The block-scaled MMA in f32 arithmetic for a card without it: the same fragments, gathered by
// shuffles. A test build's only (`COZY_SVDQ_EMULATE`): it checks the layouts on any GPU.
__device__ __forceinline__ float e2m1(uint32_t nibble) {
    const uint32_t m = nibble & 7u;
    const float v = m < 4u ? 0.5f * float(m) : m == 4u ? 2.f : m == 5u ? 3.f : m == 6u ? 4.f : 6.f;
    return (nibble & 8u) ? -v : v;
}

__device__ __forceinline__ float e4m3(uint32_t scales, int block) {
    const auto byte = static_cast<__nv_fp8_storage_t>((scales >> (8 * block)) & 0xffu);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(byte, __NV_E4M3)));
}

__device__ void emulated(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1, uint32_t sa,
                         uint32_t sb, int ida, int idb) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint32_t s_row = __shfl_sync(~0u, sa, 4 * g + 2 * ida);
    const uint32_t s_row8 = __shfl_sync(~0u, sa, 4 * g + 2 * ida + 1);
    const uint32_t s_col0 = __shfl_sync(~0u, sb, 8 * t + idb);
    const uint32_t s_col1 = __shfl_sync(~0u, sb, 8 * t + 4 + idb);
    for (int tt = 0; tt < 4; ++tt) {
        const uint32_t ar[2] = {__shfl_sync(~0u, a.x, 4 * g + tt), __shfl_sync(~0u, a.z, 4 * g + tt)};
        const uint32_t ar8[2] = {__shfl_sync(~0u, a.y, 4 * g + tt),
                                 __shfl_sync(~0u, a.w, 4 * g + tt)};
        const uint32_t bc0[2] = {__shfl_sync(~0u, b0, 8 * t + tt), __shfl_sync(~0u, b1, 8 * t + tt)};
        const uint32_t bc1[2] = {__shfl_sync(~0u, b0, 8 * t + 4 + tt),
                                 __shfl_sync(~0u, b1, 8 * t + 4 + tt)};
        for (int h = 0; h < 2; ++h) {
            const int block = 2 * h + tt / 2;
            const float sr = e4m3(s_row, block), sr8 = e4m3(s_row8, block);
            const float sc0 = e4m3(s_col0, block), sc1 = e4m3(s_col1, block);
            for (int e = 0; e < 8; ++e) {
                const float x0 = e2m1(ar[h] >> (4 * e)) * sr, x1 = e2m1(ar8[h] >> (4 * e)) * sr8;
                const float w0 = e2m1(bc0[h] >> (4 * e)) * sc0, w1 = e2m1(bc1[h] >> (4 * e)) * sc1;
                d[0] += x0 * w0;
                d[1] += x0 * w1;
                d[2] += x1 * w0;
                d[3] += x1 * w1;
            }
        }
    }
}
#endif

// D += A · B over 64 k on the block-scaled FP4 tensor cores (m16n8k64, E2M1 with UE4M3 scales per
// 16): `sa` and `sb` hold each lane's four scale bytes; `ida` and `idb` pick the lanes whose
// bytes are this tile's (Nunchaku's micro-scale layout).
__device__ __forceinline__ void mma4(float (&d)[4], const uint4& a, uint32_t b0, uint32_t b1,
                                     uint32_t sa, uint32_t sb, int ida, int idb) {
#if defined(COZY_SVDQ_EMULATE) && __CUDA_ARCH__ < 1200
    emulated(d, a, b0, b1, sa, sb, ida, idb);
#else
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1."
        "e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
        "{%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(sa), "n"(0),
          "h"(static_cast<short>(ida)), "r"(sb), "n"(0), "h"(static_cast<short>(idb)));
#endif
}

// ---------------------------------------------------------------------------------------------
// quantize

struct QuantArgs {
    const void* x;
    const float* p;
    const float* g;
    const __nv_bfloat16* down;
    uint4* act;
    uint32_t* ascales;
    __nv_bfloat16* la;
    int M, K;
};

// E2M1's nearest code of `q` (sign in bit 3), ties to the even code: the law's thresholds.
__device__ __forceinline__ uint32_t e2m1_code(float q, bool negative) {
    const float a = fabsf(q);
    const uint32_t c = uint32_t(a > 0.25f) + uint32_t(a >= 0.75f) + uint32_t(a > 1.25f) +
                       uint32_t(a >= 1.75f) + uint32_t(a > 2.5f) + uint32_t(a >= 3.5f) +
                       uint32_t(a > 5.0f);
    return negative ? (c | 8u) : c;
}

// Eight elements of a 16-block (the other eight are lane ^ 1's): their codes, low nibble first,
// and the block's E4M3 scale byte.
template <bool Half>
__device__ __forceinline__ void quant8(const uint4& x, const float (&p)[8], float gx, float six_g,
                                       uint32_t& code, uint32_t& scale) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 f = to_float2<Half>(word(x, e));
        v[2 * e] = __bfloat162float(__float2bfloat16_rn(f.x * p[2 * e]));
        v[2 * e + 1] = __bfloat162float(__float2bfloat16_rn(f.y * p[2 * e + 1]));
    }
    float amax = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(v[e]));
    amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, 1));
    const float s = fminf(fmaxf(__fdiv_rn(amax, six_g), kScaleFloor), 448.f);
    const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
    const float unit = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3))) * gx;
    uint32_t c = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) c |= e2m1_code(__fdiv_rn(v[e], unit), v[e] < 0.f) << (4 * e);
    code = c;
    scale = s8;
}

template <bool Half>
__device__ __forceinline__ uint4 to_half(const uint4& v) {
    if constexpr (Half) {
        uint4 out;
        out.x = from_float2<true>(to_float2<false>(v.x).x, to_float2<false>(v.x).y);
        out.y = from_float2<true>(to_float2<false>(v.y).x, to_float2<false>(v.y).y);
        out.z = from_float2<true>(to_float2<false>(v.z).x, to_float2<false>(v.z).y);
        out.w = from_float2<true>(to_float2<false>(v.w).x, to_float2<false>(v.w).y);
        return out;
    } else {
        return v;
    }
}

template <int R>
__device__ __forceinline__ void put(float* to, const float (&la)[2][R / 8][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < R / 8; ++j) {
            float* row = to + (16 * i + g) * R + 8 * j + 2 * t;
            row[0] = la[i][j][0];
            row[1] = la[i][j][1];
            row[8 * R] = la[i][j][2];
            row[8 * R + 1] = la[i][j][3];
        }
}

template <int R>
__device__ __forceinline__ void add(const float* from, float (&la)[2][R / 8][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < R / 8; ++j) {
            const float* row = from + (16 * i + g) * R + 8 * j + 2 * t;
            la[i][j][0] += row[0];
            la[i][j][1] += row[1];
            la[i][j][2] += row[8 * R];
            la[i][j][3] += row[8 * R + 1];
        }
}

template <int R, bool Half>
__global__ void __launch_bounds__(kQuantWarps * 32) quantize_kernel(const QuantArgs a) {
    extern __shared__ float part[];  // [kSlice][R]: the partials' running sum
    const int lane = threadIdx.x & 31, kw = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int r0 = blockIdx.x * kSlice, bm = r0 / kRows, wg = (r0 % kRows) / kSlice;
    const int steps = a.K / kDepth;
    const float gx = __ldg(a.g), six_g = 6.0f * gx;
    const auto* x = static_cast<const typename Types<Half>::T*>(a.x);
    float la[2][R / 8][4] = {};
    for (int k = kw; k < steps; k += kQuantWarps) {
        uint4 X[2][2][2];  // [m16 tile][row g, g + 8][k 8t.., 32 + 8t..]
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int rh = 0; rh < 2; ++rh) {
                const int r = r0 + 16 * i + 8 * rh + g;
#pragma unroll
                for (int kh = 0; kh < 2; ++kh)
                    X[i][rh][kh] = load16(x + size_t(r) * a.K + k * kDepth + 32 * kh + 8 * t, r < a.M);
            }
        float P[2][8];
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            const float4* p = reinterpret_cast<const float4*>(a.p + k * kDepth + 32 * kh + 8 * t);
            const float4 lo = __ldg(p), hi = __ldg(p + 1);
            P[kh][0] = lo.x, P[kh][1] = lo.y, P[kh][2] = lo.z, P[kh][3] = lo.w;
            P[kh][4] = hi.x, P[kh][5] = hi.y, P[kh][6] = hi.z, P[kh][7] = hi.w;
        }
        uint32_t code[2][2][2], scale[2][2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int rh = 0; rh < 2; ++rh)
#pragma unroll
                for (int kh = 0; kh < 2; ++kh)
                    quant8<Half>(X[i][rh][kh], P[kh], gx, six_g, code[i][rh][kh], scale[i][rh][kh]);
        const size_t tile = (size_t(bm) * steps + k) * kWarps + wg;
#pragma unroll
        for (int i = 0; i < 2; ++i)
            a.act[tile * 64 + 32 * i + lane] =
                make_uint4(code[i][0][0], code[i][1][0], code[i][0][1], code[i][1][1]);
        // Lane L holds row L/4 + 8 (L % 2) of tile (L % 4) / 2, its four blocks' bytes in order;
        // block b of a row is held by the lanes 4 (L/4) + 2 (b % 2) (+1) of its k-half b / 2.
        uint32_t held[2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
            held[i] = scale[i][0][0] | scale[i][0][1] << 8 | scale[i][1][0] << 16 |
                      scale[i][1][1] << 24;
        uint32_t bytes = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int from = 4 * g + 2 * (b & 1);
            const uint32_t t0 = __shfl_sync(~0u, held[0], from);
            const uint32_t t1 = __shfl_sync(~0u, held[1], from);
            const uint32_t v = (t >> 1) ? t1 : t0;
            bytes |= ((v >> (8 * ((lane & 1) * 2 + (b >> 1)))) & 0xffu) << (8 * b);
        }
        a.ascales[tile * 32 + lane] = bytes;
        // la += x · downᵀ, lane t's k at step (kh, s) being 32 kh + 8t + 4s .. + 3.
#pragma unroll
        for (int j = 0; j < R / 8; ++j) {
            uint4 D[2];
#pragma unroll
            for (int kh = 0; kh < 2; ++kh)
                D[kh] = to_half<Half>(
                    load16(a.down + size_t(8 * j + g) * a.K + k * kDepth + 32 * kh + 8 * t, true));
#pragma unroll
            for (int kh = 0; kh < 2; ++kh)
#pragma unroll
                for (int s = 0; s < 2; ++s)
#pragma unroll
                    for (int i = 0; i < 2; ++i)
                        mma16<Half>(la[i][j], word(X[i][0][kh], 2 * s), word(X[i][1][kh], 2 * s),
                                    word(X[i][0][kh], 2 * s + 1), word(X[i][1][kh], 2 * s + 1),
                                    word(D[kh], 2 * s), word(D[kh], 2 * s + 1));
        }
    }
    // ((w0 + w1) + w2) + ... + w7, one warp at a time: the last holds the sum.
    for (int w = 0; w < kQuantWarps; ++w) {
        if (kw == w) {
            if (w > 0) add<R>(part, la);
            if (w + 1 < kQuantWarps) put<R>(part, la);
        }
        __syncthreads();
    }
    if (kw != kQuantWarps - 1) return;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < R / 8; ++j) {
            __nv_bfloat16* row = a.la + size_t(r0 + 16 * i + g) * R + 8 * j + 2 * t;
            *reinterpret_cast<__nv_bfloat162*>(row) = __floats2bfloat162_rn(la[i][j][0], la[i][j][1]);
            *reinterpret_cast<__nv_bfloat162*>(row + 8 * R) =
                __floats2bfloat162_rn(la[i][j][2], la[i][j][3]);
        }
}

// ---------------------------------------------------------------------------------------------
// gemm

struct GemmArgs {
    const uint4* act;
    const uint32_t* ascales;
    const uint4* w;
    const uint4* wscales;
    const float* gx;
    const float* gw;
    const float* c;
    const __nv_bfloat16* la;
    const __nv_bfloat16* up;
    const void* bias;
    void* out;
    int M, N, K;
};

// One k-step's operands for a warp: its two m16 activation tiles, the block's eight n16 weight
// tiles and both sides' scale bytes.
struct Stage {
    uint4 a[2];
    uint4 w[8];
    uint32_t sa;
    uint4 sw;
};

__device__ __forceinline__ void load(Stage& s, const uint4* act, const uint32_t* asc,
                                     const uint4* w, const uint4* ws, int k, bool ok) {
    s.a[0] = load16(act + size_t(k) * 512, ok);
    s.a[1] = load16(act + size_t(k) * 512 + 32, ok);
#pragma unroll
    for (int j = 0; j < 8; ++j) s.w[j] = load16(w + size_t(k) * 256 + 32 * j, ok);
    s.sa = ok ? __ldg(asc + size_t(k) * 256) : 0u;
    s.sw = load16(ws + size_t(k) * 32, ok);
}

__device__ __forceinline__ void compute(float (&acc)[2][16][4], const Stage& s) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t sb = word(s.sw, j / 2);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mma4(acc[i][2 * j], s.a[i], s.w[j].x, s.w[j].y, s.sa, sb, i, 2 * (j % 2));
            mma4(acc[i][2 * j + 1], s.a[i], s.w[j].z, s.w[j].w, s.sa, sb, i, 2 * (j % 2) + 1);
        }
    }
}

template <int R, bool Half>
__global__ void __launch_bounds__(kWarps * 32, 1) gemm_kernel(const GemmArgs a) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int bm = blockIdx.x, bn = blockIdx.y, steps = a.K / kDepth;
    const uint4* act = a.act + (size_t(bm) * steps * kWarps + warp) * 64 + lane;
    const uint32_t* asc = a.ascales + (size_t(bm) * steps * kWarps + warp) * 32 + lane;
    const uint4* w = a.w + size_t(bn) * steps * 256 + lane;
    const uint4* ws = a.wscales + size_t(bn) * steps * 32 + lane;

    float acc[2][16][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int q = 0; q < 16; ++q)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][q][e] = 0.f;
    Stage s0, s1;
    load(s0, act, asc, w, ws, 0, true);
    for (int k = 0; k < steps; k += 2) {
        load(s1, act, asc, w, ws, k + 1, k + 1 < steps);
        compute(acc, s0);
        load(s0, act, asc, w, ws, k + 2, k + 2 < steps);
        compute(acc, s1);  // past the last step its operands are zeros: it adds nothing
    }

    const int row = bm * kRows + warp * kSlice, col = bn * kCols;
    const float alpha = __ldg(a.gx) * __ldg(a.gw);
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        float2 s = make_float2(alpha, alpha);
        if (a.c != nullptr) {
            const float2 c = __ldg(reinterpret_cast<const float2*>(a.c + col + 8 * q + 2 * t));
            s.x *= c.x;
            s.y *= c.y;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            acc[i][q][0] *= s.x;
            acc[i][q][1] *= s.y;
            acc[i][q][2] *= s.x;
            acc[i][q][3] *= s.y;
        }
    }
    // + la · upᵀ, lane t's k at step s being 16s + 4t .. + 3 in both operands.
#pragma unroll 1
    for (int s = 0; s < R / 16; ++s) {
        uint32_t A[2][4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const __nv_bfloat16* p = a.la + size_t(row + 16 * i + g) * R + 16 * s + 4 * t;
            const uint2 lo = *reinterpret_cast<const uint2*>(p);
            const uint2 hi = *reinterpret_cast<const uint2*>(p + 8 * R);
            A[i][0] = lo.x, A[i][1] = hi.x, A[i][2] = lo.y, A[i][3] = hi.y;
        }
#pragma unroll
        for (int q = 0; q < 16; ++q) {
            const uint2 u = __ldg(reinterpret_cast<const uint2*>(
                a.up + size_t(col + 8 * q + g) * R + 16 * s + 4 * t));
#pragma unroll
            for (int i = 0; i < 2; ++i)
                mma16<false>(acc[i][q], A[i][0], A[i][1], A[i][2], A[i][3], u.x, u.y);
        }
    }
    using T = typename Types<Half>::T;
    const auto* bias = static_cast<const T*>(a.bias);
    auto* out = static_cast<T*>(a.out);
#pragma unroll
    for (int q = 0; q < 16; ++q) {
        const int n = col + 8 * q + 2 * t;
        float2 b = make_float2(0.f, 0.f);
        if (bias != nullptr) b = to_float2<Half>(*reinterpret_cast<const uint32_t*>(bias + n));
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = row + 16 * i + g;
            if (r < a.M)
                *reinterpret_cast<uint32_t*>(out + size_t(r) * a.N + n) =
                    from_float2<Half>(acc[i][q][0] + b.x, acc[i][q][1] + b.y);
            if (r + 8 < a.M)
                *reinterpret_cast<uint32_t*>(out + size_t(r + 8) * a.N + n) =
                    from_float2<Half>(acc[i][q][2] + b.x, acc[i][q][3] + b.y);
        }
    }
}

int rows(int M) { return (M + kRows - 1) / kRows * kRows; }

template <int R, bool Half>
cudaError_t launch_quantize(const QuantArgs& a, cudaStream_t stream) {
    quantize_kernel<R, Half><<<rows(a.M) / kSlice, kQuantWarps * 32, kSlice * R * sizeof(float),
                               stream>>>(a);
    return cudaGetLastError();
}

template <int R, bool Half>
cudaError_t launch_gemm(const GemmArgs& a, cudaStream_t stream) {
    gemm_kernel<R, Half><<<dim3(rows(a.M) / kRows, a.N / kCols), kWarps * 32, 0, stream>>>(a);
    return cudaGetLastError();
}

template <bool Half>
cudaError_t ranked_quantize(const QuantArgs& a, int R, cudaStream_t stream) {
    switch (R) {
        case 32: return launch_quantize<32, Half>(a, stream);
        case 64: return launch_quantize<64, Half>(a, stream);
        case 128: return launch_quantize<128, Half>(a, stream);
        default: return cudaErrorInvalidValue;
    }
}

template <bool Half>
cudaError_t ranked_gemm(const GemmArgs& a, int R, cudaStream_t stream) {
    switch (R) {
        case 32: return launch_gemm<32, Half>(a, stream);
        case 64: return launch_gemm<64, Half>(a, stream);
        case 128: return launch_gemm<128, Half>(a, stream);
        default: return cudaErrorInvalidValue;
    }
}

}  // namespace

cudaError_t quantize(const void* x, bool x_half, const float* p, const float* g, const void* down,
                     void* act, void* ascales, void* la, int M, int K, int R,
                     cudaStream_t stream) {
    if (M < 0 || !fits(kCols, K, R)) return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    const QuantArgs a{x, p, g, static_cast<const __nv_bfloat16*>(down), static_cast<uint4*>(act),
                      static_cast<uint32_t*>(ascales), static_cast<__nv_bfloat16*>(la), M, K};
    return x_half ? ranked_quantize<true>(a, R, stream) : ranked_quantize<false>(a, R, stream);
}

cudaError_t gemm(const void* act, const void* ascales, const void* w, const void* wscales,
                 const float* gx, const float* gw, const float* c, const void* la, const void* up,
                 const void* bias, void* out, bool out_half, int M, int N, int K, int R,
                 cudaStream_t stream) {
    if (M < 1 || !fits(N, K, R)) return cudaErrorInvalidValue;
    const GemmArgs a{static_cast<const uint4*>(act), static_cast<const uint32_t*>(ascales),
                     static_cast<const uint4*>(w), static_cast<const uint4*>(wscales), gx, gw, c,
                     static_cast<const __nv_bfloat16*>(la), static_cast<const __nv_bfloat16*>(up),
                     bias, out, M, N, K};
    return out_half ? ranked_gemm<true>(a, R, stream) : ranked_gemm<false>(a, R, stream);
}

}  // namespace cozy_svdq
