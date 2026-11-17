// SPDX-License-Identifier: Apache-2.0
// int8-convrot convolution (cr-327): Cozy's NDHWC implicit-GEMM conv on the int8 tensor cores (mma.sync m16n8k32, sm80+),
// and the passes that feed it. Activations are int8 with one scale per voxel, weights int8 with one scale per (output
// channel, tap), both rotated over input channels by the regular H256 (int8-convrot256/1, per tap). The voxels a
// output sums have different scales, so each tap's int32 sum is promoted to f32 with that tap's scales before the
// next; the epilogue adds the bias and a residual. The passes: a tensor's GroupNorm statistics, and the pass that
// normalizes (a per-(n, channel) affine), applies SiLU, rotates and quantizes each voxel (C = 512), alone or after a
// depthwise temporal conv.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

namespace conv8 {

struct ConvParams {
  const int8_t *x;    // [N, D, H, W, C]
  const float *xs;    // [N, D, H, W]
  const int8_t *w;    // [K, taps, C]
  const float *ws;    // [K, taps] (ws_tap = 1) or [K] (ws_tap = 0)
  const float *bias;  // [K] or null
  const void *res;    // [N, Z, P, Q, K] or null
  void *out;          // [N, Z, P, Q, K]
  int N, D, H, W, C, K;
  int T, R, S;     // filter extents
  int pt, pr, ps;  // padding before (frames, rows, columns); stride 1
  int Z, P, Q;     // output extents
  int M;           // N * Z * P * Q
  int ws_tap;
};

template <typename T> struct Out2;
template <> struct Out2<__nv_bfloat16> {
  using type = __nv_bfloat162;
  static __device__ __forceinline__ type pack(float a, float b) { return __floats2bfloat162_rn(a, b); }
  static __device__ __forceinline__ float2 unpack(type v) { return __bfloat1622float2(v); }
};
template <> struct Out2<__half> {
  using type = __half2;
  static __device__ __forceinline__ type pack(float a, float b) { return __floats2half2_rn(a, b); }
  static __device__ __forceinline__ float2 unpack(type v) { return __half22float2(v); }
};

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void *src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A tile row of BK bytes holds BK/16 chunks of 16; chunk c of row r is stored at chunk c ^ f(r), so the 8 rows one
// ldmatrix reads land in distinct banks.
template <int BK> __device__ __forceinline__ int swz(int r, int c) {
  if constexpr (BK == 64) return c ^ ((r >> 1) & 3);
  else return c ^ (r & 7);
}

template <typename OutT, int BM, int BN, int BK, int WM, int WN, int STAGES>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32, 1) conv_kernel(const ConvParams p) {
  constexpr int WARPS_N = BN / WN;
  constexpr int THREADS = (BM / WM) * WARPS_N * 32;
  constexpr int MF = WM / 16, NF = WN / 8, KS = BK / 32;
  constexpr int CHUNKS = BK / 16;
  constexpr int A_LOADS = (BM * CHUNKS + THREADS - 1) / THREADS, B_LOADS = (BN * CHUNKS + THREADS - 1) / THREADS;
  static_assert(THREADS % CHUNKS == 0, "a thread keeps its chunk column");
  static_assert(THREADS >= BM + BN, "a thread a scale");
  static_assert(NF % 2 == 0, "B frags load in pairs");

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t *sa = smem;                     // [STAGES][BM][BK]
  uint8_t *sb = smem + STAGES * BM * BK;  // [STAGES][BN][BK]
  // Each tap's scales, loaded with the tap's first stage into a ring of NBUF taps: a tap's are still there when it
  // promotes as long as (NBUF - 1) * C / BK > STAGES - 2 (`conv_fits`).
  constexpr int NBUF = STAGES <= 3 ? 2 : 4;
  float *srow = reinterpret_cast<float *>(sb + STAGES * BN * BK);  // [NBUF][BM]
  float *scol = srow + NBUF * BM;                                  // [NBUF][BN]

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WARPS_N, wn = warp % WARPS_N;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int taps = p.T * p.R * p.S, CB = p.C / BK, iters = taps * CB;
  const int kc = tid % CHUNKS;

  // An output voxel's input offset (the tap at its own position) and which taps stay inside the input: bit dt,
  // 8 + dr, 16 + ds. Nothing past M.
  auto locate = [&](int m, int64_t &base, uint32_t &ok) {
    base = 0, ok = 0;
    if (m >= p.M) return;
    int t = m;
    const int x = t % p.Q; t /= p.Q;
    const int y = t % p.P; t /= p.P;
    const int z = t % p.Z, n = t / p.Z;
    base = (((int64_t)n * p.D + z) * p.H + y) * p.W + x;
    for (int k = 0; k < p.T; ++k) ok |= (uint32_t)(z + k - p.pt >= 0 && z + k - p.pt < p.D) << k;
    for (int k = 0; k < p.R; ++k) ok |= (uint32_t)(y + k - p.pr >= 0 && y + k - p.pr < p.H) << (8 + k);
    for (int k = 0; k < p.S; ++k) ok |= (uint32_t)(x + k - p.ps >= 0 && x + k - p.ps < p.W) << (16 + k);
  };
  // This thread's chunks of each stage: A rows (input offsets, in-bounds taps, smem offset) and B rows.
  int64_t abase[A_LOADS], bbase[B_LOADS];
  uint32_t aok[A_LOADS], aoff[A_LOADS], boff[B_LOADS];
  bool bok[B_LOADS];
#pragma unroll
  for (int i = 0; i < A_LOADS; ++i) {
    const int r = (tid + i * THREADS) / CHUNKS;
    locate(r < BM ? m0 + r : p.M, abase[i], aok[i]);
    abase[i] = abase[i] * p.C + kc * 16;
    aoff[i] = r * BK + swz<BK>(r, kc) * 16;
  }
#pragma unroll
  for (int i = 0; i < B_LOADS; ++i) {
    const int r = (tid + i * THREADS) / CHUNKS;
    bok[i] = r < BN && n0 + r < p.K;
    bbase[i] = (int64_t)(n0 + r) * taps * p.C + kc * 16;
    boff[i] = r * BK + swz<BK>(r, kc) * 16;
  }
  // The scale this thread loads each tap: row `tid`'s voxel, or column `tid - BM`'s weight.
  int64_t vbase = 0;
  uint32_t vok = 0;
  if (tid < BM) locate(m0 + tid, vbase, vok);
  const bool col_ok = tid >= BM && tid < BM + BN && n0 + tid - BM < p.K;
  const float *col = p.ws + (int64_t)(n0 + tid - BM) * (p.ws_tap ? taps : 1);

  // The next stage to load: its chunk of input channels and its tap.
  int l_cb = 0, l_tap = 0, l_dt = 0, l_dr = 0, l_ds = 0, l_stage = 0;
  int64_t l_delta = 0;  // input offset of the tap, in voxels
  uint32_t l_need = 0;
  auto aim = [&] {
    l_delta = ((int64_t)(l_dt - p.pt) * p.H + (l_dr - p.pr)) * p.W + (l_ds - p.ps);
    l_need = (1u << l_dt) | (1u << (8 + l_dr)) | (1u << (16 + l_ds));
  };
  auto advance = [&] {
    if (++l_stage == STAGES) l_stage = 0;
    if (++l_cb < CB) return;
    l_cb = 0, ++l_tap;
    if (++l_ds == p.S) {
      l_ds = 0;
      if (++l_dr == p.R) l_dr = 0, ++l_dt;
    }
    aim();
  };
  // Part `part` of `parts` of the next stage's copies (each k-step issues one part, between its MMAs).
  auto load = [&](int part, int parts) {
    uint8_t *ta = sa + l_stage * BM * BK;
    uint8_t *tb = sb + l_stage * BN * BK;
    const int64_t a_at = l_delta * p.C + l_cb * BK, b_at = (int64_t)l_tap * p.C + l_cb * BK;
    if (part == 0 && l_cb == 0) {
      if (tid < BM) {
        const bool ok = (vok & l_need) == l_need;
        cp_async4(smem_u32(srow + (l_tap & (NBUF - 1)) * BM + tid), ok ? p.xs + vbase + l_delta : p.xs, ok);
      } else if (tid < BM + BN) {
        cp_async4(smem_u32(scol + (l_tap & (NBUF - 1)) * BN + tid - BM), col_ok ? col + (p.ws_tap ? l_tap : 0) : p.ws,
                  col_ok);
      }
    }
#pragma unroll
    for (int i = 0; i < A_LOADS; ++i) {
      if (i % parts != part) continue;
      if (BM * CHUNKS % THREADS && tid + i * THREADS >= BM * CHUNKS) break;
      const bool ok = (aok[i] & l_need) == l_need;
      cp_async16(smem_u32(ta + aoff[i]), ok ? p.x + abase[i] + a_at : p.x, ok);
    }
#pragma unroll
    for (int i = 0; i < B_LOADS; ++i) {
      if (i % parts != part) continue;
      if (BN * CHUNKS % THREADS && tid + i * THREADS >= BN * CHUNKS) break;
      cp_async16(smem_u32(tb + boff[i]), bok[i] ? p.w + bbase[i] + b_at : p.w, bok[i]);
    }
  };

  int acc[MF][NF][4];
  float facc[MF][NF][4];
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0, facc[i][j][e] = 0.f;

  aim();
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < iters) {
      load(0, 1);
      advance();
    }
    cp_commit();
  }

  const int g = lane >> 2, t4 = lane & 3;
  int c_cb = 0, c_tap = 0, c_stage = 0;
  // Each lane's ldmatrix rows and chunk columns, A then B.
  const int ar = wm * WM + (lane & 7) + ((lane >> 3) & 1) * 8, ac = lane >> 4;
  const int br = wn * WN + (lane & 7) + (lane >> 4) * 8, bc = (lane >> 3) & 1;
  for (int it = 0; it < iters; ++it) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const bool more = it + STAGES - 1 < iters;
    const uint8_t *ta = sa + c_stage * BM * BK;
    const uint8_t *tb = sb + c_stage * BN * BK;
    uint32_t af[2][MF][4], bf[2][NF][2];
    auto frags = [&](int buf, int kk) {
#pragma unroll
      for (int i = 0; i < MF; ++i) {
        const int r = ar + i * 16;
        ldsm_x4(smem_u32(ta + r * BK + swz<BK>(r, kk * 2 + ac) * 16), af[buf][i][0], af[buf][i][1], af[buf][i][2],
                af[buf][i][3]);
      }
#pragma unroll
      for (int j = 0; j < NF; j += 2) {
        const int r = br + j * 8;
        ldsm_x4(smem_u32(tb + r * BK + swz<BK>(r, kk * 2 + bc) * 16), bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0],
                bf[buf][j + 1][1]);
      }
    };
    frags(0, 0);
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
      if (kk + 1 < KS) frags((kk + 1) & 1, kk + 1);
      if (more) load(kk, KS);
#pragma unroll
      for (int i = 0; i < MF; ++i)
#pragma unroll
        for (int j = 0; j < NF; ++j) mma_s8(acc[i][j], af[kk & 1][i], bf[kk & 1][j][0], bf[kk & 1][j][1]);
    }
    if (more) advance();
    cp_commit();
    if (++c_stage == STAGES) c_stage = 0;

    if (++c_cb == CB) {
      const float *sr = srow + (c_tap & (NBUF - 1)) * BM;
      const float *sc = scol + (c_tap & (NBUF - 1)) * BN;
#pragma unroll
      for (int i = 0; i < MF; ++i) {
        const float r0 = sr[wm * WM + i * 16 + g], r1 = sr[wm * WM + i * 16 + g + 8];
#pragma unroll
        for (int j = 0; j < NF; ++j) {
          const float c0 = sc[wn * WN + j * 8 + t4 * 2], c1 = sc[wn * WN + j * 8 + t4 * 2 + 1];
          facc[i][j][0] = fmaf(static_cast<float>(acc[i][j][0]), r0 * c0, facc[i][j][0]);
          facc[i][j][1] = fmaf(static_cast<float>(acc[i][j][1]), r0 * c1, facc[i][j][1]);
          facc[i][j][2] = fmaf(static_cast<float>(acc[i][j][2]), r1 * c0, facc[i][j][2]);
          facc[i][j][3] = fmaf(static_cast<float>(acc[i][j][3]), r1 * c1, facc[i][j][3]);
          acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;
        }
      }
      c_cb = 0, ++c_tap;
    }
  }
  cp_wait<0>();

  using O2 = Out2<OutT>;
  using V2 = typename O2::type;
  const V2 *res = reinterpret_cast<const V2 *>(p.res);
  V2 *out = reinterpret_cast<V2 *>(p.out);
#pragma unroll
  for (int i = 0; i < MF; ++i)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int m = m0 + wm * WM + i * 16 + g + hh * 8;
      if (m >= p.M) continue;
#pragma unroll
      for (int j = 0; j < NF; ++j) {
        const int k = n0 + wn * WN + j * 8 + t4 * 2;
        if (k >= p.K) continue;
        float v0 = facc[i][j][hh * 2], v1 = facc[i][j][hh * 2 + 1];
        if (p.bias) v0 += __ldg(p.bias + k), v1 += __ldg(p.bias + k + 1);
        const int64_t o = ((int64_t)m * p.K + k) / 2;
        if (res) {
          const float2 rv = O2::unpack(res[o]);
          v0 += rv.x, v1 += rv.y;
        }
        out[o] = O2::pack(v0, v1);
      }
    }
}

template <typename OutT, int BM, int BN, int BK, int WM, int WN, int STAGES>
inline size_t conv_smem() {
  return STAGES * (BM + BN) * BK + (STAGES <= 3 ? 2 : 4) * (BM + BN) * sizeof(float);
}

// Whether the config serves a conv of C input channels: whole BK chunks, and the scale ring outlives a tap.
template <int BK, int STAGES>
inline bool conv_fits(int C) {
  const int nbuf = STAGES <= 3 ? 2 : 4;
  return C % BK == 0 && (nbuf - 1) * (C / BK) > STAGES - 2;
}

template <typename OutT, int BM, int BN, int BK, int WM, int WN, int STAGES>
inline cudaError_t launch_conv(const ConvParams &p, cudaStream_t stream) {
  auto kernel = conv_kernel<OutT, BM, BN, BK, WM, WN, STAGES>;
  const size_t smem = conv_smem<OutT, BM, BN, BK, WM, WN, STAGES>();
  cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  if (e != cudaSuccess) return e;
  dim3 grid((p.M + BM - 1) / BM, (p.K + BN - 1) / BN);
  kernel<<<grid, (BM / WM) * (BN / WN) * 32, smem, stream>>>(p);
  return cudaGetLastError();
}

// ---------------------------------------------------------------- GroupNorm statistics

// acc[n][g] += (sum, sum of squares) over the n-th sample's voxels of x [N * V, C] (NDHWC), each group C / G channels.
template <typename InT>
__global__ void __launch_bounds__(256) gn_stats_kernel(const InT *x, double *acc, int64_t V, int C, int G) {
  extern __shared__ float red[];  // [rows][G][2]
  const int n = blockIdx.y;
  const int lanes = C / 8, rows = 256 / lanes;  // threads per voxel; voxels at once
  const int tid = threadIdx.x, col = tid % lanes, row = tid / lanes;
  const int per = C / G;  // channels per group; a thread's 8 lie in one group when per % 8 == 0
  float s = 0.f, ss = 0.f;
  if (row < rows) {
    const InT *base = x + (int64_t)n * V * C + col * 8;
    for (int64_t v = (int64_t)blockIdx.x * rows + row; v < V; v += (int64_t)gridDim.x * rows) {
      const uint4 raw = __ldg(reinterpret_cast<const uint4 *>(base + v * C));
      const typename Out2<InT>::type *h2 = reinterpret_cast<const typename Out2<InT>::type *>(&raw);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = Out2<InT>::unpack(h2[k]);
        s += f.x + f.y;
        ss += f.x * f.x + f.y * f.y;
      }
    }
  }
  for (int i = tid; i < rows * G * 2; i += 256) red[i] = 0.f;
  __syncthreads();
  if (row < rows) {
    const int gi = col * 8 / per;
    atomicAdd(&red[(row * G + gi) * 2], s);
    atomicAdd(&red[(row * G + gi) * 2 + 1], ss);
  }
  __syncthreads();
  for (int i = tid; i < G * 2; i += 256) {
    float t = 0.f;
    for (int r = 0; r < rows; ++r) t += red[r * G * 2 + i];
    atomicAdd(acc + (int64_t)n * G * 2 + i, (double)t);
  }
}

template <typename InT>
inline cudaError_t launch_gn_stats(const InT *x, double *acc, int N, int64_t V, int C, int G, int sms,
                                   cudaStream_t stream) {
  const int lanes = C / 8, rows = 256 / lanes;
  int blocks = (int)std::min<int64_t>((V + rows - 1) / rows, (int64_t)sms * 8);
  blocks = std::max(blocks, 1);
  gn_stats_kernel<InT><<<dim3(blocks, N), 256, rows * G * 2 * sizeof(float), stream>>>(x, acc, V, C, G);
  return cudaGetLastError();
}

// ---------------------------------------------------------------- normalize, activate, rotate, quantize

// A voxel's 512 values (16 a lane: channels 16 * lane + k) times the orthonormal regular H256 per 256 channels when
// `rotate`, then quantized: int8 codes to `q` (512 bytes), their scale to `*qs`. `tile` and `codes` are the warp's.
__device__ __forceinline__ void rotate_quant(float (&y)[16], bool rotate, float (*tile)[16][17], int8_t *codes,
                                             int8_t *q, float *qs) {
  const int lane = threadIdx.x & 31, grp = lane >> 4, li = lane & 15;
  if (rotate) {
    // channel within its 256-group = 64a + 16b + 4c + d: this lane is (a, b) = li, its registers (c, d).
    // M4 v = sum(v) - 2 reverse(v); along d, then c, in registers.
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float *u = y + 4 * c;
      const float s = u[0] + u[1] + u[2] + u[3];
      const float u0 = u[0], u1 = u[1];
      u[0] = s - 2.f * u[3]; u[1] = s - 2.f * u[2]; u[2] = s - 2.f * u1; u[3] = s - 2.f * u0;
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const float s = y[d] + y[4 + d] + y[8 + d] + y[12 + d];
      const float u0 = y[d], u1 = y[4 + d];
      y[d] = s - 2.f * y[12 + d]; y[4 + d] = s - 2.f * y[8 + d]; y[8 + d] = s - 2.f * u1; y[12 + d] = s - 2.f * u0;
    }
    // Transpose: this lane takes (c, d) = li and every (a, b).
#pragma unroll
    for (int k = 0; k < 16; ++k) tile[grp][li][k] = y[k];
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 16; ++k) y[k] = tile[grp][k][li];
    __syncwarp();
#pragma unroll
    for (int a4 = 0; a4 < 4; ++a4) {  // along b
      float *u = y + 4 * a4;
      const float s = u[0] + u[1] + u[2] + u[3];
      const float u0 = u[0], u1 = u[1];
      u[0] = s - 2.f * u[3]; u[1] = s - 2.f * u[2]; u[2] = s - 2.f * u1; u[3] = s - 2.f * u0;
    }
#pragma unroll
    for (int b4 = 0; b4 < 4; ++b4) {  // along a
      const float s = y[b4] + y[4 + b4] + y[8 + b4] + y[12 + b4];
      const float u0 = y[b4], u1 = y[4 + b4];
      y[b4] = s - 2.f * y[12 + b4]; y[4 + b4] = s - 2.f * y[8 + b4]; y[8 + b4] = s - 2.f * u1; y[12 + b4] = s - 2.f * u0;
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) amax = fmaxf(amax, fabsf(y[k]));
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  if (rotate) {
    // register k is (a, b) = k of channel grp * 256 + 64a + 16b + li
#pragma unroll
    for (int k = 0; k < 16; ++k) codes[grp * 256 + k * 16 + li] = (int8_t)__float2int_rn(y[k] * inv);
    __syncwarp();
    *reinterpret_cast<uint4 *>(q + lane * 16) = *reinterpret_cast<const uint4 *>(codes + lane * 16);
    __syncwarp();
  } else {
    uint32_t packed[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      uint32_t w = 0;
#pragma unroll
      for (int e = 0; e < 4; ++e) w |= (uint32_t)(uint8_t)(int8_t)__float2int_rn(y[4 * k + e] * inv) << (8 * e);
      packed[k] = w;
    }
    *reinterpret_cast<uint4 *>(q + lane * 16) = make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
  if (lane == 0) *qs = (amax / 127.f) / (rotate ? 16.f : 1.f);
}

template <typename InT> __device__ __forceinline__ void load16(const InT *row, float (&y)[16]) {
  const uint4 r0 = __ldg(reinterpret_cast<const uint4 *>(row));
  const uint4 r1 = __ldg(reinterpret_cast<const uint4 *>(row + 8));
  const typename Out2<InT>::type *h0 = reinterpret_cast<const typename Out2<InT>::type *>(&r0);
  const typename Out2<InT>::type *h1 = reinterpret_cast<const typename Out2<InT>::type *>(&r1);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f0 = Out2<InT>::unpack(h0[k]), f1 = Out2<InT>::unpack(h1[k]);
    y[2 * k] = f0.x, y[2 * k + 1] = f0.y, y[8 + 2 * k] = f1.x, y[8 + 2 * k + 1] = f1.y;
  }
}

// y = a[n][c] * x + b[n][c], SiLU when `silu`; each 256 channels times the orthonormal regular H256 when `rotate`;
// then each voxel's int8 codes and scale (q * scale ~ y). C = 512: a warp per voxel, 16 channels a lane.
template <typename InT>
__global__ void __launch_bounds__(256) norm_quant_kernel(const InT *x, const float *a, const float *b, int8_t *q,
                                                         float *qs, int64_t V, int64_t per_n, bool silu, bool rotate) {
  constexpr int C = 512;
  __shared__ float tile[8][2][16][17];
  __shared__ __align__(16) int8_t codes[8][C];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int64_t v = (int64_t)blockIdx.x * 8 + warp; v < V; v += (int64_t)gridDim.x * 8) {
    const int64_t n = v / per_n;
    float y[16];
    load16(x + v * C + lane * 16, y);
    const float4 *av = reinterpret_cast<const float4 *>(a + n * C + lane * 16);
    const float4 *bv = reinterpret_cast<const float4 *>(b + n * C + lane * 16);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4 aa = __ldg(av + k), bb = __ldg(bv + k);
      y[4 * k] = fmaf(aa.x, y[4 * k], bb.x);
      y[4 * k + 1] = fmaf(aa.y, y[4 * k + 1], bb.y);
      y[4 * k + 2] = fmaf(aa.z, y[4 * k + 2], bb.z);
      y[4 * k + 3] = fmaf(aa.w, y[4 * k + 3], bb.w);
    }
    if (silu) {
#pragma unroll
      for (int k = 0; k < 16; ++k) y[k] = y[k] / (1.f + __expf(-y[k]));
    }
    rotate_quant(y, rotate, tile[warp], codes[warp], q + v * C, qs + v);
  }
}

// TemporalConv's first half for its int8 pointwise conv: y = silu(a[n][c] * x + b[n][c]), the depthwise temporal conv
// `w` [C][KT] (zero padding KT / 2 frames each side) plus `bias`, then rotated and quantized per voxel (`rotate_quant`).
// x [N, T, HW, 512]; a warp walks one (n, position) through time, holding the last KT activated frames.
template <typename InT, int KT>
__global__ void __launch_bounds__(256) temporal_quant_kernel(const InT *x, const float *a, const float *b,
                                                             const float *w, const float *bias, int8_t *q, float *qs,
                                                             int N, int T, int HW) {
  constexpr int C = 512, PAD = KT / 2;
  __shared__ float tile[8][2][16][17];
  __shared__ __align__(16) int8_t codes[8][C];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float wk[KT][16], bk[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    bk[k] = bias ? __ldg(bias + lane * 16 + k) : 0.f;
#pragma unroll
    for (int t = 0; t < KT; ++t) wk[t][k] = __ldg(w + (lane * 16 + k) * KT + t);
  }
  for (int64_t pos = (int64_t)blockIdx.x * 8 + warp; pos < (int64_t)N * HW; pos += (int64_t)gridDim.x * 8) {
    const int n = (int)(pos / HW), hw = (int)(pos % HW);
    float aa[16], bb[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) aa[k] = __ldg(a + n * C + lane * 16 + k), bb[k] = __ldg(b + n * C + lane * 16 + k);
    float win[KT][16];
#pragma unroll
    for (int s = 0; s < KT; ++s)
#pragma unroll
      for (int k = 0; k < 16; ++k) win[s][k] = 0.f;
    // Slot s holds frame t with t % KT == s; the loop is unrolled by KT so every index is static.
    for (int t0 = 0; t0 < T + PAD; t0 += KT) {
#pragma unroll
      for (int s = 0; s < KT; ++s) {
        const int t = t0 + s;  // the frame read now; the output is t - PAD
        if (t >= T + PAD) break;
        if (t < T) {
          float y[16];
          load16(x + (((int64_t)n * T + t) * HW + hw) * C + lane * 16, y);
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const float v = fmaf(aa[k], y[k], bb[k]);
            win[s][k] = v / (1.f + __expf(-v));
          }
        } else {
#pragma unroll
          for (int k = 0; k < 16; ++k) win[s][k] = 0.f;
        }
        const int out = t - PAD;
        if (out < 0) continue;
        float y[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) y[k] = bk[k];
        // frames out - PAD .. out + PAD = t - 2 PAD .. t, in slots (s - 2 PAD + j) mod KT
#pragma unroll
        for (int j = 0; j < KT; ++j) {
          const int slot = (s + 1 + j) % KT;  // frame t - KT + 1 + j
#pragma unroll
          for (int k = 0; k < 16; ++k) y[k] = fmaf(wk[j][k], win[slot][k], y[k]);
        }
        const int64_t v = ((int64_t)n * T + out) * HW + hw;
        rotate_quant(y, true, tile[warp], codes[warp], q + v * C, qs + v);
      }
    }
  }
}

template <typename InT, int KT>
inline cudaError_t launch_temporal_quant(const InT *x, const float *a, const float *b, const float *w,
                                         const float *bias, int8_t *q, float *qs, int N, int T, int HW, int sms,
                                         cudaStream_t stream) {
  const int blocks = (int)std::max<int64_t>(1, std::min<int64_t>(((int64_t)N * HW + 7) / 8, (int64_t)sms * 8));
  temporal_quant_kernel<InT, KT><<<blocks, 256, 0, stream>>>(x, a, b, w, bias, q, qs, N, T, HW);
  return cudaGetLastError();
}

template <typename InT>
inline cudaError_t launch_norm_quant(const InT *x, const float *a, const float *b, int8_t *q, float *qs, int64_t V,
                                     int64_t per_n, bool silu, bool rotate, int sms, cudaStream_t stream) {
  const int blocks = (int)std::max<int64_t>(1, std::min<int64_t>((V + 7) / 8, (int64_t)sms * 16));
  norm_quant_kernel<InT><<<blocks, 256, 0, stream>>>(x, a, b, q, qs, V, per_n, silu, rotate);
  return cudaGetLastError();
}

}  // namespace conv8
