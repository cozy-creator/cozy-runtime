// SPDX-License-Identifier: Apache-2.0
// `_kernels._conv`'s int8-convrot launchers (cr-327, `conv8.cuh`): C entry points the binding (`../conv.cpp`) calls with
// raw device pointers. `config` names a tile shape the Python side picks by tensor-core family (`accel.fusion._conv8`).
#include "conv8.cuh"

namespace {

// -1 when the config does not serve the shape (input channels not whole K steps) or this card's shared memory.
template <typename OutT, int BM, int BN, int BK, int WM, int WN, int STAGES>
int run(const conv8::ConvParams &p, cudaStream_t stream) {
  if (!conv8::conv_fits<BK, STAGES>(p.C)) return -1;
  int device = 0, optin = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  if (conv8::conv_smem<OutT, BM, BN, BK, WM, WN, STAGES>() > static_cast<size_t>(optin)) return -1;
  return conv8::launch_conv<OutT, BM, BN, BK, WM, WN, STAGES>(p, stream);
}

template <typename OutT>
int conv_config(const conv8::ConvParams &p, int config, cudaStream_t stream) {
  switch (config) {
    // 128x128 output tiles, eight 64x32 warps: K steps of 128 bytes in three stages (99 KB, every card), or of
    // 64 bytes in four (65 KB), or of 128 bytes in four (132 KB: sm80, sm90 and sm100's larger shared memory).
    case 0: return run<OutT, 128, 128, 128, 64, 32, 3>(p, stream);
    case 1: return run<OutT, 128, 128, 64, 64, 32, 4>(p, stream);
    case 2: return run<OutT, 128, 128, 128, 64, 32, 4>(p, stream);
    default: return -1;
  }
}

}  // namespace

// 0 when it ran, -1 when `config` does not serve the shape on this card, else CUDA's error.
extern "C" int cozy_conv8(const int8_t *x, const float *xs, const int8_t *w, const float *ws, const float *bias,
                          const void *res, void *out, int n, int d, int h, int wd, int c, int k, int t, int r, int s,
                          int pt, int pr, int ps, int z, int p, int q, bool ws_tap, bool half, int config,
                          cudaStream_t stream) {
  conv8::ConvParams prm{};
  prm.x = x, prm.xs = xs, prm.w = w, prm.ws = ws, prm.bias = bias, prm.res = res, prm.out = out;
  prm.N = n, prm.D = d, prm.H = h, prm.W = wd, prm.C = c, prm.K = k, prm.T = t, prm.R = r, prm.S = s;
  prm.pt = pt, prm.pr = pr, prm.ps = ps, prm.Z = z, prm.P = p, prm.Q = q, prm.M = n * z * p * q;
  prm.ws_tap = ws_tap ? 1 : 0;
  return half ? conv_config<__half>(prm, config, stream) : conv_config<__nv_bfloat16>(prm, config, stream);
}

extern "C" int cozy_conv8_stats(const void *x, double *acc, int n, int64_t voxels, int c, int groups, bool half, int sms,
                                cudaStream_t stream) {
  return half ? conv8::launch_gn_stats<__half>(static_cast<const __half *>(x), acc, n, voxels, c, groups, sms, stream)
              : conv8::launch_gn_stats<__nv_bfloat16>(static_cast<const __nv_bfloat16 *>(x), acc, n, voxels, c, groups,
                                                      sms, stream);
}

extern "C" int cozy_conv8_quant(const void *x, const float *a, const float *b, int8_t *q, float *qs, int64_t voxels,
                                int64_t per_n, bool silu, bool half, int sms, cudaStream_t stream) {
  return half ? conv8::launch_norm_quant<__half>(static_cast<const __half *>(x), a, b, q, qs, voxels, per_n, silu,
                                                 true, sms, stream)
              : conv8::launch_norm_quant<__nv_bfloat16>(static_cast<const __nv_bfloat16 *>(x), a, b, q, qs, voxels,
                                                        per_n, silu, true, sms, stream);
}

extern "C" int cozy_conv8_temporal(const void *x, const float *a, const float *b, const float *w, const float *bias,
                                   int8_t *q, float *qs, int n, int t, int hw, bool half, int sms, cudaStream_t stream) {
  return half ? conv8::launch_temporal_quant<__half, 5>(static_cast<const __half *>(x), a, b, w, bias, q, qs, n, t, hw,
                                                        sms, stream)
              : conv8::launch_temporal_quant<__nv_bfloat16, 5>(static_cast<const __nv_bfloat16 *>(x), a, b, w, bias, q,
                                                               qs, n, t, hw, sms, stream);
}
