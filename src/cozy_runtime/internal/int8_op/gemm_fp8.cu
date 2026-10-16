// SPDX-License-Identifier: Apache-2.0
// The fp8-rowwise GEMM on TMA cards (`gemm_tma.cuh`): E4M3 operands, f32 accumulation.
#include "gemm_tma.cuh"

namespace cozy_int8 {
namespace {

using E4M3 = cutlass::float_e4m3_t;

#ifdef COZY_TCGEN05
// The int8 table's tcgen05 tiles, measured on a B300 at our shapes: the two-SM 256x256 runs
// 2.5-2.9 PF where its tiles fill 90% of their last wave, the 256x128 leads where they do not
// (1536-4096 rows, SDXL), and the one-SM 64x128 up to 256 rows.
template <typename Out>
const Fn kConfigs[3] = {
    &TmaGemm<E4M3, Out, Shape<_256, _256, _128>, Shape<_2, _1, _1>, TwoSm>::run,  // 0
    &TmaGemm<E4M3, Out, Shape<_256, _128, _128>, Shape<_2, _1, _1>, TwoSm>::run,  // 1
    &TmaGemm<E4M3, Out, Shape<_64, _128, _128>, Shape<_1, _1, _1>, OneSm>::run,   // 2
};

int choose(int M, int N, int, int sms) {
    if (M <= 256) return 2;
    const auto filled = [&](int tile_n) {
        const int tiles = ((M + 255) / 256) * ((N + tile_n - 1) / tile_n), pairs = sms / 2;
        return double(tiles) / (double((tiles + pairs - 1) / pairs) * pairs);
    };
    return filled(256) < 0.9 && filled(128) >= filled(256) ? 1 : 0;
}
#elif defined(COZY_WGMMA)
// vLLM's `scaled_mm_sm90_fp8_dispatch.cuh` (Apache-2.0) up to 128 rows, then measured on an H100
// SXM at our shapes against cuBLASLt's `_scaled_mm`: the 2x1-cluster cooperative 128x256 matches
// it at H3's 52k rows (1.24-1.32 PF), the 1x2-cluster ping-pong leads the up projections, the
// cluster-free cooperative the down projections.
template <typename Out>
const Fn kConfigs[6] = {
    &TmaGemm<E4M3, Out, Shape<_128, _128, _128>, Shape<_2, _1, _1>, Pingpong>::run,     // 0
    &TmaGemm<E4M3, Out, Shape<_64, _128, _128>, Shape<_2, _1, _1>, Pingpong>::run,      // 1
    &TmaGemm<E4M3, Out, Shape<_64, _64, _128>, Shape<_1, _8, _1>, Basic>::run,          // 2
    &TmaGemm<E4M3, Out, Shape<_128, _256, _128>, Shape<_1, _1, _1>, Cooperative>::run,  // 3
    &TmaGemm<E4M3, Out, Shape<_128, _256, _128>, Shape<_2, _1, _1>, Cooperative>::run,  // 4
    &TmaGemm<E4M3, Out, Shape<_128, _128, _128>, Shape<_1, _2, _1>, Pingpong>::run,     // 5
};

int choose(int M, int N, int K, int) {
    if (M <= 64) return 2;
    if (M <= 128) return 1;
    if (M >= 16384) return 4;
    if (K >= 4 * N && K >= 5120) return 3;
    return N >= 2 * K ? 5 : 0;
}
#else
// Block-scaled mma.sync (RTX 50): vLLM's sm120 tile (`scaled_mm_sm120_fp8_dispatch.cuh`, Apache-2.0); no clusters on RTX 50.
template <typename Out>
const Fn kConfigs[1] = {
    &TmaGemm<E4M3, Out, Shape<_128, _128, _128>, Shape<_1, _1, _1>, Cooperative>::run,  // 0
};

int choose(int, int, int, int) { return 0; }
#endif

}  // namespace

cudaError_t fp8_gemm(const void* q, const void* w, const float* scales, const float* w_scale,
                     const void* bias, void* out, bool out_half, int M, int N, int K, int sms,
                     int config, cudaStream_t stream) {
    return launch(kConfigs<cutlass::bfloat16_t>, kConfigs<cutlass::half_t>,
                  config < 0 ? choose(M, N, K, sms) : config, out_half, q, w, scales, w_scale,
                  bias, out, M, N, K, sms, stream);
}

}  // namespace cozy_int8
