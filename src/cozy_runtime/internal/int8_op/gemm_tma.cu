// SPDX-License-Identifier: Apache-2.0
// The int8 GEMM on TMA cards (`gemm_tma.cuh`). Hopper (90): tiles after vLLM's
// `scaled_mm_sm90_int8_dispatch.cuh` (Apache-2.0). Datacenter Blackwell (100): tcgen05, after
// SGLang's `convrot_int8` (Apache-2.0); a tcgen05 row without int8 (B300) builds none.
#include <algorithm>

#include "gemm_tma.cuh"

namespace cozy_int8::COZY_FAMILY {
namespace {

#ifdef COZY_TCGEN05
// Measured on a B200 at our shapes (SGLang's 128x128 and 128x256 one-SM tiles ran 10-20% below
// the two-SM 256x256): small problems take the cluster-free 64x128.
template <typename Out>
const Fn kConfigs[3] = {
    &TmaGemm<int8_t, Out, Shape<_256, _256, _128>, Shape<_2, _1, _1>, TwoSm>::run,  // 0
    &TmaGemm<int8_t, Out, Shape<_256, _128, _128>, Shape<_2, _1, _1>, TwoSm>::run,  // 1
    &TmaGemm<int8_t, Out, Shape<_64, _128, _128>, Shape<_1, _1, _1>, OneSm>::run,   // 2
};
int choose(int M, int N, int K) {
    return M <= 128 || int64_t(N) * K <= (4 << 20) ? 2 : 0;
}
#else
// Measured on an H100 SXM at our shapes against vLLM's table (`scaled_mm_sm90_int8_dispatch`,
// whose 128x128 2x1-cluster ping-pong is 0) and cooperative wide tiles: no cluster is as fast
// below ~4k rows and its launch costs the host ~2 us less.
template <typename Out>
const Fn kConfigs[4] = {
    &TmaGemm<int8_t, Out, Shape<_128, _128, _128>, Shape<_2, _1, _1>, Pingpong>::run,     // 0
    &TmaGemm<int8_t, Out, Shape<_64, _64, _256>, Shape<_1, _1, _1>, Basic>::run,          // 1
    &TmaGemm<int8_t, Out, Shape<_128, _128, _128>, Shape<_1, _1, _1>, Pingpong>::run,     // 2
    &TmaGemm<int8_t, Out, Shape<_128, _256, _128>, Shape<_1, _1, _1>, Cooperative>::run,  // 3
};
int choose(int M, int N, int K) {
    if (M <= 128) return 1;
    // 16k+ rows (H3's 52k, SeedVR2's 16k): the cooperative 128x256, 1.1-1.45x the 128x128 there.
    if (M >= 16384 || (K >= 4 * N && K >= 5120)) return 3;
    return M >= 4096 && std::min(N, K) >= 3072 ? 0 : 2;
}
#endif

}  // namespace

cudaError_t gemm(const int8_t* q, const int8_t* w, const float* scales, const float* w_scale,
                 const void* bias, void* out, bool out_half, int M, int N, int K, int sms,
                 int config, Workspace, cudaStream_t stream) {
    return launch(kConfigs<cutlass::bfloat16_t>, kConfigs<cutlass::half_t>,
                  config < 0 ? choose(M, N, K) : config, out_half, q, w, scales, w_scale, bias,
                  out, M, N, K, sms, stream);
}

}  // namespace cozy_int8::COZY_FAMILY
