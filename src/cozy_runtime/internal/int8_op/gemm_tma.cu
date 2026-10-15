// SPDX-License-Identifier: Apache-2.0
// The int8 GEMM on TMA cards, CUTLASS 3.x with the dequant in an EVT epilogue,
// D = out(acc * scales[m] * w_scale[n] + bias[n]); COZY_SM picks the card's table.
// Hopper (90): warp-specialized wgmma, tiles after vLLM's `scaled_mm_sm90_int8_dispatch.cuh`
// (Apache-2.0). Datacenter Blackwell (100): tcgen05, after SGLang's `convrot_int8` (Apache-2.0).
#include <cuda.h>

// CUTLASS asks the runtime for the driver's tensor-map encoder on every TMA descriptor; ask
// once. Defined before CUTLASS so its direct-call wrapper calls this.
namespace cozy_int8 {
inline CUresult encode_tiled(CUtensorMap* map, CUtensorMapDataType type, cuuint32_t rank,
                             void* address, const cuuint64_t* dims, const cuuint64_t* strides,
                             const cuuint32_t* box, const cuuint32_t* element_strides,
                             CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                             CUtensorMapL2promotion promotion, CUtensorMapFloatOOBfill fill) {
    using Fn = decltype(&cuTensorMapEncodeTiled);
    static Fn fn = [] {
        void* found = nullptr;
        cudaDriverEntryPointQueryResult status;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &found, 12000,
                                         cudaEnableDefault, &status);
        return reinterpret_cast<Fn>(found);
    }();
    return fn(map, type, rank, address, dims, strides, box, element_strides, interleave, swizzle,
              promotion, fill);
}
}  // namespace cozy_int8
#define CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 1
#define cuTensorMapEncodeTiled cozy_int8::encode_tiled

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <atomic>

#include "int8.h"

namespace cozy_int8 {
namespace {

using namespace cute;
using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using Cooperative = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Basic = cutlass::gemm::KernelTmaWarpSpecialized;

#if COZY_SM == 100
using Arch = cutlass::arch::Sm100;
using OneSm = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using TwoSm = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
template <typename Schedule>
using EpilogueOf = std::conditional_t<std::is_same_v<Schedule, TwoSm>,
                                      cutlass::epilogue::TmaWarpSpecialized2Sm,
                                      cutlass::epilogue::TmaWarpSpecialized1Sm>;
#else
using Arch = cutlass::arch::Sm90;
template <typename Schedule>
using EpilogueOf = std::conditional_t<std::is_same_v<Schedule, Cooperative>,
                                      cutlass::epilogue::TmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecialized>;
#endif

template <typename Out, typename Tile, typename Cluster, typename Schedule>
struct TmaGemm {
    using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, Tile, float, float,
                                                                Stride<_1, _0, _0>>;
    using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, Tile, float, float,
                                                                Stride<_0, _1, _0>>;
    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, Tile, Out, Out, Stride<_0, _1, _0>>;
    using Scaled = cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float,
                                               cutlass::FloatRoundStyle::round_to_nearest>,
        WScale, cutlass::epilogue::fusion::Sm90AccFetch>;
    using EVT = cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<cutlass::multiply_add, Out, float,
                                               cutlass::FloatRoundStyle::round_to_nearest>,
        XScale, Scaled, Bias>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, int32_t, float, void,
        cutlass::layout::RowMajor, 16 / sizeof(Out), Out, cutlass::layout::RowMajor,
        16 / sizeof(Out), EpilogueOf<Schedule>, EVT>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp, int8_t, cutlass::layout::RowMajor,
        16, int8_t, cutlass::layout::ColumnMajor, 16, int32_t, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        Schedule>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static bool run(const int8_t* q, const int8_t* w, const float* xs, const float* ws,
                    const Out* bias, Out* out, int M, int N, int K, int sms, cudaStream_t s) {
        static std::atomic<uint64_t> configured{0};
        int device = 0;
        cudaGetDevice(&device);
        const uint64_t bit = uint64_t(1) << (device & 63);
        if (!(configured.load(std::memory_order_relaxed) & bit)) {
            if (cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     Kernel::SharedStorageSize) != cudaSuccess)
                return false;
            configured.fetch_or(bit);
        }
        using StrideA = typename Kernel::StrideA;
        using StrideB = typename Kernel::StrideB;
        using StrideD = typename Kernel::StrideD;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {q, cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1)), w,
             cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1))},
            {{}, nullptr, {}, out,
             cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1))}};
        args.epilogue.thread = {{xs}, {{ws}, {}, {}}, {bias}, {}};
        args.hw_info.device_id = device;
        args.hw_info.sm_count = sms;
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess) return false;
        auto params = Kernel::to_underlying_arguments(args, nullptr);
        // A programmatic dependent of the rotate kernel before it (`rotate.cu`).
        return Gemm::run(params, s, nullptr, true) == cutlass::Status::kSuccess;
    }
};

template <typename Out>
using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const Out*, Out*,
                    int, int, int, int, cudaStream_t);

#if COZY_SM == 100
// Measured on a B200 at our shapes (SGLang's 128x128 and 128x256 one-SM tiles ran 10-20% below
// the two-SM 256x256): small problems take the cluster-free 64x128.
template <typename Out>
const Fn<Out> kConfigs[] = {
    &TmaGemm<Out, Shape<_256, _256, _128>, Shape<_2, _1, _1>, TwoSm>::run,  // 0
    &TmaGemm<Out, Shape<_256, _128, _128>, Shape<_2, _1, _1>, TwoSm>::run,  // 1
    &TmaGemm<Out, Shape<_64, _128, _128>, Shape<_1, _1, _1>, OneSm>::run,   // 2
};
constexpr int kTmaConfigs = 3;

int choose(int M, int N, int K) {
    return M <= 128 || int64_t(N) * K <= (4 << 20) ? 2 : 0;
}
#else
// Measured on an H100 SXM at our shapes against vLLM's table (`scaled_mm_sm90_int8_dispatch`,
// whose 128x128 2x1-cluster ping-pong is 0) and cooperative wide tiles: no cluster is as fast
// below ~4k rows and its launch costs the host ~2 us less.
template <typename Out>
const Fn<Out> kConfigs[] = {
    &TmaGemm<Out, Shape<_128, _128, _128>, Shape<_2, _1, _1>, Pingpong>::run,     // 0
    &TmaGemm<Out, Shape<_64, _64, _256>, Shape<_1, _1, _1>, Basic>::run,          // 1
    &TmaGemm<Out, Shape<_128, _128, _128>, Shape<_1, _1, _1>, Pingpong>::run,     // 2
    &TmaGemm<Out, Shape<_128, _256, _128>, Shape<_1, _1, _1>, Cooperative>::run,  // 3
};
constexpr int kTmaConfigs = 4;

int choose(int M, int N, int K) {
    if (M <= 128) return 1;
    if (K >= 4 * N && K >= 5120) return 3;
    return M >= 4096 && std::min(N, K) >= 3072 ? 0 : 2;
}
#endif

template <typename Out>
bool run_as(const int8_t* q, const int8_t* w, const float* xs, const float* ws, const void* bias,
            void* out, int M, int N, int K, int sms, int config, cudaStream_t s) {
    if (config < 0) config = choose(M, N, K);
    if (config >= kTmaConfigs) return false;
    return kConfigs<Out>[config](q, w, xs, ws, static_cast<const Out*>(bias),
                                 static_cast<Out*>(out), M, N, K, sms, s);
}

}  // namespace

bool gemm(const int8_t* q, const int8_t* w, const float* scales, const float* w_scale,
          const void* bias, void* out, bool out_half, int M, int N, int K, int sms, int config,
          Workspace, cudaStream_t stream) {
    return out_half ? run_as<cutlass::half_t>(q, w, scales, w_scale, bias, out, M, N, K, sms,
                                              config, stream)
                    : run_as<cutlass::bfloat16_t>(q, w, scales, w_scale, bias, out, M, N, K, sms,
                                                  config, stream);
}

}  // namespace cozy_int8
