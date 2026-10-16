// SPDX-License-Identifier: Apache-2.0
// The 8-bit GEMM on TMA cards, CUTLASS 3.x with the dequant in an EVT epilogue,
// D = out(acc * scales[m] * w_scale[n] + bias[n]), for int8 (`gemm_tma.cu`) and E4M3
// (`gemm_fp8.cu`) operands; the row's family picks the builder: wgmma warp-specialized,
// tcgen05, or block-scaled mma.sync (RTX 50) through its TMA warp-specialized mma.
#pragma once

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

#include <atomic>
#include <type_traits>

#include "int8.h"

namespace cozy_int8 {

using namespace cute;
using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using Cooperative = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Basic = cutlass::gemm::KernelTmaWarpSpecialized;

template <typename Schedule>
struct Mainloop {
    using type = Schedule;
};

#ifdef COZY_TCGEN05
using Arch = cutlass::arch::Sm100;
using OneSm = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using TwoSm = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
template <typename Schedule>
using EpilogueOf = std::conditional_t<std::is_same_v<Schedule, TwoSm>,
                                      cutlass::epilogue::TmaWarpSpecialized2Sm,
                                      cutlass::epilogue::TmaWarpSpecialized1Sm>;
#else
#ifdef COZY_MMA_SYNC
using Arch = cutlass::arch::Sm120;
#else
using Arch = cutlass::arch::Sm90;
// Hopper's fp8 wgmma keeps an fp32 sum only by promoting every few k-blocks; the fp8 leaf's
// law is the fast accumulation `_scaled_mm(use_fast_accum=True)` measured (#517b).
template <>
struct Mainloop<Pingpong> {
    using type = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
};
template <>
struct Mainloop<Cooperative> {
    using type = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
};
template <>
struct Mainloop<Basic> {
    using type = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
};
#endif
template <typename Schedule>
using EpilogueOf = std::conditional_t<std::is_same_v<Schedule, Cooperative>,
                                      cutlass::epilogue::TmaWarpSpecializedCooperative,
                                      cutlass::epilogue::TmaWarpSpecialized>;
#endif

// One tile's launcher: `(q, w, scales, w_scale, bias, out, M, N, K, sms, stream)`.
using Fn = cudaError_t (*)(const void*, const void*, const float*, const float*, const void*,
                           void*, int, int, int, int, cudaStream_t);

template <typename In, typename Out, typename Tile, typename Cluster, typename Schedule>
struct TmaGemm {
    static constexpr bool kInt8 = std::is_same_v<In, int8_t>;
    using Acc = std::conditional_t<kInt8, int32_t, float>;
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
        cutlass::epilogue::collective::EpilogueTileAuto, Acc, float, void,
        cutlass::layout::RowMajor, 16 / sizeof(Out), Out, cutlass::layout::RowMajor,
        16 / sizeof(Out), EpilogueOf<Schedule>, EVT>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp, In, cutlass::layout::RowMajor, 16, In,
        cutlass::layout::ColumnMajor, 16, Acc, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        std::conditional_t<kInt8, Schedule, typename Mainloop<Schedule>::type>>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    // The adapter's launch, keeping its error: the shared-memory opt-in once per card, then one
    // `cudaLaunchKernelEx` with the cluster and the programmatic dependency on the quantize
    // kernel before it (`rotate.cu`).
    static cudaError_t run(const void* q, const void* w, const float* xs, const float* ws,
                           const void* bias, void* out, int M, int N, int K, int sms,
                           cudaStream_t s) {
        static std::atomic<uint64_t> configured{0};
        int device = 0;
        cudaGetDevice(&device);
        const uint64_t bit = uint64_t(1) << (device & 63);
        if (!(configured.load(std::memory_order_relaxed) & bit)) {
            const cudaError_t e = cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       Kernel::SharedStorageSize);
            if (e != cudaSuccess) return e;
            configured.fetch_or(bit);
        }
        using StrideA = typename Kernel::StrideA;
        using StrideB = typename Kernel::StrideB;
        using StrideD = typename Kernel::StrideD;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {static_cast<const In*>(q),
             cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1)),
             static_cast<const In*>(w),
             cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1))},
            {{}, nullptr, {}, static_cast<Out*>(out),
             cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1))}};
        args.epilogue.thread = {{xs}, {{ws}, {}, {}}, {static_cast<const Out*>(bias)}, {}};
        args.hw_info.device_id = device;
        args.hw_info.sm_count = sms;
        if (!Kernel::can_implement(args)) return cudaErrorNotSupported;
        auto params = Kernel::to_underlying_arguments(args, nullptr);
        cudaLaunchAttribute attributes[2];
        attributes[0].id = cudaLaunchAttributeClusterDimension;
        attributes[0].val.clusterDim = {unsigned(size<0>(Cluster{})), unsigned(size<1>(Cluster{})), 1};
        attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{Kernel::get_grid_shape(params), Kernel::get_block_shape(),
                                  size_t(Kernel::SharedStorageSize), s, attributes, 2};
        return cudaLaunchKernelEx(&config, cutlass::device_kernel<Kernel>, params);
    }
};

// `config` of a table's bf16 or fp16 half, or the shape's `choose` when config < 0.
template <int Count>
cudaError_t launch(const Fn (&bf16)[Count], const Fn (&fp16)[Count], int config, bool out_half,
                   const void* q, const void* w, const float* xs, const float* ws,
                   const void* bias, void* out, int M, int N, int K, int sms, cudaStream_t s) {
    if (config >= Count) return cudaErrorNotSupported;
    return (out_half ? fp16 : bf16)[config](q, w, xs, ws, bias, out, M, N, K, sms, s);
}

}  // namespace cozy_int8
