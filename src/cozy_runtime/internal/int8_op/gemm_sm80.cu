// SPDX-License-Identifier: Apache-2.0
// The int8 GEMM for every card without wgmma or tcgen05 (sm80, sm86, sm89, sm120): CUTLASS 2.x
// mma.sync with the dequant in an EVT epilogue, D = out(acc * scales[m] * w_scale[n] +
// bias[n]). Tile table, shape tree and lean banded stream-K swizzle are comfy-kitchen's
// (`cutlass_gemm_int8.cu`, `cutlass_gemm_common.cuh`, v0.2.37, Apache-2.0, Copyright (c) 2025
// Comfy Org), fitted on Ada and RTX 50 cards, plus vLLM's sm89 64x64 tile (Apache-2.0).
// clang-format off: visitors.hpp needs the GEMM headers first
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
// clang-format on

#include <algorithm>
#include <cmath>

#include "int8.h"

namespace cozy_int8 {
namespace {

using namespace cute;

template <cudaDeviceAttr Attr>
int device_attribute() {
    static int values[64] = {};
    int device = 0;
    cudaGetDevice(&device);
    int& value = values[device & 63];
    if (value == 0) cudaDeviceGetAttribute(&value, Attr, device);
    return value;
}

// N tiles per band when the L2 holds one; a card with under 16 MB of L2 takes the width that
// leaves one wave the least to read.
int band_tiles(int retained, int sms, cutlass::gemm::GemmCoord tile) {
    if (device_attribute<cudaDevAttrL2CacheSize>() >= (16 << 20)) return retained;
    return std::max(1, int(std::lround(std::sqrt(double(sms) * tile.m() / tile.n()))));
}

// Stream-K over the partial last wave only, data-parallel elsewhere; tiles walk N in bands so a
// wave's weight slice stays in L2.
template <int BandN>
struct LeanStreamK {
    using StreamkFeature = void;
    template <typename GemmKernel>
    struct KernelTraits {};
    enum ReductionStrategy { kNone, kAtomic, kMixed };
    static constexpr ReductionStrategy kReductionStrategy = kAtomic;

    cutlass::gemm::GemmCoord problem_size, tiled_shape_;
    int iterations_per_tile, sk_blocks, sk_iterations, avail_sms, dp_blocks;
    int dp_first_wave_tiles = 1, reduction_blocks = 0, sk_tiles, sk_waves;
    bool cohort_raster = false;
    int band = BandN;

    LeanStreamK() = default;
    LeanStreamK(cutlass::gemm::GemmUniversalMode, cutlass::gemm::GemmCoord problem,
                cutlass::gemm::GemmCoord tile, int, int, int device_sms, int available_sms, size_t,
                size_t, size_t, int)
        : problem_size(problem),
          tiled_shape_((problem.m() + tile.m() - 1) / tile.m(),
                       (problem.n() + tile.n() - 1) / tile.n(), 1),
          iterations_per_tile((problem.k() + tile.k() - 1) / tile.k()),
          avail_sms(available_sms < 0 || available_sms > device_sms ? device_sms
                                                                     : available_sms) {
        band = band_tiles(BandN, avail_sms, tile);
        const int tiles = tiled_shape_.m() * tiled_shape_.n();
        const int partial = tiles % avail_sms;
        if (partial == 0) {
            sk_tiles = sk_blocks = sk_waves = 0;
            dp_blocks = tiles;
        } else {
            sk_tiles = tiles < avail_sms ? tiles : avail_sms + partial;
            sk_iterations = sk_tiles * iterations_per_tile;
            sk_blocks = std::max(1, std::min(sk_iterations / 2, avail_sms));
            sk_waves = (sk_blocks + avail_sms - 1) / avail_sms;
            dp_blocks = tiles - sk_tiles;
        }
        sk_iterations = sk_tiles * iterations_per_tile;
    }

    CUTLASS_HOST_DEVICE cutlass::gemm::GemmCoord tiled_shape() const { return tiled_shape_; }
    CUTLASS_HOST_DEVICE int iters_per_tile() const { return iterations_per_tile; }
    CUTLASS_HOST_DEVICE int sk_regions() const { return sk_blocks == 0 ? 0 : 1; }
    CUTLASS_HOST_DEVICE int sk_blocks_per_region() const { return sk_blocks; }
    dim3 get_grid_dims() const { return dim3(sk_waves * avail_sms + dp_blocks, 1, 1); }
    CUTLASS_DEVICE int get_sk_tile_idx(int iteration) const { return iteration / iterations_per_tile; }
    CUTLASS_DEVICE int get_block_idx() const { return blockIdx.x; }

    CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int tile) const {
        int m, n;
        if (tiled_shape_.m() < tiled_shape_.n()) {
            n = tile / tiled_shape_.m();
            m = tile - n * tiled_shape_.m();
        } else if (band > 0) {
            const int b = tile / (band * tiled_shape_.m());
            const int width = min(band, tiled_shape_.n() - b * band);
            const int r = tile - b * band * tiled_shape_.m();
            m = r / width;
            n = b * band + (r - m * width);
        } else {
            m = tile / tiled_shape_.n();
            n = tile - m * tiled_shape_.n();
        }
        return {m, n, 0};
    }

    CUTLASS_DEVICE int get_sk_block_idx(int iteration) const {
        const int small = sk_iterations / sk_blocks;
        const int big_blocks = sk_iterations - small * sk_blocks;
        if (iteration < big_blocks * (small + 1)) return iteration / (small + 1);
        return big_blocks + (iteration - big_blocks * (small + 1)) / small;
    }

    CUTLASS_DEVICE void get_iter_extents(int block, int& begin, int& end) const {
        const int small = sk_iterations / sk_blocks;
        const int big_blocks = sk_iterations - small * sk_blocks;
        begin = block * small + min(block, big_blocks);
        end = begin + small + (block < big_blocks ? 1 : 0);
    }

    CUTLASS_DEVICE int get_first_block_idx(int tile, int block) const {
        return tile < sk_tiles ? get_sk_block_idx(tile * iterations_per_tile) : block;
    }
};

using Identity = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using Banded = LeanStreamK<32>;

template <typename Out, int TBM, int TBN, int TBK, int WM, int WN, int WK, int Stages,
          typename Swizzle>
struct Sm80Gemm {
    using TB = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using Warp = cutlass::gemm::GemmShape<WM, WN, WK>;
    static constexpr int AlignC = 16 / sizeof(Out);
    using Map = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, Out, AlignC, 1>;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float,
                                                                        Stride<_1, _0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float,
                                                                        Stride<_0, _1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, Out,
                                                                      Stride<_0, _1, int32_t>>;
    using Mul = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Add = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus, Out, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Scaled = cutlass::epilogue::threadblock::Sm80EVT<
        Mul, cutlass::epilogue::threadblock::Sm80EVT<
                 Mul, cutlass::epilogue::threadblock::VisitorAccFetch, XScale>,
        WScale>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        Map, Out, cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, int64_t>>;
    using EVT = cutlass::epilogue::threadblock::Sm80EVT<
        Store, cutlass::epilogue::threadblock::Sm80EVT<Add, Scaled, Bias>>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t,
        cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16, Out,
        cutlass::layout::RowMajor, AlignC, int32_t, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, TB, Warp, cutlass::gemm::GemmShape<16, 8, 32>, EVT, Swizzle, Stages,
        cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static constexpr bool kStreamK = !std::is_same_v<Swizzle, Identity>;

    static bool run(const int8_t* q, const int8_t* w, const float* xs, const float* ws,
                    const Out* bias, Out* out, int M, int N, int K, Workspace workspace,
                    cudaStream_t s) {
        typename EVT::Arguments callbacks{
            {{{{}, {const_cast<float*>(xs), 0.f, {_1{}, _0{}, M}}, {}},
              {const_cast<float*>(ws), 0.f, {_0{}, _1{}, N}},
              {}},
             {const_cast<Out*>(bias), Out(0), {_0{}, _1{}, N}},
             {}},
            {out, {int64_t(N), _1{}, int64_t(M) * N}}};
        const cutlass::gemm::GemmCoord problem(M, N, K);
        typename Gemm::Arguments args = [&] {
            if constexpr (kStreamK)
                return typename Gemm::Arguments(
                    cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, callbacks,
                    const_cast<int8_t*>(q), const_cast<int8_t*>(w), nullptr, nullptr,
                    int64_t(M) * K, int64_t(N) * K, 0, 0, K, K, 0, 0, -1);
            else
                return typename Gemm::Arguments(
                    cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, callbacks,
                    const_cast<int8_t*>(q), const_cast<int8_t*>(w), nullptr, nullptr,
                    int64_t(M) * K, int64_t(N) * K, 0, 0, K, K, 0, 0);
        }();
        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        const size_t bytes = Gemm::get_workspace_size(args);
        void* scratch = bytes ? workspace.allocate(bytes, workspace.context) : nullptr;
        if (bytes && scratch == nullptr) return false;
        if (gemm.initialize(args, scratch, s) != cutlass::Status::kSuccess) return false;
        return gemm.run(s) == cutlass::Status::kSuccess;
    }
};

template <typename Out>
using Fn = bool (*)(const int8_t*, const int8_t*, const float*, const float*, const Out*, Out*,
                    int, int, int, Workspace, cudaStream_t);

template <typename Out>
const Fn<Out> kConfigs[] = {
    &Sm80Gemm<Out, 128, 256, 64, 64, 64, 64, 3, Identity>::run,  // 0
    &Sm80Gemm<Out, 128, 128, 64, 64, 64, 64, 4, Identity>::run,  // 1
    &Sm80Gemm<Out, 64, 128, 64, 32, 64, 64, 4, Identity>::run,   // 2
    &Sm80Gemm<Out, 16, 128, 64, 16, 64, 64, 4, Identity>::run,   // 3
    &Sm80Gemm<Out, 128, 128, 64, 64, 64, 64, 4, Banded>::run,    // 4
    &Sm80Gemm<Out, 128, 256, 64, 64, 64, 64, 3, Banded>::run,    // 5
    &Sm80Gemm<Out, 64, 64, 128, 32, 64, 64, 5, Identity>::run,   // 6: vLLM's sm89 M64 tile
};
constexpr int kSm80Configs = 7;

// comfy-kitchen's fitted tree (its configs 0, 1, 2, 6, 12, 13 are ours 0-5), wave guard and
// L2 reroute; up to 128 rows vLLM's 64x64 tile, measured as fast on a 4090 and an RTX 5090
// as kitchen's stream-K, which clears its workspace with an extra launch every call.
int tree(int m, int n, int k) {
    const int64_t mn = int64_t(m) * n;
    if (n <= 24832) {
        if (mn <= 1477632) {
            if (mn <= 259072) return k <= 7296 ? 3 : 4;
            return int64_t(n) * k <= 16252928 ? 2 : 4;
        }
        if (mn <= 4193792) return int64_t(n) * k <= 5275648 ? 1 : 4;
        return int64_t(m) * k <= int64_t(n) * 5675 ? 0 : 5;
    }
    if (int64_t(n) * k <= int64_t(m) * 11096) return 0;
    return mn <= 108003328 ? 0 : 5;
}

int choose(int m, int n, int k, int sms) {
    if (m <= 16) return 3;
    if (m <= 128) return 6;
    int config = tree(m, n, k);
    if (config != 0 && config != 1) return config;
    // At ~2k rows the tree's 128x256 can leave a 2.1-wave grid where 128x128 wins.
    const auto cost = [&](int tile_n, double per_tile) {
        const double waves = double(((m + 127) / 128) * ((n + tile_n - 1) / tile_n)) / sms;
        return std::ceil(waves) / waves * per_tile;
    };
    config = cost(256, 1.0) <= cost(128, 1.08) ? 0 : 1;
    // Identity order re-reads the activation per weight column once it outgrows L2.
    const int64_t reroute =
        std::max<int64_t>(int64_t(device_attribute<cudaDevAttrL2CacheSize>()) * 3 / 4, 48 << 20);
    const bool bands = (m + 127) / 128 >= (n + 255) / 256;
    return bands && int64_t(m) * k > reroute ? 5 : config;
}

template <typename Out>
bool run_as(const int8_t* q, const int8_t* w, const float* xs, const float* ws, const void* bias,
            void* out, int M, int N, int K, int sms, int config, Workspace workspace,
            cudaStream_t s) {
    if (config < 0) config = choose(M, N, K, sms);
    if (config >= kSm80Configs) return false;
    return kConfigs<Out>[config](q, w, xs, ws, static_cast<const Out*>(bias),
                                 static_cast<Out*>(out), M, N, K, workspace, s);
}

}  // namespace

bool gemm(const int8_t* q, const int8_t* w, const float* scales, const float* w_scale,
          const void* bias, void* out, bool out_half, int M, int N, int K, int sms, int config,
          Workspace workspace, cudaStream_t stream) {
    return out_half ? run_as<cutlass::half_t>(q, w, scales, w_scale, bias, out, M, N, K, sms,
                                              config, workspace, stream)
                    : run_as<cutlass::bfloat16_t>(q, w, scales, w_scale, bias, out, M, N, K, sms,
                                                  config, workspace, stream);
}

}  // namespace cozy_int8
