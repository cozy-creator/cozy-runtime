// SPDX-License-Identifier: Apache-2.0
// `_kernels._gemm`: the 8-bit linears (`gemm/int8.h`) for every card in one module. Each
// tensor-core family's GEMMs are compiled for that family's targets only (`CMakeLists.txt`) and
// the caller names the device's family (`FAMILIES`). Arguments are raw device pointers, sizes
// and a stream, so neither torch nor the Python version is a build input; the caller allocates
// and checks. Every call returns its CUDA error code: 0 ran, NOT_SUPPORTED has no kernel for
// this card or shape, OUT_OF_MEMORY is the device's.
#include <nanobind/nanobind.h>

#include <cuda_runtime.h>

#include <cstdint>

#include "int8.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// `device` current for one call and the caller's restored after: torch keeps its own.
class OnDevice {
  public:
    explicit OnDevice(int device) {
        error_ = cudaGetDevice(&previous_);
        if (error_ == cudaSuccess && previous_ != device) error_ = cudaSetDevice(device);
    }
    ~OnDevice() {
        int now = -1;
        if (error_ == cudaSuccess && cudaGetDevice(&now) == cudaSuccess && now != previous_)
            cudaSetDevice(previous_);
    }
    cudaError_t error() const { return error_; }

  private:
    int previous_ = -1;
    cudaError_t error_ = cudaSuccess;
};

int sm_count(int device) {
    static int counts[64] = {};
    int& count = counts[device & 63];
    if (count == 0) cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    return count;
}

template <typename T>
T* at(uintptr_t address) {
    return reinterpret_cast<T*>(address);
}

cudaStream_t on(uintptr_t stream) { return reinterpret_cast<cudaStream_t>(stream); }

// A GEMM's scratch (stream-K partials) from the caller's `allocate(bytes)`, which returns an
// object with `data_ptr()` that stays alive until the call returns (a stream-ordered allocator).
struct Scratch {
    nb::handle allocate;
    nb::object held;
};

void* allocate(size_t bytes, void* context) {
    auto& scratch = *static_cast<Scratch*>(context);
    if (scratch.allocate.is_none()) return nullptr;
    scratch.held = scratch.allocate(bytes);
    return at<void>(nb::cast<uintptr_t>(scratch.held.attr("data_ptr")()));
}

#ifdef COZY_TMA
constexpr cozy_int8::Gemm* kInt8[] = {&cozy_int8::mma_sync::gemm, &cozy_int8::wgmma::gemm,
                                      &cozy_int8::tcgen05::gemm};
constexpr cozy_int8::Fp8Gemm* kFp8[] = {&cozy_int8::mma_sync::fp8_gemm,
                                        &cozy_int8::wgmma::fp8_gemm, &cozy_int8::tcgen05::fp8_gemm};
#else
// Without CUTLASS's TMA GEMMs (an MSVC build): int8 on mma.sync only (`_kernels.GEMM`).
constexpr cozy_int8::Gemm* kInt8[] = {&cozy_int8::mma_sync::gemm, nullptr, nullptr};
constexpr cozy_int8::Fp8Gemm* kFp8[] = {nullptr, nullptr, nullptr};
#endif

// The activation rows' width each format takes: the kernels index registers by it.
bool fits(bool fp8, int K) { return K > 0 && K <= cozy_int8::kMaxK && K % (fp8 ? 16 : 256) == 0; }

int quantize(int device, bool fp8, uintptr_t x, bool x_half, uintptr_t q, uintptr_t scales,
             int M, int K, uintptr_t scale, uintptr_t shift, int64_t scale_stride,
             int64_t shift_stride, int rows_per_entry, int norm, float eps, int activation,
             uintptr_t stream) {
    if (!fits(fp8, K) || M < 0 || norm < cozy_int8::kNormNone || norm > cozy_int8::kNormRms ||
        activation < cozy_int8::kActNone || activation > cozy_int8::kActGeluTanh ||
        rows_per_entry < 1)
        return cudaErrorInvalidValue;
    if (M == 0) return cudaSuccess;
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    cozy_int8::Prologue prologue;
    prologue.scale = at<const void>(scale);
    prologue.shift = at<const void>(shift);
    prologue.scale_stride = scale_stride;
    prologue.shift_stride = shift_stride;
    prologue.rows_per_entry = rows_per_entry;
    prologue.norm = norm;
    prologue.eps = eps;
    prologue.activation = activation;
    if (fp8)
        return cozy_int8::fp8_quantize(at<const void>(x), x_half, at<void>(q), at<float>(scales),
                                       M, K, prologue, on(stream));
    return cozy_int8::rotate_quantize(at<const void>(x), x_half, at<int8_t>(q), at<float>(scales),
                                      M, K, prologue, on(stream));
}

int gemv(int device, uintptr_t x, bool x_half, uintptr_t w, uintptr_t w_scale, uintptr_t bias,
         uintptr_t out, bool out_half, int M, int N, int K, uintptr_t stream) {
    if (!fits(false, K) || N < 1) return cudaErrorInvalidValue;
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    return cozy_int8::gemv(at<const void>(x), x_half, at<const int8_t>(w), at<const float>(w_scale),
                           at<const void>(bias), at<void>(out), out_half, M, N, K,
                           sm_count(device), on(stream));
}

int gemm(int device, int family, bool fp8, uintptr_t q, uintptr_t scales, uintptr_t w,
         uintptr_t w_scale, uintptr_t bias, uintptr_t out, bool out_half, int M, int N, int K,
         int config, nb::handle allocator, uintptr_t stream) {
    if (family < 0 || family > 2 || M < 1 || N < 1 || !fits(fp8, K)) return cudaErrorInvalidValue;
    if (fp8 ? kFp8[family] == nullptr : kInt8[family] == nullptr) return cudaErrorNotSupported;
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    const int sms = sm_count(device);
    if (fp8)
        return kFp8[family](at<const void>(q), at<const void>(w), at<const float>(scales),
                            at<const float>(w_scale), at<const void>(bias), at<void>(out),
                            out_half, M, N, K, sms, config, on(stream));
    Scratch scratch{allocator, {}};
    return kInt8[family](at<const int8_t>(q), at<const int8_t>(w), at<const float>(scales),
                         at<const float>(w_scale), at<const void>(bias), at<void>(out), out_half,
                         M, N, K, sms, config, {&allocate, &scratch}, on(stream));
}

int w4a8_gemm(int device, uintptr_t q, uintptr_t scales, uintptr_t w, uintptr_t s,
              uintptr_t codebook, uintptr_t w_scale, uintptr_t bias, uintptr_t out, bool out_half,
              int M, int N, int K, int config, uintptr_t stream) {
    if (M < 1 || N < 8 || N % 8 || !fits(false, K)) return cudaErrorInvalidValue;
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    return cozy_int8::w4a8_gemm(at<const int8_t>(q), at<const uint8_t>(w), at<const uint8_t>(s),
                                at<const float>(codebook), at<const float>(scales),
                                at<const float>(w_scale), at<const void>(bias), at<void>(out),
                                out_half, M, N, K, config, on(stream));
}

int w4a8_unpack(int device, uintptr_t w, uintptr_t s, uintptr_t codebook, uintptr_t out, int N,
                int K, uintptr_t stream) {
    if (N < 1 || !fits(false, K)) return cudaErrorInvalidValue;
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    return cozy_int8::w4a8_unpack(at<const uint8_t>(w), at<const uint8_t>(s),
                                  at<const float>(codebook), at<int8_t>(out), N, K,
                                  sm_count(device), on(stream));
}

}  // namespace

NB_MODULE(_gemm, m) {
    m.attr("NOT_SUPPORTED") = int(cudaErrorNotSupported);
    m.attr("OUT_OF_MEMORY") = int(cudaErrorMemoryAllocation);
    m.attr("GEMV_ROWS") = cozy_int8::kGemvRows;
    m.attr("MAX_K") = cozy_int8::kMaxK;
#ifdef COZY_TMA
    m.attr("TMA") = true;
#else
    m.attr("TMA") = false;
#endif
    m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
    m.def("quantize", &quantize, "device"_a, "fp8"_a, "x"_a, "x_half"_a, "q"_a, "scales"_a, "M"_a,
          "K"_a, "scale"_a, "shift"_a, "scale_stride"_a, "shift_stride"_a, "rows_per_entry"_a,
          "norm"_a, "eps"_a, "activation"_a, "stream"_a);
    m.def("gemv", &gemv, "device"_a, "x"_a, "x_half"_a, "w"_a, "w_scale"_a, "bias"_a, "out"_a,
          "out_half"_a, "M"_a, "N"_a, "K"_a, "stream"_a);
    m.def("w4a8_gemm", &w4a8_gemm, "device"_a, "q"_a, "scales"_a, "w"_a, "s"_a, "codebook"_a,
          "w_scale"_a, "bias"_a, "out"_a, "out_half"_a, "M"_a, "N"_a, "K"_a, "config"_a, "stream"_a);
    m.def("w4a8_unpack", &w4a8_unpack, "device"_a, "w"_a, "s"_a, "codebook"_a, "out"_a, "N"_a,
          "K"_a, "stream"_a);
    m.def("gemm", &gemm, "device"_a, "family"_a, "fp8"_a, "q"_a, "scales"_a, "w"_a, "w_scale"_a,
          "bias"_a, "out"_a, "out_half"_a, "M"_a, "N"_a, "K"_a, "config"_a, "allocate"_a.none(),
          "stream"_a);
}
