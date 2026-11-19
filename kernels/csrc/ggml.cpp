// SPDX-License-Identifier: Apache-2.0
// `_kernels._ggml`: llama.cpp's quantized matmuls (MMVQ, few-row MMQ, MMQ, MMVQ's MUL_MAT_ID; `ggml/`, vendored at
// b12466) for the `ggml.*` lanes, on the caller's buffers: raw device pointers, sizes, a stream and a stream-ordered allocator for
// the temporaries, so neither torch nor the Python version is a build input. Every call returns its CUDA error code:
// 0 ran, NOT_SUPPORTED has no kernel for this card, type or shape; `error()` names a failure ggml raised.
#include <nanobind/nanobind.h>

#include <cuda_runtime.h>

#include <cstdint>

#include "ggml/ids.h"
#include "ggml/matmul.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

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

// Each temporary from the caller's `allocate(bytes)`: an object with `data_ptr()`, held until the call returns.
struct Held {
    nb::handle allocate;
    nb::list objects;
};

void* allocate(size_t bytes, void* context) {
    auto& held = *static_cast<Held*>(context);
    nb::object made = held.allocate(bytes);
    held.objects.append(made);
    return reinterpret_cast<void*>(nb::cast<uintptr_t>(made.attr("data_ptr")()));
}

int kernel(int device, int type, int64_t K, int64_t N, int64_t M, uintptr_t w) {
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return cozy_ggml::kNone;
    return cozy_ggml::kernel(device, type, reinterpret_cast<const void*>(w), K, N, M);
}

int matmul(int device, int type, uintptr_t w, uintptr_t x, int x_type, uintptr_t out, int64_t K, int64_t N, int64_t M,
           nb::handle allocator, uintptr_t stream) {
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    Held held{allocator, {}};
    return cozy_ggml::matmul(device, type, reinterpret_cast<const void*>(w), reinterpret_cast<const void*>(x), x_type,
                             reinterpret_cast<float*>(out), K, N, M, &allocate, &held,
                             reinterpret_cast<cudaStream_t>(stream));
}

int quantize(int device, uintptr_t x, int x_type, uintptr_t q8_1, int64_t K, int64_t M, uintptr_t stream) {
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    return cozy_ggml::quantize(device, reinterpret_cast<const void*>(x), x_type, reinterpret_cast<void*>(q8_1), K, M,
                               reinterpret_cast<cudaStream_t>(stream));
}

int matmul_quantized(int device, int type, uintptr_t w, uintptr_t q8_1, uintptr_t out, int64_t K, int64_t N, int64_t M,
                     nb::handle allocator, uintptr_t stream) {
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    Held held{allocator, {}};
    return cozy_ggml::matmul_quantized(device, type, reinterpret_cast<const void*>(w), reinterpret_cast<const void*>(q8_1),
                                       reinterpret_cast<float*>(out), K, N, M, &allocate, &held,
                                       reinterpret_cast<cudaStream_t>(stream));
}

int ids_batch(int device, int type) {
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return 0;
    return cozy_ggml::ids_batch(device, type);
}

int matmul_ids(int device, int type, uintptr_t table, uintptr_t x, int64_t rows, uintptr_t ids, uintptr_t out, int64_t K,
               int64_t N, int64_t T, int64_t k, nb::handle allocator, uintptr_t stream) {
    const OnDevice guard(device);
    if (guard.error() != cudaSuccess) return guard.error();
    Held held{allocator, {}};
    return cozy_ggml::matmul_ids(device, type, reinterpret_cast<const void*>(table), reinterpret_cast<const float*>(x),
                                 rows, reinterpret_cast<const int32_t*>(ids), reinterpret_cast<float*>(out), K, N, T, k,
                                 &allocate, &held, reinterpret_cast<cudaStream_t>(stream));
}

}  // namespace

NB_MODULE(_ggml, m) {
    m.attr("NOT_SUPPORTED") = int(cudaErrorNotSupported);
    m.attr("OUT_OF_MEMORY") = int(cudaErrorMemoryAllocation);
    m.attr("UNKNOWN") = int(cudaErrorUnknown);
    m.attr("KERNELS") = nb::make_tuple("", "mmvq", "mmq_few", "mmq");
    m.attr("UPSTREAM") = "llama.cpp b12466 (ede6106780fef959e40ae309afb68da7b6442883)";
    m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
    m.def("error", [] { return cozy_ggml::error(); });
    m.def("kernel", &kernel, "device"_a, "type"_a, "K"_a, "N"_a, "M"_a, "w"_a);
    m.attr("ACTIVATIONS") = nb::make_tuple("float32", "bfloat16", "float16");
    m.def("matmul", &matmul, "device"_a, "type"_a, "w"_a, "x"_a, "x_type"_a, "out"_a, "K"_a, "N"_a, "M"_a,
          "allocate"_a.none(), "stream"_a);
    m.def("quantized_bytes", &cozy_ggml::quantized_bytes, "K"_a, "M"_a);
    m.def("quantize", &quantize, "device"_a, "x"_a, "x_type"_a, "q8_1"_a, "K"_a, "M"_a, "stream"_a);
    m.def("ids_batch", &ids_batch, "device"_a, "type"_a);
    m.def("matmul_ids", &matmul_ids, "device"_a, "type"_a, "table"_a, "x"_a, "rows"_a, "ids"_a, "out"_a, "K"_a, "N"_a,
          "T"_a, "k"_a, "allocate"_a.none(), "stream"_a);
    m.def("matmul_quantized", &matmul_quantized, "device"_a, "type"_a, "w"_a, "q8_1"_a, "out"_a, "K"_a, "N"_a, "M"_a,
          "allocate"_a.none(), "stream"_a);
}
