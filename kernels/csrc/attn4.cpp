// SPDX-License-Identifier: Apache-2.0
// `_kernels._attn4`: INT4 Q·Kᵀ attention on Ampere and Ada, the cards with s4 tensor cores
// (`attn4/fwd.cu`, `attn4/quant.cu`). Arguments are raw device pointers, element strides and a
// stream; the caller allocates (`cozy_runtime.internal.attention_int4`). Returns the CUDA error
// code.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include "attn4/attn4.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

int attention(int device, uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, uintptr_t q4,
              uintptr_t q_scale, uintptr_t k4, uintptr_t k_scale, uintptr_t k_sum,
              uintptr_t v8, uintptr_t v_scale, int batch, int heads, int heads_k, int len_q,
              int len_k, int cta_k, bool bf16, float scale, cozy::attn4::Strides q_strides,
              cozy::attn4::Strides k_strides, cozy::attn4::Strides v_strides,
              cozy::attn4::Strides out_strides, uintptr_t stream) {
  const auto at = [](uintptr_t p) { return reinterpret_cast<void *>(p); };
  const auto floats = [](uintptr_t p) { return reinterpret_cast<float *>(p); };
  const cozy::attn4::Call call{
      at(q),         at(k),          at(v),         at(out),    at(q4),    at(k4),
      at(v8),        floats(q_scale), floats(k_scale), floats(v_scale), floats(k_sum),
      batch,         heads,          heads_k,       len_q,      len_k,     cta_k,
      bf16,          scale,          q_strides,     k_strides,  v_strides, out_strides};
  return cozy::attn4::attention(device, call, reinterpret_cast<cudaStream_t>(stream));
}

}  // namespace

NB_MODULE(_attn4, m) {
  m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
  m.def("attention", &attention, "device"_a, "q"_a, "k"_a, "v"_a, "out"_a, "q4"_a, "q_scale"_a,
        "k4"_a, "k_scale"_a, "k_sum"_a, "v8"_a, "v_scale"_a, "batch"_a, "heads"_a, "heads_k"_a,
        "len_q"_a, "len_k"_a, "cta_k"_a, "bf16"_a, "scale"_a, "q_strides"_a, "k_strides"_a,
        "v_strides"_a, "out_strides"_a, "stream"_a);
}
