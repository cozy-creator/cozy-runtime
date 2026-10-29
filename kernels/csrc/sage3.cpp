// SPDX-License-Identifier: Apache-2.0
// `_kernels._sage3`: SageAttention3's Blackwell forward and FP4 quantizers (thu-ml/SageAttention
// d1a57a5, `sageattention3_blackwell`, Apache-2.0) for RTX 50 (sm_120a) and DGX Spark (sm_121a),
// reached without torch (`sage3/fwd.cu`, `sage3/quant.cu`). Arguments are raw device pointers,
// element strides and a stream; the caller allocates as upstream's `api.py` and `mha_fwd` do
// (`cozy_runtime.internal.sage3`). Returns the CUDA error code; NOT_SUPPORTED where no unit here
// serves the call.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include <cuda_runtime.h>

#include "sage3/sage3.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_sage3, m) {
    m.attr("NOT_SUPPORTED") = int(cudaErrorNotSupported);
    m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
    m.def("fwd", &cozy::sage3::fwd, "device"_a, "q"_a, "k"_a, "v"_a, "sfq"_a, "sfk"_a, "sfv"_a,
          "delta_s"_a, "out"_a, "lse"_a, "semaphore"_a, "batch"_a, "heads"_a, "heads_k"_a,
          "seqlen_q"_a, "seqlen_k"_a, "unpadded_k"_a, "head_dim"_a, "q_strides"_a, "k_strides"_a,
          "v_strides"_a, "sfq_strides"_a, "sfk_strides"_a, "sfv_strides"_a, "ds_strides"_a,
          "o_strides"_a, "scale"_a, "causal"_a, "per_block_mean"_a, "bf16"_a, "stream"_a);
    m.def("quant", &cozy::sage3::quant, "kind"_a, "device"_a, "input"_a, "output"_a,
          "output_sf"_a, "bf16"_a, "batch"_a, "heads"_a, "tokens"_a, "head_dim"_a,
          "in_strides"_a, "out_strides"_a, "sf_strides"_a, "stream"_a);
}
