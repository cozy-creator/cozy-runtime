// SPDX-License-Identifier: Apache-2.0
// `_kernels._ggml`'s MUL_MAT_ID (`ids.cu`): a mixture of experts' routed products, each expert's weight read through a
// table of addresses on the device, in plain C++ for the binding.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cozy_ggml {

// The tokens one MMVQ launch of MUL_MAT_ID takes for `type` on `device` (ggml's `get_mmvq_mmid_max_batch`).
int ids_batch(int device, int type);

// out[T * k, N] (f32) = each (token t, slot j) pair's product: x's row for it times the [N, K] `type` blocks of expert
// ids[t * k + j], read at table[expert] (device memory, one address per expert). x is f32, `rows` rows a token: 1 (every
// slot reads the token's row) or k (slot j reads row t * k + j). By MMVQ, `ids_batch` tokens a launch; temporaries from
// `allocate(bytes, context)`. A CUDA error code; a failure ggml itself raises is cudaErrorUnknown, its message in
// `error()`.
int matmul_ids(int device, int type, const void* table, const float* x, int64_t rows, const int32_t* ids, float* out,
               int64_t K, int64_t N, int64_t T, int64_t k, void* (*allocate)(size_t, void*), void* context,
               cudaStream_t stream);

}  // namespace cozy_ggml
