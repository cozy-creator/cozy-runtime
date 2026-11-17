// SPDX-License-Identifier: Apache-2.0
// `_kernels._ggml`'s calls into llama.cpp's quantized matmuls (`matmul.cu`), in plain C++ for the binding.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cozy_ggml {

// The kernel ggml's own MUL_MAT rule takes (`ggml_cuda_mul_mat`'s quantized branch): few-row MMQ where it is
// tuned to win, else MMVQ up to its batch, else MMQ.
enum Kernel { kNone = 0, kMmvq = 1, kMmqFew = 2, kMmq = 3 };

// The kernel for the `type` weights [N, K] at `w` times M rows of a 256-byte aligned activation on `device`.
int kernel(int device, int type, const void* w, int64_t K, int64_t N, int64_t M);

// out[M, N] (f32) = x[M, K] (f32) times the [N, K] ggml blocks at `w`, on `stream`; temporaries from
// `allocate(bytes, context)`, held by the caller until this returns. A CUDA error code; a failure ggml itself
// raises is cudaErrorUnknown, its message in `error()`.
int matmul(int device, int type, const void* w, const float* x, float* out, int64_t K, int64_t N, int64_t M,
           void* (*allocate)(size_t, void*), void* context, cudaStream_t stream);

const char* error();

}  // namespace cozy_ggml
