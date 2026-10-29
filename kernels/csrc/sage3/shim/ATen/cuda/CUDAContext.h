// SPDX-License-Identifier: Apache-2.0
// The ATen that SageAttention3's `launch.h` names, without torch: its two CUDA checks, which keep the
// call's first error for `_sage3` to return (`sage3.h`) where ATen's throw.
#pragma once

#include <cuda_runtime.h>

namespace cozy::sage3 {
inline thread_local cudaError_t failed = cudaSuccess;
}  // namespace cozy::sage3

#define C10_CUDA_CHECK(expr)                                                                \
    do {                                                                                    \
        const cudaError_t cozy_error_ = (expr);                                             \
        if (cozy_error_ != cudaSuccess && cozy::sage3::failed == cudaSuccess)               \
            cozy::sage3::failed = cozy_error_;                                              \
    } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
