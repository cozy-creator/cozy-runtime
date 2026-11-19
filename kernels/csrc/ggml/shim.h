// SPDX-License-Identifier: Apache-2.0
// `shim.cu`'s side the binding uses: one call of the stock ggml-cuda entry points on the caller's stream, its
// temporaries from the caller's allocator.
#pragma once

#include "ggml/src/ggml-cuda/common.cuh"

namespace cozy_ggml {

struct Allocate {
    void* (*fn)(size_t bytes, void* context);
    void* context;
};

// While it lives, this thread's ggml context for `device` runs on `stream` and takes temporaries from `allocate`.
class Call {
  public:
    Call(int device, cudaStream_t stream, Allocate* allocate);
    ~Call();
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;
    ggml_backend_cuda_context& ctx() { return *ctx_; }

  private:
    ggml_backend_cuda_context* ctx_;
};

}  // namespace cozy_ggml
