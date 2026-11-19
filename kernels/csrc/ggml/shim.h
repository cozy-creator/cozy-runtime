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

// The caller's stream as a launch names it: a null one is torch's default stream, the legacy NULL stream, which ggml's
// context reads as "none yet" and replaces with a non-blocking stream of its own, unordered with the caller's.
inline cudaStream_t launch_stream(cudaStream_t stream) { return stream == nullptr ? cudaStreamLegacy : stream; }

// While it lives, this thread's ggml context for `device` runs on `stream` (`launch_stream`) and takes temporaries from
// `allocate`.
class Call {
  public:
    Call(int device, cudaStream_t stream, Allocate* allocate);
    ~Call();
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;
    ggml_backend_cuda_context& ctx() { return *ctx_; }
    // The stream this call's own launches name: the context's.
    cudaStream_t stream() const { return ctx_->streams[ctx_->device][0]; }

  private:
    ggml_backend_cuda_context* ctx_;
};

}  // namespace cozy_ggml
