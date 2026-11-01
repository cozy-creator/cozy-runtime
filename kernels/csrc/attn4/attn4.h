// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>

namespace cozy::attn4 {

// (batch, head, token) element strides; channels are contiguous.
using Strides = std::array<int64_t, 3>;

// One attention call over device pointers the caller allocated (`internal.attention_int4`):
// Q/K/V and the output in bf16 or fp16, head dimension 128; the int4 Q [B, H, Lq, 64] and
// K [B, Hk, Lk, 64], their scales, K's channel sums [B, Hk, 128], and the INT8 V
// [B * Hk * 128, Lk padded to cta_k] with its channel scales.
struct Call {
  const void *q, *k, *v;
  void *out, *q4, *k4, *v8;
  float *q_scale, *k_scale, *v_scale, *k_sum;
  int batch, heads, heads_k, len_q, len_k, cta_k;
  bool bf16;
  float scale;
  Strides q_strides, k_strides, v_strides, out_strides;
};

void quantize_qk(const Call &call, cudaStream_t stream);
// The CUDA error of the call's launches; std::runtime_error for a call none serves.
int attention(int device, const Call &call, cudaStream_t stream);

}  // namespace cozy::attn4
