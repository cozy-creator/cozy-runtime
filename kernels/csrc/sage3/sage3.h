// SPDX-License-Identifier: Apache-2.0
// `_sage3`'s units (`fwd.cu`, `quant.cu`) as its binding (`sage3.cpp`) calls them: raw device
// pointers, element strides and a stream; each returns the CUDA error code, NOT_SUPPORTED where no
// unit serves the call.
#pragma once

#include <array>
#include <cstdint>

namespace cozy::sage3 {

using Strides = std::array<int64_t, 3>;  // batch, head, row; the last dimension is dense

// SageAttention3's `fp4attn_cuda.fwd` (`blackwell/api.cu`, `mha_fwd`): packed FP4 Q, K and V^T
// with their FP8 block scales, `delta_s`, into `out` (bf16 or fp16) and `lse` (float32).
int fwd(int device, uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t sfq, uintptr_t sfk,
        uintptr_t sfv, uintptr_t delta_s, uintptr_t out, uintptr_t lse, uintptr_t semaphore,
        int batch, int heads, int heads_k, int seqlen_q, int seqlen_k, int unpadded_k,
        int head_dim, Strides qs, Strides ks, Strides vs, Strides sfqs, Strides sfks, Strides sfvs,
        Strides dss, Strides os, float scale, bool causal, bool per_block_mean, bool bf16,
        uintptr_t stream);

// SageAttention3's `fp4quant_cuda.scaled_fp4_quant` (kind 0), `_permute` (1) and `_trans` (2)
// (`quantization/fp4_quantization_4d.cu`) at its tensor layout 1, (batch, heads, tokens, dim).
int quant(int kind, int device, uintptr_t input, uintptr_t output, uintptr_t output_sf, bool bf16,
          int batch, int heads, int tokens, int head_dim, Strides in, Strides out, Strides sf,
          uintptr_t stream);

}  // namespace cozy::sage3
