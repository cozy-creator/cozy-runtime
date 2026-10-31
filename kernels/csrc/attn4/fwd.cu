// SPDX-License-Identifier: Apache-2.0
// INT4 Q·Kᵀ attention: comfy-kitchen's pure-integer attention kernel (Comfy-Org/comfy-kitchen
// f8b7e97, `backends/cuda/sage_attention`, Apache-2.0, derived from SageAttention) at its
// `DataType::kInt4` instantiation, which upstream's template carries (s4 mma.sync m16n8k64) and
// no upstream launcher builds. Everything after Q·Kᵀ is Kitchen's INT8 path unchanged: its V
// quantizer, unsigned INT8 P, INT32 P·V and FP32 online softmax. Q/K come from `quant.cu`.
#include "qk_int_sv_i8_cuda.cuh"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "attn4.h"

extern "C" void launch_quant_v_int8_kernel(const void *v, void *out, void *scale, int B, int H,
                                           int N, int D, int padded_N, int64_t sb, int64_t sh,
                                           int64_t sn, int input_dtype_code, cudaStream_t stream);

namespace cozy::attn4 {
namespace {

constexpr uint32_t HEAD_DIM = 128, CTA_Q = 128, WARP_Q = 16;

template <uint32_t CTA_K, typename Out, typename Offset>
void launch(const Call &c, int padded_k, cudaStream_t stream) {
  auto kernel = qk_int_sv_i8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, CTA_K, HEAD_DIM, DataType::kInt4, QuantGranularity::kPerThread,
      QuantGranularity::kPerThread, float, false, Out, ComputeUnit::kCudaCore, MaskMode::kNone,
      false, true, false, false, true, Offset>;
  // Q and K at half a byte a channel, V at one, or the FP16 output tile staged at the end.
  const int smem = static_cast<int>(std::max<uint32_t>(
      (CTA_Q + CTA_K) * HEAD_DIM / 2 + CTA_K * HEAD_DIM, CTA_Q * HEAD_DIM * sizeof(half)));
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
    return;  // the call's error, as cudaGetLastError reports it
  const int64_t row = HEAD_DIM / 2;
  const auto &o = c.out_strides;
  kernel<<<dim3(div_ceil(c.len_q, CTA_Q), c.heads, c.batch), dim3(32, CTA_Q / WARP_Q), smem,
           stream>>>(
      static_cast<int8_t *>(c.q4), static_cast<int8_t *>(c.k4), static_cast<int8_t *>(c.v8),
      static_cast<Out *>(c.out), nullptr, c.q_scale, c.k_scale, c.v_scale, nullptr, nullptr, 0,
      0, 0, 0, 0, c.len_q, c.len_k, c.heads / c.heads_k, Offset(int64_t(c.heads) * c.len_q * row), row,
      Offset(c.len_q * row), Offset(int64_t(c.heads_k) * c.len_k * row), row, Offset(c.len_k * row),
      Offset(int64_t(c.heads_k) * HEAD_DIM * padded_k), Offset(int64_t(HEAD_DIM) * padded_k),
      padded_k, Offset(o[0]), o[2], Offset(o[1]), c.scale, nullptr);
}

template <uint32_t CTA_K, typename Out>
void dispatch(const Call &c, int padded_k, cudaStream_t stream) {
  // The kernel's within-head offsets are 32-bit; its batch and head bases 64-bit when they must be.
  const int64_t row = HEAD_DIM / 2, o_bz = c.out_strides[0], o_h = c.out_strides[1];
  const bool wide =
      int64_t(c.batch - 1) * std::max({int64_t(c.heads) * c.len_q * row, int64_t(c.heads_k) * c.len_k * row,
                                       int64_t(c.heads_k) * HEAD_DIM * padded_k, o_bz}) > UINT32_MAX ||
      int64_t(c.heads - 1) * std::max(c.len_q * row, o_h) > UINT32_MAX ||
      int64_t(c.heads_k - 1) * std::max(c.len_k * row, int64_t(HEAD_DIM) * padded_k) > UINT32_MAX;
  if (wide)
    launch<CTA_K, Out, uint64_t>(c, padded_k, stream);
  else
    launch<CTA_K, Out, uint32_t>(c, padded_k, stream);
}

}  // namespace

int attention(int device, const Call &c, cudaStream_t stream) {
  if (c.cta_k != 64 && c.cta_k != 128)
    throw std::runtime_error("attn4: cta_k must be 64 or 128");
  if (c.batch < 1 || c.heads < 1 || c.heads_k < 1 || c.heads % c.heads_k || c.len_q < 1 ||
      c.len_k < 1)
    throw std::runtime_error("attn4: empty shape or Q heads not a multiple of K/V heads");
  if (int64_t(c.len_q) * c.out_strides[2] > INT_MAX || int64_t(c.len_k) * HEAD_DIM / 2 > INT_MAX)
    throw std::overflow_error("attn4: within-head offsets exceed the int32 range");
  cudaError_t error = cudaSetDevice(device);
  if (error != cudaSuccess) return error;
  const int padded_k = div_ceil(c.len_k, c.cta_k) * c.cta_k;
  quantize_qk(c, stream);
  launch_quant_v_int8_kernel(c.v, c.v8, c.v_scale, c.batch, c.heads_k, c.len_k, HEAD_DIM,
                             padded_k, c.v_strides[0], c.v_strides[1], c.v_strides[2],
                             c.bf16 ? 2 : 1, stream);
  if (c.cta_k == 128)
    c.bf16 ? dispatch<128, nv_bfloat16>(c, padded_k, stream)
           : dispatch<128, half>(c, padded_k, stream);
  else
    c.bf16 ? dispatch<64, nv_bfloat16>(c, padded_k, stream)
           : dispatch<64, half>(c, padded_k, stream);
  return cudaGetLastError();
}

}  // namespace cozy::attn4
