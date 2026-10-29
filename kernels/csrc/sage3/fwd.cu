// SPDX-License-Identifier: Apache-2.0
// SageAttention3's Blackwell forward (thu-ml/SageAttention d1a57a5, `sageattention3_blackwell`,
// Apache-2.0) without torch: upstream's kernel headers unmodified (their ATen names from
// `shim/`), and this file filling `Flash_fwd_params` as `blackwell/api.cu`'s `set_params_fprop`
// and `mha_fwd` do, field for field. The four units are its dispatch's: head dims 64 and 128,
// bf16 and fp16 out, causal and per-block mean switched at run time as upstream's `launch.h` does.
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

#include <cmath>

#include "launch.h"
#include "params.h"
#include "sage3.h"
#include "static_switch.h"

namespace cozy::sage3 {
namespace {

using FP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

template <bool IsBF16>
void dispatch_dtype(Flash_fwd_params& params, cudaStream_t stream) {
    using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
    if (params.d == 64) {
        run_mha_fwd_<FP4, 64, OType>(params, stream);
    } else {
        run_mha_fwd_<FP4, 128, OType>(params, stream);
    }
}

}  // namespace

int fwd(int device, uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t sfq, uintptr_t sfk,
        uintptr_t sfv, uintptr_t delta_s, uintptr_t out, uintptr_t lse, uintptr_t semaphore,
        int batch, int heads, int heads_k, int seqlen_q, int seqlen_k, int unpadded_k,
        int head_dim, Strides qs, Strides ks, Strides vs, Strides sfqs, Strides sfks, Strides sfvs,
        Strides dss, Strides os, float scale, bool causal, bool per_block_mean, bool bf16,
        uintptr_t stream) {
    // Upstream dispatches 64 and 128 only (a 256 passes its checks and launches nothing).
    if ((head_dim != 64 && head_dim != 128) || heads_k != heads || !semaphore)
        return cudaErrorNotSupported;
    if (batch < 1 || heads < 1 || seqlen_q < 1 || seqlen_k < 0) return cudaErrorInvalidValue;
    int previous = -1;
    cudaError_t error = cudaGetDevice(&previous);
    if (error == cudaSuccess && previous != device) error = cudaSetDevice(device);
    if (error != cudaSuccess) return error;
    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, device);
    if (error == cudaSuccess && !(props.major == 12 && (props.minor == 0 || props.minor == 1)))
        error = cudaErrorNotSupported;
    if (error == cudaSuccess && seqlen_k > 0) {
        auto round = [](int x, int m) { return (x + m - 1) / m * m; };
        Flash_fwd_params params = {};
        params.q_ptr = reinterpret_cast<void*>(q);
        params.k_ptr = reinterpret_cast<void*>(k);
        params.v_ptr = reinterpret_cast<void*>(v);
        params.delta_s_ptr = reinterpret_cast<void*>(delta_s);
        params.sfq_ptr = reinterpret_cast<void*>(sfq);
        params.sfk_ptr = reinterpret_cast<void*>(sfk);
        params.sfv_ptr = reinterpret_cast<void*>(sfv);
        // Packed FP4: two elements a byte.
        params.q_row_stride = qs[2] * 2, params.k_row_stride = ks[2] * 2;
        params.v_row_stride = vs[2] * 2;
        params.q_head_stride = qs[1] * 2, params.k_head_stride = ks[1] * 2;
        params.v_head_stride = vs[1] * 2;
        params.ds_row_stride = dss[2], params.ds_head_stride = dss[1];
        params.sfq_row_stride = sfqs[2], params.sfk_row_stride = sfks[2];
        params.sfv_row_stride = sfvs[2];
        params.sfq_head_stride = sfqs[1], params.sfk_head_stride = sfks[1];
        params.sfv_head_stride = sfvs[1];
        params.o_ptr = reinterpret_cast<void*>(out);
        params.o_row_stride = os[2], params.o_head_stride = os[1];
        params.q_batch_stride = qs[0] * 2, params.k_batch_stride = ks[0] * 2;
        params.v_batch_stride = vs[0] * 2;
        params.ds_batch_stride = dss[0];
        params.sfq_batch_stride = sfqs[0], params.sfk_batch_stride = sfks[0];
        params.sfv_batch_stride = sfvs[0];
        params.o_batch_stride = os[0];
        params.cu_seqlens_q = nullptr, params.cu_seqlens_k = nullptr, params.seqused_k = nullptr;
        params.p_ptr = nullptr;
        params.softmax_lse_ptr = reinterpret_cast<void*>(lse);
        params.b = batch;
        params.h = heads, params.h_k = heads_k, params.h_h_k_ratio = heads / heads_k;
        params.seqlen_q = seqlen_q, params.seqlen_k = seqlen_k;
        params.unpadded_seqlen_k = unpadded_k;
        params.seqlen_q_rounded = round(seqlen_q, 128), params.seqlen_k_rounded = round(seqlen_k, 128);
        params.d = params.d_rounded = head_dim;
        params.head_divmod = cutlass::FastDivmod(heads);
        params.scale_softmax = scale;
        params.scale_softmax_log2 = scale * M_LOG2E;  // in double, as upstream's
        __half log2_half = __float2half(params.scale_softmax_log2);
        __half2 log2_half2 = __half2(log2_half, log2_half);
        params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(log2_half2);
        // p_dropout 0: keep every activation.
        params.p_dropout = 1.f;
        params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
        params.rp_dropout = 1.f / params.p_dropout;
        params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
        // mha_fwd's window: left -1, right 0 when causal else -1.
        int left = -1, right = causal ? 0 : -1;
        params.is_causal = left < 0 && right == 0;
        params.per_block_mean = per_block_mean;
        params.seqlen_s = per_block_mean ? seqlen_q : 128;  // BLOCK_M
        if (left < 0 && right >= 0) left = seqlen_k;
        if (left >= 0 && right < 0) right = seqlen_k;
        params.window_size_left = left, params.window_size_right = right;
        params.is_seqlens_k_cumulative = true;
        params.is_bf16 = bf16;
        params.tile_count_semaphore = reinterpret_cast<int*>(semaphore);
        const auto s = reinterpret_cast<cudaStream_t>(stream);
        failed = cudaSuccess;
        BOOL_SWITCH(params.is_bf16, IsBF16, ([&] { dispatch_dtype<IsBF16>(params, s); }));
        error = failed != cudaSuccess ? failed : cudaGetLastError();
    }
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
    return error;
}

}  // namespace cozy::sage3
