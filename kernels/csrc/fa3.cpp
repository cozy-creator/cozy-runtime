// SPDX-License-Identifier: Apache-2.0
// `cozy_kernels._fa3`: FlashAttention-3's Hopper forward (Dao-AILab flash-attention v2.8.3.post1,
// `hopper/`, BSD-3-Clause), its instantiation units compiled unmodified for sm_90a, reached
// without torch: this file fills `Flash_fwd_params` as upstream's `mha_fwd` does for dense,
// fixed-length, unpaged, unsplit attention (K/V heads dividing Q heads), head dims 64 and 128,
// bf16 and fp16. Arguments are raw device pointers, element strides and a stream; the
// caller allocates `out` like `q` and `lse` as float32 [b, h, seqlen_q], and for causal
// attention a zeroed int32 semaphore. Returns the CUDA error code; NOT_SUPPORTED where no unit
// here serves the call.
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <array>
#include <cstdint>

#include "flash.h"

template <int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA,
          bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);

namespace nb = nanobind;
using namespace nb::literals;
using Strides = std::array<int64_t, 3>;  // batch, row (sequence), head; the head dim is dense

namespace {

int fwd(int device, uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t out, uintptr_t lse,
        bool bf16, int batch, int seqlen_q, int seqlen_k, int heads, int heads_k, int head_dim,
        Strides qs, Strides ks, Strides vs, Strides os, float scale, bool causal,
        uintptr_t semaphore, uintptr_t stream) {
    if (heads_k < 1 || heads % heads_k || (head_dim != 64 && head_dim != 128) ||
        (causal && !semaphore))
        return cudaErrorNotSupported;
    if (batch < 1 || seqlen_q < 1 || seqlen_k < 1 || heads < 1) return cudaErrorInvalidValue;
    int previous = -1;
    cudaError_t error = cudaGetDevice(&previous);
    if (error == cudaSuccess && previous != device) error = cudaSetDevice(device);
    if (error != cudaSuccess) return error;
    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, device);
    if (error == cudaSuccess && props.major * 10 + props.minor != 90) error = cudaErrorNotSupported;
    if (error == cudaSuccess) {
        auto round = [](int x, int m) { return (x + m - 1) / m * m; };
        Flash_fwd_params p = {};
        p.is_bf16 = bf16;
        p.q_ptr = reinterpret_cast<void*>(q);
        p.k_ptr = reinterpret_cast<void*>(k);
        p.v_ptr = reinterpret_cast<void*>(v);
        p.o_ptr = reinterpret_cast<void*>(out);
        p.softmax_lse_ptr = reinterpret_cast<void*>(lse);
        p.q_batch_stride = qs[0], p.q_row_stride = qs[1], p.q_head_stride = qs[2];
        p.k_batch_stride = ks[0], p.k_row_stride = ks[1], p.k_head_stride = ks[2];
        p.v_batch_stride = vs[0], p.v_row_stride = vs[1], p.v_head_stride = vs[2];
        p.o_batch_stride = os[0], p.o_row_stride = os[1], p.o_head_stride = os[2];
        p.v_dim_stride = 1;
        p.b = p.b_k = batch;
        p.h = heads, p.h_k = heads_k;
        p.seqlen_q = seqlen_q, p.seqlen_k = seqlen_k;
        p.seqlen_q_rounded = round(seqlen_q, 128), p.seqlen_k_rounded = round(seqlen_k, 128);
        p.total_q = batch * seqlen_q, p.total_k = batch * seqlen_k;
        p.d = p.dv = p.d_rounded = p.dv_rounded = head_dim;
        p.scale_softmax = scale;
        p.p_dropout = 1.f;
        p.p_dropout_in_uint8_t = 255;
        p.rp_dropout = 1.f;
        p.is_causal = causal;
        p.window_size_left = seqlen_k - 1;
        p.window_size_right = causal ? 0 : seqlen_q - 1;
        p.num_splits = 1;
        p.tile_count_semaphore = causal ? reinterpret_cast<int*>(semaphore) : nullptr;
        p.arch = 90;
        p.num_sm = props.multiProcessorCount;
        const auto s = reinterpret_cast<cudaStream_t>(stream);
        if (bf16) {
            if (head_dim == 64)
                run_mha_fwd_<90, cutlass::bfloat16_t, 64, 64, false, false, false, false>(p, s);
            else
                run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(p, s);
        } else {
            if (head_dim == 64)
                run_mha_fwd_<90, cutlass::half_t, 64, 64, false, false, false, false>(p, s);
            else
                run_mha_fwd_<90, cutlass::half_t, 128, 128, false, false, false, false>(p, s);
        }
        error = cudaGetLastError();
    }
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
    return error;
}

}  // namespace

NB_MODULE(_fa3, m) {
    m.attr("NOT_SUPPORTED") = int(cudaErrorNotSupported);
    m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
    m.def("fwd", &fwd, "device"_a, "q"_a, "k"_a, "v"_a, "out"_a, "lse"_a, "bf16"_a, "batch"_a,
          "seqlen_q"_a, "seqlen_k"_a, "heads"_a, "heads_k"_a, "head_dim"_a, "q_strides"_a,
          "k_strides"_a, "v_strides"_a, "o_strides"_a, "scale"_a, "causal"_a, "semaphore"_a,
          "stream"_a);
}
