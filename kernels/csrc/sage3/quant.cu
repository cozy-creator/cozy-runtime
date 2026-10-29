// SPDX-License-Identifier: Apache-2.0
// SageAttention3's FP4 quantizers (thu-ml/SageAttention d1a57a5,
// `sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu`, Apache-2.0) without
// torch: `quant_kernels.cuh` is upstream's file up to its first torch wrapper, its torch includes
// dropped (kernels/CMakeLists.txt cuts it and checks the cut's bytes). These launches are its
// wrappers' at tensor layout 1, the one its `api.py` uses: (batch, heads, tokens, dim).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "quant_kernels.cuh"
#include "sage3.h"

namespace cozy::sage3 {

int quant(int kind, int device, uintptr_t input, uintptr_t output, uintptr_t output_sf, bool bf16,
          int batch, int heads, int tokens, int head_dim, Strides in, Strides out, Strides sf,
          uintptr_t stream) {
    if ((head_dim != 64 && head_dim != 128) || kind < 0 || kind > 2) return cudaErrorNotSupported;
    if (batch < 1 || heads < 1 || tokens < 0) return cudaErrorInvalidValue;
    if (tokens == 0) return cudaSuccess;
    int previous = -1;
    cudaError_t error = cudaGetDevice(&previous);
    if (error == cudaSuccess && previous != device) error = cudaSetDevice(device);
    if (error != cudaSuccess) return error;
    constexpr int BLOCK_SIZE = 128;
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    auto launch = [&](auto element, auto dim) {
        using T = decltype(element);
        constexpr int HEAD_DIM = decltype(dim)::value;
        dim3 block(BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1);
        dim3 grid((tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, batch, heads);
        const T* x = reinterpret_cast<const T*>(input);
        uint8_t* o = reinterpret_cast<uint8_t*>(output);
        uint8_t* f = reinterpret_cast<uint8_t*>(output_sf);
        if (kind == 2) {
            scaled_fp4_quant_trans_kernel<HEAD_DIM, BLOCK_SIZE, T><<<grid, block, 0, s>>>(
                x, o, f, batch, heads, tokens, int(in[0]), int(in[1]), int(in[2]), int(out[0]),
                int(out[1]), int(out[2]), int(sf[0]), int(sf[1]), int(sf[2]));
        } else if (kind == 1) {
            scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, true, T><<<grid, block, 0, s>>>(
                x, o, f, batch, heads, tokens, int(in[0]), int(in[1]), int(in[2]), int(out[0]),
                int(out[1]), int(out[2]), int(sf[0]), int(sf[1]), int(sf[2]));
        } else {
            scaled_fp4_quant_kernel<HEAD_DIM, BLOCK_SIZE, false, T><<<grid, block, 0, s>>>(
                x, o, f, batch, heads, tokens, int(in[0]), int(in[1]), int(in[2]), int(out[0]),
                int(out[1]), int(out[2]), int(sf[0]), int(sf[1]), int(sf[2]));
        }
    };
    using D64 = std::integral_constant<int, 64>;
    using D128 = std::integral_constant<int, 128>;
    if (bf16) {
        head_dim == 64 ? launch(nv_bfloat16{}, D64{}) : launch(nv_bfloat16{}, D128{});
    } else {
        head_dim == 64 ? launch(half{}, D64{}) : launch(half{}, D128{});
    }
    error = cudaGetLastError();
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
    return error;
}

}  // namespace cozy::sage3
