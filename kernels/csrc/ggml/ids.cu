// SPDX-License-Identifier: Apache-2.0
// llama.cpp's MUL_MAT_ID for a few tokens (`ggml_cuda_mul_mat_id`'s MMVQ branch, ggml-cuda.cu, b12466, MIT) over
// experts held apart: the stock `ggml_cuda_mul_mat_vec_q` with ids, given a weight whose data is a table of each
// expert's address and whose channel stride is 0, which the patched mmvq.cu reads as that table (UPSTREAM.md). The
// routing stays on the device, so a captured step holds it. x is f32, as llama.cpp's activations are; ggml quantizes it
// to q8_1 itself.
#include "ids.h"
#include "matmul.h"
#include "shim.h"

#include "ggml/src/ggml-cuda/mmvq.cuh"

#include <algorithm>
#include <exception>

namespace cozy_ggml {

int ids_batch(int device, int type) {
    const int cc = ggml_cuda_info().devices[device].cc;
    return std::min(MMVQ_MAX_BATCH_SIZE, get_mmvq_mmid_max_batch(ggml_type(type), cc));
}

int matmul_ids(int device, int type, const void* table, const float* x, int64_t rows, const int32_t* ids, float* out,
               int64_t K, int64_t N, int64_t T, int64_t k, void* (*allocate)(size_t, void*), void* context,
               cudaStream_t stream) {
    if (!runs(type) || K <= 0 || N <= 0 || T <= 0 || k <= 0 || K % MATRIX_ROW_PADDING) return cudaErrorInvalidValue;
    if (rows != 1 && rows != k) return cudaErrorInvalidValue;
    try {
        Allocate temporaries{allocate, context};
        Call call(device, stream, &temporaries);
        auto& ctx = call.ctx();
        const int64_t batch = ids_batch(device, type);
        for (int64_t t0 = 0; t0 < T; t0 += batch) {
            const int64_t n = std::min(batch, T - t0);
            // MUL_MAT_ID's tensors (ggml.c `ggml_mul_mat_id`): weights [K, N, experts], x [K, rows, n], ids [k, n],
            // out [N, k, n]. The weights' data is the table; their channel stride 0 says so.
            ggml_tensor w{}, src1{}, chosen{}, dst{};
            w.type = ggml_type(type);
            w.ne[0] = K, w.ne[1] = N, w.ne[2] = 1, w.ne[3] = 1;
            w.nb[0] = ggml_type_size(w.type);
            w.nb[1] = ggml_row_size(w.type, K);
            w.nb[2] = w.nb[3] = 0;
            w.data = const_cast<void*>(table);
            src1.type = GGML_TYPE_F32;
            src1.ne[0] = K, src1.ne[1] = rows, src1.ne[2] = n, src1.ne[3] = 1;
            src1.nb[0] = sizeof(float);
            src1.nb[1] = sizeof(float) * K;
            src1.nb[2] = src1.nb[1] * rows;
            src1.nb[3] = src1.nb[2] * n;
            src1.data = const_cast<float*>(x + t0 * rows * K);
            chosen.type = GGML_TYPE_I32;
            chosen.ne[0] = k, chosen.ne[1] = n, chosen.ne[2] = 1, chosen.ne[3] = 1;
            chosen.nb[0] = sizeof(int32_t);
            chosen.nb[1] = sizeof(int32_t) * k;
            chosen.nb[2] = chosen.nb[3] = chosen.nb[1] * n;
            chosen.data = const_cast<int32_t*>(ids + t0 * k);
            dst.type = GGML_TYPE_F32;
            dst.ne[0] = N, dst.ne[1] = k, dst.ne[2] = n, dst.ne[3] = 1;
            dst.nb[0] = sizeof(float);
            dst.nb[1] = sizeof(float) * N;
            dst.nb[2] = dst.nb[1] * k;
            dst.nb[3] = dst.nb[2] * n;
            dst.data = out + t0 * k * N;
            dst.op = GGML_OP_MUL_MAT_ID;
            dst.src[0] = &w, dst.src[1] = &src1, dst.src[2] = &chosen;
            ggml_cuda_mul_mat_vec_q(ctx, &w, &src1, &chosen, &dst);
        }
        return cudaGetLastError();
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (const std::exception& failure) {
        failed(failure.what());
        return cudaErrorUnknown;
    }
}

}  // namespace cozy_ggml
