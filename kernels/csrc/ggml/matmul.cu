// SPDX-License-Identifier: Apache-2.0
// llama.cpp's quantized matmuls on the caller's buffers: the stock entry points of MMVQ and few-row MMQ, MMQ's
// dense host launch, and ggml's own choice among them, given ggml tensors that describe the caller's pointers.
// ggml quantizes the activation to q8_1 itself, as it does in llama.cpp.
#include "matmul.h"
#include "shim.h"

#include "ggml/src/ggml-cuda/mmq-few.cuh"
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"
#include "ggml/src/ggml-cuda/quantize.cuh"

#include <climits>
#include <cstring>
#include <exception>
#include <string>

namespace cozy_ggml {
namespace {

thread_local std::string last_error;

bool runs(int type) {
    switch (type) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// ggml's MUL_MAT node over the caller's buffers: weights [N, K] in `type`, x [M, K] f32, out [M, N] f32.
struct Node {
    ggml_tensor w{}, x{}, out{};

    Node(ggml_type type, const void* wd, const float* xd, float* od, int64_t K, int64_t N, int64_t M) {
        w.type = type;
        w.ne[0] = K, w.ne[1] = N, w.ne[2] = 1, w.ne[3] = 1;
        w.nb[0] = ggml_type_size(type);
        w.nb[1] = ggml_row_size(type, K);
        w.nb[2] = w.nb[3] = w.nb[1] * N;
        w.data = const_cast<void*>(wd);
        x.type = GGML_TYPE_F32;
        x.ne[0] = K, x.ne[1] = M, x.ne[2] = 1, x.ne[3] = 1;
        x.nb[0] = sizeof(float);
        x.nb[1] = sizeof(float) * K;
        x.nb[2] = x.nb[3] = x.nb[1] * M;
        x.data = const_cast<float*>(xd);
        out.type = GGML_TYPE_F32;
        out.ne[0] = N, out.ne[1] = M, out.ne[2] = 1, out.ne[3] = 1;
        out.nb[0] = sizeof(float);
        out.nb[1] = sizeof(float) * N;
        out.nb[2] = out.nb[3] = out.nb[1] * M;
        out.data = od;
        out.op = GGML_OP_MUL_MAT;
        out.src[0] = &w;
        out.src[1] = &x;
    }
};

// `ggml_cuda_should_use_mmq` (mmq.cu) for the NVIDIA cards that have tensor cores: MMQ's tiles need 48 KiB.
bool should_use_mmq(int cc, int device) {
    return turing_mma_available(cc) && ggml_cuda_info().devices[device].smpbo >= 48 * 1024;
}

int choose(int device, const Node& node) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const ggml_type type = node.w.type;
    const bool few = ggml_cuda_should_use_mmq_few(&node.w, &node.x, cc);
    if (!few && ggml_cuda_should_use_mmvq(type, cc, node.x.ne[1])) return kMmvq;
    if (few) return kMmqFew;
    if (should_use_mmq(cc, device)) return kMmq;
    return kNone;
}

// `ggml_cuda_mul_mat_q`'s dense branch (mmq.cu, llama.cpp b12466, MIT): q8_1 activations, no fusion, no
// experts; the kernels are mmq.cuh's, instantiated by template-instances/mmq-instance-<type>.cu.
void mul_mat_q(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    cudaStream_t stream = ctx.stream();
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_dst = ggml_type_size(dst->type);
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t s01 = nb01 / ts_src0, s02 = nb02 / ts_src0, s03 = nb03 / ts_src0;
    const int64_t s1 = nb1 / ts_dst, s2 = nb2 / ts_dst, s3 = nb3 / ts_dst;
    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    // ggml_cuda_mmq_get_J_best: the tile width with the fewest tiles that fits in shared memory.
    int J_best = 0, nthreads_best = 0, ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc, GGML_PREC_Q8);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) continue;
        const int ntiles = (ne11 + config.J - 1) / config.J;
        if (ntiles < ntiles_best) {
            J_best = J;
            nthreads_best = config.nthreads;
            ntiles_best = ntiles;
        }
    }
    GGML_ASSERT(J_best > 0);

    const size_t chunk = nthreads_best * sizeof(int);
    const size_t padding = ((J_best * sizeof(block_q8_1_mmq) + chunk - 1) / chunk) * chunk;
    const int ntj = (ne1 + J_best - 1) / J_best;
    ggml_cuda_pool_alloc<char> src1_q8_1(
        ctx.pool(), ne13 * ne12 * ne11 * ne10_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ + padding);
    quantize_mmq_q8_1_cuda((const float*)src1->data, nullptr, src1_q8_1.get(), src0->type, ne10,
                           nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float), ne10_padded, ne11, ne12,
                           ne13, stream);
    CUDA_CHECK(cudaGetLastError());
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12 * s12;
    const mmq_args args = {(const char*)src0->data, src0->type, (const int*)src1_q8_1.ptr, nullptr, nullptr,
                           (float*)dst->data, nullptr, ne00, ne01, ne1, s01, ne11, s1, ne02, ne12, s02, s12, s2,
                           ne03, ne13, s03, s13, s3, ntj, J_best, {}};
    switch (src0->type) {
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ctx, args, stream); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ctx, args, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ctx, args, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream); break;
        default: GGML_ABORT("type %d is not one _ggml runs", int(src0->type));
    }
}

}  // namespace

int kernel(int device, int type, const void* w, int64_t K, int64_t N, int64_t M) {
    try {
        if (!runs(type) || K <= 0 || N <= 0 || M <= 0) return kNone;
        // The weight where it is; the activation and output as `matmul`'s caller allocates them, 256-byte aligned.
        const auto aligned = reinterpret_cast<float*>(uintptr_t(256));
        const Node node(ggml_type(type), w, aligned, aligned, K, N, M);
        return choose(device, node);
    } catch (const std::exception& failure) {
        last_error = failure.what();
        return kNone;
    }
}

int matmul(int device, int type, const void* w, const float* x, float* out, int64_t K, int64_t N, int64_t M,
           void* (*allocate)(size_t, void*), void* context, cudaStream_t stream) {
    if (!runs(type) || K <= 0 || N <= 0 || M <= 0 || K % MATRIX_ROW_PADDING) return cudaErrorInvalidValue;
    try {
        Allocate temporaries{allocate, context};
        Call call(device, stream, &temporaries);
        Node node(ggml_type(type), w, x, out, K, N, M);
        switch (choose(device, node)) {
            case kMmvq: ggml_cuda_mul_mat_vec_q(call.ctx(), &node.w, &node.x, nullptr, &node.out); break;
            case kMmqFew: ggml_cuda_mul_mat_q_few(call.ctx(), &node.w, &node.x, &node.out); break;
            case kMmq: mul_mat_q(call.ctx(), &node.w, &node.x, &node.out); break;
            default: return cudaErrorNotSupported;
        }
        return cudaGetLastError();
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (const std::exception& failure) {
        last_error = failure.what();
        return cudaErrorUnknown;
    }
}

const char* error() { return last_error.c_str(); }

}  // namespace cozy_ggml
