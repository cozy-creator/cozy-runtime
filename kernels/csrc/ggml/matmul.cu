// SPDX-License-Identifier: Apache-2.0
// llama.cpp's quantized matmuls on the caller's buffers: the stock entry points of MMVQ and few-row MMQ, MMQ's
// dense host launch, and ggml's own choice among them, given ggml tensors that describe the caller's pointers.
// ggml quantizes an f32 activation to q8_1 itself, as it does in llama.cpp; a bf16 or f16 one is quantized here by the
// same arithmetic (`quantize_rows`), with no f32 copy of it.
#include "matmul.h"
#include "shim.h"

#include "ggml/src/ggml-cuda/mmq-few.cuh"
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"
#include "ggml/src/ggml-cuda/quantize.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float widen(half v) { return __half2float(v); }

// `quantize_q8_1` (quantize.cu, llama.cpp b12466, MIT) over M dense rows of K 16-bit values, each widened to f32 (exact),
// so its blocks are ggml's for the same values: rows of `K_padded` (whole 512-column chunks), zeros past K.
template <typename T>
__global__ void __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
    quantize_q8_1_rows(const T* __restrict__ x, block_q8_1* __restrict__ y, int64_t K, int64_t K_padded) {
    const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= K_padded) return;
    const int64_t row = blockIdx.y;
    const float xi = i0 < K ? widen(x[row * K + i0]) : 0.0f;
    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int sum = warp_reduce_sum<QK8_1>((int)q);
    const int64_t i_cont = row * K_padded + i0;
    y[i_cont / QK8_1].qs[i_cont % QK8_1] = q;
    if (i_cont % QK8_1 == 0) y[i_cont / QK8_1].ds = make_half2(d, d * sum);
}

// `quantize_mmq_q8_1` (quantize.cu, llama.cpp b12466, MIT) for one channel of M dense rows of 16-bit values, as above.
template <mmq_q8_1_ds_layout ds_layout, typename T>
__global__ void quantize_mmq_q8_1_rows(const T* __restrict__ x, void* __restrict__ vy, int64_t K, int64_t K_padded,
                                       int M) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= K_padded) return;
    const int64_t row = blockIdx.x;
    block_q8_1_mmq* y = (block_q8_1_mmq*)vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs = i0 % QK8_1_MMQ;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < K) {
        const T* p = x + row * K + i0;
        xi = make_float4(widen(p[0]), widen(p[1]), widen(p[2]), widen(p[3]));
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, ggml_cuda_shfl_xor(amax, offset, WARP_SIZE));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        int sum_q = int(q.x) + int(q.y) + int(q.z) + int(q.w);
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum_q += ggml_cuda_shfl_xor(sum_q, offset, WARP_SIZE);
        }
        sum = d * float(sum_q);
    }
    const int64_t ib = k_block * M + row;
    char4* yqs4 = (char4*)y[ib].qs;
    yqs4[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d;
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

template <typename T>
__global__ void widen_rows(const T* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = widen(x[i]);
}

// The activation as the kernels read it: f32 rows (ggml quantizes them), or 16-bit ones (quantized here).
struct Rows {
    const void* data;
    int type;  // kF32, kBF16, kF16

    template <typename F>
    void as(F&& f) const {
        switch (type) {
            case kBF16: f(static_cast<const __nv_bfloat16*>(data)); break;
            case kF16: f(static_cast<const half*>(data)); break;
            default: GGML_ABORT("not a 16-bit activation");
        }
    }
};

void quantize_mmq(const Rows& x, void* y, ggml_type type, int64_t K, int64_t K_padded, int64_t M, cudaStream_t stream) {
    const int64_t block_num_y = (K_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 blocks(M, block_num_y, 1), threads(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    x.as([&](auto rows) {
        switch (mmq_get_q8_1_ds_layout(type)) {
            case MMQ_Q8_1_DS_LAYOUT_D4:
                quantize_mmq_q8_1_rows<MMQ_Q8_1_DS_LAYOUT_D4><<<blocks, threads, 0, stream>>>(rows, y, K, K_padded, M);
                break;
            case MMQ_Q8_1_DS_LAYOUT_DS4:
                quantize_mmq_q8_1_rows<MMQ_Q8_1_DS_LAYOUT_DS4><<<blocks, threads, 0, stream>>>(rows, y, K, K_padded, M);
                break;
            case MMQ_Q8_1_DS_LAYOUT_D2S6:
                quantize_mmq_q8_1_rows<MMQ_Q8_1_DS_LAYOUT_D2S6><<<blocks, threads, 0, stream>>>(rows, y, K, K_padded, M);
                break;
            default: GGML_ABORT("no q8_1 layout for type %d", int(type));
        }
    });
}

// `ggml_cuda_mul_mat_q`'s dense branch (mmq.cu, llama.cpp b12466, MIT): q8_1 activations, no fusion, no
// experts; the kernels are mmq.cuh's, instantiated by template-instances/mmq-instance-<type>.cu.
void mul_mat_q(ggml_backend_cuda_context& ctx, const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst,
               const Rows& rows) {
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
    if (rows.type == kF32) {
        quantize_mmq_q8_1_cuda((const float*)src1->data, nullptr, src1_q8_1.get(), src0->type, ne10,
                               nb11 / sizeof(float), nb12 / sizeof(float), nb13 / sizeof(float), ne10_padded, ne11,
                               ne12, ne13, stream);
    } else {
        quantize_mmq(rows, src1_q8_1.get(), src0->type, ne10, ne10_padded, ne11, stream);
    }
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

int matmul(int device, int type, const void* w, const void* x, int x_type, float* out, int64_t K, int64_t N,
           int64_t M, void* (*allocate)(size_t, void*), void* context, cudaStream_t stream) {
    if (!runs(type) || K <= 0 || N <= 0 || M <= 0 || K % MATRIX_ROW_PADDING) return cudaErrorInvalidValue;
    if (x_type != kF32 && x_type != kBF16 && x_type != kF16) return cudaErrorInvalidValue;
    try {
        Allocate temporaries{allocate, context};
        Call call(device, stream, &temporaries);
        auto& ctx = call.ctx();
        const Rows rows{x, x_type};
        Node node(ggml_type(type), w, static_cast<const float*>(x), out, K, N, M);
        const int kernel = choose(device, node);
        if (kernel == kNone) return cudaErrorNotSupported;
        if (x_type == kF32) {
            switch (kernel) {
                case kMmvq: ggml_cuda_mul_mat_vec_q(ctx, &node.w, &node.x, nullptr, &node.out); break;
                case kMmqFew: ggml_cuda_mul_mat_q_few(ctx, &node.w, &node.x, &node.out); break;
                default: mul_mat_q(ctx, &node.w, &node.x, &node.out, rows); break;
            }
        } else if (kernel == kMmvq) {
            // MMVQ's own launch over q8_1 rows quantized here (`ggml_cuda_op_mul_mat_vec_q`).
            const int64_t K_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
            ggml_cuda_pool_alloc<char> q8_1(ctx.pool(), M * K_padded / QK8_1 * sizeof(block_q8_1));
            const dim3 blocks((K_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, M, 1);
            rows.as([&](auto r) {
                quantize_q8_1_rows<<<blocks, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(r, (block_q8_1*)q8_1.get(), K,
                                                                                    K_padded);
            });
            CUDA_CHECK(cudaGetLastError());
            ggml_cuda_op_mul_mat_vec_q(ctx, &node.w, &node.x, &node.out, static_cast<const char*>(w), nullptr,
                                       q8_1.get(), out, 0, N, M, K_padded, stream);
        } else if (kernel == kMmqFew) {
            // Few rows: widened to f32 (a few rows' bytes), then ggml's own few-row MMQ.
            ggml_cuda_pool_alloc<float> wide(ctx.pool(), M * K);
            rows.as([&](auto r) {
                widen_rows<<<(M * K + 255) / 256, 256, 0, stream>>>(r, wide.get(), M * K);
            });
            CUDA_CHECK(cudaGetLastError());
            node.x.data = wide.get();
            ggml_cuda_mul_mat_q_few(ctx, &node.w, &node.x, &node.out);
        } else {
            mul_mat_q(ctx, &node.w, &node.x, &node.out, rows);
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
