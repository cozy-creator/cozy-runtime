// SPDX-License-Identifier: Apache-2.0
// `torch.ops.cozy.int8_linear`: the int8-convrot256/1 linear as one torch op. One Python call
// checks, allocates and launches: M <= 4 rows in one kernel (rotate, quantize, multiply), more
// in two (rotate + quantize, then the GEMM with both scales and the bias in its epilogue).
// `int8_quantize` and `int8_gemm` are its two halves, for producers that quantize elsewhere.
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <tuple>

#include "int8.h"

namespace {

using at::Tensor;

int sm_count(int device) {
    static int counts[64] = {};
    int& count = counts[device & 63];
    if (count == 0) cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    return count;
}

bool is_half(at::ScalarType type) {
    TORCH_CHECK(type == at::kHalf || type == at::kBFloat16,
                "int8_linear: activations and outputs are bf16 or fp16, not ", type);
    return type == at::kHalf;
}

bool aligned(const Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

// The activation as contiguous, 16-byte aligned [M, K] rows.
Tensor rows(const Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.dim() >= 1, "int8_linear: x must be a CUDA tensor");
    const int64_t K = x.size(-1);
    TORCH_CHECK(K % 256 == 0 && K <= cozy_int8::kMaxK,
                "int8_linear: K must be a multiple of 256 up to ", cozy_int8::kMaxK, ", not ", K);
    Tensor flat = x.reshape({-1, K}).contiguous();
    return aligned(flat) ? flat : flat.clone();
}

void check_weight(const Tensor& data, const Tensor& scale, std::optional<Tensor>& bias,
                  int64_t K, at::ScalarType out_dtype) {
    TORCH_CHECK(data.dtype() == at::kChar && data.dim() == 2 && data.size(1) == K &&
                    data.is_contiguous() && aligned(data),
                "int8_linear: data must be a contiguous int8 [N, ", K, "]");
    const int64_t N = data.size(0);
    TORCH_CHECK(N % 8 == 0, "int8_linear: N must be a multiple of 8, not ", N);
    TORCH_CHECK(scale.dtype() == at::kFloat && scale.numel() == N && scale.is_contiguous(),
                "int8_linear: scale must be a contiguous float32 with ", N, " values");
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == N, "int8_linear: bias must have ", N, " values");
        // The epilogue reads it in the output dtype.
        if (bias->scalar_type() != out_dtype || !bias->is_contiguous())
            bias = bias->to(out_dtype).contiguous();
    }
}

struct Scratch {
    at::TensorOptions options;
    Tensor held;
};

void* allocate(size_t bytes, void* context) {
    auto* scratch = static_cast<Scratch*>(context);
    scratch->held = at::empty({int64_t(bytes)}, scratch->options);
    return scratch->held.data_ptr();
}

Tensor output(const Tensor& x, int64_t N, at::ScalarType out_dtype) {
    auto sizes = x.sizes().vec();
    sizes.back() = N;
    return at::empty(sizes, x.options().dtype(out_dtype));
}

void gemm(const int8_t* q, const float* scales, const Tensor& data, const Tensor& scale,
          const std::optional<Tensor>& bias, Tensor& out, int64_t M, int64_t config,
          cudaStream_t stream) {
    Scratch scratch{data.options().dtype(at::kByte), {}};
    const int64_t N = data.size(0), K = data.size(1);
    const bool ran = cozy_int8::gemm(
        q, data.const_data_ptr<int8_t>(), scales, scale.const_data_ptr<float>(),
        bias.has_value() ? bias->const_data_ptr() : nullptr, out.data_ptr(),
        is_half(out.scalar_type()), int(M), int(N), int(K), sm_count(data.get_device()),
        int(config), {&allocate, &scratch}, stream);
    TORCH_CHECK(ran, "int8_linear: no int8 GEMM ran for M=", M, " N=", N, " K=", K,
                " config=", config);
}

Tensor int8_linear(const Tensor& x, const Tensor& data, const Tensor& scale,
                   std::optional<Tensor> bias, at::ScalarType out_dtype) {
    const c10::cuda::OptionalCUDAGuard guard(x.device());
    const Tensor flat = rows(x);
    const int64_t M = flat.size(0), K = flat.size(1);
    check_weight(data, scale, bias, K, out_dtype);
    const bool x_half = is_half(flat.scalar_type()), out_half = is_half(out_dtype);
    Tensor out = output(x, data.size(0), out_dtype);
    if (M == 0) return out;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    const void* b = bias.has_value() ? bias->const_data_ptr() : nullptr;
    if (x_half == out_half &&
        cozy_int8::gemv(flat.const_data_ptr(), x_half, data.const_data_ptr<int8_t>(),
                        scale.const_data_ptr<float>(), b, out.data_ptr(), out_half, int(M),
                        int(data.size(0)), int(K), sm_count(x.get_device()), stream))
        return out;
    // One buffer: the int8 rows, then their f32 scales 16-byte aligned.
    const int64_t scales_at = (M * K + 15) / 16 * 16;
    Tensor buffer = at::empty({scales_at + 4 * M}, flat.options().dtype(at::kChar));
    int8_t* q = buffer.mutable_data_ptr<int8_t>();
    float* scales = reinterpret_cast<float*>(q + scales_at);
    cozy_int8::rotate_quantize(flat.const_data_ptr(), x_half, q, scales, int(M), int(K), stream);
    gemm(q, scales, data, scale, bias, out, M, -1, stream);
    return out;
}

std::tuple<Tensor, Tensor> int8_quantize(const Tensor& x) {
    const c10::cuda::OptionalCUDAGuard guard(x.device());
    const Tensor flat = rows(x);
    const int64_t M = flat.size(0), K = flat.size(1);
    Tensor q = at::empty({M, K}, flat.options().dtype(at::kChar));
    Tensor scales = at::empty({M}, flat.options().dtype(at::kFloat));
    if (M > 0)
        cozy_int8::rotate_quantize(flat.const_data_ptr(), is_half(flat.scalar_type()),
                                   q.mutable_data_ptr<int8_t>(), scales.mutable_data_ptr<float>(),
                                   int(M), int(K),
                                   at::cuda::getCurrentCUDAStream(x.get_device()).stream());
    return {q, scales};
}

Tensor int8_gemm(const Tensor& q, const Tensor& scales, const Tensor& data, const Tensor& scale,
                 std::optional<Tensor> bias, at::ScalarType out_dtype, int64_t config) {
    const c10::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.is_cuda() && q.dtype() == at::kChar && q.dim() == 2 && q.is_contiguous() &&
                    aligned(q),
                "int8_gemm: q must be a contiguous int8 [M, K]");
    const int64_t M = q.size(0);
    TORCH_CHECK(scales.dtype() == at::kFloat && scales.numel() == M && scales.is_contiguous(),
                "int8_gemm: scales must be a contiguous float32 with ", M, " values");
    check_weight(data, scale, bias, q.size(1), out_dtype);
    is_half(out_dtype);
    Tensor out = at::empty({M, data.size(0)}, q.options().dtype(out_dtype));
    if (M > 0)
        gemm(q.const_data_ptr<int8_t>(), scales.const_data_ptr<float>(), data, scale, bias, out,
             M, config, at::cuda::getCurrentCUDAStream(q.get_device()).stream());
    return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cozy, m) {
    m.def("int8_linear(Tensor x, Tensor data, Tensor scale, Tensor? bias, ScalarType out_dtype)"
          " -> Tensor");
    m.def("int8_quantize(Tensor x) -> (Tensor, Tensor)");
    m.def("int8_gemm(Tensor q, Tensor scales, Tensor data, Tensor scale, Tensor? bias, "
          "ScalarType out_dtype, int config=-1) -> Tensor");
}

TORCH_LIBRARY_IMPL(cozy, CUDA, m) {
    m.impl("int8_linear", &int8_linear);
    m.impl("int8_quantize", &int8_quantize);
    m.impl("int8_gemm", &int8_gemm);
}
