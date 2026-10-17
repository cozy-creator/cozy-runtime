// SPDX-License-Identifier: Apache-2.0
// `torch.ops.cozy.int8_linear` and `fp8_linear`: the int8-convrot256/1 and fp8-rowwise/1
// linears, each one torch op. One Python call checks, allocates and launches: the activation
// quantized per row in one kernel (int8 after its rotation), then the GEMM with both scales and
// the bias in its epilogue; int8's M <= 4 rows take one kernel for all of it. `*_quantize` and
// `*_gemm` are the two halves, for producers that quantize once for several linears;
// `*_quantize` also folds the producer's norm, modulation and activation into the same pass
// (`cozy_int8::Prologue`).
// A launch out of device memory raises torch's OutOfMemoryError, which Runtime recovers; a GEMM
// tile that cannot run hands that call to cuBLASLt (`_int_mm`, `_scaled_mm`), said once. A
// build without the fp8 CUTLASS GEMM (sm89, sm121) serves fp8 on `_scaled_mm` by design.
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_int_mm.h>
#include <ATen/ops/_scaled_mm.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <atomic>
#include <cstdio>
#include <string>
#include <tuple>

#include "int8.h"

namespace {

using at::Tensor;
// The int8 GEMM this build carries: its TMA family's, else mma.sync's (`gemm_sm80.cu`).
#ifdef COZY_TMA
namespace int8_family = cozy_int8::COZY_FAMILY;
#else
namespace int8_family = cozy_int8::mma_sync;
#endif

// One 8-bit operand format: its name in messages, its torch dtype, the K multiple it takes.
struct Format {
    const char* name;
    at::ScalarType type;
    int64_t k_multiple;
};
const Format kInt8{"int8", at::kChar, 256};
const Format kFp8{"fp8", at::kFloat8_e4m3fn, 16};

int sm_count(int device) {
    static int counts[64] = {};
    int& count = counts[device & 63];
    if (count == 0) cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    return count;
}

bool is_half(at::ScalarType type) {
    TORCH_CHECK(type == at::kHalf || type == at::kBFloat16,
                "8-bit linear: activations and outputs are bf16 or fp16, not ", type);
    return type == at::kHalf;
}

bool aligned(const Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

// The activation as contiguous, 16-byte aligned [M, K] rows.
Tensor rows(const Format& f, const Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.dim() >= 1, f.name, "_linear: x must be a CUDA tensor");
    const int64_t K = x.size(-1);
    TORCH_CHECK(K % f.k_multiple == 0 && K <= cozy_int8::kMaxK, f.name,
                "_linear: K must be a multiple of ", f.k_multiple, " up to ", cozy_int8::kMaxK,
                ", not ", K);
    Tensor flat = x.reshape({-1, K}).contiguous();
    return aligned(flat) ? flat : flat.clone();
}

void check_weight(const Format& f, const Tensor& data, const Tensor& scale,
                  std::optional<Tensor>& bias, int64_t K, at::ScalarType out_dtype) {
    TORCH_CHECK(data.scalar_type() == f.type && data.dim() == 2 && data.size(1) == K &&
                    data.is_contiguous() && aligned(data),
                f.name, "_linear: data must be a contiguous ", f.type, " [N, ", K, "]");
    const int64_t N = data.size(0);
    TORCH_CHECK(N % 8 == 0, f.name, "_linear: N must be a multiple of 8, not ", N);
    TORCH_CHECK(scale.dtype() == at::kFloat && scale.numel() == N && scale.is_contiguous(),
                f.name, "_linear: scale must be a contiguous float32 with ", N, " values");
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == N, f.name, "_linear: bias must have ", N, " values");
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

void check(cudaError_t error, const char* what) {
    if (error == cudaSuccess) return;
    TORCH_CHECK_WITH(OutOfMemoryError, error != cudaErrorMemoryAllocation, "8-bit linear: ", what,
                     ": ", cudaGetErrorString(error));
    TORCH_CHECK(false, "8-bit linear: ", what, ": ", cudaGetErrorString(error));
}

// The CUTLASS GEMM of this build for `f`; cudaErrorNotSupported where it carries none.
cudaError_t gemm(const Format& f, const void* q, const float* scales, const Tensor& data,
                 const Tensor& scale, const std::optional<Tensor>& bias, Tensor& out, int64_t M,
                 int64_t config, cudaStream_t stream) {
    const void* b = bias.has_value() ? bias->const_data_ptr() : nullptr;
    const int N = int(data.size(0)), K = int(data.size(1)), sms = sm_count(data.get_device());
    const bool half = is_half(out.scalar_type());
    if (f.type == at::kChar) {
        Scratch scratch{data.options().dtype(at::kByte), {}};
        return int8_family::gemm(static_cast<const int8_t*>(q), data.const_data_ptr<int8_t>(),
                                 scales, scale.const_data_ptr<float>(), b, out.data_ptr(), half,
                                 int(M), N, K, sms, int(config), {&allocate, &scratch}, stream);
    }
#ifdef COZY_FP8_GEMM
    return cozy_int8::COZY_FAMILY::fp8_gemm(q, data.const_data_ptr(), scales,
                                            scale.const_data_ptr<float>(), b, out.data_ptr(), half,
                                            int(M), N, K, sms, int(config), stream);
#else
    return cudaErrorNotSupported;
#endif
}

// The GEMM's `error`: out of memory raises; anything else, cuBLASLt's GEMM (`_int_mm` with
// int32 out and more than 16 rows, or `_scaled_mm`) and the rest of the epilogue in torch ops.
// Said once, except where cuBLASLt is this build's fp8 GEMM.
void fall_back(const Format& f, cudaError_t error, const Tensor& q, const Tensor& scales,
               const Tensor& data, const Tensor& scale, const std::optional<Tensor>& bias,
               Tensor& out) {
    (void)cudaGetLastError();
    check(error == cudaErrorMemoryAllocation ? error : cudaSuccess, "the GEMM");
    static std::atomic<bool> said[2] = {false, false};
    const bool fp8 = f.type != at::kChar;
#ifdef COZY_FP8_GEMM
    const bool by_design = false;
#else
    const bool by_design = fp8;
#endif
    if (!by_design && !said[fp8].exchange(true))
        std::fprintf(stderr, "[%s] the CUTLASS GEMM did not run (M=%lld N=%lld K=%lld: %s); "
                     "cuBLASLt serves such calls\n", f.name, (long long)q.size(0),
                     (long long)data.size(0), (long long)q.size(1), cudaGetErrorString(error));
    const int64_t M = q.size(0);
    if (fp8) {
        Tensor rows = out.view({M, -1});
        at::_scaled_mm_out(rows, q, data.t(), scales.view({M, 1}), scale.view({1, -1}), {}, {},
                           out.scalar_type(), true);
        if (bias.has_value()) rows.add_(*bias);
        return;
    }
    const Tensor rows = M > 16 ? q : at::constant_pad_nd(q, {0, 0, 0, 17 - M});
    Tensor value = at::_int_mm(rows, data.t()).narrow(0, 0, M).to(at::kFloat);
    value.mul_(scales.view({M, 1})).mul_(scale.view({1, -1}));
    if (bias.has_value()) value.add_(*bias);
    out.view({M, -1}).copy_(value);
}

Tensor int8_linear(const Tensor& x, const Tensor& data, const Tensor& scale,
                   std::optional<Tensor> bias, at::ScalarType out_dtype) {
    const c10::cuda::OptionalCUDAGuard guard(x.device());
    const Tensor flat = rows(kInt8, x);
    const int64_t M = flat.size(0), K = flat.size(1);
    check_weight(kInt8, data, scale, bias, K, out_dtype);
    const bool x_half = is_half(flat.scalar_type()), out_half = is_half(out_dtype);
    Tensor out = output(x, data.size(0), out_dtype);
    if (M == 0) return out;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    const void* b = bias.has_value() ? bias->const_data_ptr() : nullptr;
    const cudaError_t one = cozy_int8::gemv(
        flat.const_data_ptr(), x_half, data.const_data_ptr<int8_t>(),
        scale.const_data_ptr<float>(), b, out.data_ptr(), out_half, int(M), int(data.size(0)),
        int(K), sm_count(x.get_device()), stream);
    if (one != cudaErrorNotSupported) {
        check(one, "the GEMV");
        return out;
    }
    // One buffer: the int8 rows, then their f32 scales 16-byte aligned.
    const int64_t scales_at = (M * K + 15) / 16 * 16;
    Tensor buffer = at::empty({scales_at + 4 * M}, flat.options().dtype(at::kChar));
    int8_t* q = buffer.mutable_data_ptr<int8_t>();
    float* scales = reinterpret_cast<float*>(q + scales_at);
    check(cozy_int8::rotate_quantize(flat.const_data_ptr(), x_half, q, scales, int(M), int(K),
                                     {}, stream),
          "the rotation");
    const cudaError_t error = gemm(kInt8, q, scales, data, scale, bias, out, M, -1, stream);
    if (error != cudaSuccess)
        fall_back(kInt8, error, buffer.narrow(0, 0, M * K).view({M, K}),
                  buffer.narrow(0, scales_at, 4 * M).view(at::kFloat), data, scale, bias, out);
    return out;
}

// A modulation table as `[E, K]` rows the kernel reads 16 bytes at a time: E divides M.
std::optional<Tensor> table(const std::optional<Tensor>& t, const Tensor& flat, const char* name) {
    if (!t.has_value()) return t;
    const int64_t M = flat.size(0), K = flat.size(1);
    TORCH_CHECK(t->device() == flat.device() && t->scalar_type() == flat.scalar_type() &&
                    t->size(-1) == K,
                "8-bit quantize: ", name, " must be a ", flat.scalar_type(), " table of ", K,
                "-wide rows on the activation's device");
    Tensor rows = t->reshape({-1, K});
    TORCH_CHECK(rows.size(0) > 0 && M % rows.size(0) == 0, "8-bit quantize: ", name, "'s ",
                rows.size(0), " rows do not divide the activation's ", M);
    if (rows.stride(1) != 1 || rows.stride(0) % 8 || !aligned(rows)) rows = rows.contiguous();
    return aligned(rows) ? rows : rows.clone();
}

std::tuple<Tensor, Tensor> quantize(const Format& f, const Tensor& x,
                                    const std::optional<Tensor>& scale,
                                    const std::optional<Tensor>& shift, int64_t norm, double eps,
                                    int64_t activation) {
    const c10::cuda::OptionalCUDAGuard guard(x.device());
    const Tensor flat = rows(f, x);
    const int64_t M = flat.size(0), K = flat.size(1);
    TORCH_CHECK(norm >= cozy_int8::kNormNone && norm <= cozy_int8::kNormRms &&
                    activation >= cozy_int8::kActNone && activation <= cozy_int8::kActGeluTanh,
                f.name, "_quantize: norm ", norm, " / activation ", activation);
    Tensor q = at::empty({M, K}, flat.options().dtype(f.type));
    Tensor scales = at::empty({M}, flat.options().dtype(at::kFloat));
    if (M == 0) return {q, scales};
    const auto s = table(scale, flat, "scale"), b = table(shift, flat, "shift");
    TORCH_CHECK(!s || !b || s->size(0) == b->size(0), f.name, "_quantize: scale and shift differ");
    const auto& any = s ? s : b;
    cozy_int8::Prologue prologue;
    prologue.scale = s ? s->const_data_ptr() : nullptr;
    prologue.shift = b ? b->const_data_ptr() : nullptr;
    prologue.scale_stride = s ? s->stride(0) : 0;
    prologue.shift_stride = b ? b->stride(0) : 0;
    prologue.rows_per_entry = any ? int(M / any->size(0)) : 1;
    prologue.norm = int(norm);
    prologue.eps = float(eps);
    prologue.activation = int(activation);
    const bool half = is_half(flat.scalar_type());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    check(f.type == at::kChar
              ? cozy_int8::rotate_quantize(flat.const_data_ptr(), half, q.mutable_data_ptr<int8_t>(),
                                           scales.mutable_data_ptr<float>(), int(M), int(K),
                                           prologue, stream)
              : cozy_int8::fp8_quantize(flat.const_data_ptr(), half, q.mutable_data_ptr(),
                                        scales.mutable_data_ptr<float>(), int(M), int(K),
                                        prologue, stream),
          "the quantization");
    return {q, scales};
}

Tensor gemm_op(const Format& f, const Tensor& q, const Tensor& scales, const Tensor& data,
               const Tensor& scale, std::optional<Tensor> bias, at::ScalarType out_dtype,
               int64_t config) {
    const c10::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == f.type && q.dim() == 2 && q.is_contiguous() &&
                    aligned(q),
                f.name, "_gemm: q must be a contiguous ", f.type, " [M, K]");
    const int64_t M = q.size(0);
    TORCH_CHECK(scales.dtype() == at::kFloat && scales.numel() == M && scales.is_contiguous(),
                f.name, "_gemm: scales must be a contiguous float32 with ", M, " values");
    check_weight(f, data, scale, bias, q.size(1), out_dtype);
    is_half(out_dtype);
    Tensor out = at::empty({M, data.size(0)}, q.options().dtype(out_dtype));
    if (M == 0) return out;
    const cudaError_t error =
        gemm(f, q.const_data_ptr(), scales.const_data_ptr<float>(), data, scale, bias, out, M,
             config, at::cuda::getCurrentCUDAStream(q.get_device()).stream());
    if (error != cudaSuccess) {
        TORCH_CHECK(config < 0 || error != cudaErrorNotSupported, f.name, "_gemm: config ",
                    config, " cannot take M=", M, " N=", data.size(0), " K=", q.size(1));
        fall_back(f, error, q, scales, data, scale, bias, out);
    }
    return out;
}

std::tuple<Tensor, Tensor> int8_quantize(const Tensor& x, const std::optional<Tensor>& scale,
                                         const std::optional<Tensor>& shift, int64_t norm,
                                         double eps, int64_t activation) {
    return quantize(kInt8, x, scale, shift, norm, eps, activation);
}

std::tuple<Tensor, Tensor> fp8_quantize(const Tensor& x, const std::optional<Tensor>& scale,
                                        const std::optional<Tensor>& shift, int64_t norm,
                                        double eps, int64_t activation) {
    return quantize(kFp8, x, scale, shift, norm, eps, activation);
}

Tensor int8_gemm(const Tensor& q, const Tensor& scales, const Tensor& data, const Tensor& scale,
                 std::optional<Tensor> bias, at::ScalarType out_dtype, int64_t config) {
    return gemm_op(kInt8, q, scales, data, scale, std::move(bias), out_dtype, config);
}

Tensor fp8_gemm(const Tensor& q, const Tensor& scales, const Tensor& data, const Tensor& scale,
                std::optional<Tensor> bias, at::ScalarType out_dtype, int64_t config) {
    return gemm_op(kFp8, q, scales, data, scale, std::move(bias), out_dtype, config);
}

Tensor fp8_linear(const Tensor& x, const Tensor& data, const Tensor& scale,
                  std::optional<Tensor> bias, at::ScalarType out_dtype) {
    const auto [q, scales] = quantize(kFp8, x, {}, {}, 0, 0., 0);
    auto sizes = x.sizes().vec();
    sizes.back() = data.size(0);
    return gemm_op(kFp8, q, scales, data, scale, std::move(bias), out_dtype, -1).view(sizes);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(cozy, m) {
    for (const char* f : {"int8", "fp8"}) {
        const std::string name(f);
        m.def((name + "_linear(Tensor x, Tensor data, Tensor scale, Tensor? bias, "
                      "ScalarType out_dtype) -> Tensor").c_str());
        m.def((name + "_quantize(Tensor x, Tensor? scale=None, Tensor? shift=None, int norm=0, "
                      "float eps=0., int activation=0) -> (Tensor, Tensor)").c_str());
        m.def((name + "_gemm(Tensor q, Tensor scales, Tensor data, Tensor scale, Tensor? bias, "
                      "ScalarType out_dtype, int config=-1) -> Tensor").c_str());
    }
}

TORCH_LIBRARY_IMPL(cozy, CUDA, m) {
    m.impl("int8_linear", &int8_linear);
    m.impl("int8_quantize", &int8_quantize);
    m.impl("int8_gemm", &int8_gemm);
    m.impl("fp8_linear", &fp8_linear);
    m.impl("fp8_quantize", &fp8_quantize);
    m.impl("fp8_gemm", &fp8_gemm);
}
