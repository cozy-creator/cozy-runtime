// SPDX-License-Identifier: Apache-2.0
// `_kernels._conv`: convolution kernels. A causal-conv video encoder's, comfy-kitchen's (Comfy-Org/comfy-kitchen #167 at
// Runtime's `kernel_sources` pin, `csrc/conv/UPSTREAM.md`), unmodified: per-frame GroupNorm, SiLU and the convolution's
// padding (reflect or zero in space, zero frames before) in one pass, NDHWC in and out (`group_norm_pad3d.cu`), and an
// NDHWC conv3d accumulating in fp16 with its bias and a residual in the epilogue (`cutlass_conv3d_fp16.cu`). And Cozy's
// int8-convrot conv3d with the passes that feed it (`conv/conv8.cuh`, cr-327). Arguments are raw device pointers,
// shapes, element strides and a stream; the caller allocates (`cozy_runtime.accel.fusion._conv`, `_conv8`).
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>

extern "C" void launch_group_norm_silu_pad3d(
    const void *x, const void *gamma, const void *beta, void *out, void *workspace, const void *stats_in, int B,
    int C, int T, int H, int W, int G, float eps, int left, int right, int top, int bottom, int front, bool silu,
    bool zero_pad, int64_t row_stride, int dtype_code, cudaStream_t stream);

extern "C" bool launch_cutlass_fp16_conv3d(
    const void *x, const void *w, const void *bias, const void *resid, bool resid_full, void *out, int N, int D,
    int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q, int sd, int sh, int sw, int config,
    int xs_w, int xs_h, int xs_d, int xs_n, int os_w, int os_h, int os_d, int os_n, cudaStream_t stream);

extern "C" int cozy_conv8(const int8_t *x, const float *xs, const int8_t *w, const float *ws, const float *bias,
                          const void *res, void *out, int n, int d, int h, int wd, int c, int k, int t, int r, int s,
                          int pt, int pr, int ps, int z, int p, int q, bool ws_tap, bool half, int config,
                          cudaStream_t stream);
extern "C" int cozy_conv8_stats(const void *x, double *acc, int n, int64_t voxels, int c, int groups, bool half, int sms,
                                cudaStream_t stream);
extern "C" int cozy_conv8_quant(const void *x, const float *a, const float *b, int8_t *q, float *qs, int64_t voxels,
                                int64_t per_n, bool silu, bool half, int sms, cudaStream_t stream);
extern "C" int cozy_conv8_temporal(const void *x, const float *a, const float *b, const float *w, const float *bias,
                                   int8_t *q, float *qs, int n, int t, int hw, bool half, int sms, cudaStream_t stream);

namespace nb = nanobind;
using namespace nb::literals;

namespace {

void *at(uintptr_t p) { return reinterpret_cast<void *>(p); }

// The kernel's launch, then CUDA's error code: 0 when it launched. A shape the launcher refuses raises ValueError
// here, before it: MSVC assumes an extern "C" function never throws (/EHsc), so its throw is never caught across it.
int norm_pad(uintptr_t x, uintptr_t gamma, uintptr_t beta, uintptr_t out, uintptr_t workspace, int batch,
             int channels, int frames, int height, int width, int groups, float eps, int left, int right,
             int top, int bottom, int front, bool silu, bool zero_pad, int64_t row_stride, bool bf16,
             uintptr_t stream) {
  if (channels % 8 != 0 || channels / 8 > 256 || (gamma && (groups < 1 || channels % groups != 0 || groups > 1024)))
    throw nb::value_error("norm_pad: unsupported channel/group count");
  if (left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0)
    throw nb::value_error("norm_pad: padding must be non-negative");
  if (!zero_pad && (left >= width || right >= width || top >= height || bottom >= height))
    throw nb::value_error("norm_pad: reflect padding must be smaller than the input");
  if (static_cast<int64_t>(batch) * (frames + front) > 65535)
    throw nb::value_error("norm_pad: batch * frames must be <= 65535");
  launch_group_norm_silu_pad3d(at(x), gamma ? at(gamma) : nullptr, beta ? at(beta) : nullptr, at(out),
                               at(workspace), nullptr, batch, channels, frames, height, width, groups, eps, left,
                               right, top, bottom, front, silu, zero_pad, row_stride, bf16 ? 2 : 1,
                               reinterpret_cast<cudaStream_t>(stream));
  return cudaGetLastError();
}

// 0 when it ran, -1 when no tile config serves the shape (the caller convolves another way), else CUDA's error.
int conv(uintptr_t x, uintptr_t weight, uintptr_t bias, uintptr_t residual, bool residual_full, uintptr_t out,
         int n, int d, int h, int w, int c, int k, int t, int r, int s, int z, int p, int q, int sd, int sh, int sw,
         int config, std::array<int, 4> x_strides, std::array<int, 4> out_strides, uintptr_t stream) {
  const bool ran = launch_cutlass_fp16_conv3d(
      at(x), at(weight), at(bias), at(residual), residual_full, at(out), n, d, h, w, c, k, t, r, s, z, p, q, sd,
      sh, sw, config, x_strides[0], x_strides[1], x_strides[2], x_strides[3], out_strides[0], out_strides[1],
      out_strides[2], out_strides[3], reinterpret_cast<cudaStream_t>(stream));
  const int code = cudaGetLastError();
  return code ? code : ran ? 0 : -1;
}

// int8-convrot conv3d, NDHWC: x int8 [n, d, h, w, c] with one f32 scale a voxel (xs), the filter int8 [k, t, r, s, c]
// with one f32 scale a (row, tap) (ws_tap) or a row; stride 1, zero padding (pt, pr, ps) before and (z, p, q) out. The
// bias (f32, or 0) and a residual like `out` (or 0) in the epilogue. 0 when it ran, -1 when `config` does not serve the
// shape on this card, else CUDA's error.
int conv8(uintptr_t x, uintptr_t xs, uintptr_t weight, uintptr_t ws, uintptr_t bias, uintptr_t residual, uintptr_t out,
          int n, int d, int h, int w, int c, int k, int t, int r, int s, int pt, int pr, int ps, int z, int p, int q,
          bool ws_tap, bool half, int config, uintptr_t stream) {
  if (c % 64 != 0 || k % 8 != 0 || t > 8 || r > 8 || s > 8 || pt < 0 || pr < 0 || ps < 0 || z <= 0 || p <= 0 ||
      q <= 0 || static_cast<int64_t>(n) * z * p * q > INT32_MAX || z + t - 1 - 2 * pt != d ||
      p + r - 1 - 2 * pr != h || q + s - 1 - 2 * ps != w)
    throw nb::value_error("conv8: unsupported shape");
  const int code = cozy_conv8(static_cast<const int8_t *>(at(x)), static_cast<const float *>(at(xs)),
                              static_cast<const int8_t *>(at(weight)), static_cast<const float *>(at(ws)),
                              bias ? static_cast<const float *>(at(bias)) : nullptr, residual ? at(residual) : nullptr,
                              at(out), n, d, h, w, c, k, t, r, s, pt, pr, ps, z, p, q, ws_tap, half, config,
                              reinterpret_cast<cudaStream_t>(stream));
  return code ? code : static_cast<int>(cudaGetLastError());
}

// acc [n, groups, 2] f64 += each group's (sum, sum of squares) over x [n, voxels, c] (fp16 or bf16).
int conv8_stats(uintptr_t x, uintptr_t acc, int n, int64_t voxels, int c, int groups, bool half, int sms,
                uintptr_t stream) {
  if (c % 8 != 0 || c / 8 > 256 || 256 % (c / 8) != 0 || groups < 1 || c % groups != 0 || (c / groups) % 8 != 0 ||
      n < 1 || n > 65535)
    throw nb::value_error("conv8_stats: unsupported channel/group count");
  return cozy_conv8_stats(at(x), static_cast<double *>(at(acc)), n, voxels, c, groups, half, sms,
                          reinterpret_cast<cudaStream_t>(stream));
}

// q int8 [voxels, 512] and qs f32 [voxels]: silu(a x + b) (or a x + b), rotated by H256 and quantized a voxel; a, b f32
// [voxels / per_n, 512].
int conv8_quant(uintptr_t x, uintptr_t a, uintptr_t b, uintptr_t q, uintptr_t qs, int64_t voxels, int64_t per_n,
                bool silu, bool half, int sms, uintptr_t stream) {
  if (per_n < 1 || voxels % per_n != 0) throw nb::value_error("conv8_quant: voxels are whole samples");
  return cozy_conv8_quant(at(x), static_cast<const float *>(at(a)), static_cast<const float *>(at(b)),
                          static_cast<int8_t *>(at(q)), static_cast<float *>(at(qs)), voxels, per_n, silu, half, sms,
                          reinterpret_cast<cudaStream_t>(stream));
}

// The same of the depthwise temporal conv (kernel 5, zero padding 2, w f32 [512, 5], bias f32 or 0) of
// silu(a x + b), x [n, t, hw, 512].
int conv8_temporal(uintptr_t x, uintptr_t a, uintptr_t b, uintptr_t w, uintptr_t bias, uintptr_t q, uintptr_t qs, int n,
                   int t, int hw, bool half, int sms, uintptr_t stream) {
  if (n < 1 || t < 1 || hw < 1) throw nb::value_error("conv8_temporal: empty");
  return cozy_conv8_temporal(at(x), static_cast<const float *>(at(a)), static_cast<const float *>(at(b)),
                             static_cast<const float *>(at(w)), bias ? static_cast<const float *>(at(bias)) : nullptr,
                             static_cast<int8_t *>(at(q)), static_cast<float *>(at(qs)), n, t, hw, half, sms,
                             reinterpret_cast<cudaStream_t>(stream));
}

}  // namespace

NB_MODULE(_conv, m) {
  m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
  m.def("norm_pad", &norm_pad, "x"_a, "gamma"_a, "beta"_a, "out"_a, "workspace"_a, "batch"_a, "channels"_a,
        "frames"_a, "height"_a, "width"_a, "groups"_a, "eps"_a, "left"_a, "right"_a, "top"_a, "bottom"_a,
        "front"_a, "silu"_a, "zero_pad"_a, "row_stride"_a, "bf16"_a, "stream"_a);
  m.def("conv", &conv, "x"_a, "weight"_a, "bias"_a, "residual"_a, "residual_full"_a, "out"_a, "n"_a, "d"_a,
        "h"_a, "w"_a, "c"_a, "k"_a, "t"_a, "r"_a, "s"_a, "z"_a, "p"_a, "q"_a, "sd"_a, "sh"_a, "sw"_a, "config"_a,
        "x_strides"_a, "out_strides"_a, "stream"_a);
  m.def("conv8", &conv8, "x"_a, "xs"_a, "weight"_a, "ws"_a, "bias"_a, "residual"_a, "out"_a, "n"_a, "d"_a, "h"_a, "w"_a,
        "c"_a, "k"_a, "t"_a, "r"_a, "s"_a, "pt"_a, "pr"_a, "ps"_a, "z"_a, "p"_a, "q"_a, "ws_tap"_a, "half"_a, "config"_a,
        "stream"_a);
  m.def("conv8_stats", &conv8_stats, "x"_a, "acc"_a, "n"_a, "voxels"_a, "c"_a, "groups"_a, "half"_a, "sms"_a,
        "stream"_a);
  m.def("conv8_quant", &conv8_quant, "x"_a, "a"_a, "b"_a, "q"_a, "qs"_a, "voxels"_a, "per_n"_a, "silu"_a, "half"_a,
        "sms"_a, "stream"_a);
  m.def("conv8_temporal", &conv8_temporal, "x"_a, "a"_a, "b"_a, "w"_a, "bias"_a, "q"_a, "qs"_a, "n"_a, "t"_a, "hw"_a,
        "half"_a, "sms"_a, "stream"_a);
}
