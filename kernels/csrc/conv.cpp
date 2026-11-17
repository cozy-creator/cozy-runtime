// SPDX-License-Identifier: Apache-2.0
// `_kernels._conv`: a causal-conv video encoder's kernels, comfy-kitchen's (Comfy-Org/comfy-kitchen #167 at Runtime's
// `kernel_sources` pin, `csrc/conv/UPSTREAM.md`), unmodified: per-frame GroupNorm, SiLU and the convolution's padding
// (reflect or zero in space, zero frames before) in one pass, NDHWC in and out (`group_norm_pad3d.cu`), and an NDHWC
// conv3d accumulating in fp16 with its bias and a residual in the epilogue (`cutlass_conv3d_fp16.cu`). Arguments are raw
// device pointers, shapes, element strides and a stream; the caller allocates (`cozy_runtime.accel.fusion._conv`).
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

}  // namespace

NB_MODULE(_conv, m) {
  m.def("error_name", [](int code) { return cudaGetErrorName(cudaError_t(code)); });
  m.def("norm_pad", &norm_pad, "x"_a, "gamma"_a, "beta"_a, "out"_a, "workspace"_a, "batch"_a, "channels"_a,
        "frames"_a, "height"_a, "width"_a, "groups"_a, "eps"_a, "left"_a, "right"_a, "top"_a, "bottom"_a,
        "front"_a, "silu"_a, "zero_pad"_a, "row_stride"_a, "bf16"_a, "stream"_a);
  m.def("conv", &conv, "x"_a, "weight"_a, "bias"_a, "residual"_a, "residual_full"_a, "out"_a, "n"_a, "d"_a,
        "h"_a, "w"_a, "c"_a, "k"_a, "t"_a, "r"_a, "s"_a, "z"_a, "p"_a, "q"_a, "sd"_a, "sh"_a, "sw"_a, "config"_a,
        "x_strides"_a, "out_strides"_a, "stream"_a);
}
