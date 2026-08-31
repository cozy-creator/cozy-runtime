/*
 * Derived from Comfy-Org/comfy-kitchen v0.2.31
 * (7c6ca3a5b63857d42c2d49777d6afb69de23f13f).
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * Modifications Copyright (c) 2026 Cozy contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;

int map_dtype_to_code(const nb::dlpack::dtype &dtype) {
  if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Float) {
    if (dtype.bits == 32) return 0;
    if (dtype.bits == 16) return 1;
  } else if (dtype.code == (uint8_t)nb::dlpack::dtype_code::Bfloat &&
             dtype.bits == 16) {
    return 2;
  }
  return -1;
}

extern "C" void launch_rms_rope_kernel(
    const void *q, const void *k, const void *freqs, const void *q_scale,
    const void *k_scale, void *q_out, void *k_out, int64_t batch,
    int64_t dim1, int64_t dim2, int64_t head_dim, int64_t rot_dim,
    int64_t freqs_batch, int64_t freqs_dim1, int64_t freqs_dim2,
    int64_t q_s0, int64_t q_s1, int64_t q_s2, int64_t q_s3,
    int64_t k_s0, int64_t k_s1, int64_t k_s2, int64_t k_s3,
    int64_t qo_s0, int64_t qo_s1, int64_t qo_s2, int64_t qo_s3,
    int64_t ko_s0, int64_t ko_s1, int64_t ko_s2, int64_t ko_s3,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3,
    int64_t f_s4, int64_t f_s5, int64_t qs_stride, int64_t ks_stride,
    float epsilon, int input_dtype_code, int freqs_dtype_code,
    int scale_dtype_code, bool has_k, bool split_half, cudaStream_t stream);

void rms_rope_split_half(
    nb::ndarray<nb::device::cuda> q, nb::ndarray<nb::device::cuda> k,
    nb::ndarray<nb::device::cuda> freqs,
    nb::ndarray<nb::device::cuda> q_scale,
    nb::ndarray<nb::device::cuda> k_scale,
    nb::ndarray<nb::device::cuda> q_out,
    nb::ndarray<nb::device::cuda> k_out, float epsilon,
    uintptr_t stream_ptr, int64_t rot_dim = 0) {
  if (q.ndim() != 4 || k.ndim() != 4 || q_out.ndim() != 4 ||
      k_out.ndim() != 4) {
    throw std::runtime_error("Q/K inputs and outputs must be 4D");
  }
  for (int axis = 0; axis < 4; ++axis) {
    if (k.shape(axis) != q.shape(axis) || q_out.shape(axis) != q.shape(axis) ||
        k_out.shape(axis) != q.shape(axis)) {
      throw std::runtime_error("Q/K input and output shapes must match");
    }
  }

  const int64_t batch = q.shape(0), dim1 = q.shape(1), dim2 = q.shape(2);
  const int64_t head_dim = q.shape(3);
  if (head_dim < 32 || head_dim % 32 != 0) {
    throw std::runtime_error("head_dim must be a positive multiple of 32");
  }
  const int64_t rot = rot_dim > 0 ? rot_dim : head_dim;
  if (rot % 2 != 0 || rot > head_dim) {
    throw std::runtime_error("rot_dim must be even and <= head_dim");
  }
  if (freqs.ndim() != 6 ||
      (freqs.shape(0) != 1 && freqs.shape(0) != batch) ||
      (freqs.shape(1) != 1 && freqs.shape(1) != dim1) ||
      (freqs.shape(2) != 1 && freqs.shape(2) != dim2) ||
      freqs.shape(3) != rot / 2 || freqs.shape(4) != 2 ||
      freqs.shape(5) != 2) {
    throw std::runtime_error("freqs shape must broadcast to Q/K");
  }
  if (q_scale.ndim() != 1 || k_scale.ndim() != 1 ||
      q_scale.shape(0) != head_dim || k_scale.shape(0) != head_dim) {
    throw std::runtime_error("scales must be 1D head_dim tensors");
  }

  const int input_code = map_dtype_to_code(q.dtype());
  const int freqs_code = map_dtype_to_code(freqs.dtype());
  const int scale_code = map_dtype_to_code(q_scale.dtype());
  if ((input_code != 1 && input_code != 2) ||
      map_dtype_to_code(k.dtype()) != input_code ||
      map_dtype_to_code(q_out.dtype()) != input_code ||
      map_dtype_to_code(k_out.dtype()) != input_code) {
    throw std::runtime_error("Q/K inputs and outputs must share FP16/BF16 dtype");
  }
  if (freqs_code < 0 || scale_code < 0 ||
      map_dtype_to_code(k_scale.dtype()) != scale_code) {
    throw std::runtime_error("frequency/scale dtype is unsupported");
  }

  launch_rms_rope_kernel(
      q.data(), k.data(), freqs.data(), q_scale.data(), k_scale.data(),
      q_out.data(), k_out.data(), batch, dim1, dim2, head_dim, rot,
      freqs.shape(0), freqs.shape(1), freqs.shape(2), q.stride(0),
      q.stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1),
      k.stride(2), k.stride(3), q_out.stride(0), q_out.stride(1),
      q_out.stride(2), q_out.stride(3), k_out.stride(0), k_out.stride(1),
      k_out.stride(2), k_out.stride(3), freqs.stride(0), freqs.stride(1),
      freqs.stride(2), freqs.stride(3), freqs.stride(4), freqs.stride(5),
      q_scale.stride(0), k_scale.stride(0), epsilon, input_code, freqs_code,
      scale_code, true, true, reinterpret_cast<cudaStream_t>(stream_ptr));
}

NB_MODULE(_C, m) {
  m.def("rms_rope_split_half", &rms_rope_split_half,
        nb::arg("q"), nb::arg("k"), nb::arg("freqs"), nb::arg("q_scale"),
        nb::arg("k_scale"), nb::arg("q_out"), nb::arg("k_out"),
        nb::arg("epsilon"), nb::arg("stream_ptr"), nb::arg("rot_dim") = 0);
}
