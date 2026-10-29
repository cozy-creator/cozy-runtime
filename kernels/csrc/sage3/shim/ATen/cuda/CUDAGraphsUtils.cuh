// SPDX-License-Identifier: Apache-2.0
// SageAttention3's `params.h` includes this for `at::cuda::philox::unpack`, which it never calls.
#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
