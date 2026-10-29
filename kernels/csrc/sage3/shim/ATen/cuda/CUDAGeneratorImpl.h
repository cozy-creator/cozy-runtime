// SPDX-License-Identifier: Apache-2.0
// SageAttention3's `params.h` holds ATen's Philox state for dropout, which `_sage3` never runs.
#pragma once

namespace at {
struct PhiloxCudaState {};
}  // namespace at
