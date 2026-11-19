// SPDX-License-Identifier: Apache-2.0
// The step launcher's device code (`engine.cpp`).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cozy_engine {

// A ring slot: its sequence number (0 while it is written), the record's length, then the record.
constexpr int64_t kHeader = 16;

// The WHILE body's last node: `again(handle, go)` sets the loop's condition to `*go != 0`.
const void* again_kernel();

// One record of `nbytes` from device `src` into slot `*counter % slots` of the ring, then `*counter += 1`.
cudaError_t report(void* ring, int64_t slots, int64_t slot_bytes, unsigned long long* counter, const void* src,
                   int64_t nbytes, cudaStream_t stream);

}  // namespace cozy_engine
