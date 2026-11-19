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

// A WHILE body's memset and memcpy nodes, as kernels (`engine.cpp`'s `lower`, as cozy-runtime #1860 lowered
// ResidentRepeat's): driver 615 crashes in a conditional body holding a memcpy node (xs-213, engine-routes
// 44614698658), which CUDA's guide allows, so a body holds kernel, empty and child-graph nodes only, on every
// driver. `fill_rows(dst, pitch, row_bytes, rows, word, wide)`: `word` repeats the value to four bytes.
// `copy_rows(dst, src, dst_pitch, dst_slice, src_pitch, src_slice, width, height, depth, wide)`. `wide`
// moves 16 bytes a thread, for 16-aligned addresses, pitches and rows.
const void* fill_rows_kernel();
const void* copy_rows_kernel();

// One record of `nbytes` from device `src` into slot `*counter % slots` of the ring, then `*counter += 1`.
cudaError_t report(void* ring, int64_t slots, int64_t slot_bytes, unsigned long long* counter, const void* src,
                   int64_t nbytes, cudaStream_t stream);

}  // namespace cozy_engine
