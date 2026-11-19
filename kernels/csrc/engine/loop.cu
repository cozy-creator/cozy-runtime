// SPDX-License-Identifier: Apache-2.0
#include "engine/loop.h"

namespace cozy_engine {
namespace {

__global__ void again(cudaGraphConditionalHandle handle, const int* go) {
    cudaGraphSetConditional(handle, *go != 0 ? 1u : 0u);
}

// One thread writes the slot, so the order the host reads it in is __threadfence_system()'s own: the
// sequence number cleared, the length and record, then the sequence number. A record is a step's tokens
// and flags: small.
__global__ void write_record(unsigned char* ring, unsigned long long slots, unsigned long long slot_bytes,
                             unsigned long long* counter, const unsigned char* src, unsigned long long nbytes) {
    const unsigned long long n = *counter;
    unsigned char* slot = ring + (n % slots) * slot_bytes;
    volatile unsigned long long* head = reinterpret_cast<volatile unsigned long long*>(slot);
    head[0] = 0;
    __threadfence_system();
    head[1] = nbytes;
    unsigned char* body = slot + kHeader;
    if ((reinterpret_cast<uintptr_t>(src) | nbytes) % 16 == 0) {
        const uint4* from = reinterpret_cast<const uint4*>(src);
        uint4* to = reinterpret_cast<uint4*>(body);
#pragma unroll 8
        for (unsigned long long i = 0; i < nbytes / 16; ++i) to[i] = from[i];
    } else {
        for (unsigned long long i = 0; i < nbytes; ++i) body[i] = src[i];
    }
    __threadfence_system();
    head[0] = n + 1;
    *counter = n + 1;
}

__global__ void fill_rows(unsigned char* dst, size_t pitch, size_t row_bytes, size_t rows, unsigned word, bool wide) {
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (wide) {
        const uint4 value = make_uint4(word, word, word, word);
        const size_t units = row_bytes / 16, total = units * rows;
        for (; i < total; i += stride) reinterpret_cast<uint4*>(dst + (i / units) * pitch)[i % units] = value;
        return;
    }
    for (const size_t total = row_bytes * rows; i < total; i += stride) {
        const size_t column = i % row_bytes;
        dst[(i / row_bytes) * pitch + column] = static_cast<unsigned char>(word >> (8 * (column % 4)));
    }
}

__global__ void copy_rows(unsigned char* dst, const unsigned char* src, size_t dst_pitch, size_t dst_slice,
                          size_t src_pitch, size_t src_slice, size_t width, size_t height, size_t depth, bool wide) {
    const size_t unit = wide ? 16 : 1, per_row = width / unit, total = per_row * height * depth;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const size_t row = i / per_row, column = (i % per_row) * unit;
        const size_t y = row % height, z = row / height;
        unsigned char* to = dst + z * dst_slice + y * dst_pitch + column;
        const unsigned char* from = src + z * src_slice + y * src_pitch + column;
        if (wide) *reinterpret_cast<uint4*>(to) = *reinterpret_cast<const uint4*>(from);
        else *to = *from;
    }
}

}  // namespace

const void* again_kernel() { return reinterpret_cast<const void*>(&again); }
const void* fill_rows_kernel() { return reinterpret_cast<const void*>(&fill_rows); }
const void* copy_rows_kernel() { return reinterpret_cast<const void*>(&copy_rows); }

cudaError_t report(void* ring, int64_t slots, int64_t slot_bytes, unsigned long long* counter, const void* src,
                   int64_t nbytes, cudaStream_t stream) {
    write_record<<<1, 1, 0, stream>>>(static_cast<unsigned char*>(ring), static_cast<unsigned long long>(slots),
                                      static_cast<unsigned long long>(slot_bytes), counter,
                                      static_cast<const unsigned char*>(src),
                                      static_cast<unsigned long long>(nbytes));
    return cudaGetLastError();
}

}  // namespace cozy_engine
