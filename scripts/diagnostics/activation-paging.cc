// Diagnostic only: scoped managed allocations with a separately bounded owner.
#include <cuda_runtime_api.h>
#include <sys/types.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {
std::mutex mutex;
std::unordered_map<void*, size_t> owned;
// alloc_calls, free_calls, live_bytes, peak_bytes, largest_request, limit_denials,
// last_alloc_rc, last_free_rc, invalid_frees, sync_failures, limit, successful_frees.
std::array<uint64_t, 12> stats{};
}

extern "C" {
int probe_set_limit(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!owned.empty() || bytes == 0) return 1;
  stats[10] = bytes;
  return 0;
}

uint64_t probe_stat(int index) {
  std::lock_guard<std::mutex> lock(mutex);
  return index >= 0 && index < 12 ? stats[index] : 0;
}

void* probe_managed_alloc(ssize_t size, int device, cudaStream_t stream) {
  (void)stream;
  std::lock_guard<std::mutex> lock(mutex);
  ++stats[0];
  const size_t bytes = size > 0 ? static_cast<size_t>(size) : 0;
  if (bytes > stats[4]) stats[4] = bytes;
  if (device != 0 || bytes == 0 || bytes > stats[10] ||
      stats[2] > stats[10] - bytes) {
    ++stats[5];
    stats[6] = cudaErrorMemoryAllocation;
    return nullptr;
  }
  void* ptr = nullptr;
  cudaError_t rc = cudaSetDevice(device);
  if (rc == cudaSuccess) rc = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
  stats[6] = rc;
  if (rc != cudaSuccess) return nullptr;
  owned.emplace(ptr, bytes);
  stats[2] += bytes;
  if (stats[2] > stats[3]) stats[3] = stats[2];
  return ptr;
}

void probe_managed_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex);
  ++stats[1];
  const auto it = owned.find(ptr);
  if (device != 0 || it == owned.end() || size < 0 ||
      it->second != static_cast<size_t>(size)) {
    ++stats[8];
    stats[7] = cudaErrorInvalidValue;
    return;
  }
  cudaError_t rc = cudaSetDevice(device);
  if (rc == cudaSuccess) rc = cudaStreamSynchronize(stream);
  if (rc != cudaSuccess) {
    ++stats[9];
    stats[7] = rc;
    return;
  }
  rc = cudaFree(ptr);
  stats[7] = rc;
  if (rc == cudaSuccess) {
    stats[2] -= it->second;
    owned.erase(it);
    ++stats[11];
  }
}

int probe_pointer_info(void* ptr, uint64_t* result) {
  std::lock_guard<std::mutex> lock(mutex);
  if (owned.find(ptr) == owned.end()) return cudaErrorInvalidValue;
  cudaPointerAttributes attributes{};
  const cudaError_t rc = cudaPointerGetAttributes(&attributes, ptr);
  if (rc == cudaSuccess) {
    result[0] = attributes.type;
    result[1] = attributes.device;
    result[2] = reinterpret_cast<uintptr_t>(attributes.devicePointer);
    result[3] = reinterpret_cast<uintptr_t>(attributes.hostPointer);
  }
  return rc;
}
}
