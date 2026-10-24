// SPDX-License-Identifier: Apache-2.0
// Executor-local inactive weight owners for PyTorch 2.14.1's native allocator.
// The callback runs under PyTorch's recursive device allocator lock. It must not
// acquire the GIL, call Python, release another device's tensors or wait on a
// thread that could need that allocator. This is a cache-miss callback, before
// cudaMalloc, not an OOM observer; it receives no allocation size or stream.
#include <ATen/core/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace {
struct StorageUse {
  c10::StorageImpl* storage;
  size_t tensor;
  size_t owners;
  size_t bytes;
};

struct Pack {
  c10::DeviceIndex device;
  std::vector<at::Tensor> tensors;
  std::vector<StorageUse> storages;

  bool exclusive() const {
    // Fresh TensorImpls below have no Python object or autograd metadata. A
    // StorageImpl can itself carry a Python wrapper; never decref that here.
    // Exclusive native strong ownership also excludes escaped tensor aliases.
    for (const auto& use : storages) {
      if (use.storage->pyobj_slot()->load_pyobj() != nullptr ||
          tensors[use.tensor].storage().use_count() != use.owners)
        return false;
    }
    return true;
  }
};

struct Counters {
  int64_t misses = 0;
  int64_t packs = 0;
  int64_t storage_bytes = 0;
  int64_t aliased = 0;
};

struct Cache {
  std::mutex mutex;
  int64_t next = 1;
  std::map<int64_t, Pack> packs;
  std::map<c10::DeviceIndex, Counters> counters;
};

Cache& cache() {
  // Runtime explicitly drops its tokens. Avoid interpreter/library teardown
  // ordering at process exit; the OS releases any remaining executor storage.
  static auto* value = new Cache;
  return *value;
}

int64_t put(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(!tensors.empty(), "cannot cache an empty weight group");
  TORCH_CHECK(c10::cuda::CUDACachingAllocator::name() == "native",
              "weight reclamation requires PyTorch's native CUDA allocator");
  Pack pack{tensors.front().device().index(), {}, {}};
  pack.tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() &&
                    tensor.device().index() == pack.device &&
                    tensor.layout() == at::kStrided && !tensor.is_conj() &&
                    !tensor.is_neg() && !tensor.requires_grad() &&
                    !tensor.unsafeGetTensorImpl()->is_python_dispatch() &&
                    !tensor.unsafeGetTensorImpl()->has_symbolic_sizes_strides(),
                "inactive weights must be ordinary concrete strided CUDA tensors");
    const auto& storage = tensor.storage();
    TORCH_CHECK(storage.allocator() == c10::cuda::CUDACachingAllocator::get(),
                "inactive weights must own PyTorch CUDA storage");
    // Do not retain a Python-associated TensorImpl. Rebuild only the plain
    // view metadata over its original storage, preserving storage aliases.
    auto impl = c10::make_intrusive<c10::TensorImpl>(
        c10::Storage(storage), c10::DispatchKey::CUDA, tensor.dtype());
    impl->set_sizes_and_strides(tensor.sizes(), tensor.strides(), tensor.storage_offset());
    pack.tensors.emplace_back(std::move(impl));
    auto* pointer = storage.unsafeGetStorageImpl();
    auto found = std::find_if(pack.storages.begin(), pack.storages.end(),
                             [pointer](const auto& use) { return use.storage == pointer; });
    if (found == pack.storages.end())
      pack.storages.push_back({pointer, pack.tensors.size() - 1, 1, storage.nbytes()});
    else
      ++found->owners;
  }
  auto& held = cache();
  std::lock_guard<std::mutex> lock(held.mutex);
  TORCH_CHECK(held.next < std::numeric_limits<int64_t>::max(), "weight cache token exhausted");
  const auto token = held.next++;
  held.counters.try_emplace(pack.device);
  held.packs.emplace(token, std::move(pack));
  return token;
}

std::vector<at::Tensor> take(int64_t token) {
  auto& held = cache();
  std::map<int64_t, Pack>::node_type found;
  {
    std::lock_guard<std::mutex> lock(held.mutex);
    found = held.packs.extract(token);
  }
  // Remove before exposing TensorImpls to Python. No tensor decrefs or allocator
  // calls occur while the cache mutex is held.
  return found.empty() ? std::vector<at::Tensor>{} : std::move(found.mapped().tensors);
}

bool present(int64_t token) {
  auto& held = cache();
  std::lock_guard<std::mutex> lock(held.mutex);
  return held.packs.count(token) != 0;
}

void drop(int64_t token) {
  // Explicit owner teardown runs outside an allocation callback, so escaped
  // aliases/Python Storage wrappers can use their ordinary destruction path.
  auto tensors = take(token);
}

std::vector<int64_t> counters(int64_t device) {
  auto& held = cache();
  std::lock_guard<std::mutex> lock(held.mutex);
  const auto found = held.counters.find(static_cast<c10::DeviceIndex>(device));
  const auto value = found == held.counters.end() ? Counters{} : found->second;
  return {value.misses, value.packs, value.storage_bytes, value.aliased};
}

class Reclaim final : public c10::FreeMemoryCallback {
 public:
  bool Execute() override {
    c10::DeviceIndex device;
    if (c10::cuda::GetDevice(&device) != cudaSuccess)
      return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(c10::cuda::getCurrentCUDAStream(device).stream(), &capture) !=
            cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
      return false;

    auto& held = cache();
    std::vector<std::map<int64_t, Pack>::node_type> retired;
    {
      std::unique_lock<std::mutex> lock(held.mutex, std::try_to_lock);
      if (!lock.owns_lock())
        return false;
      auto count = held.counters.find(device);
      if (count == held.counters.end())
        return false;
      ++count->second.misses;
      retired.reserve(held.packs.size());
      // There is no requested size and no second callback after cudaMalloc
      // fails. Release the finite eligible set; guessing one victim can still
      // fail the allocation even when enough inactive weights are available.
      for (auto at = held.packs.begin(); at != held.packs.end();) {
        auto current = at++;
        auto& pack = current->second;
        if (pack.device != device)
          continue;
        if (!pack.exclusive()) {
          ++count->second.aliased;
          continue;
        }
        ++count->second.packs;
        for (const auto& use : pack.storages)
          count->second.storage_bytes += static_cast<int64_t>(use.bytes);
        retired.push_back(held.packs.extract(current));
      }
    }
    // Destruction uses only plain native tensor/storage owners of this device.
    // Torch's recursive allocator lock permits its normal free path, and its
    // recorded streams protect pending consumers. Never hold our mutex here.
    const bool released = !retired.empty();
    retired.clear();
    return released;
  }
};
} // namespace

namespace c10 {
REGISTER_FREE_MEMORY_CALLBACK(cozy_residency, Reclaim);
} // namespace c10

TORCH_LIBRARY(cozy_residency, m) {
  m.def("put(Tensor[] tensors) -> int", &put);
  m.def("take(int token) -> Tensor[]", &take);
  m.def("present(int token) -> bool", &present);
  m.def("drop(int token) -> ()", &drop);
  m.def("counters(int device) -> int[]", &counters);
}
