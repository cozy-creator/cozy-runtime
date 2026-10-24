# Native inactive-weight cache candidate

This is a buildable candidate for allocator-path reclamation, not an enabled Runtime memory
backend. Python residency has not yet transferred its inactive tensor ownership to these tokens.
The design choice against aimdo or a local Rust VMM implementation remains open.

The library targets exactly PyTorch 2.14.1, its native CUDA caching allocator, and CUDA 13.0.
It has no Python entrypoint, CUDA kernels, alternate allocator, VMM reservations or serving-time
compiler. The existing native-wheel build installs `_kernels/_residency.so` (Windows: `.dll`).
A caller must verify the Torch version, CUDA version and allocator backend before loading it
with `torch.ops.load_library` after importing Torch.

The operators are:

- `put(Tensor[]) -> int`: retain fresh plain native views over an inactive group's existing
  Torch storage, preserving shape, stride, storage offset, dtype and storage aliases. Reject
  unsupported tensors. The caller must drop its ordinary Python references after publishing
  the token; holding them makes the group ineligible for callback reclamation.
- `take(int) -> Tensor[]`: remove the token before exposing any tensor to Python. An empty
  result means the callback discarded it, so the owner must refill before its next use.
- `present(int) -> bool`: report whether the token still exists; this is an observation,
  not a lifetime pin. An active forward or cached graph must own the result of `take`.
- `drop(int) -> ()`: release a token at an explicit owner boundary.
- `counters(int device) -> int[]`: process-cumulative cache misses observed, packs discarded,
  discarded storage bytes, and alias-ineligible pack observations. Storage bytes are not a
  claim about driver free bytes or completed consumers.

PyTorch invokes `FreeMemoryCallback` under its recursive device allocator mutex, on a cache
miss before attempting a new driver allocation. The interface provides no requested size,
device or stream and no second callback after driver allocation fails. The candidate therefore
discards the finite set of eligible packs on the current device on that miss. This can evict
useful weights despite ample VRAM; it does not match aimdo's victim precision. No speed or
thrashing result has been measured.

The callback tries the cache mutex rather than waiting, removes selected packs, releases that
mutex, then destroys their owners. It skips capture on the current stream, other devices,
storage with Python wrappers, and storage with strong owners outside the pack. Fresh native
TensorImpls avoid Python tensor destruction; the storage checks avoid Python storage destruction
or invalidating escaped aliases under the allocator lock. Callers must already record every
consumer stream, and must never put active or graph-retained weights into the cache.

Qualification still requires a real unwrapped Torch allocation recovering under a local
fraction, numerical refill, preserved active/graph/escaped-alias storage, and capture and
multithread lifetime checks. The prior Python-residency GPU receipts do not prove this candidate.

The initial checkpoint compiles and links with GCC against the installed PyTorch
2.14.1+cu130 headers/libraries and the pinned CUDA 13.0 headers. Loading the resulting shared
library registers the operator schemas and returns zero counters without initializing CUDA.
Its ELF exports no defined dynamic symbols. This is source/build/import evidence only; the
full native-wheel CMake/CI build and GPU callback execution remain unqualified.
