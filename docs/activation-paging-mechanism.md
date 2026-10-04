# Activation paging mechanism gate

Tracker320; parent319/324. Ownergpu_qualification, branch
`codex/activation-paging-mechanism-20261004`, fetched baseb2a93731 advanced to the
reviewed2aaac54e candidate. This is a staged design, with no product change or
allocator/GPU prototype executed.

The objective is one full native SDXL VAE decode of the immutable131072-byte
FP16 latent SHA87cad3ed70cfae838271022697730e7d3c627d75ef6979b4d8e8c2e4aab72899,
shape[1,4,128,128], original seed1114002. Preserve the selected BF16 source and
ordinary SDXL scale/upcast, all global normalization/attention, full1024-square
output, the35dB rule and no whole-forward replay. Ilulu's external holder113 /
birth33024000 /6,439,436,288 allocated bytes stays unchanged at1.5GiB available
VRAM; only root authorizes a prototype.

## Confirmed evidence

GPU4280 completed20 denoise steps but an untiled VAE operator needed512MiB with
502,951,936 allocatable bytes. H0c tiled512 then completed4281 but scored30.111dB
against its preserved control. Diagnostic4282 produced the immutable latent and
observed nine512-sample/border tiles. On the same5080/H0c, independent untiled4283
scored48.588dB to control while tiled5124284 scored30.362dB; their mutual score is
30.366dB. Producer3070 tile vs independent5080 tile scored49.415dB. These isolate
stock tiling as the main fidelity loss, with cross-device limits kept explicit.
Evidence resides in`memory/fresh-pools-1p5-20261004/LATENT-DECODE-COMPARISON.json`.

Actual ilulu Torch is2.14.0+cu130, commit08187d9e0fba026dc8217405802ab5381dc88d90,
CUDA13.0, driver580.65.06. C++/g++ and the installed cu13 runtime headers/library
exist; nvcc does not. No allocation or kernel is required to inspect these facts.

## Supported API and the accounting gate

[PyTorch2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
documents a scoped CUDAPluggableAllocator/MemPool route, pool lifetime references
and optional fallback. It also warns that managed-memory migration can be costly.
Use a scope, not an environment or global allocator replacement, and retain the
pool until all tensors and stream work are complete.

[CUDA memory management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
permits managed oversubscription only with nonzero concurrentManagedAccess;
query the actual device instead of assuming support from its product name.
[CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
does not permit CUDA IPC handles for cudaMallocManaged allocations. TensorFS/D2
weight buffers must therefore remain ordinary device/IPC allocations. Only new
transient decode activations are candidates for managed backing.

The [exact installed Torch source](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/c10/cuda/CUDACachingAllocator.cpp)
rejects virtual segment totals above allowed_memory_maximum inalloc_block before
the custom MemPool callback (lines3609–3612,3665/3667). A custom pool therefore
does not bypass the SDK's present logical fraction. This source-proven integration
gate is separate from hardware managed-memory support. No cap increase, hidden
environment switch, ABI/vtable trick or preallocated-cache workaround may pass it.

## Proposed reviewed phases

1. Read-only actual-device attributes and software/holder identity, including
   managedMemory, concurrentManagedAccess, pageableMemoryAccess and host-page-table
   support. Capture host/cgroup availability independently. Missing support yields
   a typed unsupported result; it does not qualify a fallback.
2. After root review, build a tiny owned C++ bridge against installed cu13 headers
   and libcudart. Explicit callbacks call cudaMallocManaged/cudaFree and report
   requested virtual bytes, pointer attributes, return codes and stream lifetime.
   Create a scoped MemPool through the documented interface. No kernel or model
   operation starts during compilation.
3. Probe allocator accounting under the unchanged SDK fraction and physical holder:
   small managed allocation plus a declared larger segment which tests the cap
   boundary. Separate pre-callback rejection from CUDA allocation refusal; capture
   virtual PyTorch counters/pool snapshots, actual physical free/used and host
   charges. These are allocator diagnostics, not smaller substitute inference.
4. Only if a supported bounded activation path can be established without raising
   the ordinary device cap, run ONE full native decode with the saved latent and
   standard decoder input128x128. Establish/convert weights outside the managed
   scope; verify their pointer kind and ordinary TensorFS/D2 path. Record managed
   pointers, physical GPU ceiling, host/cgroup peaks, time and complete output.
   Compare RGB with the independent untiled reference and control; no35dB change.
5. A failure after entry retires the affected operation/process; it never retries
   the decoder or changes its geometry. Archive all failures and scope/pool release
   receipts before any next reviewed attempt.

Current stock MemPool source predicts phase3 will refuse virtual oversubscription
at the SDK fraction. This is not a hardware-limit claim. Phase4 is consequently
not authorized by this design alone. If virtual charge is the only gate, present
an explicit scoped accounting change for review: an activation resource needs
separate bounded host/virtual accounting and physical residency authority, while
ordinary allocations retain their existing protection. No general fraction/cap
increase or disabled ordinary protection is acceptable. The supported interface
or minimal integration needed to distinguish the two charges must be proved
first; it is not an implemented feature or an excuse to bypass admission.

If managed backing cannot provide a supported path, propose exact layer streaming:
offload complete activations between layers, preserve whole-image group-normalization
statistics and full attention K/V, and use bounded convolution halos/output chunks
only with proved equivalent math. Cropping global context or using stock tiled_decode
does not close the fidelity gate. No implementation starts before a reviewed
mechanism can execute the immutable full latent and retain numerical quality.
