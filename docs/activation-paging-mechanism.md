# Activation paging mechanism gate

Tracker #320; parent #319/#324. Owner gpu_qualification, branch
`codex/activation-paging-mechanism-20261004`, fetched base b2a93731 advanced to the
reviewed 2aaac54e candidate. This is a staged design, with no product change or
allocator/GPU prototype executed.

The objective is one full native SDXL VAE decode of the immutable 131072-byte
FP16 latent SHA 87cad3ed70cfae838271022697730e7d3c627d75ef6979b4d8e8c2e4aab72899,
shape [1,4,128,128], original seed 1114002. Preserve the selected BF16 source and
ordinary SDXL scale/upcast, all global normalization/attention, full1024-square
output, the 35 dB rule and no whole-forward replay. Ilulu's external holder 113 /
birth 33024000 / 6,439,436,288 allocated bytes stays unchanged at 1.5 GiB available
VRAM; only root authorizes a prototype.

## Confirmed evidence

GPU 4280 completed 20 denoise steps but an untiled VAE operator needed 512 MiB with
502,951,936 allocatable bytes. H0c tiled512 then completed 4281 but scored 30.111 dB
against its preserved control. Diagnostic 4282 produced the immutable latent and
observed nine 512-sample/border tiles. On the same 5080/H0c, independent untiled 4283
scored 48.588 dB to control while tiled512 4284 scored 30.362 dB; their mutual score is
30.366 dB. Producer 3070 tile vs independent 5080 tile scored 49.415 dB. These isolate
stock tiling as the main fidelity loss, with cross-device limits kept explicit.
Evidence resides in `memory/fresh-pools-1p5-20261004/LATENT-DECODE-COMPARISON.json`.

Actual ilulu Torch is 2.14.0+cu130, commit 08187d9e0fba026dc8217405802ab5381dc88d90,
CUDA 13.0, driver 580.65.06. C++/g++ and the installed cu13 runtime headers/library
exist; nvcc does not. No allocation or kernel is required to inspect these facts.

## Supported API and the accounting gate

[PyTorch 2.14 CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)
documents a scoped CUDAPluggableAllocator/MemPool route, pool lifetime references
and optional fallback. It also warns that managed-memory migration can be costly.
Use a scope, not an environment or global allocator replacement, and retain the
pool until all tensors and stream work are complete.

The [installed Python pool interface](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/torch/cuda/memory.py)
accepts an allocator, use_on_oom and no_split, with no separate pool fraction.
The supported fraction setter applies to the device. Do not infer a scoped
physical-accounting exemption from the pool's separate identity.

[CUDA memory management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)
permits managed oversubscription only with nonzero concurrentManagedAccess;
query the actual device instead of assuming support from its product name.
[CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
does not permit CUDA IPC handles for cudaMallocManaged allocations. TensorFS/D2
weight buffers must therefore remain ordinary device/IPC allocations. Only new
transient decode activations are candidates for managed backing.

The [exact installed Torch source](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/c10/cuda/CUDACachingAllocator.cpp)
rejects virtual segment totals above allowed_memory_maximum in alloc_block before
the custom MemPool callback (lines 3609–3612, 3665/3667). A custom pool therefore
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
   standard decoder input 128x128. Establish/convert weights outside the managed
   scope; verify their pointer kind and ordinary TensorFS/D2 path. Record managed
   pointers, physical GPU ceiling, host/cgroup peaks, time and complete output.
   Compare RGB with the independent untiled reference and control; no 35 dB change.
   H0c's cold VAE wrapper currently selects tiled512 from physical room, so this
   phase also requires a reviewed activation-aware admission plan. Disabling the
   stock tiling flag alone does not prove a full decode. Preserve the native
   untiled post_quant_conv then decoder math and record the actual input shape
   and a single decoder invocation.
5. A failure after entry retires the affected operation/process; it never retries
   the decoder or changes its geometry. Archive all failures and scope/pool release
   receipts before any next reviewed attempt.

Current stock MemPool source predicts phase 3 will refuse virtual oversubscription
at the SDK fraction. This is not a hardware-limit claim. Phase 4 is consequently
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
