# Activation paging mechanism gate

Tracker #320; parent #319/#324. Owner gpu_qualification, branch
`codex/activation-paging-mechanism-20261004`, fetched base b2a93731 advanced to the
reviewed 2aaac54e candidate. This records a staged design and the authorized
allocator diagnostic. No product implementation or full decode has run.

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

A subsequent read-only CUDA query returned managedMemory=1,
concurrentManagedAccess=1 and pageableMemoryAccess=1 on that actual 3070, with
host-page-table and direct-host managed-access flags both zero. All return codes
were zero; the CUDA process list remained holder PID 113 and owned idle executor
PID 2059, with no new tensor allocation or kernel. This confirms the hardware
capability flag, not actual oversubscription or a full decode.

The container memory.max is 17,999,998,976 bytes, memory.current was
15,603,408,896 bytes, historical memory.peak equals that limit, and swap.max is
zero. Host MemAvailable is a separate machine-wide fact and cannot replace this
container limit. Record reclaimable/file and anonymous charges before assigning
a bounded managed-host allowance. Evidence: activation-managed-capabilities.json.

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

Stock MemPool source predicted, and the authorized diagnostic confirmed, virtual
cap refusal before its callback. This is not a hardware-limit claim. Phase 4 is consequently
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

## Authorized allocator diagnostic result

Clean probe source 180d9b57 built with g++ against the installed CUDA runtime
13.0.96 and isolated official nvidia-cuda-crt 13.0.88 headers. The initial missing
CRT-header build failure and subsequent header-path failure remain archived.
No generated executor dependencies or SDK files changed. The bridge binary SHA is
fe0c9b5e2e764888c84cd8176788a3c0206e85727d9bac63c36d6a462f7fd274.

On ilulu, one 1 MiB uint8 tensor used a scoped managed pool. Its actual pointer
type was cudaMemoryTypeManaged; one fill and CPU copy verified its contents.
The bridge obtained one 2 MiB segment. A declared 514 MiB allocation under this
new diagnostic process's 512 MiB Torch cap raised OOM with 606.81 MiB CUDA free.
Custom allocation callbacks remained 1 before and after the request, with zero
bridge budget denials. This is actual pre-callback virtual-limit evidence, not
managed oversubscription or full-decode qualification.

After tensor/pool release, Torch reserved bytes and bridge live bytes were zero;
one cudaFree succeeded, with no invalid free or stream-sync failure. Probe PID
2564 exited naturally. Holder PID 113/birth 33024000 and original idle executor
PID 2059/birth 33529715 were preserved. Shared container current samples rose by
340,963,328 bytes; probe RSS high-water was 646,107,136 bytes. Cgroup max/OOM
events did not change, and historical peak already equaled the limit before entry.

Physical cleanup is not fully classified: NVML free remained 794 MiB after exit
versus 920 MiB before, while both original compute-app charges were unchanged.
FB reserved remained 352 MiB. Device FD inventory also includes owned
cozy-machine PID 1268/birth 33267283, absent from compute-apps. Attribute the
126 MiB FB-used delta between machine/contextless allocations and driver/UVM
state before claiming full physical restoration. No reclaim, reset or process
kill was performed. Evidence: activation-probe-1/result, RESOURCE-SUMMARY.json,
physical-after-process.json, post-stability.json and post-device-clients.json.

A source-backed inference now explains the exact delta. The actual Open Kernel
Module is 580.65.06; its [UVM physical allocator](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/580.65.06/kernel-open/nvidia-uvm/uvm_pmm_gpu.c)
uses 2 MiB root chunks and batches non-pinned user allocations by 2^order, leaving
unused chunks on a per-GPU free list. Actual read-only module parameters are
order=6 and global_oversubscription=1. One first batch can therefore allocate
128 MiB; freeing the used 2 MiB chunk leaves 63 cached chunks, exactly 126 MiB.
The source's free path releases one available root chunk, and its device teardown
releases remaining free roots. Classification: PLAUSIBLE SOURCE+GPU-consistent
UVM batch cache. No PMM trace or physical eviction test ran, so actual cache
reclamation and full physical restoration remain unqualified. Module settings
were only read. Evidence: uvm-module-readonly.json and UVM-CACHE-INFERENCE.json.

## Next source and CPU gate

The small managed pool works, but stock MemPool cannot distinguish host-backed
logical bytes from ordinary physical bytes at its guard. No supported per-pool
fraction API has been established. Do not proceed by raising the per-device
fraction around a scope: allocations on another thread could still use the
ordinary pool, so that would not preserve its protection.

A potential alternative is exact activation streaming between native operators,
using the [documented Torch dispatch extension](https://docs.pytorch.org/docs/2.14/notes/extending.html).
An owned activation wrapper would preserve its original logical CUDA metadata
while retaining complete host backing. Before each original native operator,
materialize only its required operands under the existing physical cap; afterward
offload newly derived outputs and release their device storage when stream work
and readers are complete. Model/parameter sources remain ordinary TensorFS/D2
allocations. Views and aliases must retain the same owner and mutation semantics.
This is a proposed mechanism, not an implemented or qualified alternative.

The next CPU gate should inspect the actual selected VAE configuration and run
metadata-only analysis at full [1,4,128,128] latent shape to enumerate operators,
aliases, live input/output bytes and full normalization/attention domains. Native
temporary/workspace bounds require separate evidence; a geometry estimate alone
does not establish them. Preserve the exact scale/BF16 math, original decoder
control flow and one forward. If a native operator cannot fit, a reviewed exact
layer-streaming implementation must preserve full-image normalization statistics,
full attention K/V and convolution halos. Never return CPU metadata to model code
where the original tensor was CUDA, crop global context or silently choose tiling.

Externally managed storage remains another research route, with an explicit
interop gate. [PyTorch DLPack](https://docs.pytorch.org/docs/2.14/dlpack.html)
supports shared external storage, but the
[exact installed converter](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/DLConvertor.cpp)
does not accept DLPack's dedicated managed-memory device tag. Do not silently
relabel a producer's memory to get past that boundary. Nor assume that an ATen
out variant removes temporary allocations: the
[exact out code generator](https://raw.githubusercontent.com/pytorch/pytorch/08187d9e0fba026dc8217405802ab5381dc88d90/torchgen/native_function_generation.py)
implements generated out variants by computing a functional temporary then
copying it. Any public external-storage route must prove truthful device typing,
ownership/stream cleanup and native temporary bounds before a full GPU decode.

Only this source/CPU assessment is proposed next. Full-decode GPU execution and
product implementation require another concrete review; phases 2–3 do not grant it.
