# Owned managed allocator probes

Tracker #320, draft PR1169. These scripts are diagnostic tools and are not loaded
by Runtime or installed into an executor environment. Root authorized allocator
phases 2–3 only on ilulu. No model decode, SDK mutation or global allocator switch
is authorized here.

Build activation-paging.cc with g++ against the actual generated environment's
cu13 headers and versioned libcudart. nvcc is unnecessary. Keep source, binary
hashes and build receipt in the owned operation directory.

Run activation-paging.py with that generated environment's existing interpreter.
It requires unchanged holder PID113/birth33024000/6,439,436,288 bytes and separately
records the idle owned executor PID2059/birth33529715. These existing allocations
mean the full 1.5 GiB pool is not immediately free.

The declared diagnostic Torch cap is 512 MiB; preflight requires another 64 MiB
of actual free room after creating this probe's context. This new process's
fraction does not change any existing SDK/device grant. The bridge separately
limits live managed virtual allocations to 16 MiB, and requires at least 1 GiB
of container headroom before importing Torch. No allocation above that bridge
bound can occur even if the expected stock fraction rejection is absent.

The probe allocates a 1 MiB uint8 tensor through a scoped CUDAPluggableAllocator /
MemPool, performs one fill, verifies a CPU copy and actual managed pointer type,
then requests 514 MiB under the 512 MiB virtual cap. It records custom-callback
counts before and after that request. A pre-callback OOM confirms the source
accounting gate; it is not proof of managed oversubscription or a full decode.

The allocator frees only pointers it owns, synchronizes their allocation stream,
and keeps counters for failed/unknown frees. After tensors and the pool are
released, live managed bytes must be zero. The probe exits naturally; it never
stops the holder, idle executor or durable work. Capture a separate post-process
physical receipt to verify this probe's context has gone.

Every stage journals physical GPU/app state, own process high-water RSS, current
and historical peak container memory, OOM events and Torch virtual counters.
Container peak is shared and already at its historical limit; do not call it this
probe's exclusive peak or infer host/source-lifetime closure. Full decode remains
gated on a separately reviewed supported accounting/admission mechanism.
