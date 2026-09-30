# Prove known-target room before evicting another unit

Dropping live GPU tensors can lower allocated bytes while allocator reserve and driver
free stay unchanged. A known-target offload previously continued evicting until its
final cap update trimmed the cache, even when fewer released tensors could have made
sufficient room.

Each existing known-target victim pass now owns a short-lived callback. It accumulates
net fresh process-allocator decreases across victims and subtracts observed increases.
Only when this measured decrease could cover the remaining gap does it invoke the
existing device synchronization/cache trim. It then reads actual allocatable room
again; only that fresh result may stop offload. No logical weight size, cached bytes,
or release witness is used as admission credit.

The witness is a process allocator observation, not proof that one named weight unit
caused every byte of the decrease. Retained views may prevent a decrease; fragmentation
or another device consumer may prevent trim from producing enough room. Both cases
continue through the existing finite victim/unit sequence without fabricated capacity.
The witness resets after each successful trim. `dropped` clears then too; later actual
homes set it again, preserving final cap cleanup without duplicating a completed drain.
A failed trim poisons the executor and retains the failure as an unreusable device fault.

Unknown-work plans with no target, grouped execution, compiled/pinned/mutable eligibility
and whole-store atomic movement retain their existing policy. A whole-store movement
can be measured only after the complete atomic operation. No new Worker message or
persistent per-request cache is introduced.

CPU tests use registered real Torch units and controlled allocator/driver observations.
They cover partial success across two victims, retained views, partial/zero physical
release, external free-memory changes, net reallocation, whole-store atomicity, group
and unknown-plan behavior, and failed-drain quarantine. They do not establish real CUDA
fragmentation behavior or an inference speedup. GPU qualification must separately measure
trim cost, retained weights, repeated staging, exact outputs and complete request totals.
