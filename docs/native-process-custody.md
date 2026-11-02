# Establish native custody before Runtime CUDA work

Owner: `/root/p3_owner_restart`, `codex/xs138-p4-process-custody`.
Frozen parent:616acb27f. Creator companion uses private Python129 plus native4c/e8.

Runtime must perform a private CPU-only allocator bind, create and validate its
explicit negotiated cooperation Owner, and bind that service to Working lifetime
before torch.cuda.init, library warm-up, Memory publication or package work. Memory
adopts the same staged execution Owner instead of creating another unknown client.
Failures preserve original errors and native custody without exposing partial
host.memory or changing an existing consumer's configuration.

Public Torch install now activates device0 by default; Runtime must not use that
standalone path or silently create Shared0 when negotiation is missing. CPU/MPS
and import-only/Hello keep their existing behavior. Unsupported CUDA cases refuse
before group/device work: missing native configuration/capability, degree>1 without
per-rank configuration, and a GPU-granted Runtime environment without Torch.
These remain explicit open qualification lanes, not completed policy retirement.

The native passive hold is not an execution client. It retains publication of
Working Ready/Pending storage across visible owner closure and process-lifetime
Torch allocator roots, without fabricating idle activity or training H. Actual
Runtime requests retain their existing boundary/error contract.

Local source/CPU checks and real GPU qualification are separate. Root owns paired
artifacts, GPU execution and tracker status. No GPU/rental/release in this lane.
