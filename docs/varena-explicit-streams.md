# Runtime owns its execution stream

Owner: `/root/explicit_stream_owner`; branch `codex/xs138-explicit-stream-owner`, draft PR2033.
Fresh origin/master base `cb6c66bf343d15637eaeb4af35731a37fc56d61c`, followed by frozen
integration `61fe32331968f89a1585d25263924ef15cc04a39`. This is the pre-launch explicit-stream
cut authorized in tracker commits40cd9957/bb93c6f8. Old artifacts and their receipts are unchanged.

The previous initializer selected a CUDA device but left eager work on implicit current
streams. Pool workers and background kernel probes each have independent CUDA stream state.
The initializer now installs Varena before CUDA initialization, selects the rank device,
creates one unique non-blocking stream and makes it current before Memory construction,
first library launches, native INT8 priming or package work. This stream belongs to the
executor process. It remains alive until process exit because library workspaces and tensor
finalizers may retain it after an ordinary model unload.

Commands (including follower commands), idle graph work, legacy idle-room work and shutdown
enter that actual stream explicitly. Every readiness opener receives the same stream, including
INT8's background thread. Runtime's exported thread pool and remote group-call handoff carry
the submitter's actual stream, preserving a deliberate graph or framework stream selection.
Plain author-created Python threads retain their existing unsupported context behavior.

First launches, request drains, follower completion, explicit offload and shutdown record and
wait on a completion event. Quiescence queries the owned execution stream, including its joined
producers. It does not query a fresh thread's default stream. Existing capture input, control,
library and replay/output bridges remain explicit event/stream edges. A caller using additional
framework streams must join their consumers before releasing a use or crossing its completion
boundary. No external stream handle is reinterpreted or silently replaced.

Domain's library/control streams now use the same unique non-blocking creation helper. They
retain the existing never-replayed library-owner graph. Teardown still joins users, resets
executables and the owner graph, drains retirement events, calls `vt.retire_stream` and only
then destroys the streams. This preserves the Torch library-workspace lifetime contract.
The device-wide fence in the failure-only accelerator-health probe is unchanged; it is not a
request-completion or allocator-retirement proof.

## Current API basis

The CUDA13.4 [driver stream contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html)
defines `cuStreamCreate(..., CU_STREAM_NON_BLOCKING)` and explicit event dependencies.
The [default-stream contract](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)
explains why legacy and per-thread default streams are not substitutes for that ownership.

The historical Torch2.14.0+cu130 cohort exposes only blocking creation in its
[cudart Python binding](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/cuda/shared/cudart.cpp).
`accel.nonblocking_stream` therefore uses the installed driver's `cuStreamCreate` and wraps the
returned handle unchanged with `torch.cuda.ExternalStream`. Driver loading remains lazy past
Runtime's allocator/device boundary. `accel.destroy_stream` calls `cuStreamDestroy_v2` only
after owner retirement. No Torch/CUDA package is upgraded by this change; latest API contract
review is not a latest-stack execution qualification.

## Verification and remaining gates

`research/varena-explicit-streams/cpu-check.py` exercises actual Runtime command/background
thread boundaries with simulated CUDA calls; its receipt explicitly says CPU contracts only.
It confirms initialization order, non-blocking creation/failure cleanup, readiness propagation,
actual stream preservation in pools and remote calls, completion events and shutdown order.
CUDA remains uninitialized and the CUDA driver loader remains unused. Focused Ruff and parse
checks cover the changed source and fixture. The identical SDK type-check comparison records
all pre-existing errors and any new errors separately; this is not a clean full-project check.

`research/varena-explicit-streams/gpu` is an ordinary Cozy application, **unrun**. It checks actual
stream flags/handles, the real startup INT8 provider, full matrix-output contents, request and
readiness/pool ordering, Runtime Domain capture and replay on execution and external streams,
escaped output contents after domain destruction, and live-allocation return to its baseline.
It contains no simulated CUDA or replacement inference implementation. Root owns packaging,
exact source hashes, GPU selection, rental/lifecycle receipts and execution. Run it twice in a
single executor, then exercise ordinary unload/offload and shutdown; keep the first failure.
The root must also run the unchanged actual full-model gate, including initialization, capture,
replay, outputs and cleanup. This bounded fixture does not establish model parity/performance.

Remaining independent seams: P4 owns public TensorFS stream signatures/default removal;
native owners own asynchronous source-fill and retirement contracts. Their changes are not
bundled here. Actual multi-rank execution, Windows execution and full-model GPU behavior remain
unqualified by this source/CPU change. No GPU or rental was used by this owner.
