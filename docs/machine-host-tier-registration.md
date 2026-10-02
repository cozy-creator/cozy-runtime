# Machine-created host tier registration

`host_tiers.owner/1` adds optional `Load.host_tier_owner` (default false/omitted); current
world 1 executors advertise it without opening CUDA/NVML. The existing model library emits
its native traversal, selected physical parts and partition roster before asking for a host
tier. A typed `HostTierPlan` is canonical JSON in a fully sealed memfd. Its `HostTierPrepare`
control envelope carries manifest/name/layout and raw SHA-256/length, so large model metadata
is not subject to the 64 KiB control-frame cap. No environment variable selects this behavior.

The machine must select the actor/immutable plan/model Header components independently,
verify descriptor seals/bytes/envelope, build the native ReadPlan/Layout and create/fill the
native host backing. Existing `HostTier` asks adopt that exact ready tier. Older owners can
refuse this optional prepare and return a cache miss; older executors omit the selector and
continue the baseline. Model/Loader/Diffusers code and latest compute/copy/placement safety
remain unchanged. Model partitioning is still the existing implementation, not a copied
Rust model interpretation.

46 CPU gates pass with the safe TensorFS descriptor extension (PR303) and released native
Rust owner. The actual SDK `Executor._durable` talks to `host-plane-cpu-owner`, obtains real
native source descriptors and a machine-filled tier, then actual TensorFS `register(host_fd)`
adopts it: declared bytes match, recipient native fills=0 and buffered source-copy bytes=0.
Four additional SDK gates cover actual socket/SCM_RIGHTS, sealed metadata over 64 KiB, omitted
selectors/advisory evolution and optional older-owner refusal. Existing source/integrity,
failed parallel verification cleanup, readlease, policy and skew gates pass. The fd failure
test collects unrelated fixture finalizers before taking its exact descriptor baseline;
the exact no-leak assertion remains intact.

This is shared allocation/fill/adoption, not full Degree 1. Cache misses and fine/refined
layouts still use the existing private pinned/staging path; source descriptors retain mapped
pageable bytes separately. Only exact process exit authorizes machine reclaim in this
checkpoint. Live cooperative release/refill with DMA/compute fences, complete machine memory
accounting, actual SDXL/Anima/H3, multi-GPU/NCCL, GPU crash tests and ordinary CLI qualification
remain open. Root owns GPU/rental tests; no GPU/NVML was used for these gates.

The bounded follow-up adds omitted-default `initial_host_budget_bytes` and optional
`host_tier_disk`, with distinct `host_tiers.initial_budget/1` and `host_tiers.disk/1`
capabilities. The initial nonnegative budget is applied before backend construction,
registration and automatic fill; omission retains the old behavior. Authenticated readonly
Tier grants select native `register_machine_host` and skip executor weight object descriptors,
hash traversal and private host-data fills. The general untrusted source path still verifies
bytes. CPU readonly weights bind their file mapping without requiring full host residency.

The native disk candidate and SDK socket/file path have CPU proof only. Thirty-one paired SDK
transport/initial-budget/readonly-file and real older/current-bound inference tests pass with
native PR305; three native disk and twelve existing host-tier tests pass. This does not
qualify standalone machine disk wiring or GPU behavior. The previous 46-gate source/owner
checkpoint remains available separately. A temporary CPU Torch import test was removed
because a CUDA-enabled Torch wheel mapped libcuda; the final clean gate uses actual host-only
native budget allocation and has no driver mapping.

Root has held expansion pending a matched current/new-stack value comparison. Legacy loading
must remain the default until descriptor loading is equal or faster under matched cache,
allocator and resource settings without weakening integrity. Initial-budget safety can
benefit either stack; these drafts are reviewable candidates, not a migration requirement.
