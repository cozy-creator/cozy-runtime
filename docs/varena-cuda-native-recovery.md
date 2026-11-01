# CUDA recovery belongs to native allocation

This development cut follows the exact dfb/2626 owner qualification: ordinary Cozy
runs25587–25591 passed independent audit, recorded in Creator a9920511a. Typed held-GIL
refusal, later explicit resize, capture-aware close, ordinary and explicitly guarded
NativeSequence/graph retirement, escaped output storage and actual physical-OOM recovery
are separate proof boundaries. The allocation test observed192MiB physically released by
allocation return,64MiB still Ready, and zero peer credit; peer-exhausted telemetry did not
prevent the successful real local driver retry.

The matched control is Runtime2626e3f48bc147f1f3d46b19bd385356e628680c. This branch starts
from freshly fetched origin/mastercb6c66bf3 and fast-forwards that exact integration before
this cut. Native source, kernels, source reads/fills, Plan generation and owner callbacks
are unchanged. The required native cohort is the existing CacheOwner capability; no
unpublished release version or product Git-SHA lockstep is invented. A CUDA Memory refuses
an SDK missing that capability before changing limits or creating its Arena.

CUDA no longer sorts Python cache/weight owners after a failed allocation, retries
`torch.empty` or `Region.prepare` in Python, or performs a second `room()`-based reclaim
before grouped native input admission. The installed native managed limit remains the
physical admission boundary. The existing free-only native preparation precheck remains.

`Memory.reclaim` retains the existing non-CUDA owner order and measurement behavior.
Explicit CUDA `Memory.trim` is a lifecycle discard-all request: retire eligible graph
owners, drop idle disposable owners/views, reclaim idle native backing and trim working
cache. Active pins and escaped storage retain their ordinary lifetimes. Host-tier cleanup
stays part of explicit offload and its source policy remains unchanged.

`Request.reclaim` returns no additional CUDA capacity to the legacy cut machinery. P2's
remaining estimated sizing and operation split/retry path is not migrated by this change;
closed native consumers retain their existing no-operation-retry contracts. LM-specific
KV/DerivedCache lifecycle and v3 Cooperative/Foreign service accounting remain separately
qualified work, not conclusions inferred from the five componentless parent fixtures.

Source checks and independent review precede packaging. The previously passing control
artifacts remain immutable; the cut itself still needs matched runtime/model qualification.
