# Reliable inference under memory pressure

Audit base: Runtime `b2a93731`, machine `cf902bf`. Program audit:
[cozy-machine #30](https://github.com/cozy-creator/cozy-machine/pull/30).

The requested model, dimensions, steps and numerical policy must survive recovery.
Paging model weights does not by itself make an indivisible activation or kernel
workspace fit. Failure to implement a recovery contract is separate from measured
physical capacity. Neither condition justifies silently changing the request.

## Confirmed findings

1. `WeightResidency._block` inferred complete-forward replay safety from one
   observed call's input mutations. Its subsequent quiet path did not watch module
   buffers, Python state or RNG, and its input version check omitted inference
   tensors. A later conditional input mutation was also invisible to that check.
2. The operator recovery mode retried mutable, random and opaque custom operators
   without knowing whether a failed kernel had already changed state. Real CPU
   dispatcher faults demonstrate partial mutation and a consumed random draw.
3. Executor startup charged the device-wide memory delta as this process's CUDA
   context. Concurrent foreign allocations therefore changed subsequent budgets.
   Context measurements took maxima and were persisted without an expiry.
4. Machine group learning persisted rank 0's context on every GPU despite having
   per-rank facts. Heterogeneous devices or asymmetric library work could be charged
   incorrectly.
5. The finer 16 MiB module grain does not bound childless modules. A real meta
   embedding produces a 2 GiB region. This affects a catalog model: MiniMax H3's
   text config is vocabulary 151936, width 5120, and its publishing recipe retains
   `model.language_model.embed_tokens.weight` at bf16. That leaf alone is
   1,555,824,640 bytes (1.449 GiB). Anima's width 1024 embedding is 0.290 GiB, so
   the same limit does not establish the cause of its 1 GiB failures.

Inherited reports of six successful Anima requests at 1.5 GiB are GPU evidence for
the inherited branch, not qualification of these changes. The 1.25/1 GiB cells and
safe recovery on continued H3 groups remain unqualified.

## Decisions and planned work

| Work | Implementation | Completion gate |
|---|---|---|
| Atomic recovery | Never replay a whole forward from inferred purity. Retry pure ATen operations; failed mutable, random or opaque ops identify `recovery_contract_unavailable`. | Real CPU fault tests, exact-output constrained CUDA inference, matched launch overhead |
| Nested dispatch | One recovery mode per Python thread, avoiding duplicate nested block modes without suppressing another thread. | Dispatcher ordering tests and GPU launch timings |
| Idle components | Implemented: track executing components and reclaim other idle regions while a failed safe op retains its block lease. Operand-backed components stay; live executing cursors never close or replan. | Added real GPU plane gate; bit-identical grouped and alternating Anima at 1.5, 1.25 and 1 GiB remain pending |
| Trusted context | Use per-device attributable process memory. Unknown stays unknown. Add measurement provenance, discard only old untrusted persisted context rows, and learn each rank's own context. | CPU migration/rank-selection tests and concurrent foreign-allocation CUDA startup |
| Collective recovery | Keep the inherited whole-call rerun branch unmerged. Define an explicit collective operation transaction with rollback or restart from an authored checkpoint. | Mid-collective failure on two real GPUs with state/RNG/output proof |
| Broader replay support | Add explicit stock random-op generator rollback and supported mutable-op contracts only where useful real workloads require them. | Fault injected after state advance and exact request output |
| Physical capacity | Diagnose the remaining indivisible operator peak. Implement semantics-preserving streaming kernels or authored tiling where necessary. | Original requests at every supported pool, no smaller substitutes |

## First degraded leaf: selected-row embedding

The smallest concrete next slice is the H3 token embedding, not a whole-model CPU
fallback. The lookup needs requested rows, not the complete vocabulary. For a plain
bf16/f16 embedding with `max_norm=None`, read each requested row from the existing
immutable model-source object descriptor and copy exact stored values to the output
on the execution device. Preserve index order, duplicate indices, dtype, padding row
and shape; no arithmetic or numerical tolerance is required for this lookup.

Do not allocate or pin the complete embedding on either host or device. Batch source
reads into a buffer bounded by the current host grant, with at most one buffer in
flight and no speculative cache. Choose the normal resident path whenever the leaf
fits. Enter the selected-row path only when the actual remaining weight budget cannot
hold the leaf. The output is still required by the original request; a failure to
hold that output is a separate downstream activation constraint.

Before implementation, verify the published H3 manifest's plain bf16 part and the
existing model-source descriptor's exact object offsets. A plane host memfd is not
automatically a valid source: unfilled optional-tier pages must never be mistaken for
stored weights. Keep TensorFS ownership on the machine and consume its immutable
read grant. GPU qualification should compare lookup bytes exactly with ordinary
`torch.nn.functional.embedding`, then run the original H3 conditioning request under
the available 1 GiB pool. Support for mutating `max_norm` or encoded embedding storage
requires a separate concrete contract, rather than silently changing those semantics.

Oversized Linear leaves are a later slice, after catalog evidence identifies the
first actual failing projection. Output-row chunking preserves reduction dimensions,
but native kernel choices can change floating-point rounding; record the numerical
acceptance boundary and measured host/device peaks before claiming equivalence.

## Performance and acceptance

The first CPU microbenchmark uses one thread, alternating plain/dispatch arms, and
exactly matching projection outputs. Four tiny operators cost about 12–13 us more
per call with dispatch; this is substantial for tiny kernels and does not predict
CUDA inference time. GPU launch overhead and full request timings remain gates.

Use an ordinary default-home Cozy CLI on a rental with recorded Runtime, machine,
TensorFS, package and model versions. Run repeated grouped and alternating cells,
retaining unchanged seeds, prompts, dimensions and steps. Measure preparation,
execution, output delivery, paging traffic, lowest free memory and executor reuse.
Compare against ComfyUI with identical workload semantics and independently
recorded cache state. Success requires every request completing correctly and the
matched workload winning; mechanism tests alone do not establish either outcome.

Unsafe failed operators latch process poison even before handler entry or when authored code catches the refusal. Retained executors cannot serve a subsequent call from unproven state; the machine also fences held GPU allocations until reader release or observed exit. New CPU retained-executor cases cover uncaught and caught failures. Unknown future tag/schema metadata is isolated to typed recovery capability failure, without a Torch version gate.
