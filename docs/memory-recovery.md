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

Inherited reports of six successful Anima requests at 1.5 GiB are GPU evidence for
the inherited branch, not qualification of these changes. The 1.25/1 GiB cells and
safe recovery on continued H3 groups remain unqualified.

## Decisions and planned work

| Work | Implementation | Completion gate |
|---|---|---|
| Atomic recovery | Never replay a whole forward from inferred purity. Retry pure ATen operations; failed mutable, random or opaque ops identify `recovery_contract_unavailable`. | Real CPU fault tests, exact-output constrained CUDA inference, matched launch overhead |
| Nested dispatch | One recovery mode per active execution, avoiding duplicate nested block modes. | Dispatcher ordering tests and GPU launch timings |
| Idle components | Reclaim idle component regions while a failed safe op retains its executing block's lease. Exclude all executing components and operand-backed regions; never replan their live cursors. | Bit-identical grouped and alternating Anima at 1.5, 1.25 and 1 GiB |
| Trusted context | Use per-device attributable process memory. Unknown stays unknown. Add measurement provenance, discard only old untrusted persisted context rows, and learn each rank's own context. | CPU migration/rank-selection tests and concurrent foreign-allocation CUDA startup |
| Collective recovery | Keep the inherited whole-call rerun branch unmerged. Define an explicit collective operation transaction with rollback or restart from an authored checkpoint. | Mid-collective failure on two real GPUs with state/RNG/output proof |
| Broader replay support | Add explicit stock random-op generator rollback and supported mutable-op contracts only where useful real workloads require them. | Fault injected after state advance and exact request output |
| Physical capacity | Diagnose the remaining indivisible operator peak. Implement semantics-preserving streaming kernels or authored tiling where necessary. | Original requests at every supported pool, no smaller substitutes |

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
