# Published machine execution

The preparation record below describes the retired Python-worker implementation. TensorD
now owns preparation, acceptance and the durable journal; Runtime constructs and executes
models from supplied data/buffers. See [Machine and Runtime roles](machine-agent-runtime.md).
The execution-duration section describes Runtime observations that TensorD records.

## Historical preparation record

Published jobs keep their existing `PackageCodeDigest`: the digest of the canonical
`Environment` whose locked-requirements reference binds the release's exact code
closure. Published execution does not invent a `LocalPackageRevision` or rename the
package into a local namespace.

Wire54 adds published package/environment/interface references beside private wheel
revisions. Runtime verifies both inventories against the same prepared revisions,
requires an immutable installed receipt, and journals that exact preparation with
accepted work. Inference roots select the exact prepared placement ID and model
bindings. Jobs, inference, child calls, retries, events, and collection share the
existing workspace journal and AttemptEngine; inference results are never memoized.

Published preparation from a local SDK installs the release's hash-pinned wheel
selection and copies SDK dependencies through the existing immutable RECORD snapshot
reader. Snapshot bytes join environment content identity, while public build identity
remains the canonical Environment digest. Receipt restoration needs neither the
source SDK directory nor the package index. The older describe-only local install
adapter still opens an ordinary venv; it cannot establish machine-execution custody.

Qualification in this branch uses real Runtime/worker executors for private,
published base-image, and published SDK roots: submit, query, collect, worker
replacement, resume with the package index stopped, and fresh repeated published
inference all pass. A separate test removes the SDK and exported wheel, then opens
and executes from the retained receipt. These tests supplement ordinary Creator CLI
qualification, which remains required before a product completion claim.

Creator routing in this increment is limited to explicit same-machine selection.
Existing default and cross-machine published orchestration remains in place. Durable
external delegation and default placement policy remain followups in cr170/cl259;
package publication alone must not choose an execution coordinator.

## Execution duration

`run.timing` records the elapsed union of cooperative author execution across a root
and its children. Model installation, preparation, GPU admission, and periods when all
author event loops are blocked do not count. Synchronous author work and callbacks in
`asyncio.to_thread` / `run_in_executor` do count, including work concurrent with another
call's queue wait. This is elapsed executor work, not a CPU/GPU utilization meter.

The observation lives at the shared invocation boundary. Selector transitions stay on
the private executor channel; ordinary progress and execution boundaries publish a
cumulative snapshot. The journal keeps only the latest snapshot per root attempt,
preserving settled attempts across retry and never adding their paused gaps.

Older executors, a lost activity observation, an interrupted clock, new untracked Python
threads, and unsupported event-loop policies report unknown execution duration.
Threads already present at the executor's first author call form its baseline;
independent work on those pre-existing threads is outside this cooperative measurement.
Windows and custom loop policies retain their normal execution behavior. The client
must not replace unknown execution with admission-to-outcome wall time or advance a
stale snapshot using its own clock. Timing failures cannot fail accepted execution.

Authors that keep their own background pools can import `ThreadPoolExecutor` from
`cozy_runtime.author`. It preserves the standard executor's constructor, futures,
cancellation, initializer, and shutdown behavior. Each submission captures its current
invocation for timing; only executing callbacks count, so an idle pool can survive many
calls without adding wait time. Submissions outside an invocation remain ordinary work.
`asyncio.to_thread` and `run_in_executor` participate automatically. A joined helper
thread nested inside tracked work is covered by that work's interval; a raw thread that
continues after its tracked caller becomes idle makes the measurement unknown.
The plain-script adapter uses the same observed loop and ends its author clock before
framework shutdown. A synchronous script can still create its own event loop; managed
child waits on an unobserved loop make its execution duration unknown.
