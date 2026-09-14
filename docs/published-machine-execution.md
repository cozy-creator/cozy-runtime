# Published machine execution

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
