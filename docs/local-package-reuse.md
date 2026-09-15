# Retained local package preparation

Local wheel transfer is operation-scoped. Once preparation succeeds, Host removes
the transfer carriers and retains a preparation receipt. That receipt alone does
not establish readiness after Runtime restarts.

Host reissues `PrepareLocalPackage` to Runtime for every retained preparation
request. The original ordered wheel identities travel with empty paths when the
carriers have been removed. Runtime reopens its exact immutable environment,
verifies the retained interface and current base identity, restores job plans,
and registers the revision and operation before acknowledging preparation.

A caller asks about its exact revision under the current operation before sending
wheels. Runtime reopens by revision and registers that operation even when Host has
no previous transfer record. A hit does not manufacture received-wheel receipts.
`local_package_reuse_unavailable` means the caller must upload the missing revision.
A fresh operation accepts that upload normally. A retained prepared transfer is
terminal, so its repair needs a fresh transfer operation, retained across retries.
A successful preparation reuses installed code and dependencies only. Every root
still creates and executes its own request.

Runtime derives the environment receipt from the revision's exact project and
dependency wheels, the current base, and any frozen SDK snapshot reference. The
installation index stores no independent receipt ID. Local serving and local
machine execution may snapshot an installed SDK once; that snapshot is an immutable
installation fact and can reopen after the source SDK has gone. Updating captured
dependencies requires a new revision/generation, rather than mutating an installed
generation. Its content digest includes the snapshot and remains part of execution
and memoization identity.

Missing interface metadata is a recoverable installation miss. Missing job plans
are regenerated from the verified installed interface. An existing conflicting job
plan raises `local_package_job_plan_conflict`; uploading wheels cannot repair an
immutable-plan conflict and never overwrites it.

The independently negotiated `supports_local_installation_reuse` capability is
required before making that probe. Older Hosts rebuild `ProtocolInfoResult` from
their supported fields, so they cannot pass this capability through from a newer
Runtime. A new Host with an old Runtime also reports false.
