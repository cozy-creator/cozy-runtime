# Retained local package preparation

Local wheel transfer is operation-scoped. Once preparation succeeds, Host removes
the transfer carriers and retains a preparation receipt. That receipt alone does
not establish readiness after Runtime restarts.

Host reissues `PrepareLocalPackage` to Runtime for every retained preparation
request. The original ordered wheel identities travel with empty paths when the
carriers have been removed. Runtime reopens its exact immutable environment,
verifies the retained interface and current base identity, restores job plans,
and registers the revision and operation before acknowledging preparation.

A caller may probe a previously successful operation before sending wheels for a
fresh root. `local_package_reuse_unavailable` means the retained environment cannot
be reopened; the caller must upload under a new operation. A successful preparation
reuses code and dependencies only. Every root still creates and executes its own
request.
