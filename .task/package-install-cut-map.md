# Package installation hard cut

## Wire changes coordinated with Creator and Host

- Delete `LocalPackageRevision` and `PublishedPackageRevision` canonical documents.
- `DevelopmentPackage` retains package/release and an opaque `installation_id`, without source or revision hashes. The ID identifies an installed generation, never source content.
- Private preparation joins model bindings to a retained installation through operation/installation IDs.
- `MachineExecutionCapture` lists installed packages and its root installation ID. Callable binding uses caller/callee installation IDs, module, export and entrypoint. Remove interface/revision digest equality from callable admission.
- `MachineModelDefault` references the callee installation ID while retaining real TensorFS manifest IDs.
- Local file-transfer source and revision fingerprints disappear. Per-wheel download hash/length checks remain ordinary installation integrity; authorization and operation replay IDs remain.
- Installed interface bytes come from the worker environment after uv installation. Creator's host SDK AST is not an authority over remote package code.

## Runtime deletion sites

- `worker/package_prepare.py`: `_verify_local_revision`, source/revision length gates, digest-derived placement IDs and revision job build identity.
- `worker/local_installations.py`: digest-addressed revision receipts, composite re-derivation and repeated wheel closure comparisons.
- `worker/session.py`: prepared-revision equality, development source fences, and exact environment receipt admission. Retain lifecycle lookup and request authorization.
- `worker/machine_capture.py`, `machine_child_target.py`, `machine_builtin.py`: captured revision/interface hashes become installation lookups and callable metadata validation.
- `internal/local.py` and `cli/run.py`: `development_source_digest` traversal and source-hash execution fences.
- `internal/package_environment.py`: composite content/receipt digest store and snapshot equality; ordinary uv installs into per-package generations replace it. The installation table stores package/release, interpreter path, interface metadata and generation ownership.
- `internal/installed_interfaces.py`: regenerated source and terminal-export byte comparisons removed. Semantic declaration checks remain.
- `internal/_installed_dependencies.py`: hand-copied SDK member manifest/hash validation replaced by ordinary uv dependency installation; model/input object integrity is unrelated.

## Lifecycle

Published installs select package name/version; uv owns resolution, environment isolation and wheel integrity. A source sync updates the editable project's owned staging tree and declared dependency selection, then prepares the next environment before an invocation. Existing invocations retain their installed generation and process; update never mutates their code directory. Idle executors switch to the new generation before the next invocation. An opaque install generation is only lifetime plumbing, never compared to hashes of content.

## Keep separate

TensorFS checkpoint/object/retention digests, input/output integrity, authentication/signature checks, operation request IDs, and actual operation implementation/input memoization keys are not package installation identity. Do not substitute random installation IDs into computation identity or delete those memo owners while migrating admission.

No tests, CI, lint or vet run. No global SDK/service mutation.
