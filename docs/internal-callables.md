# Internal package callables

`@app.entrypoint(internal=True)` and `@app.job(internal=True)` retain executable
metadata and model bindings while restricting invocation to managed children of
the same installed package revision. Managed child dispatch currently requires a
job parent; this flag does not add serving-parent orchestration. An internal
callable cannot be a root
request or a call from another package revision.

The package interface retains the callable with optional `internal: true`.
Public discovery filters those entries, while installation and same-package
capture keep them. `hidden=True` continues to mean unavailable and unbound.

The worker enforces this using admitted capture and child journal identities;
caller-supplied names or a parent request ID alone confer no authority.

Runtime 0.18.3 adds this field. Older Runtime interface readers reject it, so
an older worker cannot silently expose an internal callable. Public interface
wheels omit internal functions; installation retains their complete metadata.
