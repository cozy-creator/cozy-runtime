# Cozy machine agent

`cozy-machine` is the per-machine API and Runtime supervisor. It runs independently
of the personal Cozy controller, using the same binary and machine API on laptops
and rented machines. Runtime remains the execution-journal writer and scheduler.

Build from this directory with `go build .`; run with `cozy-machine run` and the
machine grant in its environment. `cozy-machine version --json` reports identity.
No Creator checkout or client configuration is required.

This extraction starts from Creator Host commit
`6ea0916e87c731365cd01652cec68657fcb6c595`. Pure encoding/output helpers and the public
worker protocol are copied temporarily; they have no dependency on Creator client
configuration. The protocol SOURCE identifies its upstream schema. These are
source provenance records, not deployment-equality requirements.

Delivery and lifecycle integration are tracked at
https://github.com/cozy-creator/tracker/issues/249.
