# Published machine execution

Published jobs keep their existing `PackageCodeDigest`: the digest of the canonical
`Environment` whose locked-requirements reference binds the release's exact code
closure. Published execution does not invent a `LocalPackageRevision` or rename the
package into a local namespace.

The capture inventory adds published package/environment/interface references next
to private wheel revisions. Runtime validates both against the same prepared
revision inventory, retains the immutable environment receipt with accepted work,
and restores it through the existing package-environment reader. Submission,
attempts, events, retries and collection use the same workspace journal and engine.

`PreparePackageSet` already creates immutable receipts for rented preparation. The
local preinstalled interpreter path has no such receipt; machine acceptance must
first establish a Runtime-owned immutable environment, without treating a mutable
Creator installation as retained custody.

Explicitly external callable placement continues to require external coordination.
The capture/transport contract must distinguish this from captured same-machine
execution; package publication alone must not choose an execution coordinator.
