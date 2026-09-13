# Immutable dependencies across edited captures

Owner: `/root/dependency_snapshot_fix`  
Runtime branch: `fix/immutable-dependency-generations-20260914`  
Runtime base: `730bf1f15afb7aba50ebb3704224dfc6935b5989`  
Creator branch: `fix/captured-dependency-selection-20260914`  
Creator base: `80e8ed42e8866f42cf7e2196b41ab2d0bc979419`

The merged carrier-scope and hash-pass fixes are independent of this follow-up.
They may be released while the work below is implemented and qualified.

## Exact captured selection already has an owner

Current Creator calls `CaptureUnpublishedClosure` with its installed closure,
selected root extras and Python version. `ActiveWheelRequirements` evaluates
the already selected metadata; `PinDependencies` replaces the root wheel's
Requires-Dist with the selected flat dependency requirements and removes
Provides-Extra. The resulting wheel bytes already carry selection into Runtime.
No additional protocol field, metadata format, or package resolver is needed.

Creator already installs the selected locked closure into its authored venv.
Runtime preserves that complete selected roster and shares its immutable bytes.
`exact_dependencies` keeps its established no-online-resolution meaning; it is not
reinterpreted as a claim that a raw caller flattened root extras. Fixed-carrier
pruning remains separate. No new marker, schema or version floor is needed.

## Reuse retained bytes, not mutable SDK files

Final per-install generations remain self-contained. The proposed byte reuse is
an internal immutable dependency payload cache keyed by the already verified file
RECORD content hash and executable bit. Required source bytes are still verified on reuse;
unchanged Torch/CUDA payloads need no new copies. Metadata and launchers remain
private because generation construction rewrites RECORD and console wrappers.

Install captured wheels before attaching shared dependency payloads so installers
cannot modify shared inodes. Preserve final dependency verification, and change
Runtime cleanup to make only directories writable. Creator's Unix cleanup already
follows that rule. Hardlinks in completed generations retain bytes independently
of cache/source deletion; OS link counts provide lifetime facts without a database.

Creator supplies the explicit `local-packages/dependency-objects` cache path as
`COZY_DEPENDENCY_CACHE`. Older local Runtime versions ignore that additive config
value and retain full-copy behavior; it never reaches rental bootstrap or device
executors. Cache entries are disposable and completed generation links retain the
bytes independently. Qualification remains in progress. Required proof includes a selected root extra, an edited
script sharing unchanged dependency bytes, changed-source rejection, deletion of
the original source, and execution of both retained captures.
