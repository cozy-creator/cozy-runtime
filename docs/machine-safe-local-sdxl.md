# Safe local descriptor cohort

This candidate is based on fetched Runtime `0d67384a`, preserving context-room admission
`027a73ed`, measured-memory-progress retries `40eab632`, compute quiescence/resident sealing
`722913ee` and binding/decoded-copy recovery `1c7554a2`. Descriptor integration changes only
typed source records/readonly transport, Checkpoint's optional source, and PlaneBackend's
selected source/teardown. It does not replace current weight placement or safety behavior.
Bounded full verification retains all integrity checks, serialized main-thread exchange,
atomic received-FD custody and failure cleanup. No model/worker implementation is copied.

Coherent native dependency is DCE's TensorFS `822c950` (PR #303), based on release0.3.91
`6c61a06` with failed-copy drain `614967d` and unclosed-plane teardown `325d9dc`, plus the
additive descriptor API/readonly mapping support. Native wheel SHA-256 is
`b77968f6caa19ac903d80330561800b2f1a0ad7517727d278fad08b3d5697ae4`.
Runtime pure wheel SHA-256 is
`464faea80fead21657827821960e16b797fcecc4be4e806ee32ec11eec84c5d0`.
Both are private owned artifacts, not published releases or deployment equality gates.

41 CPU source/read-lease/skew/SDXL-policy checks passed against that exact compiled native
dependency, including full verification, readonly transport, corruption/drain/FD/mapping
cleanup, older Load defaults and capability isolation. Changed source files pass Ruff,
Python compilation and whitespace checks. All four required Runtime safety commits are
verified ancestors; the source integration diff leaves their implementations intact.
No package code was executed to describe it. No GPU/NVML/rental/process operations were
performed by this task. Root owns unchanged1024x1024/20-step local inference qualification.

For the first normal single-executor local8GB run, prefer `plane_budget_bytes=-1` so stage
entry does not overwrite Runtime's measured activation-growth policy with a fixed5GiB
grant. Keep the unchanged three published SDXL2.4.0 requests and one interpreter. Root's
fresh idle-device measurement/authorized ceiling must precede CUDA context creation;
the standalone pilot does not implement the full worker context scheduler. A fixed grant
is not an activation reserve and can undo the preceding stage's OOM cut.

The pilot sends pinned/device Budget after Load. Current load initializes the weight
policy before that and may prefill host memory up to the SDK's available-host policy;
`pinned_budget_bytes=4GiB` therefore caps inference cache, not the preceding load fill.
Source mappings/page-cache RSS, private pinned/staging/activation memory and complete
Degree1 accounting remain separate. Do not reuse the old Runtime67/TensorFS74 install,
old pre-safety Runtime100 candidate or pressure/fault tests on the display GPU. No local
GPU success, no universal-size recovery and no performance conclusion is claimed here.
