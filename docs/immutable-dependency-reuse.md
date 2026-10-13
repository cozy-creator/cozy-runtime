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

Final per-install generations remain self-contained. Byte reuse uses
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
bytes independently.

## Verification

The ordinary Creator CLI test selected `model-tools[cu130]`, executed two edited
scripts, ran four real model calls on the local RTX 4070, reused native production
and cancelled the final call. Distinct captured generations shared the same
read-only `libtorch_cpu.so` (446,314,593 bytes) and `libtorch_cuda.so`
(502,842,281 bytes) inodes, different from the source SDK inodes. After fixture
processes stopped and source venvs/cache were removed, both retained captures
imported their model code and executed CUDA arithmetic. The test took 626.75 s
including environment setup, capture, execution, cancellation and teardown.

The CLI wheel SHA256 was
`8f462778e3fa15bf44fbb9775fbab01a355b217030ac606a884206ac976683ea`.
The subsequent source change keeps authored selection conservative instead of
reinterpreting `exact_dependencies`; the raw exact-caller regression covers it.
The byte-sharing mechanism exercised by the CLI remains unchanged.

All 33 dependency/carrier/core-capture tests pass, including concurrent cache
publication, source changes with restored mtime, source/cache deletion, cleanup
without chmod of shared payloads, mutable cache refusal and staged collisions.
The additional real cross-filesystem case verifies copy fallback without caching.
Ruff, focused mypy and the environment-authority fence pass.

## Measured repeated capture

A retained 12-distribution SDK supplied 225,616,067 bytes in 1,013 files. Running
the real snapshot writer plus final verification twice against one cache produced
identical manifests.

| measurement | first capture | repeated capture |
| --- | ---: | ---: |
| process bytes read (`rchar`) | 451,564,098 | 451,550,194 |
| process bytes written (`wchar`) | 225,676,834 | 529,530 |
| physical bytes written | 228,007,936 | 868,352 |
| elapsed | 8.121 s | 4.933 s |
| peak RSS | 57,640 KiB | 58,612 KiB |

Repeated capture reduced logical writes by 99.77%; private metadata and launchers
still get written. Mutable source payloads and final captured content are still
verified, so read volume remains approximately unchanged. These timings describe
this local SDK fixture, not a projected H3 inference or remote speedup.

## Compatibility and rollout

Runtime 0.16.10 accepted the new config value for both version query and actual
builtin capture, retained all 12 source distributions / 225,614,517 bytes through
its original full-copy path, and did not create the supplied cache directory.
Neither a version floor nor a remote protocol change is required for the cache.

Runtime PR469 and Creator PR530 are source changes under review; cr-175 tracks
delivery. No shared SDK, installed CLI, Calcifer, rental or public release was
modified by this work. The cache is disposable; automatic cache space policy is
not introduced here.

Evidence: `outputs/immutable-dependency-generations-20260914/`, especially
`cli-proof.log`, `old-compatibility.json`, `profile-first.json`,
`profile-second.json`, `concurrent-tests.log` and `final-mypy.log`.
