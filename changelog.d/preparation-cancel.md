Wire 73: an explicit preparation may name its requester with `operation_id`. Its work runs
off the RPC: a disconnect detaches the caller, and the same `operation_id` attaches again.
`CancelPreparation` (PodHost, forwarded to Runtime) cancels only that requester's TensorFS
token and answers its last per-model progress. A shared transfer continues for any other
requester (identified preparation or run), and landed objects stay. Identified models-only
preparation no longer waits on the package preparation lock. Live identified work holds
idle release, guarded restart and store collection. Calls without `operation_id` behave as
before, and an older Runtime answers `capability_unavailable`.
