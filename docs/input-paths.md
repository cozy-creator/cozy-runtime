# Input media paths

Owner: `/root/input_runtime`; branch: `direct-input-paths`.
Base: `10fe0598709687b8387bfcf8ba6ed0020fe4743b` (`origin/master`).

The worker verifies granted local media at its existing path and lends that path to the
executor. It does not copy the media, change source permissions, or delete the source at
attempt cleanup. Source identity is checked before consuming borrowed bytes; callers must
keep sources available and unchanged until execution finishes.

Network downloads belong to the granting host, not Runtime. There is no second worker
input-media cache.
Native retained byte/tree results retain their existing workspace and checkout ownership.

Validation covers actual paths/inodes, size/hash/media refusals, changes after verification,
cleanup preservation, isolated execution, and temporary derivative reuse.

The file handoff records device, inode, size, modification time and change time after
SHA-256, length and media verification. Reads and media decoding check those facts around
consumption; a disappeared or changed source raises a typed refusal. This does not lock out
another process editing the original, so it is not a filesystem immutability guarantee.
Raw-byte reads are bounded by the verified length. Media handle reads (`read`, `stream`,
`probe`) check the same facts and keep their decode limits.
