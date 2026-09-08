# Input media paths

Owner: `/root/input_runtime`; branch: `direct-input-paths`.
Base: `10fe0598709687b8387bfcf8ba6ed0020fe4743b` (`origin/master`).

The worker verifies granted local media at its existing path and lends that path to the
executor. It does not copy the media, change source permissions, or delete the source at
attempt cleanup. Source identity is checked before consuming borrowed bytes; callers must
keep sources available and unchanged until execution finishes.

Network downloads belong to the granting host, not Runtime. Downloaded media and generated
image-preparation derivatives use the operating system temporary directory's `cozy/`
subdirectory with content-hash filenames. There is no second worker input-media cache.
Native retained byte/tree results retain their existing workspace and checkout ownership.

Validation covers actual paths/inodes, size/hash/media refusals, changes after verification,
cleanup preservation, isolated execution, and temporary derivative reuse.

The file handoff records device, inode, size, modification time and change time after
SHA-256, length and media verification. Reads and media decoding check those facts around
consumption; a disappeared or changed source raises a typed refusal. This does not lock out
another process editing the original, so it is not a filesystem immutability guarantee.
Raw-byte reads are bounded by the verified length. Existing decoder limits remain in force.

The image-prepare stdin request no longer accepts `output_directory`; its `prepared` result
names the one hash-addressed temporary derivative. Small images return the original path.
Existing derivative bytes must match the hash before reuse; publication never replaces an
existing inode. Attempt cleanup leaves temporary shared media in place for other readers.
The host or user can reclaim those disposable files when idle. Runtime 0.8.3 is the client
helper version boundary for this change.
