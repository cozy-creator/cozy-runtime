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
