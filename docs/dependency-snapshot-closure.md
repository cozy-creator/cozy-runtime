# Local dependency snapshot scope

Owner: `/root/dependency_snapshot_fix`  
Branch: `fix/dependency-snapshot-closure-20260913`  
Base: `76dcf0bc44fd1c159c04e7af674c3cbb2d398a35`

Unpublished package preparation currently copies every distribution installed in
its dependency interpreter. The fixed Runtime operations carrier therefore copies
Torch and CUDA from an SDK even though its dependency graph requires only Runtime
and NumPy. The mixed-model CLI qualification observed 33.9 GB of process reads and
5.7 GB of writes before a tiny numerical call.

The capture must retain the declared transitive dependency closure, including
selected extras and applicable platform markers. Unrelated distributions are not
part of that closure. Captured package wheels supply their own metadata and take
precedence over editable source installations. Required dependency bytes remain
immutable copies verified against their installed RECORD; source deletion or later
editing must not affect the captured invocation.

Worker-image ownership remains unchanged: rented workers use their base image for
the protected Torch/CUDA family. A mutable local SDK is not an immutable base-image
lease, so local scripts that actually require Torch still need retained bytes.
Sharing those retained bytes across different package revisions needs an explicit
immutable dependency generation rather than a symlink into a mutable SDK.

Implementation and measured verification are pending.
