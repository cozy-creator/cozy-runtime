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

## Delivered closure selection

`_installed_dependencies.required_distributions` traverses the exact installed
`Requires-Dist` metadata. It does not resolve or download packages. Captured wheels
take precedence, dependency extras propagate across cycles, false platform markers
are skipped, and absent/incompatible required dependencies refuse before copying.
Runtime itself is a required executor dependency even for a wheel that omits it.
The fixed offline operations capture retains its existing exception for absent
NumPy; owned operation preparation installs it later. After installation, the
copied dependency graph and files are verified again before publication.

The installed-dependency and carrier tests pass (22 cases). The real core-only SDK
test also passes: capture works offline, unrelated missing bytes are never touched,
changing source Runtime code cannot affect the retained capture, and later operation
preparation installs NumPy using the original retained Runtime code.

## Real SDK measurement

A read-only capture used the mixed-model qualification's actual control SDK, which
contained Torch 2.14.0, NumPy 2.5.3 and its CUDA distribution family. The candidate
ran into a separate owned output root without changing that SDK or the live fixture.

| measurement | selected closure |
| --- | ---: |
| retained dependency bytes | 136,368,127 |
| retained files | 1,675 |
| retained distributions | 12 |
| Torch / NVIDIA / CUDA distributions retained | 0 |
| elapsed capture | 85.26 s |
| peak process RSS | 666,856 KiB |

Evidence lives in `outputs/dependency-snapshot-closure-20260913/`:
`real-sdk-capture-summary.json`, `real-sdk-capture-time.log`, the exact CLI result,
and the retained immutable generation. The original fixture observed roughly
5.7 GB of copying and six minutes for a full SDK capture. These runs shared a busy
disk with other work, so the byte reduction is direct evidence; the timings are
context, not an isolated speedup measurement. Base observation still imports Torch
from the source SDK, which contributes to RSS even though its bytes are not captured.

## Remaining work

Local model packages that actually declare Torch still retain it. Repeated edited
scripts should eventually reuse an existing immutable dependency generation while
keeping code revision identity separate. Reusing the mutable SDK directly, or
treating same version strings as proof of same bytes, would break capture isolation.
The existing package environment/content identities should own that reuse; a new
preparation database is unnecessary. Repeated file hashing inside snapshot creation
is a separate measurable optimization, not part of the closure-selection fix.
