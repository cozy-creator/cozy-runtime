# Private executor compatibility

Owner: /root/evolution_tolerance_audit
Branch: fix/executor-protocol-compatibility-20260926
Base: e01247e1b196768a07ba942be65b6f176d2b7d02
Worktree: /home/fidika/cozy/.worktrees/cozy-runtime/executor-protocol-compatibility-20260926

The worker launches the package environment's executor over internal.seam's
length-framed JSON socket. The hello currently exposes the installed Runtime
version, and worker.child.require_same_runtime rejects any different public release
before dispatch even when the private command/reply implementation is compatible.

Introduce an explicit private executor protocol revision, independent of package
version and source identity. Keep it stable for additive compatible changes.
Unsupported revisions refuse only the affected executor before dispatch; process,
parent, containment and environment-seal identity checks remain intact. The last
reviewed pre-field Runtime release 0.18.33 receives a bounded legacy compatibility
range; unknown older implementations are not guessed compatible.

Implemented: hello now names executor_protocol_revision=1, owned beside framing
in internal.seam. The worker validates that revision, not SDK version equality.
The legacy range is >=0.18.33,<0.18.34; source review confirmed executor.py,
seam.py and worker/child.py were unchanged between published v0.18.33 and the
base used here before this patch. This review evidence is not an admission hash.

Validation: five focused tests pass, including real executor subprocesses over
actual framed JSON sockets: protocol1 with different Runtime version metadata,
unsupported protocol2 refusal without sending work, legacy .33 hello acceptance,
and unknown older hello refusal. Malformed revisions refuse. Existing SDK spawn
regression now expects compatible protocol with different installed metadata.
Ruff check passed for affected files. No GPU execution, whole CI, SDK publication,
installed CLI replacement or rental occurred. Root owns the combined .34 release.
