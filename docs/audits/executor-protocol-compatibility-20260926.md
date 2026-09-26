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
