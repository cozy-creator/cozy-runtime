# Private executor compatibility

Owner: /root/evolution_tolerance_audit
Branch: fix/executor-protocol-compatibility-20260926
Base: e01247e1b196768a07ba942be65b6f176d2b7d02
Worktree: /home/fidika/cozy/.worktrees/cozy-runtime/executor-protocol-compatibility-20260926

The worker launches the package environment's executor over internal.seam's
length-framed JSON socket. The hello exposes the installed Runtime version for
diagnosis, while worker.child.require_executor_protocol validates the private
command/reply revision and the native-interface field. A public Runtime version
difference is not itself a refusal when the private seam remains compatible.

The explicit private executor protocol revision is independent of package version
and source identity. Keep it stable for additive compatible changes. Unsupported
or malformed revisions, and hellos without the native-interface field, refuse only
the affected executor before dispatch; process, parent, containment and
environment-seal identity checks remain intact. Additive memory capability fields
are advisory: a revision-1 peer may omit them, in which case the worker uses the
all-resident placement contract and does not attempt unsupported host-room or
rank-memory operations.

Implemented: hello now names executor_protocol_revision=1, owned beside framing
in internal.seam. The worker validates that revision and the native-interface
field, not SDK version equality. This review evidence is not an admission hash.

Validation: focused tests pass, including real executor subprocesses over actual
framed JSON sockets: protocol1 with different Runtime version metadata, unsupported
protocol2 refusal without sending work, malformed revision refusal, and a published
0.18.51 hello without memory fields remaining compatible. Published 0.18.86 and
0.18.87 peers also pass the Start/Load/Activate/PrepareRequest/Invoke/Probe/Restore
sequence in both directions on the CPU path. These checks exercise a tiny
weightless invoke; weight-bearing mixed-GPU qualification remains a separate gate.
Ruff and mypy pass for the current combined tree. No rental or default-machine
operation is implied by these protocol checks.
