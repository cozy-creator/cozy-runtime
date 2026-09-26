# Worker startup evolution tolerance

Owner: /root/evolution_tolerance_audit
Branch: fix/worker-startup-evolution-20260926
Base: 7d71a3720d4658667caebdcf780fe2769304e3e1
Worktree: /home/fidika/cozy/.worktrees/cozy-runtime/worker-startup-evolution-20260926

A newer host may provide additional platform settings or owner-auth metadata.
Runtime must read the fields it consumes without turning unused additions into a
boot refusal. Required worker identity, owner key, media hashes and source-injection
checks remain enforced. Child processes still receive only the existing narrow
allowlist; ignoring an extra parent environment field does not inherit it.

Scope: Runtime host environment and owner-auth object readers only. The privileged
publication-authority document, actual execution requirements, signature/digest
checks and wire capability admission remain independently enforced. This is not a
claim of fleet-wide mixed-version qualification or permission to publish a release.

Validation: the focused worker-boot subprocess regression passed using the existing
Python 3.12 test interpreter with this worktree's source on PYTHONPATH. It proves
additive env/auth fields do not affect parsed authority or reach child processes;
missing/malformed owner keys, Python source injection, wrong privileged worker
identity, invalid publication tokens/origins and unsupported authority version
still refuse. Ruff check and formatting pass for the two affected Python files.
No full CI, product invocation, SDK installation, release or rental was performed.
