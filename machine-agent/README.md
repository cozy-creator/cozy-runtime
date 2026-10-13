# Cozy machine agent

`cozy-machine` is the per-machine API and Runtime supervisor. It runs independently
of the personal Cozy controller, using the same binary and machine API on laptops
and rented machines. Runtime remains the execution-journal writer and scheduler.

Build from this directory with `go build .`; run with `cozy-machine run` and the
machine grant in its environment. `cozy-machine version --json` reports identity.
No Creator checkout or client configuration is required.

The glibc Linux x86-64 and ARM64 `cozy-runtime` PyPI wheels include the `cozy-machine`
executable. `uv pip install cozy-runtime` installs it alongside the Python commands;
installation does not start a service. The wheel uses the standard scripts
installation directory and contains the native executable, with no Python launcher
or first-run binary download. The portable wheel and source distribution remain
toolchain-free and do not contain the Linux agent.

The personal controller and Tensorhub both connect to this machine API. A local
machine uses `COZY_MACHINE_LIFETIME=persistent`, a stable locally generated
`COZY_WORKER_ID`, its root/port, and `COZY_AUTHORIZED_KEYS`. It needs no Hub
registration, worker token, or rental lifecycle. An optional launch HMAC key seals
its identity receipt before Runtime starts, so status and repair remain reachable
when execution is unavailable. Rentals retain measured readiness and idle release.

Runtime is the sole execution journal writer and owns its queue, executor parking,
model preparation, and memory policy. The agent supervises it, validates fresh
process-incarnation readiness, and serves authenticated APIs and media. It keeps
no second job journal. While Runtime is down, the guardian reads its versioned
public SQLite views using `mode=ro`; it never reads private journal tables.
No durable Runtime ownership history is consumed. Structural startup refusals
remain visible as failures; transient exits retry with bounded backoff. The
resident coordinator survives compute idleness and controller disconnection.

`POST /v1/hubs/access` accepts a `Cozy-Cap` with action `hub-access` and a scoped
execution credential bound by that Hub to the machine certificate. Its JSON body
contains `origin`, `token`, `expires_at` (Unix seconds), `environment`, and optional
`ca_der_b64url`. Refresh atomically updates Runtime's hot-readable access document;
responses never return tokens. Access authorizes content operations, not rental
registration or lifecycle.

This extraction starts from Creator Host commit
`6ea0916e87c731365cd01652cec68657fcb6c595`. Pure encoding/output helpers and the public
worker protocol are copied temporarily; they have no dependency on Creator client
configuration. The protocol SOURCE identifies its upstream schema. These are
source provenance records, not deployment-equality requirements.

Delivery and lifecycle integration are tracked at
https://github.com/cozy-creator/tracker/issues/249.

The Runtime PyPI publisher builds both static agents from its release source and
includes them in Linux wheels. The included executable reports that Runtime release
version; transport compatibility still uses capabilities and worker-protocol ranges.
The native-kernel x86-64 wheel (glibc 2.28+) also contains the agent. A Python-independent
Linux ARM64 wheel (glibc 2.17+) carries the agent without CUDA kernels. macOS and Windows
continue to install the portable Python wheel.

The Runtime publisher invokes
`scripts/build-release.sh --version VERSION --output /absolute/output` from
committed source. Its static Linux amd64/arm64 executables and local
`machine-agent.json` hash manifest feed the wheel bundler in that same release.
The machine-agent workflow runs checks only. There is no separate agent release
selection or publication path; images and clients obtain the executable from
the Runtime wheel. Capabilities and worker-protocol ranges determine compatibility.

Rental control authority is refreshed from the owning Hub before every new gRPC
call, privileged HTTP request, and WebRTC handshake. The worker's latest-attempt
credential reads `/v1/worker/rental/authorized-keys`; user login credentials never
reach this endpoint. Existing client transports are revoked when their signer
leaves the set or its server-declared lease expires. A failed authority read
refuses new control and clears cached authority. This never cancels durable work
or changes the agent's private Runtime-owner claim. Standalone machines retain
locally owned keys. Deploy the Hub authority endpoint before rolling out agents
that require it; older Hubs do not provide rental authorization fallback.

At boot, automatic discovery (the default) makes one attempt to select a
stable Runtime/TensorFS pair within their installed compatible lines (major.minor for 0.x). Selection
checks the real interpreter, mandatory dependencies, installed media/model
extras, and reverse dependencies on the pair; other dependencies are never
upgraded implicitly. Missing dependencies are resolved as an additive wheel-only
closure with all installed versions fixed, including reverse constraints from
unrelated frameworks. The pair and added wheels are checked against PyPI HTTPS
metadata and SHA-256, staged privately, and probed together. Existing-file
collisions and incompatible installed dependencies defer before mutation and
require a compatible base. Rollback removes only added distributions whose
version, private wheel URL/digest and installed RECORD hashes still prove
transaction ownership. Later operator changes or uncertain partial files remain
in place with an explicit repair-required result.
The installer-owned `etc/cozy/software-policy.json` accepts
`{"startup_update":"auto","agent":"bundled"}` or
`{"startup_update":"off","agent":"explicit"}` (both fields are required). Off preserves an
explicitly selected installation without disabling explicit maintenance. Missing
policy defaults to auto; unknown or unreadable policy defers discovery. Local,
development and prerelease software also skips automatic discovery.
Maintenance requests require `agent: "bundled" | "explicit"` and a boolean `pin`: true for explicit wheels or versions,
false for default-latest selection. The server never infers this intent from the
resolved version fields. The committed software-policy file is authoritative. A manual transaction journals
its policy and entry changes with the wheel installation; rollback restores only
its own changes and preserves later operator edits. Explicit agent selection
retains the exact running application binary; bundled selection uses the candidate
Runtime wheel. The client never rewrites this policy or entry after server commit.
Automatic rental updates stop after any accepted work, including after collection,
so ordinary restarts and retries retain software. Explicit owner maintenance may
update a used rental once Runtime-owned quiescence is proven.

The existing machine and Runtime-root locks protect admission. The Runtime-owned
software-update view applies retention rules across all owners, child attempts,
native calls, holds, weight intents and input intakes. Missing or unreadable
update state defers changes. The boot ID, certificate and execution journal are
preserved. An unavailable index or incompatible candidate leaves the verified
installed pair in service. The existing guardian is the stable parent, with private ABI
`machine-bootstrap/1`. It owns the Runtime and a replaceable unprivileged public
agent. A candidate replaces the public agent before admitting work; candidate
agent exit or measured stall before commit restores the previous application
and wheel pair. Public-agent disconnect or crash leaves an accepted Runtime
running, with the same control signer, incarnation, workspace, TLS and boot ID.
The parent is base infrastructure, updated on image/service bootstrap, and is
not advertised as newly replaced application code. Authenticated runtime status
reports the actual public agent version, SHA-256 and selection separately from
the bootstrap ABI/version. The immutable identity receipt remains historical;
rental readiness and authenticated runtime status prove the selected application.

The launcher retains hashed rollback wheels and a fsynced transaction under
`var/lib/cozy/startup-update`, outside the unprivileged agent's writable state.
Startup and explicit maintenance use this same transaction and readiness gate.
Interrupted installation is repaired offline before launching, even when updates
are now disabled; a later explicit pair supersedes a stale recovery journal. Success requires the actual new Runtime's authenticated fresh
readiness, workspace and unused-submission closure before admitting any work.
A candidate that fails that boot is rolled back and the prior Runtime relaunched
in the same boot. Failed recovery blocks execution while status and repair routes stay available.
Candidate preparation measures CPU/output and download-byte progress; pending
readiness does not renew rental billing indefinitely. Previously failed pairs
are skipped during later automatic discovery. `GET /v1/machine/runtime` includes the secrets-free
`startup_update` report alongside explicit maintenance status.
