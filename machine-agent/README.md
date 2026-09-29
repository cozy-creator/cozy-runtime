# Cozy machine agent

`cozy-machine` is the per-machine API and Runtime supervisor. It runs independently
of the personal Cozy controller, using the same binary and machine API on laptops
and rented machines. Runtime remains the execution-journal writer and scheduler.

Build from this directory with `go build .`; run with `cozy-machine run` and the
machine grant in its environment. `cozy-machine version --json` reports identity.
No Creator checkout or client configuration is required.

The Linux x86-64 and ARM64 `cozy-runtime` PyPI wheels include the `cozy-machine`
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
The native-kernel x86-64 wheel also contains the agent. Additional Python-independent
Linux x86-64/ARM64 wheels carry the agent without CUDA kernels. macOS and Windows
continue to install the portable Python wheel.

Standalone artifacts remain available for existing image and client installers.
Build a standalone release from committed source with
`scripts/build-release.sh --version VERSION --output /absolute/output`.
The private repository's `machine-agent` workflow runs the tests and makes the
same artifacts on manual dispatch. It does not publish or need a public-repo
credential. The output contains static Linux amd64/arm64 executables and
`machine-agent.json` with each platform's immutable public URL and SHA-256.

Publish those exact artifacts to **cozy-creator/cozy**, tag **machine-vVERSION**,
with `gh release create --latest=false`. Refuse an existing tag; never overwrite
an artifact or change the CLI's latest release. Only binary artifacts and their
manifest are public; the build source remains in this private Runtime repository.
Verify downloaded manifest hashes and `cozy-machine version --json` before
promotion. Agent and Runtime versions evolve independently through advertised
capabilities and the supported worker-protocol range.
