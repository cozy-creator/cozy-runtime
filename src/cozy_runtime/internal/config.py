"""THE configuration authority (cozy-runtime.md §3.5, boundaries.md §7).

This module is the only place in the runtime permitted to read process environment. It is
called EXACTLY ONCE per process, at the executable entrypoint, and yields a frozen typed
value that is passed explicitly thereafter. Author/job code, protocol bindings, TensorFS
adapters and ordinary runtime modules receive the value as an argument or do without it.

Two properties are enforced live, not by review:
  * `env-authority` fence: no `os.environ` / `os.getenv` outside this module.
  * one-read: a second `read_config()` in the same process is a typed refusal, so a
    smuggled second config singleton fails the moment it runs.

Env carries VALUE, never decision (cozy-runtime-cli.md conventions): every field below is
a credential value or a path. Behaviour switches are flags, parsed by the CLI, never here.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cozy_runtime.internal import child_env
from cozy_runtime.internal.exits import Exit

COZY_HOME_DEFAULT = "~/.cozy"

# Credential env names, per source. Values only; presence never changes behaviour.
CREDENTIAL_ENV: tuple[tuple[str, str], ...] = (
    ("tensorhub", "TENSORHUB_TOKEN"),
    ("huggingface", "HF_TOKEN"),
    ("civitai", "CIVITAI_TOKEN"),
    ("comfy", "COMFY_TOKEN"),
)


class ConfigError(Exception):
    """A typed configuration refusal carrying its exit name and the remedy."""

    def __init__(self, name: str, message: str, remedy: str, code: Exit = Exit.validation):
        super().__init__(message)
        self.name = name
        self.message = message
        self.remedy = remedy
        self.code = code


@dataclass(frozen=True, slots=True)
class Credentials:
    """Per-source credential VALUES read once. `None` = absent, never "look it up later"."""

    tensorhub: str | None = None
    huggingface: str | None = None
    civitai: str | None = None
    comfy: str | None = None

    def present(self) -> tuple[str, ...]:
        return tuple(source for source, _ in CREDENTIAL_ENV if getattr(self, source) is not None)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """The frozen typed config every other module receives explicitly."""

    cozy_home: Path
    credentials: Credentials
    bootstrap_credential: str = ""
    """The launcher's per-spawn control credential (#449/#463): a Claim must present it as
    `proof`. STRUCTURALLY SECRET — never logged, never in a Report, and never inherited by
    the device child (it is stripped from `child_base_env` below: the executor has no
    business holding the control plane's credential).

    Fixed private-rental workers use `record_owner_public_key` instead."""
    child_base_env: tuple[tuple[str, str], ...] = ()
    """The device-child environment with the ERASE pass ALREADY APPLIED (cr-007's seal;
    class B of tracker-v2/spawn-allowlists.md #616.d).

    The erase happens here because this is the one module permitted to read the process
    environment: `internal/spawn.py` imposes the reviewed allowlist on top of this frozen
    base and hands the result to `posix_spawn`, so the child inherits nothing implicitly and
    no other module ever needs to look at `os.environ` to build it.
    """
    fill_forensics_dir: str = ""
    """Where the fill plane writes its memory captures, or "" for nowhere (cr-038).

    A PATH, not a logic gate: the plane branches on having somewhere to write, and a run
    with no directory configured executes the identical code path. Read ONCE here with the
    rest of the config and projected across the executor seal by the worker."""

    record_owner_public_key: bytes = b""
    """Raw Ed25519 key that verifies private-rental artifact delegations.

    Empty on local/public workers. It never enters an executor environment.
    """


@dataclass(frozen=True, slots=True)
class WorkerHostConfig:
    """The fixed product-host handoff to ``cozy-runtime-worker``.

    Paths are image-contract constants from ``runtime-worker-host-paths.json`` rather than
    environment configuration. The readiness HMAC remains supervisor-only. Desired placements,
    signed artifact intent, and attempts arrive after Claim over WorkerControl.
    """

    worker_id: str
    worker_internal_port: int
    media_internal_port: int
    control_public_key_ed25519: bytes
    control_public_key_ed25519_b64url: str
    media_token_sha256: tuple[str, ...]
    lease_expiry_unix: int
    artifact_origin: WorkerArtifactOrigin
    child_base_env: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class WorkerArtifactOrigin:
    """One operator-owned HTTPS authority; TLS always terminates in Tensorhub."""

    value: str
    host: str
    port: int


# The CLOSED non-HMAC child projection of PodEnvironmentContract/1. Unknown COZY_* names refuse;
# provider/image inheritance is not a configuration channel.
WORKER_HOST_ENV = {
    "COZY_WORKER_ID",
    "COZY_MEDIA_INTERNAL_PORT",
    "COZY_WORKER_LEASE_EXPIRY_UNIX",
    "COZY_RECORD_OWNER_AUTH_JSON",
    "COZY_WORKER_ARTIFACT_ORIGIN",
    "COZY_WORKER_INTERNAL_PORT",
}

# The host starts one current control worker. Provider/image inheritance is not a config
# channel: only these non-secret platform values survive. In particular RUNPOD/AWS/HF
# variables never cross.
# Class-D base inherit list — tracker-v2/spawn-allowlists.md (#616.d); keep equal to that
# row. PATH is deliberately absent: every pod exec is by absolute path.
WORKER_CHILD_PLATFORM_ENV = {
    "LANG",
    "LC_ALL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TZ",
}


def seal_snapshot(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The device-child SEAL names as this process actually received them (cr-007).

    A second, deliberately repeatable environment read, and it lives here for the same
    reason the first one does: this module is the only environment authority. The executor
    snapshots the seal before CUDA init and compares it after, which is how a late
    imposition is CAUGHT rather than silently ignored (the allocator reads its environment
    once). It exposes exactly the reviewed allowlist names — credentials are not among
    them, because the ERASE pass removed them from any child environment.
    """
    env = os.environ if environ is None else environ
    return {name: env.get(name, "") for name in sorted(child_env.ALLOWLIST)}


_read: bool = False
_worker_host_read: bool = False

_TOKEN_HASH = re.compile(r"^[0-9a-f]{64}$")
_ED25519_KEY = re.compile(r"^[A-Za-z0-9_-]{43}$")


def parse_worker_artifact_origin(value: str) -> WorkerArtifactOrigin:
    """Refuse anything except canonical ``https://host[:port]``."""

    if (
        value.strip() != value
        or len(value.encode()) > 512
        or any(ord(character) <= 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigError(
            "worker_host_config",
            "COZY_WORKER_ARTIFACT_ORIGIN is not one bounded HTTPS origin",
            "set the operator-owned Tensorhub worker API origin",
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port or 443
    except ValueError as exc:
        raise ConfigError(
            "worker_host_config",
            "COZY_WORKER_ARTIFACT_ORIGIN has an invalid port",
            "set the operator-owned Tensorhub worker API origin",
        ) from exc
    explicit_port = ""
    has_explicit_port = False
    if parsed.netloc.startswith("["):
        closing = parsed.netloc.find("]")
        if closing >= 0 and parsed.netloc[closing + 1 :].startswith(":"):
            explicit_port = parsed.netloc[closing + 2 :]
            has_explicit_port = True
    elif ":" in parsed.netloc:
        explicit_port = parsed.netloc.rsplit(":", 1)[1]
        has_explicit_port = True
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.netloc != parsed.netloc.lower()
        or value != f"https://{parsed.netloc}"
        or (has_explicit_port and explicit_port != str(port))
        or not 1 <= port <= 65535
    ):
        raise ConfigError(
            "worker_host_config",
            "COZY_WORKER_ARTIFACT_ORIGIN is not canonical https://host[:port]",
            "set the operator-owned Tensorhub worker API origin",
        )
    return WorkerArtifactOrigin(value=value, host=parsed.hostname, port=port)


def read_config(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Read + validate the process environment ONCE at the entrypoint.

    Raises ConfigError on a second call: exactly one config authority per process.
    """
    global _read
    if _read:
        raise ConfigError(
            "config_authority",
            "read_config() called twice — a process has exactly one config authority",
            "read the config once at the entrypoint and pass the frozen value explicitly",
            Exit.internal,
        )
    env = os.environ if environ is None else environ
    home_raw = env.get("COZY_HOME", COZY_HOME_DEFAULT).strip()
    if not home_raw:
        raise ConfigError(
            "config",
            "COZY_HOME is set but empty",
            f"unset COZY_HOME or point it at a directory (default {COZY_HOME_DEFAULT})",
        )
    home = Path(home_raw).expanduser()
    if not home.is_absolute():
        raise ConfigError(
            "config",
            f"COZY_HOME must be an absolute path, got {home_raw!r}",
            f"set COZY_HOME to an absolute directory (default {COZY_HOME_DEFAULT})",
        )
    creds = Credentials(**{source: (env.get(var) or None) for source, var in CREDENTIAL_ENV})
    bootstrap = env.get("COZY_BOOTSTRAP_CREDENTIAL", "").strip()
    _read = True
    return RuntimeConfig(
        cozy_home=home,
        credentials=creds,
        bootstrap_credential=bootstrap,
        child_base_env=tuple(
            sorted(
                (k, v)
                for k, v in env.items()
                if not child_env.erased(k) and k != "COZY_BOOTSTRAP_CREDENTIAL"
            )
        ),
        fill_forensics_dir=env.get(child_env.FILL_FORENSICS_DIR, "").strip(),
    )


def read_worker_host_config(environ: Mapping[str, str] | None = None) -> WorkerHostConfig:
    """Read the fixed product-host handoff once; admit no extra ``COZY_*`` name."""

    global _worker_host_read
    if _worker_host_read:
        raise ConfigError(
            "config_authority",
            "read_worker_host_config() called twice — a process has one host handoff",
            "read the host handoff once at cozy-runtime-worker",
            Exit.internal,
        )
    env = os.environ if environ is None else environ
    if any(env.get(name) for name in ("PYTHONHOME", "PYTHONPATH")):
        raise ConfigError(
            "worker_host_config",
            "PYTHONHOME/PYTHONPATH source injection is forbidden for the fixed worker",
            "execute the image-owned cozy-runtime-worker without source paths",
        )
    if unknown := sorted(
        name for name in env if name.startswith("COZY_") and name not in WORKER_HOST_ENV
    ):
        raise ConfigError(
            "worker_host_config",
            f"unknown worker environment variable {unknown[0]}",
            "use the closed six-value worker handoff",
        )

    def required(name: str) -> str:
        value = env.get(name, "")
        if not value:
            raise ConfigError(
                "worker_host_config",
                f"required worker environment variable {name} is absent",
                "launch through cozy-daemon or pod-supervisor",
            )
        return value

    def bounded(name: str) -> str:
        value = required(name)
        if value.strip() != value or len(value.encode()) > 256:
            raise ConfigError(
                "worker_host_config", f"{name} is not one bounded identifier", "fix host launch"
            )
        return value

    def decimal(name: str, maximum: int) -> int:
        raw = required(name)
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(
                "worker_host_config", f"{name} is not canonical decimal", "fix host launch"
            ) from exc
        if value <= 0 or value > maximum or str(value) != raw:
            raise ConfigError(
                "worker_host_config", f"{name} is not canonical decimal in range", "fix host launch"
            )
        return value

    try:
        record_owner_auth = json.loads(required("COZY_RECORD_OWNER_AUTH_JSON"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "worker_host_config", "RecordOwner auth is not JSON", "fix host launch"
        ) from exc
    if not isinstance(record_owner_auth, dict) or set(record_owner_auth) != {
        "control_public_key_ed25519_b64url",
        "media_token_sha256",
    }:
        raise ConfigError(
            "worker_host_config",
            "RecordOwner auth is not the closed control-key/media-hash record",
            "fix host launch",
        )
    public_key_text = record_owner_auth.get("control_public_key_ed25519_b64url")
    token_hashes = record_owner_auth.get("media_token_sha256")
    try:
        public_key = base64.b64decode(
            str(public_key_text) + "=" * ((-len(str(public_key_text))) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise ConfigError(
            "worker_host_config",
            "RecordOwner control public key is not unpadded base64url",
            "fix host launch",
        ) from exc
    if (
        not isinstance(public_key_text, str)
        or _ED25519_KEY.fullmatch(public_key_text) is None
        or len(public_key) != 32
        or not isinstance(token_hashes, list)
        or not 1 <= len(token_hashes) <= 16
        or not all(isinstance(item, str) and _TOKEN_HASH.fullmatch(item) for item in token_hashes)
        or token_hashes != sorted(set(token_hashes))
    ):
        raise ConfigError(
            "worker_host_config",
            "RecordOwner auth key or media hashes are invalid",
            "fix host launch",
        )
    worker_port = decimal("COZY_WORKER_INTERNAL_PORT", 65535)
    media_port = decimal("COZY_MEDIA_INTERNAL_PORT", 65535)
    if worker_port == media_port:
        raise ConfigError(
            "worker_host_config", "worker and media ports are equal", "fix host launch"
        )
    parsed = WorkerHostConfig(
        worker_id=bounded("COZY_WORKER_ID"),
        worker_internal_port=worker_port,
        media_internal_port=media_port,
        control_public_key_ed25519=public_key,
        control_public_key_ed25519_b64url=public_key_text,
        media_token_sha256=tuple(str(item) for item in token_hashes),
        lease_expiry_unix=decimal("COZY_WORKER_LEASE_EXPIRY_UNIX", (1 << 53) - 1),
        artifact_origin=parse_worker_artifact_origin(required("COZY_WORKER_ARTIFACT_ORIGIN")),
        child_base_env=tuple(
            sorted(
                (name, value) for name, value in env.items() if name in WORKER_CHILD_PLATFORM_ENV
            )
        ),
    )
    _worker_host_read = True
    return parsed


def runtime_config_from_host(host: WorkerHostConfig) -> RuntimeConfig:
    """Project one parsed host handoff into the existing worker core config."""

    return RuntimeConfig(
        cozy_home=Path("/run/cozy"),
        credentials=Credentials(),
        record_owner_public_key=host.control_public_key_ed25519,
        child_base_env=host.child_base_env,
    )
