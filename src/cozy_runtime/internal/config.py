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

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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

    This is the SELF-SPAWNED launch mode: the launcher minted the credential, so it knows
    it. A RENTED pod is the other mode — see `access_token_hashes`."""
    access_token_hashes: str = ""
    """The RENTED-POD launch mode (#495e/#560g): the sha256 hash SET a Claim's `proof` is
    verified against, or the path of the file that holds it, in the format cozy-media
    already reads (`sha256:<64 hex>`, one per line or comma-separated).

    NOT a secret and that is the whole point: whoever launched this worker holds only
    hashes, so it can provision a pod it can never authenticate to. The renter minted the
    token and nobody else was ever told it. A path is re-read per claim, so a rotation of
    the set converges on the control plane the way it already converges on the byte plane.

    Mutually exclusive with `bootstrap_credential`: two credential authorities over one
    door is an ambiguity, not a fallback, and it refuses here."""
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


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """The fixed base-worker-image handoff to ``cozy-materialize-launch``.

    THE POD BOOTS ON ENVIRONMENT ALONE (cr-048). The retired
    ``tensorhub.rental_provision_spec/1`` + ``_bundle/1`` pair was a second, boot-only
    delivery lane for the closure that rev-2's ``DesiredPlacementSet`` + ``ArtifactGrantUpdate``
    already carry after the RecordOwner connects.

    Of the three identity facts that document smuggled, ONE is left. The ordinal and the
    rental id existed only to be echoed in the readiness receipt for tensorhub's binder,
    and that echo proved nothing the receipt's per-attempt HMAC key had not already proven,
    so both are gone from the handoff and from the payload. ``acquisition_attempt_id``
    stays because it is not an echo: it is the worker's ``--worker-id``, the identity a
    RecordOwner's ``Claim`` is addressed to and every ``Fault`` is attributed to.
    """

    acquisition_attempt_id: str
    pod_boot_id_path: Path
    tls_cert_path: Path
    tls_key_path: Path
    receipt_payload_path: Path
    worker_internal_port: int
    media_internal_port: int
    renter_token_sha256: tuple[str, ...]
    readiness_deadline_unix: int
    child_base_env: tuple[tuple[str, str], ...]


# The CLOSED adapter handoff. It is an ALLOWLIST, not a passthrough: `read_adapter_config`
# refuses any other `COZY_ADAPTER_*` name by name, so every addition here is deliberate.
# cr-048 removed four (the two provision-document path/digest pairs) and added three
# identity facts; two of those three are gone again with the receipt echo that was their
# only reader. What remains of that trio is the attempt id, which the worker wears as its
# own name.
ADAPTER_ENV = {
    "COZY_ADAPTER_ACQUISITION_ATTEMPT_ID",
    "COZY_ADAPTER_POD_BOOT_ID_PATH",
    "COZY_ADAPTER_TLS_CERT_PATH",
    "COZY_ADAPTER_TLS_KEY_PATH",
    "COZY_ADAPTER_RECEIPT_PAYLOAD_PATH",
    "COZY_ADAPTER_WORKER_INTERNAL_PORT",
    "COZY_ADAPTER_MEDIA_INTERNAL_PORT",
    "COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON",
    "COZY_ADAPTER_READINESS_DEADLINE_UNIX",
}

# The adapter starts one current control worker. Provider/image inheritance is not a config
# channel: only these non-secret platform values survive, then the adapter imposes the two
# worker-owned COZY paths explicitly. In particular RUNPOD/AWS/HF variables never cross.
# Class-D base inherit list — tracker-v2/spawn-allowlists.md (#616.d); keep equal to that
# row. PATH is deliberately absent: every pod exec is by absolute path.
ADAPTER_CHILD_PLATFORM_ENV = {
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
_adapter_read: bool = False

_TOKEN_HASH = re.compile(r"^[0-9a-f]{64}$")


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
    hashes = env.get("COZY_ACCESS_TOKEN_SHA256", "").strip()
    if bootstrap and hashes:
        # TWO AUTHORITIES OVER ONE DOOR (#560g). These are the two launch modes, not a
        # value and its fallback: a self-spawned worker's launcher MINTED the credential
        # and a rented pod's launcher only ever held hashes of one it will never see.
        # A worker handed both cannot say which authority its owner is, and a worker that
        # cannot say that must not open the door — so this refuses before anything binds.
        raise ConfigError(
            "config",
            "COZY_BOOTSTRAP_CREDENTIAL and COZY_ACCESS_TOKEN_SHA256 are both set",
            "set exactly one: COZY_BOOTSTRAP_CREDENTIAL for a worker whose launcher minted "
            "its credential, or COZY_ACCESS_TOKEN_SHA256 for a rented pod whose renter did "
            "(#495e — the two are launch modes, not a value and a fallback)",
        )
    _read = True
    return RuntimeConfig(
        cozy_home=home,
        credentials=creds,
        bootstrap_credential=bootstrap,
        access_token_hashes=hashes,
        child_base_env=tuple(
            sorted(
                (k, v)
                for k, v in env.items()
                if not child_env.erased(k) and k != "COZY_BOOTSTRAP_CREDENTIAL"
            )
        ),
        fill_forensics_dir=env.get(child_env.FILL_FORENSICS_DIR, "").strip(),
    )


def read_adapter_config(environ: Mapping[str, str] | None = None) -> AdapterConfig:
    """Read the base-worker-image adapter seam once.

    No loose ``COZY_ADAPTER_*`` extension is admitted.
    """

    global _adapter_read
    if _adapter_read:
        raise ConfigError(
            "config_authority",
            "read_adapter_config() called twice — a process has exactly one adapter handoff",
            "read the adapter handoff once at its fixed executable entrypoint",
            Exit.internal,
        )
    env = os.environ if environ is None else environ
    if any(env.get(name) for name in ("PYTHONHOME", "PYTHONPATH")):
        raise ConfigError(
            "adapter_config",
            "PYTHONHOME/PYTHONPATH source injection is forbidden for the fixed adapter",
            "install and execute the control Runtime wheel owned by the base worker image "
            "without source paths",
        )
    if unknown := sorted(
        name for name in env if name.startswith("COZY_ADAPTER_") and name not in ADAPTER_ENV
    ):
        raise ConfigError(
            "adapter_config",
            f"unknown adapter environment variable {unknown[0]}",
            "use the closed base-worker-image adapter handoff",
        )

    def required(name: str) -> str:
        value = env.get(name, "")
        if not value:
            raise ConfigError(
                "adapter_config",
                f"required adapter environment variable {name} is absent",
                "launch only through the generic base-worker-image bootstrap",
            )
        return value

    def path(name: str) -> Path:
        raw = required(name)
        value = Path(raw)
        if not value.is_absolute() or str(value) != raw:
            raise ConfigError(
                "adapter_config", f"{name} is not one absolute normalized path", "fix bootstrap"
            )
        return value

    def bounded(name: str) -> str:
        value = required(name)
        if value.strip() != value or len(value) > 256:
            raise ConfigError(
                "adapter_config", f"{name} is not one bounded untrimmed identifier", "fix bootstrap"
            )
        return value

    def decimal(name: str, maximum: int) -> int:
        raw = required(name)
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(
                "adapter_config", f"{name} is not canonical decimal", "fix bootstrap"
            ) from exc
        if value <= 0 or value > maximum or str(value) != raw:
            raise ConfigError(
                "adapter_config", f"{name} is not canonical decimal in range", "fix bootstrap"
            )
        return value

    try:
        token_hashes = json.loads(required("COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "adapter_config", "renter token hashes are not JSON", "fix bootstrap"
        ) from exc
    if (
        not isinstance(token_hashes, list)
        or not 1 <= len(token_hashes) <= 16
        or not all(isinstance(item, str) and _TOKEN_HASH.fullmatch(item) for item in token_hashes)
        or token_hashes != sorted(set(token_hashes))
    ):
        raise ConfigError(
            "adapter_config",
            "renter token hashes are not a non-empty bounded sorted unique set",
            "fix bootstrap",
        )
    worker_port = decimal("COZY_ADAPTER_WORKER_INTERNAL_PORT", 65535)
    media_port = decimal("COZY_ADAPTER_MEDIA_INTERNAL_PORT", 65535)
    if worker_port == media_port:
        raise ConfigError("adapter_config", "worker and media ports are equal", "fix bootstrap")
    parsed = AdapterConfig(
        acquisition_attempt_id=bounded("COZY_ADAPTER_ACQUISITION_ATTEMPT_ID"),
        pod_boot_id_path=path("COZY_ADAPTER_POD_BOOT_ID_PATH"),
        tls_cert_path=path("COZY_ADAPTER_TLS_CERT_PATH"),
        tls_key_path=path("COZY_ADAPTER_TLS_KEY_PATH"),
        receipt_payload_path=path("COZY_ADAPTER_RECEIPT_PAYLOAD_PATH"),
        worker_internal_port=worker_port,
        media_internal_port=media_port,
        renter_token_sha256=tuple(str(item) for item in token_hashes),
        readiness_deadline_unix=decimal("COZY_ADAPTER_READINESS_DEADLINE_UNIX", (1 << 53) - 1),
        child_base_env=tuple(
            sorted(
                (name, value) for name, value in env.items() if name in ADAPTER_CHILD_PLATFORM_ENV
            )
        ),
    )
    _adapter_read = True
    return parsed
