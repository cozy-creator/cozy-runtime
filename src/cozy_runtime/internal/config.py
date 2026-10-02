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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from cozy_runtime.internal import child_env, host_paths
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
    """Raw Ed25519 key that verifies Claim proofs: required on any TCP listener. Empty only
    where the listener is a 0700 unix socket. It never enters an executor environment."""

    dependency_cache: Path | None = None
    """Creator-owned local cache of immutable dependency payloads; not executor authority."""

    object_storage_hosts: tuple[str, ...] = ()
    """Deployment-declared TensorFS egress hosts; never inferred from download URLs."""

    python_root: Path | None = None
    """Host-owned interpreter cache override; local roles share cozy_home/python."""

    kernel_cache: Path | None = None
    """The machine's persistent compiled-kernel store (`internal/kernel_cache.py`); none
    outside a rooted worker, where kernels compile in-process as before."""

    @property
    def managed_python_root(self) -> Path:
        return self.python_root or self.cozy_home / "python"


@dataclass(frozen=True, slots=True)
class MemoConfig:
    """`memo:` in the machine's runtime.yaml: memoized Model methods (tracker #298).

    On by default and self-managing; a zero size turns its tier off. There is no switch and
    no environment variable."""

    #: each executor's in-memory tier; -1 is min(1 GiB, host memory / 16)
    process_bytes: int = -1
    #: the machine tier on disk
    store_bytes: int = 2 << 30
    #: the largest result written to disk: worth storing is costly to produce and small
    entry_bytes: int = 512 << 10
    #: how long an entry lives on disk
    ttl_ms: int = 7 * 86_400_000


@dataclass(frozen=True, slots=True)
class WorkerHostConfig:
    """The fixed product-host handoff to ``cozy-runtime-worker``.

    The Host names one machine root; every Runtime path is the fixed layout under it
    (`host_paths.Layout`). The readiness HMAC remains Host-only. Desired placements, accepted
    placements and attempts arrive after Claim over WorkerControl.
    """

    worker_id: str
    worker_internal_port: int
    media_internal_port: int
    control_public_key_ed25519: bytes
    control_public_key_ed25519_b64url: str
    media_token_sha256: tuple[str, ...]
    child_base_env: tuple[tuple[str, str], ...]
    storage_admission_fd: int | None = None
    tensorhub_origin: str = ""
    tensorhub_worker_token: str = field(default="", repr=False)
    """Worker-only Hub capability; never inherited by a job executor."""
    object_storage_hosts: tuple[str, ...] = ()
    layout: host_paths.Layout = field(default_factory=host_paths.Layout)
    control_mode: str = "legacy"
    """The supervisor launch contract; absent keeps supported older Hosts working."""
    process_incarnation: str = ""
    """Fresh supervisor-minted child identity, never durable execution ownership."""
    visible_devices: tuple[str, ...] | None = None
    """`CUDA_VISIBLE_DEVICES` entries, by index or GPU UUID: where the machine's GPUs are.
    None (unset) is every card; () (empty) is none, served CPU only."""
    memo: MemoConfig = field(default_factory=MemoConfig)


@dataclass(frozen=True, slots=True)
class TensorhubOrigin:
    """One validated Hub authority; HTTP is limited to explicitly delegated loopback access."""

    value: str
    host: str
    port: int
    scheme: str = "https"


# The host starts one current control worker. Provider/image inheritance is not a config
# channel: only these non-secret platform values survive. In particular RUNPOD/AWS/HF
# variables never cross.
# Class-D base inherit list — tracker-v2/spawn-allowlists.md (#616.d); keep equal to that
# row. PATH is deliberately absent: every pod exec is by absolute path.
# The pod executor's base environment (spawn-allowlists.md class D -> class B): the
# supervisor-authored fixed PATH plus the platform names the supervisor let through.
# PATH is here because the executor compiles: Triton builds its launcher stub with the
# image's gcc, and gcc's collect2 finds `ld` through PATH alone — with none, every
# Triton kernel on a pod failed as `CalledProcessError: … -l:libcuda.so.1 … exit status 1`
# while the same seal on a local worker inherited a PATH and passed (cl-101). The value is
# never the provider's: pod-supervisor replaced it before this process started.
WORKER_CHILD_PLATFORM_ENV = {
    "LANG",
    "LC_ALL",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TZ",
}


def inherited_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """THIS process's whole environment, for a child that IS this process again (cr-068).

    A follower rank is spawned by rank 0 of a group lane's executor and must run under
    exactly rank 0's sealed environment - the worker sealed it once, and a second seal
    would be a second authority. Nothing is erased or imposed here: the seal already
    happened, and `seal_snapshot` is how the child proves it inherited the same one.
    """
    env = os.environ if environ is None else environ
    return dict(env)


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


def sees_no_gpu(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this process was told it has no GPU: `CUDA_VISIBLE_DEVICES` is set and empty,
    as an operator hides a host's GPUs and as the seal of a CPU executor says. Such a process
    asks no GPU driver (`accel`): each ask opens the NVIDIA device nodes."""
    env = os.environ if environ is None else environ
    return env.get("CUDA_VISIBLE_DEVICES") == ""


def restore_seal(sealed: Mapping[str, str], names: Iterable[str]) -> None:
    """Re-impose sealed values in this process; an empty sealed value means unset."""
    for name in names:
        if sealed.get(name, ""):
            os.environ[name] = sealed[name]
        else:
            os.environ.pop(name, None)


_read: bool = False
_worker_host_read: bool = False

_TOKEN_HASH = re.compile(r"^[0-9a-f]{64}$")
_ED25519_KEY = re.compile(r"^[A-Za-z0-9_-]{43}$")


def parse_tensorhub_origin(value: str, *, allow_loopback_http: bool = False) -> TensorhubOrigin:
    """Canonical HTTPS, or explicitly delegated access to a loopback HTTP development Hub."""

    if (
        value.strip() != value
        or len(value.encode()) > 512
        or any(ord(character) <= 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigError(
            "worker_host_config",
            "TENSORHUB_ORIGIN is not one bounded HTTPS origin",
            "set the Tensorhub API origin",
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port or (80 if parsed.scheme == "http" else 443)
    except ValueError as exc:
        raise ConfigError(
            "worker_host_config",
            "TENSORHUB_ORIGIN has an invalid port",
            "set the Tensorhub API origin",
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
        not (
            parsed.scheme == "https"
            or (
                allow_loopback_http
                and parsed.scheme == "http"
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            )
        )
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.netloc != parsed.netloc.lower()
        or value != f"{parsed.scheme}://{parsed.netloc}"
        or (has_explicit_port and explicit_port != str(port))
        or not 1 <= port <= 65535
    ):
        raise ConfigError(
            "worker_host_config",
            "TENSORHUB_ORIGIN is not canonical https://host[:port]",
            "set the Tensorhub API origin",
        )
    return TensorhubOrigin(value=value, host=parsed.hostname, port=port, scheme=parsed.scheme)


def parse_object_storage_hosts(value: str) -> tuple[str, ...]:
    """Consume the existing podenv bounded, sorted DNS host/suffix declaration."""
    hosts = tuple(value.split(",")) if value else ()
    if (
        len(hosts) > 8
        or hosts != tuple(sorted(set(hosts)))
        or any(
            len(host) > 253
            or re.fullmatch(
                r"\.?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*", host
            )
            is None
            for host in hosts
        )
    ):
        raise ConfigError(
            "config",
            "TENSORHUB_OBJECT_STORAGE_HOSTS is not a bounded sorted unique DNS host list",
            "use the deployment's explicit object storage hosts",
        )
    return hosts


def parse_visible_devices(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(entry.strip() for entry in value.split(",") if entry.strip())


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
    dependency_cache = env.get("COZY_DEPENDENCY_CACHE", "").strip()
    if dependency_cache and not Path(dependency_cache).is_absolute():
        raise ConfigError(
            "config",
            "COZY_DEPENDENCY_CACHE must be an absolute path",
            "provide the Creator-owned local dependency cache path",
        )
    python_root = env.get("COZY_PYTHON_ROOT", "").strip()
    if python_root and not Path(python_root).is_absolute():
        raise ConfigError(
            "config",
            "COZY_PYTHON_ROOT must be an absolute path",
            "pass the managed_root reported by python-interpreters",
        )
    _read = True
    return RuntimeConfig(
        cozy_home=home,
        credentials=creds,
        child_base_env=tuple(sorted((k, v) for k, v in env.items() if not child_env.erased(k))),
        fill_forensics_dir=env.get(child_env.FILL_FORENSICS_DIR, "").strip(),
        dependency_cache=Path(dependency_cache) if dependency_cache else None,
        python_root=Path(python_root) if python_root else None,
        object_storage_hosts=parse_object_storage_hosts(
            env.get("TENSORHUB_OBJECT_STORAGE_HOSTS", "")
        ),
    )


def read_worker_host_config(
    environ: Mapping[str, str] | None = None,
    *,
    publication_authority_path: Path | None = None,
) -> WorkerHostConfig:
    """Read the understood product-host handoff once; ignore unused additions."""

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
    # A newer host may supply settings this worker does not consume. They do
    # not become configuration or child authority merely by existing here;
    # WORKER_CHILD_PLATFORM_ENV remains the only inherited child projection.

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
    if not isinstance(record_owner_auth, dict) or not {
        "control_public_key_ed25519_b64url",
        "media_token_sha256",
    }.issubset(record_owner_auth):
        raise ConfigError(
            "worker_host_config",
            "RecordOwner auth lacks its required control-key/media-hash fields",
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
    try:
        layout = host_paths.layout(env.get("COZY_MACHINE_ROOT"), env.get("COZY_TENSORFS_ROOT"))
    except ValueError as exc:
        raise ConfigError(
            "worker_host_config", f"machine layout is invalid: {exc}", "fix host launch"
        ) from exc
    storage_fd = env.get("COZY_STORAGE_ADMISSION_FD")
    if storage_fd not in (None, "4"):
        raise ConfigError(
            "worker_host_config", "storage admission must use inherited fd 4", "fix host launch"
        )
    control_mode = env.get("COZY_RUNTIME_CONTROL_MODE", "legacy")
    if control_mode not in {"legacy", "supervisor"}:
        raise ConfigError(
            "worker_host_config", "unsupported Runtime control contract", "update the host/Runtime"
        )
    process_incarnation = (
        bounded("COZY_RUNTIME_PROCESS_INCARNATION") if control_mode == "supervisor" else ""
    )
    hub_origin, worker_token = _machine_publication_authority(
        publication_authority_path or layout.machine_publication_authority,
        bounded("COZY_WORKER_ID"),
    )
    parsed = WorkerHostConfig(
        memo=read_memo_config(layout.runtime_settings),
        object_storage_hosts=parse_object_storage_hosts(
            env.get("TENSORHUB_OBJECT_STORAGE_HOSTS", "")
        ),
        tensorhub_origin=hub_origin,
        tensorhub_worker_token=worker_token,
        storage_admission_fd=4 if storage_fd == "4" else None,
        worker_id=bounded("COZY_WORKER_ID"),
        worker_internal_port=worker_port,
        media_internal_port=media_port,
        control_public_key_ed25519=public_key,
        control_public_key_ed25519_b64url=public_key_text,
        media_token_sha256=tuple(str(item) for item in token_hashes),
        layout=layout,
        control_mode=control_mode,
        process_incarnation=process_incarnation,
        visible_devices=parse_visible_devices(env.get("CUDA_VISIBLE_DEVICES")),
        child_base_env=tuple(
            sorted(
                (name, value) for name, value in env.items() if name in WORKER_CHILD_PLATFORM_ENV
            )
        ),
    )
    if environ is None:
        os.environ.pop("COZY_STORAGE_ADMISSION_FD", None)
    _worker_host_read = True
    return parsed


_SIZE = re.compile(r"([0-9]+)\s*(|k|m|g|t)(i?)b?", re.IGNORECASE)
_DURATION = re.compile(r"([0-9]+)\s*(ms|s|m|h|d|)", re.IGNORECASE)


def read_memo_config(path: Path) -> MemoConfig:
    """`memo:` from runtime.yaml, read once at Worker start. An absent file, section or
    unreadable value keeps that default."""
    try:
        lines = path.read_text().splitlines()
    except (FileNotFoundError, NotADirectoryError):
        return MemoConfig()
    values: dict[str, str] = {}
    inside = False
    for line in lines:
        text = line.split("#", 1)[0].rstrip()
        if not text.strip():
            continue
        if not text[0].isspace():
            inside = text.strip() == "memo:"
        elif inside:
            key, _, value = text.strip().partition(":")
            values[key.strip()] = value.strip().strip("'\"")
    found: dict[str, int] = {}
    for name in ("process_bytes", "store_bytes", "entry_bytes"):
        if (match := _SIZE.fullmatch(values.get(name, ""))) is not None:
            base = 1024 if match[3] or not match[2] else 1000
            found[name] = int(match[1]) * base ** " kmgt".index(match[2].lower() or " ")
    if (match := _DURATION.fullmatch(values.get("ttl", ""))) is not None:
        unit = {"ms": 1, "s": 1000, "": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
        found["ttl_ms"] = int(match[1]) * unit[match[2].lower()]
    return MemoConfig(**found)


def _machine_publication_authority(path: Path, worker_id: str) -> tuple[str, str]:
    """Read the privileged boot document once, alongside other worker config."""
    try:
        with path.open("rb") as source:
            raw = source.read((32 << 10) + 1)
    except FileNotFoundError:
        return "", ""  # Hosts without a handoff confer no publication authority.
    try:
        if len(raw) > 32 << 10:
            raise ValueError("authority metadata exceeds its bound")
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {
            "version",
            "origin",
            "worker_id",
            "worker_token",
        }:
            raise ValueError("authority metadata is not closed")
        origin, token = value["origin"], value["worker_token"]
        if (
            type(value["version"]) is not int
            or value["version"] != 2
            or not isinstance(origin, str)
            or not isinstance(token, str)
            or value["worker_id"] != worker_id
        ):
            raise ValueError("invalid authority metadata")
        parse_tensorhub_origin(origin)
        decoded = base64.b64decode(token + "=" * (-len(token) % 4), altchars=b"-_", validate=True)
        if (
            len(decoded) != 32
            or base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != token
        ):
            raise ValueError("invalid worker capability")
        return origin, token
    except (ValueError, binascii.Error) as exc:
        raise ConfigError(
            "worker_host_config", "Hub authority handoff is invalid", "fix host launch"
        ) from exc


def runtime_config_from_host(host: WorkerHostConfig) -> RuntimeConfig:
    """Project one parsed host handoff into the existing worker core config."""

    return RuntimeConfig(
        cozy_home=host.layout.cozy_home,
        credentials=Credentials(),
        record_owner_public_key=host.control_public_key_ed25519,
        child_base_env=host.child_base_env,
        object_storage_hosts=host.object_storage_hosts,
        dependency_cache=host.layout.dependency_cache,
        python_root=host.layout.install_root / "python",
        kernel_cache=host.layout.kernel_cache,
    )


def package_install_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Explicit uv commands cannot inherit user constraints or Python import paths."""
    source = os.environ if environ is None else environ
    return {
        name: value
        for name, value in source.items()
        if not name.startswith(("UV_", "PIP_", "PYTHON"))
        and name not in {"VIRTUAL_ENV", "CONDA_PREFIX"}
    }


def cute_dsl_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """CuTe DSL's own `CUTE_DSL_*` variables. The DSL reads them at import, so each one shapes
    the objects it emits and belongs in their kernel keys (h3a-087)."""
    source = os.environ if environ is None else environ
    return {name: value for name, value in sorted(source.items()) if name.startswith("CUTE_DSL_")}


def package_python_paths() -> str | None:
    """Operator-configured finite interpreter set; absent means installed discovery."""
    return os.environ.get("COZY_PACKAGE_PYTHONS")
