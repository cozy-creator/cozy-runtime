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

import os
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
    """The device-child environment with the ERASE pass ALREADY APPLIED (cr-007's seal).

    The erase happens here because this is the one module permitted to read the process
    environment: `internal/spawn.py` imposes the reviewed allowlist on top of this frozen
    base and hands the result to `posix_spawn`, so the child inherits nothing implicitly and
    no other module ever needs to look at `os.environ` to build it.
    """


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


def forensics_dir(environ: Mapping[str, str] | None = None) -> str:
    """Where the fill plane writes its memory captures, or "" for nowhere.

    A PATH, deliberately, and read HERE because this module is the environment authority.
    It is configuration and not a logic gate: the plane branches on having somewhere to
    write, and a run with no directory configured executes the identical code path.
    """
    env = os.environ if environ is None else environ
    return env.get(child_env.FILL_FORENSICS_DIR, "").strip()


_read: bool = False


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
    )
