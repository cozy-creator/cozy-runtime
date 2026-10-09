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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from cozy_runtime.internal import child_env
from cozy_runtime.internal.exits import Exit

COZY_HOME_DEFAULT = "~/.cozy"

# Credential env names, per source. Values only; presence never changes behaviour.
CREDENTIAL_ENV: tuple[tuple[str, str], ...] = (
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

    python_root: Path | None = None
    """Host-owned interpreter cache override; local roles share cozy_home/python."""

    kernel_cache: Path | None = None
    """The machine's persistent compiled-kernel store (`internal/kernel_cache.py`); none
    outside a rooted worker, where kernels compile in-process as before."""

    @property
    def managed_python_root(self) -> Path:
        return self.python_root or self.cozy_home / "python"


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


def replace_environment(environment: Mapping[str, str]) -> None:
    """This process's whole environment becomes `environment`: a forked executor's own seal."""
    os.environ.clear()
    os.environ.update(environment)


def hide_gpus() -> None:
    """From here on this process, and what it starts, is told it has no GPU (`sees_no_gpu`)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


_read: bool = False


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
