"""Device-child environment projection: the reviewed allowlist, as DATA.

cozy-runtime.md §3.5 — the worker projects an allowlist to the device executor by
ERASE known prefixes -> IMPOSE declared values unconditionally -> SEAL, completed BEFORE
executor CUDA init. This module owns the two data tables and the allowlist gate only.
**The seal ordering and its execution belong to cr-007** — do not implement spawn here.

The gate is live, not reviewed: `project_child_env` refuses any name outside the table,
so an unallowlisted child env field fails when it runs, not when someone reads the diff.
Credential prefixes are erased and never imposed — tokens do not reach the executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cozy_runtime.internal.exits import Exit


class ProjectionError(Exception):
    """An unallowlisted child env field, or a name the erase pass would not clear."""

    def __init__(self, name: str, message: str, remedy: str, code: Exit = Exit.internal):
        super().__init__(message)
        self.name = name
        self.message = message
        self.remedy = remedy
        self.code = code


@dataclass(frozen=True, slots=True)
class ProjectedVar:
    """One reviewed row: the env name, who supplies the value, and why the child needs it."""

    name: str
    source: str
    reason: str


# ERASE pass: every child env name starting with one of these prefixes is removed before
# imposition. Credential prefixes are here on purpose — the executor gets no tokens.
# Class-B erase rule — tracker-v2/spawn-allowlists.md (#616.d) is the authority; keep
# equal to that row.
ERASED_PREFIXES: tuple[str, ...] = (
    "COZY_",
    "CUDA_",
    "PYTORCH_",
    "TORCH_",
    "NCCL_",
    "HF_",
    "HUGGINGFACE_",
    "TENSORHUB_",
    "CIVITAI_",
    "COMFY_",
    "OMP_",
    "MKL_",
)

#: The fill plane's capture directory, named here because the allowlist row and the config
#: authority's reader must spell it identically and this is the table both read.
FILL_FORENSICS_DIR = "COZY_FILL_FORENSICS_DIR"
EXECUTOR_SCOPE_ENV = "COZY_EXECUTOR_SCOPE"

# IMPOSE pass: the complete reviewed allowlist. Adding a row is a reviewed change; the
# runtime refuses to project anything not listed here. Class-B impose vocabulary
# (tracker-v2/spawn-allowlists.md #616.d; per-tool superset — COZY_BOOTSTRAP_CREDENTIAL
# is the creator->worker leg's imposition, never the executor's).
CHILD_ENV_ALLOWLIST: tuple[ProjectedVar, ...] = (
    ProjectedVar(
        "CUDA_VISIBLE_DEVICES",
        "plan.device_selection",
        "the executor sees exactly the devices the admitted plan grants",
    ),
    ProjectedVar(
        "PYTORCH_CUDA_ALLOC_CONF",
        "plan.allocator_policy",
        "the CUDA caching allocator reads this once at init; late imposition is a no-op",
    ),
    ProjectedVar(
        "OMP_NUM_THREADS",
        "plan.cpu_threads",
        "thread cap must be set before torch import to bound host CPU in the executor",
    ),
    ProjectedVar(
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "accel.MPS_CHILD_ENV_LAW",
        "FORCED to 0 on MPS executors (#450): the fallback silently runs unsupported "
        "operations on CPU, which violates the CPU-refusal law",
    ),
    ProjectedVar(
        "COZY_HOME",
        "config.cozy_home",
        "store root, so the executor resolves the same CAS as the worker",
    ),
    ProjectedVar(
        "COZY_EXECUTOR_SOCKET",
        "worker.executor_channel",
        "the worker/executor control channel path handed to the child at spawn",
    ),
    ProjectedVar(
        EXECUTOR_SCOPE_ENV,
        "worker.process_tree_scope",
        "non-secret exact process identity inherited by first-party descendants when a "
        "credential-unwritable cgroup-v1 container cannot supply a kernel subtree",
    ),
    ProjectedVar(
        FILL_FORENSICS_DIR,
        "config.fill_forensics_dir",
        "where the fill plane writes its per-boundary memory captures. The FILL runs in "
        "the executor, so a capture directory that stopped at the worker would instrument "
        "the one process that never allocates a destination. A path, not a switch: unset "
        "means nowhere to write, and the served code path is identical either way",
    ),
)

ALLOWLIST: Mapping[str, ProjectedVar] = {v.name: v for v in CHILD_ENV_ALLOWLIST}


def erased(name: str) -> bool:
    """True when the ERASE pass clears `name` from the inherited environment."""
    return name.startswith(ERASED_PREFIXES)


def project_child_env(values: Mapping[str, str]) -> dict[str, str]:
    """Validate an imposition set against the reviewed allowlist.

    Refuses an unallowlisted name (the live red arm), and refuses an allowlisted name the
    ERASE pass would not clear first — imposition on top of an uncleared name is exactly
    the silent-inheritance bug the seal exists to prevent.
    """
    projected: dict[str, str] = {}
    for name, value in values.items():
        row = ALLOWLIST.get(name)
        if row is None:
            raise ProjectionError(
                "child_env",
                f"{name!r} is not in the reviewed device-child env allowlist "
                f"({len(ALLOWLIST)} rows)",
                "add a reviewed ProjectedVar row to CHILD_ENV_ALLOWLIST, or pass the value "
                "through the typed plan instead of the child environment",
            )
        if not erased(name):
            raise ProjectionError(
                "child_env",
                f"{name!r} is allowlisted but no ERASED_PREFIXES entry clears it first",
                "extend ERASED_PREFIXES so the imposed name is erased before imposition",
            )
        projected[name] = value
    return projected
