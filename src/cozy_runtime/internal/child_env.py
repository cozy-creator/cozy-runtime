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
    "PYTHON",
    "TORCH_",
    "TORCHINDUCTOR_",
    "TRITON_",
    "NCCL_",
    "HF_",
    "HUGGINGFACE_",
    "TENSORHUB_",
    "CIVITAI_",
    "COMFY_",
    "OMP_",
    "MKL_",
    # h3a-024: FlashAttention-4 compiles on the pod, and both halves of where it keeps the
    # objects are erased then imposed, so no image or operator export can redirect a
    # compiler cache out of the worker-boot scope that owns and deletes it.
    "FLASH_ATTENTION_",
    "TMPDIR",
)

#: The fill plane's capture directory, named here because the allowlist row and the config
#: authority's reader must spell it identically and this is the table both read.
FILL_FORENSICS_DIR = "COZY_FILL_FORENSICS_DIR"
EXECUTOR_SCOPE_ENV = "COZY_EXECUTOR_SCOPE"
JIT_CACHE_ENV: tuple[tuple[str, str], ...] = (
    ("CUDA_CACHE_PATH", "cuda"),
    ("PYTHONPYCACHEPREFIX", "bytecode"),
    ("TORCH_EXTENSIONS_DIR", "extensions"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ("TRITON_CACHE_DIR", "triton"),
    # PyTorch's NVRTC kernels (jiterator ops, Qwen3-VL's `prod` in H3's conditioner). Unset,
    # an executor with no HOME compiles them again on every start and warns it cannot cache.
    ("PYTORCH_KERNEL_CACHE_PATH", "torch-kernels"),
    # FlashAttention-4 is CuTe-DSL source and compiles for the card it finds (h3a-024). Its
    # OWN cache, deliberately, and not CUTLASS's `CUTE_DSL_CACHE_DIR`: FA4 keys on
    # `sha256(pickle(compile_key))` under a fingerprint of its own sources and carries no
    # absolute path, while the CuTe-DSL file cache hashes every `envar` attribute including
    # `Path.cwd()` and the cache directory itself — so two processes that merely started in
    # different directories cannot share its entries.
    ("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", "flash-attn4"),
)

#: The other half of the row above. Upstream gates the persistent cache on a second variable
#: read at import, so the pair is ONE fact — where FA4 keeps compiled objects — in upstream's
#: spelling, not a behaviour flag of ours: nothing in this runtime reads it, and with it unset
#: the only difference is that a re-prepared model compiles again.
FLASH_ATTENTION_CACHE_ENABLED_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"

# IMPOSE pass: the complete reviewed allowlist. Adding a row is a reviewed change; the
# runtime refuses to project anything not listed here. Class-B impose vocabulary
# (tracker-v2/spawn-allowlists.md #616.d).
CHILD_ENV_ALLOWLIST: tuple[ProjectedVar, ...] = (
    ProjectedVar(
        "CUDA_VISIBLE_DEVICES",
        "worker.device_lane",
        "the executor sees exactly its LANE's devices (cr-066): the worker assigns a "
        "placement to one lane of its envelope and seals the executor to that lane's "
        "entries, so ordinal 0 inside the executor IS the selected device. A job's lane is "
        "the whole envelope. The row used to claim `plan.device_selection`; no plan ever "
        "selected a device, and every executor saw the whole envelope and used card 0",
    ),
    ProjectedVar(
        "NCCL_NVLS_ENABLE",
        "worker.device_lane",
        "FORCED to 0 in a GROUP lane's seal (cr-068, pgw#929): NCCL enables NVLink SHARP "
        "multicast by default on NVSwitch hosts, binding it needs a privilege our containers "
        "lack, and the first all-to-all then dies with CUDA error 401. Ulysses never uses "
        "it. Erased then imposed, so no image or operator export can turn it back on; the "
        "executor's process on every GPU refuses to form a communicator under any other value",
    ),
    ProjectedVar(
        "NCCL_P2P_LEVEL",
        "worker.device_lane",
        "FORCED to NVL in a GROUP lane's seal: NCCL's GPU-to-GPU peer memory is used only "
        "between GPUs joined by NVLink. Over PCI (two A40s, torch 2.14 / NCCL 2.30.7) the "
        "communicator hung at formation 4 times in 10, both GPUs spinning in the first "
        "barrier (runs 2852, 2909); with peer memory off those GPUs exchange through host "
        "shared memory (measured: all_reduce -10%, all_to_all -28% in raw transfer rate) and "
        "every formation completes. Erased then imposed, like the NVLS row",
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
    *(
        ProjectedVar(
            name,
            "worker.pod_jit_cache_scope",
            "upstream PyTorch/Triton/CUDA owns cache keys and validation; Cozy scopes mutable "
            "reuse by worker boot plus exact content and deletes it on worker shutdown. This is "
            "warm-cache ownership inside the pod's admitted trust domain, not same-UID isolation",
        )
        for name, _directory in JIT_CACHE_ENV
    ),
    ProjectedVar(
        FLASH_ATTENTION_CACHE_ENABLED_ENV,
        "worker.pod_jit_cache_scope",
        "turns on the persistent half of FlashAttention-4's compile cache, which upstream "
        "reads at import and gates on a variable separate from the directory. Imposed with "
        "the directory or not at all: alone it would name a cache with nowhere to live",
    ),
    ProjectedVar(
        "TMPDIR",
        "worker.pod_jit_cache_scope",
        "the executor's temporary files (quantization spills, compiler scratch) go to its "
        "Runtime-owned scope on the machine's disk, never an inherited TMPDIR that may be "
        "RAM-backed; deleted with the scope",
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
        "credential-unwritable container cgroup cannot supply a delegated child subtree",
    ),
    ProjectedVar(
        "COZY_KERNEL_CACHE",
        "config.kernel_cache",
        "this executor's uid namespace of the machine kernel store (h3a-087): compiled "
        "Runtime-served kernel objects keyed by every compile input, kept across worker "
        "boots, package versions and Runtime updates. A location, not a switch: unset, a "
        "kernel compiles in-process exactly as it would with an empty store",
    ),
    ProjectedVar(
        "COZY_KERNEL_CACHE_TRUSTED",
        "config.kernel_cache",
        "the worker's own namespace of the same store, read-only to an isolated executor, "
        "so objects the Runtime built are shared without any placement writing where "
        "another placement reads",
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
