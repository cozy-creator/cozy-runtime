"""Select and install attention backends for constructed diffusers modules.

Every kernel is a Runtime-local forward (`attention_upstream`, `attention_sol`) that
`attention_ulysses` registers with Diffusers, so one wrapper serves every degree. A site takes
the first kernel of its GPU family's cell (`DENSE`, `SPARSE`, by `arch` row) at its attention
level that serves it here: 16 for a request at 16 bits (`EXACT`), else the model's
`attention_bits` for the site (default 16). A kernel that is not part of torch is compiled on
this machine for this card (`kernel_sources`, `kernel_compile`) and serves once its artifact is
published, imports and launches, all of it off any request thread (`kernel_ready`). A run
without an override never waits for a compile or a self-check: a kernel still opening, absent
or failed steps to the next one, then up a level, and the selection records why
(`Applied.fallbacks`) and what it passed over (`Applied.waiting`), so the construction selects
again once it is ready (`compiled`): at the run's next step or stage boundary (a run may mix
kernels, owner ruling 2026-10-09). A request's pin
(`--attention-kernel`) is the opposite: it waits for its own kernel's build, holds it for every
step, and refuses an unsupported or failed kernel (owner ruling 2026-10-10).
"""

from __future__ import annotations

import math
import shutil
import site
import sys
from collections import deque
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import msgspec

from cozy_runtime.author._attention import _PRODUCERS, AttentionProducer
from cozy_runtime.author._attention_scope import _EXPECTED
from cozy_runtime.author._context import device as placed_device
from cozy_runtime.author.kernels import KernelsUnavailable, Launch, quantize_once
from cozy_runtime.internal import (
    arch,
    attention_kitchen_sol,
    attention_sage2_fixed,
    attention_sol,
    attention_ulysses,
    execution_evidence,
    kernel_cache,
    kernel_compile,
    kernel_ready,
    kernel_sources,
    prebuilt,
    producers,
    tools,
    verify,
)
from cozy_runtime.internal import attention_upstream as upstream
from cozy_runtime.internal.attention_ulysses import Local
from cozy_runtime.internal.encoding.formats import Tensor
from cozy_runtime.internal.encoding.selection import DeviceFacts
from cozy_runtime.internal.executor_commands import Choice

# Tensors, torch modules and Diffusers' registry, backend members and processors are `Any`:
# neither torch nor diffusers is installed in the check venv.


class AttentionRefusal(RuntimeError):
    """An unsupported explicit selection or unavailable attention implementation."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


class Lacking(AttentionRefusal):
    """Another GPU's choice names kernels this one cannot serve now: each one's refusal
    code (`attention_kernel_compiling`: not yet). Nothing changed here."""

    def __init__(self, kernels: dict[str, str]) -> None:
        super().__init__(
            "attention_kernel_lacking",
            "; ".join(f"{name}: {code}" for name, code in sorted(kernels.items())),
        )
        self.kernels = kernels


#: One self-check case: (head dimension, dtype, batch, query length, key length, causal).
type Case = tuple[int, str, int, int, int, bool]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One attention kernel the runtime knows how to reach: the cards it runs on, what it admits
    and how it is built. The row is the only statement of what a kernel admits: site selection
    (`admits`), its self-check (`cases`) and each of its calls (`refusal`) read it."""

    name: str
    backend: str
    local: Local
    #: the MMA families it runs on (`arch`); empty is every device, CPU included
    families: frozenset[str] = frozenset()
    #: the row's formats it also needs (fp8, fp4)
    formats: frozenset[str] = frozenset()
    dtypes: frozenset[str] = frozenset()  # Empty admits all dtypes.
    #: the head dimensions it serves; empty is every one up to `max_head_dim` (zero: unbounded)
    head_dims: frozenset[int] = frozenset()
    max_head_dim: int = 0
    #: the exact (head dimension, dtype) pairs its build serves; empty is every pair above
    shapes: frozenset[tuple[int, str]] = frozenset()
    #: what a call may carry besides one shape of Q/K/V: a mask, causal attention, fewer K/V
    #: heads than Q heads, a query length other than the keys', several batch rows, and a
    #: softmax scale other than D^-1/2
    masks: bool = False
    causal: bool = True
    gqa: bool = True
    cross: bool = True
    batches: bool = True
    scales: bool = True
    #: the `kernel_sources` recipe compiled for this card; empty is part of torch
    artifact: str = ""
    #: the module the kernel is imported from; empty is torch itself
    import_name: str = ""
    #: a package's producer, served through its base backend (`known`)
    producer: AttentionProducer | None = None
    #: its precision: 16 keeps the operands' own, 8 or 4 quantizes them (its coarsest operand);
    #: the one cell level it may be listed at (`DENSE`, `SPARSE`)
    level: int = 16
    #: a sparse kernel: the model's layout routes each call's rows (`attention_sol`)
    sparse: bool = False
    #: the dense kernels a sparse one computes rows with, opened beside it: its dense steps',
    #: then its protected prefix's; none: the one its site's walk binds (`_sol_dense`)
    dense: tuple[str, ...] = ()
    #: the Runtime module whose `KERNELS` (a `TritonKernels` set) it launches: it serves once
    #: they are built here (`triton_kernels`)
    kernels: str = ""

    @property
    def exact(self) -> bool:
        return self.level == 16

    def admits(self, dtype: str, head_dim: int) -> bool:
        """Whether a site of DTYPE activations at HEAD_DIM (0: unknown) is one it serves."""
        if self.shapes and (head_dim, dtype) not in self.shapes:
            return False
        if self.head_dims and head_dim not in self.head_dims:
            return False
        if self.dtypes and dtype not in self.dtypes:
            return False
        return not (self.max_head_dim and (head_dim == 0 or head_dim > self.max_head_dim))

    def cases(self) -> list[Case]:
        """Its self-check (`_launch`): each (head dimension, dtype) it admits of its own head
        dimensions, else 64 and 128, with a query longer than its keys and two batch rows where
        it serves them, and causal at equal lengths where it serves that."""
        dims = sorted(self.head_dims | {d for d, _ in self.shapes}) or (64, 128)
        names = sorted(self.dtypes) or ("bfloat16", "float16")
        batch, query = 2 if self.batches else 1, 300 if self.cross else 257
        return [
            case
            for d in dims
            for n in names
            if self.admits(n, d)
            for case in ((d, n, batch, query, 257, False), (d, n, batch, 257, 257, True))
            if self.causal or not case[5]
        ]

    def refusal(self, query: Any, key: Any, options: Mapping[str, Any]) -> str:
        """Why it does not serve this call, or empty: its row, asserted on every call."""
        dtype, dim = str(query.dtype).removeprefix("torch."), int(query.shape[-1])
        scale = options.get("scale")
        limits = (
            (not self.admits(dtype, dim), f"{dtype} activations at head dimension {dim}"),
            (options.get("attn_mask") is not None and not self.masks, "an attention mask"),
            (bool(options.get("is_causal")) and not self.causal, "causal attention"),
            (query.shape[2] != key.shape[2] and not self.gqa, "fewer K/V heads than Q heads"),
            (query.shape[1] != key.shape[1] and not self.cross, "a query length other than K's"),
            (query.shape[0] != 1 and not self.batches, "more than one batch row"),
            (
                scale is not None
                and not self.scales
                and not math.isclose(scale, dim**-0.5, rel_tol=1e-6),
                "a softmax scale other than D^-1/2",
            ),
        )
        return next((f"{self.name} does not serve {what}" for off, what in limits if off), "")


_GPU = frozenset({arch.MMA_SYNC, arch.WGMMA, arch.TCGEN05})
_MMA, _WGMMA, _TCGEN05 = (frozenset({f}) for f in (arch.MMA_SYNC, arch.WGMMA, arch.TCGEN05))
_FP8, _FP4 = frozenset({"fp8"}), frozenset({"fp4"})
_HALF, _BF16, _D128 = frozenset({"bfloat16", "float16"}), frozenset({"bfloat16"}), frozenset({128})
_D64_128 = frozenset({64, 128})
#: What the wheel's FA3 serves (`prebuilt.FlashAttention3`); `flash-attn3-full` serves more.
_FA3 = frozenset((d, dtype) for d in (64, 128) for dtype in ("bfloat16", "float16"))
_PY, _KITCHEN, _SAGE3 = kernel_sources.PYTHON, "comfy-kitchen", "sageattn3"
#: SageAttention3's API over the wheel's `_sage3` (its group mean is the per-block-mean
#: candidate's Triton set); the source build's is `sageattn3.api`.
_SAGE3_WHEEL = "cozy_runtime.internal.sage3"
_E4M3 = "cozy_runtime.internal.attention_fp8"
#: Every sparse kernel's admission: one BF16 document at head dimension 128, equal Q/K/V.
_SOL: dict[str, Any] = dict(
    sparse=True, dtypes=_BF16, max_head_dim=128, head_dims=_D128, causal=False, gqa=False,
    cross=False, batches=False,
)  # fmt: skip

KERNELS: tuple[Candidate, ...] = (
    Candidate("sdpa", "_cozy_sdpa", upstream.SDPA, masks=True),
    Candidate(
        "cudnn", "_cozy_cudnn", upstream.CUDNN, _GPU, dtypes=_HALF, max_head_dim=128, masks=True,
    ),
    Candidate(
        "flash-attn3", "_cozy_flash_attn3", upstream.FA3, _WGMMA, dtypes=_HALF, max_head_dim=256,
        shapes=_FA3, artifact="flash-attn3-full", import_name="flash_attn_interface",
    ),
    Candidate(
        "flash-attn3-fp8", "_cozy_flash_attn3_fp8", upstream.FA3_FP8, _WGMMA, level=8,
        dtypes=_HALF, max_head_dim=128, gqa=False, artifact="flash-attn3-full",
        import_name="flash_attn_interface", kernels=_E4M3,
    ),
    Candidate(
        "flash-attn4", "_cozy_flash_attn4", upstream.FA4, _WGMMA | _TCGEN05, dtypes=_HALF,
        max_head_dim=256, artifact=_PY, import_name="flash_attn.cute",
    ),
    Candidate(
        "flash-attn4-fp8", "_cozy_flash_attn4_fp8", upstream.FA4_FP8, _TCGEN05, level=8,
        dtypes=_BF16, head_dims=_D64_128, max_head_dim=128, gqa=False, artifact=_PY,
        import_name="flash_attn.cute", kernels=_E4M3,
    ),
    Candidate(
        "sageattention", "_cozy_sage_int8_fp8", upstream.SAGE, _MMA | _WGMMA, _FP8, level=8,
        dtypes=_HALF, max_head_dim=128, artifact="sageattention", import_name="sageattention",
    ),
    Candidate(
        "sageattention-fp16pv", "_cozy_sage_fp16pv", upstream.SAGE_FP16PV, _WGMMA, level=8,
        dtypes=_HALF, max_head_dim=128, artifact="sageattention-full",
        import_name="sageattention",
    ),
    Candidate(
        "kitchen-int8", "_cozy_kitchen_int8", upstream.KITCHEN, _GPU, level=8, dtypes=_HALF,
        max_head_dim=256, masks=True, causal=False, artifact=_KITCHEN,
        import_name="comfy_kitchen",
    ),
    Candidate(
        "flashinfer-bf16-fp8", "_cozy_flashinfer_bf16_fp8", upstream.FLASHINFER, _TCGEN05,
        level=8, dtypes=_BF16, max_head_dim=128, head_dims=_D128, causal=False,
        artifact=_PY, import_name="flashinfer", kernels=_E4M3,
    ),
    Candidate(
        "sageattention3-fp4", "_cozy_sage3_fp4", upstream.SAGE3, _MMA, _FP4, level=4,
        dtypes=_HALF, max_head_dim=128, head_dims=_D64_128, causal=False, gqa=False,
        scales=False, artifact=_SAGE3, import_name=_SAGE3, kernels=_SAGE3_WHEEL,
    ),
    Candidate(
        "sageattention3-fp4-global-lowmem", "_cozy_sage3_fp4_global_lowmem",
        upstream.SAGE3_GLOBAL_LOWMEM, _MMA, _FP4, level=4, dtypes=_HALF, max_head_dim=128,
        head_dims=_D64_128, causal=False, gqa=False, scales=False, artifact=_SAGE3,
        import_name=_SAGE3,
    ),
    Candidate(
        "sageattention3-fp4-fixed-means", "_cozy_sage3_fp4_fixed_means",
        upstream.SAGE3_FIXED_MEANS, _MMA, _FP4, level=4, dtypes=_BF16, max_head_dim=128,
        head_dims=_D128, causal=False, gqa=False, cross=False, batches=False, scales=False,
        artifact=_SAGE3, import_name=_SAGE3, kernels="cozy_runtime.internal.attention_sage3_fixed",
    ),
    Candidate(
        attention_sage2_fixed.DENSE_NAME, "_cozy_sage2_fixed_kmean", attention_sage2_fixed.DENSE,
        _WGMMA, level=8, dtypes=_BF16, max_head_dim=128, head_dims=_D128, causal=False,
        gqa=False, batches=False, artifact="sageattention", import_name="sageattention",
        kernels="cozy_runtime.internal.attention_sage2_fixed_mean",
    ),
    Candidate(
        "sol-attn", "_cozy_sol_bf16", attention_sol.SOL, _GPU, _FP8, artifact=_PY,
        import_name="sol_attn", **_SOL,
    ),
    Candidate(
        attention_sage2_fixed.SOL_NAME, "_cozy_sol_sage2_fixed_kmean", attention_sage2_fixed.SOL,
        _WGMMA, level=8, artifact=_PY, import_name="sol_attn",
        dense=(attention_sage2_fixed.DENSE_NAME,), **_SOL,
    ),
    Candidate(
        "kitchen-sol", "_cozy_kitchen_sol_int8", attention_kitchen_sol.KITCHEN_SOL, _GPU, level=8,
        artifact=_KITCHEN, import_name="comfy_kitchen", dense=("kitchen-int8",), **_SOL,
    ),
    Candidate(
        "kitchen-sol-fp4-lowmem", "_cozy_kitchen_sol_fp4_lowmem",
        attention_kitchen_sol.KITCHEN_SOL_FP4_LOWMEM, _MMA, _FP4, level=4, artifact=_KITCHEN,
        import_name="comfy_kitchen", dense=("sageattention3-fp4-global-lowmem", "kitchen-int8"),
        **_SOL,
    ),
    Candidate(
        "kitchen-sol-fp4-fixed-means", "_cozy_kitchen_sol_fp4_fixed_means",
        attention_kitchen_sol.KITCHEN_SOL_FP4_FIXED_MEANS, _MMA, _FP4, level=4, artifact=_KITCHEN,
        import_name="comfy_kitchen", dense=("sageattention3-fp4-fixed-means", "kitchen-int8"),
        **_SOL,
    ),
)  # fmt: skip
BY_NAME: dict[str, Candidate] = {c.name: c for c in KERNELS}

#: Each family's kernels per attention level, best first (precision-tiers §2.4; the order is
#: the release gate's measurements). A site walks its level's cell, each level above, then SDPA.
DENSE: dict[str, dict[int, tuple[str, ...]]] = {
    arch.MMA_SYNC: {
        16: ("sdpa",),
        8: ("kitchen-int8", "sageattention"),
        4: ("sageattention3-fp4",),
    },
    arch.WGMMA: {16: ("flash-attn3",), 8: ("sageattention",)},
    arch.TCGEN05: {16: ("cudnn",), 8: ("flash-attn4-fp8",)},
}
#: A sparse site's kernels (one a package producer serves: its model supplies an
#: `AttentionLayout`), tried at its level and each above before any DENSE kernel. Sol's dense rows
#: take DENSE at the site's level, its exact paths level 16 (`_sol_dense`); on tcgen05 its exact
#: tiles take its dense rows' precision (`attention_sol_fa4`), so its 8-bit cell is empty and a
#: site steps up to it. mma.sync's 8-bit cell is Kitchen's INT8 Sol: INT8 runs 2x BF16 there, 4x
#: on GeForce (half-rate FP32 accumulation), and BF16 Sol's exact blocks alone outlast Kitchen's
#: whole sparse step (H3 15 s, RTX 4090: 23 s at peak).
SPARSE: dict[str, dict[int, tuple[str, ...]]] = {
    arch.MMA_SYNC: {
        16: ("sol-attn",),
        8: ("kitchen-sol",),
        4: ("kitchen-sol-fp4-fixed-means", "kitchen-sol-fp4-lowmem"),
    },
    arch.WGMMA: {16: ("sol-attn",), 8: ("sol-attn-sage2-fixed-kmean",)},
    arch.TCGEN05: {16: ("sol-attn",)},
}
LEVELS = (4, 8, 16)
#: A cell lists only kernels of its own precision: a coarser or finer kernel there could not merge.
_MISFILED = [
    f"{family} {'sparse' if table is SPARSE else 'dense'} {level}: {name} is {BY_NAME[name].level}"
    for table in (DENSE, SPARSE)
    for family, cells in table.items()
    for level, names in cells.items()
    for name in names
    if BY_NAME[name].level != level or BY_NAME[name].sparse != (table is SPARSE)
]
if _MISFILED:
    raise ValueError(f"attention cells hold kernels of another precision: {_MISFILED}")
#: The pin of a request at 16 bits: every site in its scope at level 16.
EXACT = "exact"
#: Package producers' candidates, each derived once from its registration.
_PRODUCED: dict[str, tuple[AttentionProducer, Candidate]] = {}


def _produced(producer: AttentionProducer) -> Candidate:
    """A producer as a candidate: its base's admission, build and dense kernel, its own
    name and Diffusers member, Ulysses only when it says so."""
    held = _PRODUCED.get(producer.name)
    if held is not None and held[0] is producer:
        return held[1]
    base = BY_NAME.get(producer.base)
    if base is None or not base.sparse:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"{producer.name} is served through {producer.base!r}, which is not one of this "
            f"Runtime's sparse backends {sorted(n for n, c in BY_NAME.items() if c.sparse)}",
        )
    candidate = replace(
        base,
        name=producer.name,
        backend="_cozy_" + producer.name.replace("-", "_"),
        local=Local(base.local.forward, base.local.impl, producer.context_parallel),
        producer=producer,
    )
    _PRODUCED[producer.name] = (producer, candidate)
    return candidate


def known() -> dict[str, Candidate]:
    """Every attention name this process serves: the Runtime's, and its package's producers,
    which take precedence over a Runtime name they repeat."""
    return {**BY_NAME, **{name: _produced(p) for name, p in _PRODUCERS.items()}}


def sparse_names(names: Iterable[str]) -> list[str]:
    found = known()
    return sorted(name for name in set(names) if name in found and found[name].sparse)


def _walked(candidate: Candidate) -> bool:
    """A sparse kernel whose dense reference its site's walk binds (`_sol_dense`)."""
    return candidate.sparse and not candidate.dense


def _sol_objects(candidate: Candidate) -> bool:
    """A sparse kernel whose sparse steps run Sol's compiled CuTe objects (`attention_sol`)."""
    return candidate.sparse and candidate.import_name == "sol_attn"


#: The kernel a declaring sparse row's exact paths take (H3: the token refiner, block 1 and the
#: blocks whose key gain has outlier channels): Kitchen INT8 where the row declares it, whose
#: per-block scales carry those keys (H3 on an RTX 5090: 0.008-0.030 rel L2, FP4 0.10-0.28),
#: else BF16 SDPA.
EXACT_INT8, EXACT_DENSE = "kitchen-int8", "sdpa"


def _exact_name(candidate: Candidate) -> str:
    return EXACT_INT8 if EXACT_INT8 in candidate.dense else EXACT_DENSE


def dense_identity(name: str) -> str:
    """The dense reference a sparse kernel recomputes rows with: its row's declared kernels and
    the exact one beside them, else the one its site's walk bound."""
    dense = known()[name].dense
    if not dense:
        return attention_sol.dense_identity()
    prefix = f"; sparse-prefix={dense[-1]}" if dense[-1] != dense[0] else ""
    return f"{dense[0]}{prefix}{_EXACT_AT}{_exact_name(known()[name])}"


def device_row(device: DeviceFacts | None) -> arch.Row:
    return arch.row(device.sm) if device is not None and device.kind == "cuda" else arch.NONE


def _off_row(candidate: Candidate, row: arch.Row) -> str:
    """Empty when the row runs CANDIDATE, else why not."""
    if not candidate.families or (
        row.family in candidate.families and candidate.formats <= row.formats()
    ):
        return ""
    needs = " with " + "+".join(sorted(candidate.formats)) if candidate.formats else ""
    return (
        f"{candidate.name} runs on {'/'.join(sorted(candidate.families))}{needs}; this card is "
        f"{row.name} ({row.family or 'no tensor-core family'})"
    )


def level(bits: int | Mapping[str, int], component: str, path: str) -> int:
    """A site's attention level from the model's `attention_bits`: the integer, or the longest
    component-qualified prefix that names the site; unlisted sites 16."""
    if isinstance(bits, int):
        return bits
    site = f"{component.rsplit('/', 1)[-1]}.{path}"
    named = [p for p in bits if site == p or site.startswith(p if p.endswith(".") else p + ".")]
    return bits[max(named, key=len)] if named else 16


def _invalid(producer: AttentionProducer | None, module: object) -> str:
    """Why PRODUCER does not serve MODULE (its `validate`), or empty."""
    try:
        if producer is not None:
            producer.validate(module)
    except Exception as exc:
        return str(exc) or type(exc).__name__
    return ""


def sparse_site(module: object) -> bool:
    """A site a package producer serves: its model supplies an `AttentionLayout` there."""
    return any(not _invalid(producer, module) for producer in _PRODUCERS.values())


def walk(row: arch.Row, at: int, layout: bool = False) -> list[str]:
    """A site's kernels on ROW, best first: a LAYOUT site's sparse cells at level AT and each
    level above (its model asked for sparse attention), then the dense cells the same way, then
    SDPA, each producer standing in before its base; none the row cannot run."""
    found, names, levels = known(), list[str](), LEVELS[LEVELS.index(at) :]
    for table in (SPARSE, DENSE) if layout else (DENSE,):
        for step in levels:
            names += table.get(row.family, {}).get(step, ())
    stand_ins: dict[str, list[str]] = {}
    for name, producer in _PRODUCERS.items():
        stand_ins.setdefault(producer.base, []).append(name)
    names = [n for name in (*names, "sdpa") for n in (*stand_ins.get(name, ()), name)]
    return [n for n in dict.fromkeys(names) if n in found and not _off_row(found[n], row)]


#: An attention site: (component, module path, module, processor).
type _Record = tuple[str, str, Any, Any]


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate compiled, imported and launched on this device, with Diffusers' member."""

    candidate: Candidate
    member: Any
    #: The compiled artifact serving it: flat string fields, as an observation row carries them.
    artifact: dict[str, str] | None = None

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> Any:
    """Diffusers' attention backend registry, or a typed absence."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    except ImportError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"this environment's diffusers exposes no attention backend registry: {exc}",
        ) from exc
    return _AttentionBackendRegistry


def _member(candidate: Candidate) -> Any:
    """The Diffusers member serving this candidate, registered on first use."""
    return attention_ulysses.member(candidate.backend, _local(candidate))


#: backend -> the kernel its Diffusers member calls (`_local`)
_LOCALS: dict[str, Local] = {}


def _local(candidate: Candidate) -> Local:
    """CANDIDATE's forward as every call reaches it: its row asserted first (`refusal`), a
    sparse one's dense rows through its declared kernels' own."""
    found = _LOCALS.get(candidate.backend)
    if found is not None:
        return found
    forward = candidate.local.forward
    if candidate.dense:
        rows = [_dense_rows(BY_NAME[name]) for name in candidate.dense]
        exact = _dense_rows(BY_NAME[_exact_name(candidate)])
        forward = partial(forward, dense=rows[0], prefix_dense=rows[-1], exact=exact)

    def guarded(query: Tensor, key: Tensor, value: Tensor, **options: Any) -> Tensor:
        refused = candidate.refusal(query, key, options)
        if refused:
            raise ValueError(refused)
        out: Tensor = forward(query, key, value, **options)
        return out

    found = _LOCALS[candidate.backend] = replace(candidate.local, forward=guarded)
    return found


def _dense_rows(candidate: Candidate) -> Any:
    """A sparse kernel's dense rows on CANDIDATE: `(query=, key=, value=, scale=)`."""
    local = _local(candidate)

    def rows(*, query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
        options = attention_ulysses.Options(
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False
        )
        return attention_ulysses._call(candidate.backend, local, query, key, value, options)

    return rows


#: kernel -> its artifact's latest state on this rank: what a run's evidence reports.
_STATES: dict[str, kernel_compile.State] = {}
#: A site and the kernel it is given: (component, module path, module, processor, kernel).
type _Site = tuple[str, str, Any, Any, Ready]
#: module -> the build this process imports it from, fixed by the first import or by `reserve`:
#: a process holds one build of a module, so a kernel another build serves needs another one.
_BUILDS: dict[str, str] = {}


def _build(candidate: Candidate) -> str:
    """The build serving CANDIDATE here: its own, or the module's build when that one holds
    its kernels too; another build is a fresh executor's (`attention_kernel_needs_executor`)."""
    build = _BUILDS.get(candidate.import_name, candidate.artifact)
    if build == candidate.artifact or candidate.artifact in kernel_sources.RECIPES[build].serves:
        return build
    raise AttentionRefusal(
        "attention_kernel_needs_executor",
        f"{candidate.name} needs the {candidate.artifact} build of {candidate.import_name}; "
        f"this executor imports {build}",
    )


def reserve(pin: str, device: DeviceFacts) -> None:
    """A request pin known at load: its kernel's build becomes its module's for this process
    before any default selection imports another, and starts compiling. Never refuses: the
    request's own plan reports a kernel that cannot serve."""
    try:
        candidate = known().get(_pin_parts(pin)[1])
    except AttentionRefusal:
        return
    if candidate is None or not candidate.artifact or candidate.import_name in _BUILDS:
        return
    _BUILDS[candidate.import_name] = candidate.artifact
    store = kernel_cache.configured()
    if store is not None and not _off_row(candidate, device_row(device)):
        with suppress(AttentionRefusal):
            kernel_compile.submit(store, _job(candidate, device))


def _job(candidate: Candidate, device: DeviceFacts) -> kernel_compile.Job:
    """The candidate's build here; a build tool it lacks is the run's warning (`tools`)."""
    build = _build(candidate)
    failed = f"{candidate.name} prebuilt" in _STATES  # the wheel's kernel failed its probe
    code = "kernel_verify_failed" if failed else "kernel_prebuilt_missing"
    try:
        job = kernel_sources.job(build, device.sm)
    except kernel_sources.Absent as exc:
        for tool in exc.tools:
            tools.missing(tool, code)
        _STATES[candidate.name] = kernel_compile.State(build, "absent", detail=str(exc))
        raise AttentionRefusal("attention_kernel_absent", f"{candidate.name}: {exc}") from exc
    store = kernel_cache.configured()
    built = store is None or store.entry(job.key) is not None
    if job.kind == "setup" and not built and not shutil.which("ninja"):
        tools.missing("ninja", code)
    return job


def _compiled(candidate: Candidate, device: DeviceFacts, wait: bool) -> dict[str, str] | None:
    """Place the candidate's compiled artifact on the import path, after waiting for its build
    when `wait`, or refuse with its state: `failed` for this machine run, `absent`, or
    `compiling` for a caller that only asks. A package environment that carries the module
    itself serves it, as its pinned version always has."""
    wheel = _wheel(candidate, device)
    if wheel is not None:
        return wheel
    _build(candidate)  # another build imported here refuses before any store or compile
    store = kernel_cache.configured()
    found = util.find_spec(candidate.import_name.split(".")[0])
    origin = str(found.origin or "") if found is not None else ""
    built = store is not None and origin.startswith(
        tuple(str(n) for n in (store.own, store.trusted) if n is not None)
    )
    if found is not None and not built:
        return None
    if store is None:
        raise AttentionRefusal(
            "attention_kernel_absent", f"{candidate.name}: no machine kernel store to compile into"
        )
    job = _job(candidate, device)
    state = (
        kernel_compile.wait(store, job, candidate.name)
        if wait
        else kernel_compile.submit(store, job)
    )
    _STATES[candidate.name] = state
    if state.state != "ready":
        raise AttentionRefusal(
            f"attention_kernel_{state.state}", f"{candidate.name}: {state.line()}"
        )
    tree = Path(state.path) / kernel_cache.SITE
    if built and not origin.startswith(str(tree)):
        raise AttentionRefusal(
            "attention_kernel_needs_executor",
            f"{candidate.name}: this executor imports {candidate.import_name} from {origin}",
        )
    _BUILDS[candidate.import_name] = job.key.kernel
    site.addsitedir(str(tree))
    return {"artifact": job.key.kernel, "key": job.key.digest[:16], "compile_ms": str(state.ms)}


def _import(candidate: Candidate) -> ModuleType:
    try:
        return import_module(candidate.import_name)
    except Exception as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"{candidate.name}: {candidate.import_name} does not import: "
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _environment_key(candidate: Candidate) -> str:
    """A kernel the package environment carries, as its kept self-check knows it: the imported
    module's file, size and mtime, so an upgraded install is checked again."""
    found = Path(getattr(_import(candidate), "__file__", "") or "")
    stat = found.stat() if found.is_file() else None
    return f"environment:{found}:{stat.st_size if stat else 0}:{stat.st_mtime_ns if stat else 0}"


def _launch(ready: Ready, device: DeviceFacts) -> None:
    """`verify_op` for a compiled kernel: each (head dim, dtype) it serves against attention in
    float64 on this card, kept per machine by its build's key (`verify`). One that fails, or
    whose probe raises, is not served. Torch's own backends need none; Sol runs only inside its
    model's layout."""
    candidate = ready.candidate
    cases = candidate.cases()
    if not candidate.artifact or candidate.sparse or device.kind != "cuda" or not cases:
        return
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    def call(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        out: object = dispatch_attention_fn(q, k, v, backend=ready.member, is_causal=causal)
        return out[0] if isinstance(out, tuple) else out

    probe = partial(prebuilt.attention_probe, torch, device.index, call, candidate.exact, cases)
    key = (ready.artifact or {}).get("key", "") or _environment_key(candidate)
    verdict = verify.verified(f"attention-{candidate.name}", key, device.sm, probe)
    if not verdict.ok:
        raise AttentionRefusal(
            "attention_kernel_failed_verification",
            f"{ready.name} on {device.name}: {verdict.detail}"[:600],
        )


#: Candidates whose wheel module failed its check on this machine run: they build from source.
_WHEEL_REFUSED: set[str] = set()


def _wheel(candidate: Candidate, device: DeviceFacts) -> dict[str, str] | None:
    """The Runtime wheel's module for CANDIDATE on this card (`prebuilt.serves`), opened as a
    compiled kernel is and checked by `_launch`, unless it failed that check this run; FA3 opens
    through `_prebuilt_fa3`. Its record names it, as a build's names its artifact."""
    module = prebuilt.ATTENTION.get(candidate.name)
    if (
        module in (None, "_fa3")
        or device.kind != "cuda"
        or candidate.name in _WHEEL_REFUSED
        or not prebuilt.serves(candidate.name, device.sm)
    ):
        return None
    return {"artifact": "prebuilt", "key": prebuilt.artifact(module).partition(":")[2][:16]}


def _prebuilt_fa3(candidate: Candidate, device: DeviceFacts) -> Ready | None:
    """FlashAttention-3 from the Runtime wheel once it verifies on this Hopper card; a
    failure is the run's one warning for it, and the local build is tried next."""
    if device.kind != "cuda" or not prebuilt.serves(candidate.name, device.sm):
        return None
    import torch

    kernel, why = prebuilt.fa3(torch, device.index, device.sm, candidate.cases())
    if kernel is None:
        if not prebuilt.ABSENT:
            _STATES[f"{candidate.name} prebuilt"] = kernel_compile.State(
                "prebuilt", "failed", detail=why
            )
        return None
    digest = prebuilt.artifact("_fa3").partition(":")[2][:16]
    upstream.bind_fa3(kernel, f"prebuilt {digest}")
    _STATES[candidate.name] = kernel_compile.State("prebuilt", "ready")
    return Ready(candidate, _member(candidate), {"artifact": "prebuilt", "key": digest})


def _compile_dependencies(
    candidates: Sequence[Candidate], device: DeviceFacts, wait: bool
) -> dict[str, dict[str, str] | None]:
    """Start every independent build before waiting or returning a no-wait refusal."""
    artifacts: dict[str, dict[str, str] | None] = {}
    refused: list[tuple[Candidate, AttentionRefusal]] = []
    for candidate in candidates:
        try:
            artifacts[candidate.name] = _compiled(candidate, device, wait=False)
        except AttentionRefusal as exc:
            refused.append((candidate, exc))
    # Known failures refuse even a strict pin immediately; another dependency's
    # compile does not turn a failed recipe into a queued one.
    failed = next((exc for _, exc in refused if not exc.code.endswith("_compiling")), None)
    if failed is not None:
        raise failed
    if refused and not wait:
        raise refused[0][1]
    for candidate, _ in refused:
        artifacts[candidate.name] = _compiled(candidate, device, wait=True)
    return artifacts


#: Each candidate's `kernel_ready` name on a device: `_slot`.
_SLOTS = "attention "


def _slot(name: str, device: DeviceFacts) -> str:
    return f"{_SLOTS}{device.kind}:{device.index} sm{device.sm} {name}"


def _opened(candidate: Candidate, device: DeviceFacts) -> Ready:
    """CANDIDATE once it serves on this device (`kernel_ready`: compiled, imported and launched,
    off any request thread), else a typed refusal: `attention_kernel_compiling` while it opens,
    or why it failed, for this process's life."""
    if not (candidate.artifact or candidate.import_name or candidate.producer):
        return _waited(candidate, device)  # torch's own: nothing to build or check
    name = _slot(candidate.name, device)
    found: Ready | None = kernel_ready.ready(name, partial(_open, candidate, device))
    if found is not None:
        return found
    failed = kernel_ready.failed(name)
    if isinstance(failed, AttentionRefusal):
        raise AttentionRefusal(failed.code, str(failed))
    if failed is not None:
        raise AttentionRefusal(
            "attention_kernel_failed", f"{candidate.name}: {type(failed).__name__}: {failed}"
        )
    state = _STATES.get(candidate.name)
    line = state.line() if state is not None and state.state == "compiling" else "opening"
    raise AttentionRefusal("attention_kernel_compiling", f"{candidate.name}: {line}")


def _waited(candidate: Candidate, device: DeviceFacts) -> Ready:
    """CANDIDATE on this device, after its build: a request that names it waits for it."""
    return kernel_ready.wait(_slot(candidate.name, device), partial(_open, candidate, device))


def _open(candidate: Candidate, device: DeviceFacts, wait: bool) -> Ready | None:
    """`kernel_ready`'s opener: `_resolve`, or None while its build runs and it may not wait."""
    try:
        return _resolve(candidate, device, wait)
    except AttentionRefusal as exc:
        if exc.code == "attention_kernel_compiling" and not wait:
            return None
        raise


def _resolve(candidate: Candidate, device: DeviceFacts, wait: bool) -> Ready:
    """One candidate compiled, imported and launched on this device, or a typed refusal; its
    build waited for when `wait`."""
    _registry()
    if candidate.producer is not None:
        base = (_waited if wait else _opened)(BY_NAME[candidate.producer.base], device)
        kernel = _import(base.candidate)
        missing = [n for n in candidate.producer.requires if not callable(getattr(kernel, n, None))]
        if missing:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.name}: {base.candidate.import_name} lacks {', '.join(missing)}",
            )
        return Ready(candidate, _member(candidate), base.artifact)
    prebuilt_ready = _prebuilt_fa3(candidate, device)
    if prebuilt_ready is not None:
        return prebuilt_ready
    artifact: dict[str, str] | None
    if candidate.dense:
        dense = [BY_NAME[name] for name in dict.fromkeys(candidate.dense)]
        artifacts = _compile_dependencies((candidate, *dense), device, wait)
        artifact = dict(artifacts[candidate.name] or {})
        for role, beside in zip(("dense", "prefix"), dense, strict=False):
            opened = (_waited if wait else _opened)(beside, device).artifact or {}
            artifact.update({f"{role}_{key}": value for key, value in opened.items()})
    else:
        artifact = _compiled(candidate, device, wait) if candidate.artifact else None
    wheel = (artifact or {}).get("artifact") == "prebuilt"
    if wheel:
        upstream.bind_sage3(import_module(_SAGE3_WHEEL), f"prebuilt {(artifact or {})['key']}")
        _STATES[candidate.name] = kernel_compile.State("prebuilt", "ready")
    elif candidate.import_name:
        module = _import(candidate)
        if candidate.import_name == _SAGE3:
            upstream.bind_sage3(import_module("sageattn3.api"), (artifact or {}).get("key", ""))
        if candidate.sparse and candidate.import_name == "comfy_kitchen":
            import torch

            if not module.sol_attn_is_available(torch.device("cuda", device.index)):
                raise AttentionRefusal(
                    "attention_kernel_absent",
                    f"{candidate.name}: compiled Kitchen Sol is unavailable",
                )
        if candidate.local in (upstream.FA3, upstream.FA3_FP8):
            upstream.bind_fa3(
                cast("upstream.FlashAttn3", module), (artifact or {}).get("key", "environment")
            )
    if candidate.kernels and device.kind == "cuda":
        own = import_module(candidate.kernels)
        launch = _launcher(own, candidate.name, device, wait)
        if launch is not None:
            own.serve(device.index, launch)
    if _sol_objects(candidate):
        _sol_route(device, wait)
        if device.kind == "cuda":
            attention_sol.submit_card(device.index)
    ready = Ready(candidate, _member(candidate), artifact)
    try:
        _launch(ready, device)
    except AttentionRefusal as exc:
        if wheel:  # the run's one warning for it; the source build is tried next
            _WHEEL_REFUSED.add(candidate.name)
            _STATES[f"{candidate.name} prebuilt"] = kernel_compile.State(
                "prebuilt", "failed", detail=str(exc)
            )
            return _resolve(candidate, device, wait)
        _STATES[candidate.name] = kernel_compile.State(candidate.name, "failed", detail=str(exc))
        raise
    if candidate.artifact:
        _STATES.setdefault(candidate.name, kernel_compile.State(candidate.artifact, "ready"))
    return ready


def _launcher(module: ModuleType, name: str, device: DeviceFacts, wait: bool) -> Launch | None:
    """The Runtime's own Triton kernels `module.KERNELS`, which kernel NAME launches, built by
    the machine's compile manager and loaded onto this card (`triton_kernels.ready`); a build
    still running refuses as compiling, a failed one as absent. A module with an eager floor
    serves without kernels this machine cannot build (None)."""
    from cozy_runtime.internal import triton_kernels as built

    try:
        launch: Launch | None = built.ready(module.KERNELS, wait, device=f"cuda:{device.index}")
    except KernelsUnavailable as exc:
        if exc.code == "kernels_absent" and getattr(module, "EAGER_FLOOR", False):
            return None
        raise AttentionRefusal("attention_kernel_absent", f"{name}: {exc}") from exc
    if launch is None:
        raise AttentionRefusal(
            "attention_kernel_compiling", f"{name}: {module.KERNELS.name} compiling"
        )
    return launch


def _expected(live_tokens: int, module: Any) -> None:
    """A model's early length statement reaches Sol when a Sol site serves `module`."""
    found = known()
    if not any(_sol_objects(found[n]) for n in observed({"": module}).totals() if n in found):
        return
    import torch

    attention_sol.expect(live_tokens, torch.device(str(placed_device())))


_EXPECTED.append(_expected)


def evidence(served: Iterable[str]) -> list[dict[str, Any]]:
    """This rank's kernels for a run's execution record: every kernel it served, and every
    kernel of its chains it did not serve and why (compiling, failed, absent, unsupported)."""
    names = set(served)
    if not names:
        return []  # no attention site ran here
    states = {name: kernel_compile.State(name, "ready") for name in names}
    states.update(_STATES)
    return [
        {**state.document(), "kernel": name, "served": name in names}
        for name, state in sorted(states.items())
    ]


def failures(
    served: Iterable[str], sol: Iterable[Mapping[str, object]] = ()
) -> list[dict[str, str]]:
    """The kernels a run's fallbacks served in place of, because they failed here (compile,
    import or launch), once per kernel and why; `sol` is the run's Sol object rows, whose failed
    object serves the dense reference. One still compiling, or ruled out by this card or shape,
    is by design and is not here."""
    names = set(served)
    rows = {
        (name, state.line())
        for name, state in _STATES.items()
        if state.state == "failed" and name not in names
    }
    rows |= {
        (f"{attention_sol.KERNEL} sparse", str(row.get("state", "")))
        for row in sol
        if row.get("source") == "failed"
    }
    return [{"kernel": kernel, "reason": reason} for kernel, reason in sorted(rows)]


def _sol_route(device: DeviceFacts, wait: bool) -> None:
    """Sol serves only its CuTe routes and FA4's (`attention_sol.route`), never Triton's: FA4's
    once its merge is built (`attention_sol_fa4.KERNELS`) and its forwards compiled."""
    if device.kind == "cuda":
        route = attention_sol.backend(device.index)
        if route == "fa4_block_sparse":
            _import(known()["flash-attn4"])
            from cozy_runtime.internal import attention_sol_fa4 as fa4

            launch = _launcher(fa4, "sol-attn FA4", device, wait)
            try:
                fa4.warm(device.index, cast("Launch", launch))
            except Exception as exc:
                raise AttentionRefusal(
                    "attention_kernel_launch_failed",
                    f"sol-attn's FA4 block-sparse route did not build on {device.name}: "
                    f"{type(exc).__name__}: {exc}"[:600],
                ) from exc
        elif not route.startswith("cute_"):
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"sol-attn has no route on {device.name}: its CuTe routes need cutlass.cute "
                "and cuda.bindings",
            )


def _sol_dense(
    order: Sequence[str],
    device: DeviceFacts,
    resolved: dict[str, Ready],
    refused: dict[str, str],
) -> tuple[Ready, Ready | None]:
    """The kernel Sol recomputes dense rows with (its dense steps, protected prefix and dense
    paths): the first one after Sol in the site's walk that serves BF16 at head dimension 128
    on this rank, else SDPA; and when that one quantizes, the next exact one, which the
    model's exact paths take (level 16). It runs locally after the Ulysses exchange, so it
    needs no context parallelism. One still compiling is passed over (run 1565: SDPA served
    Sol's dense steps at 35 s where SageAttention takes about 24 s, the price of not waiting)
    and selected again once ready."""
    found, row = known(), device_row(device)
    dense: Ready | None = None
    for name in (*order, "sdpa"):
        candidate = found.get(name)
        if (
            candidate is None
            or candidate.sparse
            or _off_row(candidate, row)
            or not candidate.admits("bfloat16", 128)
            or not candidate.cross
            or (dense is not None and not candidate.exact)
            or name in refused
        ):
            continue
        if name not in resolved:
            try:
                resolved[name] = _opened(candidate, device)
            except Exception as exc:
                refused.setdefault(name, f"{type(exc).__name__}: {exc}"[:300])
                continue
        if dense is not None:
            return dense, resolved[name]
        if candidate.exact:
            return resolved[name], None
        dense = resolved[name]
    if dense is not None:
        return dense, None
    raise AttentionRefusal("attention_kernel_absent", "Sol has no ready dense reference")


#: How a dense identity names the exact kernel beside a quantizing dense one.
_EXACT_AT = "; exact="


def _bind_sol_dense(chosen: tuple[Ready, Ready | None]) -> None:
    dense, exact = chosen
    backends = _registry()._backends
    attention_sol.bind_dense(
        backends[dense.member],
        dense.name if exact is None else f"{dense.name}{_EXACT_AT}{exact.name}",
        None if exact is None else backends[exact.member],
    )


def pinned(name: str, device: DeviceFacts, wait: bool = True) -> list[Ready]:
    """The single kernel the execution-path override names, or a typed refusal (cr-125).
    A request's `plan` turns a refusal into a visible fallback, never a silent one. Not
    `wait`: one still compiling refuses (`attention_kernel_compiling`) instead."""
    candidate = known().get(name)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"{name!r} is not an attention kernel this runtime knows; it knows {sorted(known())}",
        )
    unsupported = _off_row(candidate, device_row(device))
    if unsupported:
        # THE REFUSAL IS THE DISCOVERABILITY: the same sentence says what this card serves.
        raise AttentionRefusal(
            "attention_kernel_unsupported",
            f"{unsupported}, sm{device.sm}. This environment runs {image_kernels(device)} on "
            f"{device.name}",
        )
    return [(_waited if wait else _opened)(candidate, device)]


# ------------------------------------------------------------------------------ the pin


def _trees(roots: Mapping[str, object]) -> Iterator[tuple[str, Any]]:
    """Yield each constructed module tree, preserving component labels."""
    seen: set[int] = set()
    queue: deque[tuple[str, object, int]] = deque(
        (name, root, 0) for name, root in sorted(roots.items())
    )
    while queue:
        label, value, depth = queue.popleft()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if callable(getattr(value, "modules", None)) and callable(
            getattr(value, "parameters", None)
        ):
            yield label, value
            continue
        if depth >= 3:
            continue
        if isinstance(value, Mapping):
            for key, item in value.items():
                queue.append((str(key), item, depth + 1))
        elif hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                if not name.startswith("_"):
                    queue.append((name, item, depth + 1))


def _attention_records(roots: Mapping[str, object]) -> Iterator[_Record]:
    """All Diffusers attention processors, including those with no backend dispatch."""
    trees = list(_trees(roots))
    if not trees:  # a weightless call has none, and imports no Diffusers (or torch) to say so
        return
    try:
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_processor import Attention, MochiAttention
    except ImportError:
        return
    classes = (Attention, MochiAttention, AttentionModuleMixin)
    seen: set[int] = set()
    for component, tree in trees:
        for path, module in tree.named_modules():
            processor = getattr(module, "processor", None)
            if id(module) not in seen and isinstance(module, classes) and processor is not None:
                seen.add(id(module))
                yield component, path, module, processor


class ClassicSdpa:
    """Diffusers' `AttnProcessor2_0`, the same math, through the per-site backend contract.

    Classic `Attention` sites (SDXL's UNet, AutoencoderKL) call torch SDPA directly, so no
    kernel could be selected, pinned or observed on them. `adopt` gives each one this
    processor. Auto selection keeps SDPA there, bit-identical to before; a pin may choose
    another kernel.
    """

    _attention_backend: Any = None
    _parallel_config: Any = None

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        temb: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        shape = hidden_states.shape
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, length = context.shape[:2]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, length, batch)
            attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        with quantize_once(hidden_states, encoder_hidden_states):
            query = attn.to_q(hidden_states)
            key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query, key, value = (x.view(batch, -1, attn.heads, head_dim) for x in (query, key, value))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.flatten(2).to(query.dtype)))
        if len(shape) == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(shape)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def adopt(roots: Mapping[str, object]) -> None:
    """Give every classic `AttnProcessor2_0` site its own `ClassicSdpa`. Idempotent."""
    try:
        from diffusers.models.attention_processor import AttnProcessor2_0
    except ImportError:
        return
    for _component, _path, module, processor in list(_attention_records(roots)):
        if type(processor) is AttnProcessor2_0:
            module.set_processor(ClassicSdpa())


def _site_records(roots: Mapping[str, object]) -> Iterator[_Record]:
    """Only processors implementing Diffusers' per-site backend selection contract."""
    for record in _attention_records(roots):
        if hasattr(record[3], "_attention_backend"):
            yield record


def _unmanaged_reason(component: str, path: str, processor: object) -> str:
    return (
        f"{component}/{path} uses {type(processor).__name__}, which has no per-site "
        "attention backend selection"
    )


def _sites(roots: Mapping[str, object]) -> Iterator[tuple[str, Any, Any]]:
    for component, _path, module, processor in _site_records(roots):
        yield component, module, processor


def sites(roots: Mapping[str, object]) -> list[tuple[str, object, object]]:
    """Every runtime-selectable attention site: (component, module, processor)."""
    return list(_sites(roots))


def _head_dim(module: Any) -> int:
    """This site's per-head dimension, or 0 when the module does not say.

    A site whose head dimension cannot be read is treated as unknown, which admits only the
    unbounded floor — the runtime does not guess a shape into a kernel that would fault on it.
    """
    declared = getattr(module, "head_dim", None)
    if isinstance(declared, int) and declared > 0:
        return declared
    heads = getattr(module, "heads", None)
    inner = getattr(module, "inner_dim", None)
    if inner is None:
        to_q = getattr(module, "to_q", None)
        inner = getattr(to_q, "out_features", None)
    if isinstance(heads, int) and heads > 0 and isinstance(inner, int) and inner > 0:
        return inner // heads
    return 0


def _base(projection: object) -> object:
    """A PEFT LoRA wrapper's current base layer (encoded fill can replace it after
    construction), else the projection itself."""
    peft_linear = getattr(sys.modules.get("peft.tuners.lora.layer"), "Linear", None)
    if peft_linear is not None and isinstance(projection, peft_linear):
        return projection.get_base_layer()
    return projection


_FLOATS = frozenset({"float16", "bfloat16", "float32", "float64"})


def _projection_dtype(projection: Any) -> str:
    """Encoded output contract, or an ordinary floating Linear's result dtype."""
    import torch

    # PEFT returns the base result's dtype, even with differently typed factors.
    projection = _base(projection)
    declared = getattr(projection, "out_dtype", None)
    if declared is None and type(projection) is torch.nn.Linear:
        declared = projection.weight.dtype
    name = str(declared).removeprefix("torch.")
    return name if name in _FLOATS else ""


def _dtype(module: Any) -> str:
    """The site's attention activation precision: its own `activation_dtype` when it declares
    one (a site that casts Q/K/V itself), else the active Q/K/V projection contracts.

    Never inspect unrelated norm parameters or encoded weight buffers. Unknown or
    disagreeing projection contracts admit only an unbounded dtype backend (SDPA).
    Floating Linear inference assumes no external autocast changes its output dtype.
    """
    declared = getattr(module, "activation_dtype", None)
    if declared is not None:
        name = str(declared).removeprefix("torch.")
        return name if name in _FLOATS else ""
    projections = _projections(module)
    dtypes = {_projection_dtype(projection) for projection in projections}
    return next(iter(dtypes)) if len(dtypes) == 1 and "" not in dtypes else ""


def _projections(module: object) -> tuple[object, ...]:
    """The site's active Q/K/V projections; empty when the module does not say."""
    q, k, v, kv = (getattr(module, name, None) for name in ("to_q", "to_k", "to_v", "to_kv"))
    packed = getattr(module, "to_qkv", None)
    fused = getattr(module, "fused_projections", None)
    if packed is not None and (fused is True or (fused is None and q is None)):
        return (packed,)
    if q is not None and kv is not None and fused is True:
        return (q, kv)
    if q is not None and k is not None and v is not None:
        return (q, k, v)
    return ()


def _context_parallel(member: Any) -> bool:
    """Does diffusers' registry mark this backend as one it can shard? Read at the pin and
    never restated on a row: the backend's own registration is the authority, and a static
    copy of it would be the drift `_admit_arch` exists to refuse."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    return bool(_AttentionBackendRegistry._is_context_parallel_available(member))


@dataclass(frozen=True, slots=True)
class Applied:
    """What the runtime pinned, per component and per kernel. The BOOT record, and all of it."""

    #: component -> kernel name -> attention sites pinned to it.
    hosts: dict[str, dict[str, int]]
    #: the kernels this selection resolved here, in the order it reached them
    offered: tuple[str, ...]
    #: set when the caller PINNED one through the execution-path override (cr-125).
    pin: str = ""
    artifacts: dict[str, dict[str, str]] | None = None
    #: kernels selection passed over here, and why (a compiling one's progress)
    skipped: dict[str, str] | None = None
    #: kernels this selection passed over while they compiled: ready, they are selected again
    waiting: tuple[str, ...] = ()
    #: Selection-time reasons; later compile progress must not rewrite why a fallback ran.
    fallbacks: tuple[execution_evidence.AttentionFallback, ...] = ()

    def joined(self, later: Applied) -> Applied:
        """This selection and one that served after it in the same run: every kernel each
        component served."""
        hosts = {component: dict(kernels) for component, kernels in self.hosts.items()}
        for component, kernels in later.hosts.items():
            hosts.setdefault(component, {}).update(kernels)
        artifacts = {**(self.artifacts or {}), **(later.artifacts or {})}
        fallbacks = self.fallbacks + tuple(
            row for row in later.fallbacks if row not in self.fallbacks
        )
        return replace(self, hosts=hosts, artifacts=artifacts or None, fallbacks=fallbacks)

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kernels in self.hosts.values():
            for name, sites in kernels.items():
                out[name] = out.get(name, 0) + sites
        return out

    def document(self) -> execution_evidence.Boot:
        return _boot(
            self.totals(),
            {name: dict(kernels) for name, kernels in self.hosts.items()},
            list(self.offered),
            self.pin,
            self.artifacts or {},
            self.skipped or {},
            self.fallbacks,
        )

    def line(self) -> str:
        """The ONE line a pod prints at boot. It is the whole per-run attention record."""
        return self.document()["line"]


def line(
    totals: Mapping[str, int],
    hosts: Mapping[str, Mapping[str, int]],
    offered: Sequence[str],
    pin: str,
    skipped: Collection[str] = (),
) -> str:
    """The boot line, from the facts alone — so a GROUP prepare renders the same sentence from
    its merged facts as a single one does from its own `Applied` (cr-124)."""
    ignored = f"; skipped {sorted(skipped)}" if skipped else ""
    if not totals:
        none = "NONE — this construction holds no backend-selectable diffusers attention site"
        return none + ignored
    chosen = ", ".join(f"{name} x{sites}" for name, sites in sorted(totals.items()))
    where = "; ".join(
        f"{component}: {'+'.join(sorted(kernels))}" for component, kernels in sorted(hosts.items())
    )
    how = f"PINNED {pin}" if pin else f"auto over {list(offered)}"
    return f"{chosen} ({how}) — {where}{ignored}"


def _boot(
    kernels: dict[str, int],
    hosts: dict[str, dict[str, int]],
    offered: list[str],
    pin: str,
    artifacts: Mapping[str, object],
    skipped: dict[str, str],
    fallbacks: Sequence[execution_evidence.AttentionFallback] = (),
) -> execution_evidence.Boot:
    # RENDERED HERE, not by the worker: one renderer for the one line, in the module that
    # knows what it means.
    boot = execution_evidence.Boot(
        kernels=kernels,
        hosts=hosts,
        offered=offered,
        pin=pin,
        line=line(kernels, hosts, offered, pin, skipped),
    )
    if fallbacks:
        boot["fallbacks"] = list(fallbacks)
        boot["line"] += "; " + "; ".join(
            f"{row['component']}: selected {row['selected']} instead of "
            f"{row['preferred']}: {row['reason']}"
            for row in fallbacks
        )
    if artifacts:
        boot["artifacts"] = dict(artifacts)
    if skipped:
        boot["skipped"] = skipped
    return boot


def merge(documents: Mapping[str, object]) -> execution_evidence.Boot | None:
    """A GROUP prepare's one attention fact, from its models' own `Applied.document()`s,
    each decoded here once; a model holding no attention site sends none."""
    hosts: dict[str, dict[str, int]] = {}
    kernels: dict[str, int] = {}
    offered: list[str] = []
    pins: set[str] = set()
    artifacts: dict[str, object] = {}
    skipped: dict[str, str] = {}
    fallbacks: list[execution_evidence.AttentionFallback] = []
    for parameter, document in documents.items():
        if not document:
            continue
        try:
            row = msgspec.convert(document, execution_evidence.Boot, strict=True)
        except msgspec.ValidationError as exc:
            raise AttentionRefusal(
                "attention_fact_malformed", f"{parameter}'s attention fact: {exc}"
            ) from exc
        for component, chosen in row["hosts"].items():
            hosts[f"{parameter}/{component}"] = chosen
        for name, sites in row["kernels"].items():
            kernels[name] = kernels.get(name, 0) + sites
        offered += [name for name in row["offered"] if name not in offered]
        if row["pin"]:
            pins.add(row["pin"])
        if "artifacts" in row:
            artifacts[parameter] = row["artifacts"]
        if "skipped" in row:
            skipped.update(row["skipped"])
        fallbacks.extend(
            {**held, "component": f"{parameter}/{held['component']}"}
            for held in row.get("fallbacks", ())
        )
    if not kernels:
        return None
    pin = "+".join(sorted(pins))
    return _boot(kernels, hosts, offered, pin, artifacts, skipped, fallbacks)


def _pin_parts(pin: str) -> tuple[str, str]:
    if "=" not in pin:
        return "", pin
    component, name = pin.split("=", 1)
    if not component or not name or "=" in name or any(ch.isspace() for ch in pin):
        raise AttentionRefusal(
            "attention_kernel_unknown", f"invalid attention override {pin!r}; use component=backend"
        )
    return component, name


def reaches(roots: Mapping[str, object], pin: str) -> bool:
    """Whether any attention site here falls in the pin's scope."""
    scope, _ = _pin_parts(pin)
    return any(_matches(component, scope) for component, _, _, _ in _attention_records(roots))


def for_model(pin: str, parameters: Sequence[str]) -> str:
    """Translate an aggregate model/component pin to one construction's local scope."""
    scope, backend = _pin_parts(pin)
    if "/" not in scope:
        return pin
    model, component = scope.split("/", 1)
    return f"{component}={backend}" if model in parameters else ""


def _matches(component: str, scope: str) -> bool:
    if not scope or component == scope:
        return True
    model, qualified, name = scope.partition("/")
    if not qualified:
        return component.rsplit("/", 1)[-1] == scope
    owners, owned, component_name = component.partition("/")
    return bool(owned and model in owners.split("+") and name == component_name)


def _unfit(
    candidate: Candidate,
    row: arch.Row | None,
    module: object,
    dtype: str,
    head_dim: int,
    degree: int,
) -> str:
    """Empty when the row (None: unchecked), this site and the degree admit CANDIDATE, else
    why not. Static: whether it is compiled yet is resolution's question."""
    off = "" if row is None else _off_row(candidate, row)
    if off:
        return off
    if not candidate.admits(dtype, head_dim):
        return (
            f"{candidate.name} does not serve {dtype or 'unknown'} activations at head "
            f"dimension {head_dim or 'unknown'}"
        )
    invalid = _invalid(candidate.producer, module)
    if invalid:
        return invalid
    try:
        member = _member(candidate)
        if member not in _registry()._backends:
            return f"{candidate.name} is not registered with Diffusers"
        if degree > 1 and not _context_parallel(member):
            return f"{candidate.name} cannot shard sequence-parallel degree {degree}"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:300]
    return ""


def _first(
    order: Sequence[str],
    row: arch.Row | None,
    device: DeviceFacts | None,
    component: str,
    module: object,
    degree: int,
    resolved: dict[str, Ready],
    refused: dict[str, str],
    fallbacks: list[execution_evidence.AttentionFallback],
) -> Ready | None:
    """The first kernel of ORDER that serves this site here. One its row, shape, module or
    degree rules out is passed by silently; one absent, compiling or failing its launch is
    written into REFUSED, and the kernel that serves in its place into FALLBACKS (the run's
    warning). Without a DEVICE only RESOLVED kernels serve."""
    found, dtype, head_dim = known(), _dtype(module), _head_dim(module)
    for name in order:
        candidate = found.get(name)
        if candidate is None or _unfit(candidate, row, module, dtype, head_dim, degree):
            continue
        if name in refused:
            continue
        if name not in resolved:
            if device is None:
                continue
            try:
                resolved[name] = _opened(candidate, device)
            except Exception as exc:
                # BROAD ON PURPOSE: a kernel that fails to resolve for ANY reason steps to the
                # next, never failing a prepare that only tried to go faster. A PIN keeps its
                # typed refusals (`pinned`).
                refused[name] = f"{type(exc).__name__}: {exc}"[:300]
                continue
        entry = resolved[name]
        for passed in order[: order.index(name)]:
            fallback = execution_evidence.AttentionFallback(
                component=component, preferred=passed, selected=name, reason=refused.get(passed, "")
            )
            if passed in refused and fallback not in fallbacks:
                fallbacks.append(fallback)
        return entry
    return None


def install(
    ready: list[Ready],
    roots: Mapping[str, object],
    pin: str = "",
    degree: int = 1,
    *,
    device: DeviceFacts | None = None,
    bits: int | Mapping[str, int] = 16,
    row: arch.Row | None = None,
    skipped: dict[str, str] | None = None,
    choice: Choice | None = None,
) -> Applied:
    """Validate a complete per-site selection before changing any processor.

    A site in the pin's scope takes the pinned kernel, or refuses (`EXACT`: its walk at level
    16). Given another GPU's CHOICE (a group follower takes rank 0's), each site takes the
    kernel that GPU serves there and Sol its dense reference, each resolved on DEVICE as a pin
    is, or refuses. Every other site takes the first kernel of its walk on ROW (else DEVICE's)
    at the model's `bits` for it (`walk`, `level`); a classic site keeps SDPA. Without a
    DEVICE, sites take the READY kernels in order.
    """
    scope, name = _pin_parts(pin)
    if device is not None:
        row = row or device_row(device)
    hosts: dict[str, dict[str, int]] = {}
    assignments: list[_Site] = []
    resolved = {entry.name: entry for entry in ready}
    taken = {} if choice is None else {(c, path): k for c, path, k in choice.sites}
    if choice is not None:
        if device is None:
            raise AttentionRefusal("attention_kernel_unsupported", "a choice installs on a device")
        lacking: dict[str, str] = {}
        for kernel in sorted({*taken.values(), choice.dense, choice.exact} - {""}):
            try:
                resolved[kernel] = pinned(kernel, device, wait=False)[0]
            except AttentionRefusal as exc:
                lacking[kernel] = exc.code
        if lacking:
            raise Lacking(lacking)
    refused: dict[str, str] = dict(skipped or {})
    matched = False
    sol_order: list[str] | None = None
    fallbacks: list[execution_evidence.AttentionFallback] = []
    for record in _attention_records(roots):
        component, path, module, processor = record
        forced = bool(name and _matches(component, scope))
        matched |= forced
        entry: Ready | None
        if forced and name != EXACT:
            reason = _pin_reason(resolved.get(name), f"{name} is not open here", record, degree)
            if reason:
                raise AttentionRefusal(
                    "attention_kernel_unsupported",
                    f"{reason}. A pinned kernel is never quietly replaced",
                )
            entry, order = resolved[name], [name]
        elif not hasattr(processor, "_attention_backend"):
            if forced:
                reason = _unmanaged_reason(component, path, processor)
                raise AttentionRefusal("attention_kernel_unsupported", reason)
            continue
        elif choice is not None:
            order = [taken.get((component, path), "")]
            if not order[0]:
                raise AttentionRefusal(
                    "attention_kernel_unsupported",
                    f"the choice names no kernel for {component}/{path}",
                )
            entry = _first(order, None, None, component, module, degree, resolved, refused, [])
        else:
            at = 16 if forced else level(bits, component, path)
            order = (
                _walk_order(row, at, module, processor)
                if row is not None
                else [entry.name for entry in ready]
            )
            entry = _first(
                order, row, device, component, module, degree, resolved, refused, fallbacks
            )
        if entry is None:
            detail = _site_refusal(
                order[0] if choice is not None else "no available attention kernel",
                component,
                _dtype(module),
                _head_dim(module),
            )
            if degree > 1:
                detail = AttentionRefusal(
                    detail.code,
                    f"{detail}; sequence-parallel degree {degree} requires context parallelism",
                )
            raise detail
        if _walked(entry.candidate) and sol_order is None:
            sol_order = order[order.index(entry.name) + 1 :]
        assignments.append((component, path, module, processor, entry))
        hosts.setdefault(component, {})
        hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
    if name and not matched:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{pin!r} matches no attention component"
        )
    if choice is not None and sol_order is not None:
        if not choice.dense:
            raise AttentionRefusal("attention_kernel_absent", "the choice names no dense reference")
        exact = resolved[choice.exact] if choice.exact else None
        _bind_sol_dense((resolved[choice.dense], exact))
    elif device is not None and sol_order is not None:
        _bind_sol_dense(_sol_dense(sol_order, device, resolved, refused))
    _install_sites(
        [
            (entry, component, path, module)
            for component, path, module, _processor, entry in assignments
        ]
    )
    for _component, _path, _module, processor, entry in assignments:
        processor._attention_backend = entry.member
    if skipped is not None:
        skipped.update(refused)
    artifacts = {name: entry.artifact for name, entry in resolved.items() if entry.artifact}
    waiting = tuple(
        n for n in sorted(refused) if device is not None and kernel_ready.opening(_slot(n, device))
    )
    return Applied(
        hosts, tuple(resolved), pin, artifacts or None, refused or None, waiting, tuple(fallbacks)
    )


def _walk_order(row: arch.Row, at: int, module: Any, processor: Any) -> list[str]:
    """A site's kernels at level AT: a classic site keeps SDPA, every other one walks."""
    return ["sdpa"] if isinstance(processor, ClassicSdpa) else walk(row, at, sparse_site(module))


def _pin_reason(entry: Ready | None, why: str, record: _Record, degree: int) -> str:
    """Why a site in a named pin's scope does not take its kernel ENTRY, or empty: its
    processor has no per-site selection, the kernel does not serve here (`why`, ENTRY None), or
    its row refuses this site or degree. Every pin reads it: at load a reason refuses the pin
    (`install`), at a request the site keeps its prepared kernel and says why (`plan`)."""
    component, path, module, processor = record
    if not hasattr(processor, "_attention_backend"):
        return _unmanaged_reason(component, path, processor)
    if entry is None:
        return why
    return _unfit(entry.candidate, None, module, _dtype(module), _head_dim(module), degree)


def _install_sites(records: Sequence[tuple[Ready, str, str, Any]]) -> None:
    """Install metadata hooks atomically, preserving hooks owned by preparation."""
    # Validate all producer topology/hooks before any metadata or processor mutation.
    for entry, _component, _path, module in records:
        producer = entry.candidate.producer
        if producer is not None:
            producer.validate(module)
    added: list[tuple[str, str, Any]] = []
    processors: list[tuple[Any, Any]] = []
    try:
        for entry, component, path, module in records:
            if not entry.candidate.sparse:
                continue
            local = component.rsplit("/", 1)[-1]
            if getattr(module, "_cozy_sol_site", None) is None:
                added.append((local, path, module))
            attention_sol.install_site(module, local, path)
            producer = entry.candidate.producer
            if producer is not None:
                processors.append((module, module.processor))
                kernel = entry.candidate.import_name
                producers.install(module, producer, entry.candidate.backend, kernel, local, path)
    except BaseException:
        for module, processor in reversed(processors):
            module.set_processor(processor)
        for component, path, module in reversed(added):
            if getattr(module, "_cozy_sol_site", None) is not None:
                attention_sol.remove_site(module, component, path)
        raise


def _names() -> dict[str, str]:
    """Diffusers backend value -> the kernel name it serves."""
    return {candidate.backend: candidate.name for candidate in known().values()}


def _served(processor: object, names: Mapping[str, str]) -> str:
    """The kernel a site's processor dispatches to now."""
    backend = getattr(processor, "_attention_backend", None)
    value = getattr(backend, "value", str(backend))
    return names.get(value, value)


def observed(roots: Mapping[str, object], pin: str = "") -> Applied:
    """Read actual processor choices; never substitute the prepared recommendation."""
    names = _names()
    hosts: dict[str, dict[str, int]] = {}
    for component, _module, processor in _sites(roots):
        name = _served(processor, names)
        hosts.setdefault(component, {})
        hosts[component][name] = hosts[component].get(name, 0) + 1
    served = {name for kernels in hosts.values() for name in kernels}
    artifacts = {
        ready.name: ready.artifact
        for ready in cast("list[Ready]", kernel_ready.served(_SLOTS))
        if ready.name in served and ready.artifact
    }
    return Applied(hosts, (), pin, artifacts=artifacts or None)


def chosen(roots: Mapping[str, object]) -> Choice:
    """What this GPU serves now: every site's kernel, and Sol's dense reference where Sol
    serves (`attention_sol.binding`)."""
    names = _names()
    sites = tuple(
        (component, path, _served(processor, names))
        for component, path, _module, processor in _site_records(roots)
    )
    found = known()
    if not any(kernel in found and _walked(found[kernel]) for *_, kernel in sites):
        return Choice(sites=sites)
    dense, _, exact = attention_sol.binding()[1].partition(_EXACT_AT)
    return Choice(sites=sites, dense=dense, exact=exact)


def difference(ours: Choice, theirs: Choice) -> str:
    """The first way THEIRS serves other than OURS; "" when it serves the same."""
    if (ours.dense, ours.exact) != (theirs.dense, theirs.exact):
        return (
            f"Sol's dense reference {ours.dense or 'none'} (exact {ours.exact or 'itself'}) "
            f"against {theirs.dense or 'none'} (exact {theirs.exact or 'itself'})"
        )
    held = {(component, path): kernel for component, path, kernel in theirs.sites}
    for component, path, kernel in ours.sites:
        other = held.pop((component, path), None)
        if other != kernel:
            return f"{component}/{path}: {kernel} against {other or 'none'}"
    if held:
        (component, path), kernel = sorted(held.items())[0]
        return f"{component}/{path}: none against {kernel}"
    return ""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Prepared processor choices, restored without resolution or policy re-execution, with
    the dense reference Sol ran with when one of them is Sol."""

    choices: tuple[tuple[Any, Any], ...]
    dense: attention_sol.Binding | None = None

    def restore(self) -> None:
        for processor, member in self.choices:
            processor._attention_backend = member
        if self.dense is not None:
            attention_sol.rebind(self.dense)


def snapshot(roots: Mapping[str, object]) -> Snapshot:
    names = _names()
    held = [processor for _, _, processor in _sites(roots)]
    found = known()
    sol = any(
        (kernel := _served(processor, names)) in found and _walked(found[kernel])
        for processor in held
    )
    return Snapshot(
        tuple((processor, processor._attention_backend) for processor in held),
        attention_sol.binding() if sol else None,
    )


@contextmanager
def override(
    defaults: Snapshot, roots: Mapping[str, object], swap: Swap | None, pin: str = ""
) -> Iterator[Applied]:
    """Observe one temporary request selection and restore it even when execution raises."""
    defaults.restore()
    changed = swap.sites if swap is not None else ()
    temporary_sites = [
        (component.rsplit("/", 1)[-1], path, module)
        for component, path, module, _, entry in changed
        if entry.candidate.sparse and getattr(module, "_cozy_sol_site", None) is None
    ]
    temporary_processors = [
        (module, module.processor) for _, _, module, _, entry in changed if entry.candidate.producer
    ]
    dense = attention_sol.binding()
    try:
        applied = None
        if swap is not None:
            applied = swap.apply()
        facts = observed(roots, pin)
        yield (
            replace(
                facts,
                artifacts={**(facts.artifacts or {}), **(applied.artifacts or {})} or None,
                fallbacks=applied.fallbacks,
            )
            if applied is not None
            else facts
        )
    finally:
        try:
            for component, path, module in temporary_sites:
                if getattr(module, "_cozy_sol_site", None) is not None:
                    attention_sol.remove_site(module, component, path)
        finally:
            try:
                for module, processor in reversed(temporary_processors):
                    module.set_processor(processor)
            finally:
                attention_sol.rebind(dense)
                defaults.restore()


def _site_refusal(name: str, component: str, dtype: str, head_dim: int) -> AttentionRefusal:
    return AttentionRefusal(
        "attention_kernel_unsupported",
        f"{name} does not serve {component}'s attention: {dtype or 'unknown'} activations "
        f"at head dimension {head_dim or 'unknown'}. A pinned kernel is never quietly replaced",
    )


def _baked_in(roots: Mapping[str, object]) -> str:
    for label, tree in _trees(roots):
        for module in tree.modules():
            if hasattr(module, "_orig_mod"):
                return f"{label} is an OptimizedModule wrapping {type(module._orig_mod).__name__}"
            if getattr(module, "_compiled_call_impl", None) is not None:
                return f"{label} holds {type(module).__name__} compiled in place"
    return ""


@dataclass(frozen=True, slots=True)
class Swap:
    """A validated per-request attention change, applied atomically."""

    #: (component, path, module, processor, kernel) per site it changes
    sites: tuple[_Site, ...]
    pin: str = ""
    #: sites in scope that keep their prepared kernel, and why the pin could not run there
    fallbacks: tuple[execution_evidence.AttentionFallback, ...] = ()
    #: Sol's dense reference for this request (`EXACT`, a walked pin), restored after it
    dense: tuple[Ready, Ready | None] | None = None

    def apply(self) -> Applied:
        _install_sites([(entry, c, path, module) for c, path, module, _, entry in self.sites])
        if self.dense is not None:
            _bind_sol_dense(self.dense)
        hosts: dict[str, dict[str, int]] = {}
        for component, _, _, processor, entry in self.sites:
            processor._attention_backend = entry.member
            hosts.setdefault(component, {})
            hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
        served = {entry.name: entry for *_, entry in self.sites}
        artifacts = {name: e.artifact for name, e in served.items() if e.artifact}
        return Applied(hosts, tuple(served), self.pin, artifacts or None, fallbacks=self.fallbacks)


def _is_selected(defaults: Snapshot, roots: Mapping[str, object], pin: str) -> bool:
    """A repeated prepared pin is a no-op only at every site in its nonempty scope."""
    scope, name = _pin_parts(pin)
    candidate = known().get(name)
    if candidate is None:
        return False
    prepared = {id(processor): member for processor, member in defaults.choices}
    records = [row for row in _attention_records(roots) if _matches(row[0], scope)]
    return bool(records) and all(
        getattr(processor, "_attention_backend", None) == candidate.backend
        and prepared.get(id(processor)) == candidate.backend
        for _, _, _, processor in records
    )


def admit(pin: str, degree: int) -> None:
    """Refuse a pin no construction here could serve before anything is constructed."""
    _, name = _pin_parts(pin)
    if not name or name == EXACT:
        return
    candidate = known().get(name)
    if candidate is None:
        raise AttentionRefusal(
            "attention_kernel_unknown",
            f"unknown attention kernel {name!r}; this runtime knows {sorted(known())}",
        )
    if degree > 1:
        _registry()
        if not _context_parallel(_member(candidate)):
            raise AttentionRefusal(
                "attention_kernel_unsupported",
                f"{name} does not support context parallelism, which this sequence-parallel "
                f"degree {degree} construction requires",
            )


def plan(device: DeviceFacts, roots: Mapping[str, object], name: str, degree: int = 1) -> Swap:
    """A request pin on every site in its scope its kernel can run. A site it cannot (this
    card, this shape, the group degree, a failed build, a compiled module) keeps its prepared
    kernel, recorded with why: a pin falls back visibly, never silently. An unknown kernel,
    a pin that reaches nothing and a build another executor must import refuse. `EXACT` puts
    each site in scope at level 16 (`exact`)."""
    admit(name, 1)
    scope, backend = _pin_parts(name)
    selected = {component: root for component, root in roots.items() if _matches(component, scope)}
    records = list(_attention_records(selected))
    if not records:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{name} matches no diffusers attention site"
        )
    if backend == EXACT:  # a compiled module keeps the kernel it was compiled with
        return exact(device, [] if _baked_in(selected) else records, name, degree)
    candidate = known()[backend]
    held = observed(selected).hosts
    why = _baked_in(selected)
    why = why and f"{why}; a compiled module keeps the kernel it was compiled with"
    entry = None
    if not why and degree > 1:
        _registry()
        if not _context_parallel(_member(candidate)):
            why = f"{backend} cannot shard sequence-parallel degree {degree}"
    if not why:
        try:
            entry = pinned(backend, device)[0]
        except AttentionRefusal as exc:
            if exc.code in ("attention_kernel_unknown", "attention_kernel_needs_executor"):
                raise
            why = str(exc)
    # A pinned walked kernel's dense rows are its own walk's, SDPA, for this request only: never
    # whichever reference an earlier selection in this executor bound (route by rule).
    walked = entry is not None and _walked(entry.candidate)
    dense = _sol_dense((), device, {}, {}) if walked else None
    sites: list[_Site] = []
    fallbacks: list[execution_evidence.AttentionFallback] = []
    for record in records:
        component, path, module, processor = record
        reason = _pin_reason(entry, why, record, degree)
        if reason or entry is None:
            row = execution_evidence.AttentionFallback(
                component=component,
                preferred=backend,
                selected="+".join(sorted(held.get(component) or ())) or "its own attention",
                reason=reason[:600],
            )
            if row not in fallbacks:
                fallbacks.append(row)
            continue
        sites.append((component, path, module, processor, entry))
    return Swap(tuple(sites), name, tuple(fallbacks), dense)


def exact(
    device: DeviceFacts,
    records: Sequence[_Record],
    pin: str = EXACT,
    degree: int = 1,
) -> Swap:
    """A request at 16 bits: each site its walk at level 16; a sparse site keeps its sparse
    kernel, whose dense rows go exact too."""
    row = device_row(device)
    resolved: dict[str, Ready] = {}
    refused: dict[str, str] = {}
    sites: list[_Site] = []
    fallbacks: list[execution_evidence.AttentionFallback] = []
    sol: list[str] | None = None
    for component, path, module, processor in records:
        if not hasattr(processor, "_attention_backend"):
            continue
        order = _walk_order(row, 16, module, processor)
        entry = _first(order, row, device, component, module, degree, resolved, refused, fallbacks)
        if entry is not None:
            if _walked(entry.candidate) and sol is None:
                sol = order[order.index(entry.name) + 1 :]
            sites.append((component, path, module, processor, entry))
    dense = None if sol is None else _sol_dense(sol, device, resolved, refused)
    return Swap(tuple(sites), pin, tuple(fallbacks), dense)


def select(
    device: DeviceFacts,
    roots: Mapping[str, object],
    pin: str = "",
    degree: int = 1,
    *,
    bits: int | Mapping[str, int] = 16,
    row: arch.Row | None = None,
    passed: Mapping[str, str] | None = None,
) -> Applied:
    """Select once at preparation: each site's walk at the model's `bits`, or the pin. Every
    kernel in `passed` is passed over, with why (another GPU of the group lacks it)."""
    adopt(roots)
    _, name = _pin_parts(pin)
    ready = pinned(name, device) if name and name != EXACT else []
    skipped: dict[str, str] = dict(passed or {})
    applied = install(ready, roots, pin, degree, device=device, bits=bits, row=row, skipped=skipped)
    return replace(applied, skipped=skipped or None)


def compiled(names: Iterable[str], device: DeviceFacts) -> bool:
    """Whether a kernel a selection passed over while it opened serves now (`kernel_ready`).
    It starts nothing and waits for nothing."""
    return any(kernel_ready.ready(_slot(name, device)) is not None for name in names)


def image_kernels(device: DeviceFacts) -> str:
    """What this device serves now, for a refusal message (cr-125): every kernel already
    ready here, and each one still compiling with its progress. Starts no compile."""
    ready = sorted(r.name for r in cast("list[Ready]", kernel_ready.served(_slot("", device))))
    busy = [f"{n} {s.line()}" for n, s in sorted(_STATES.items()) if s.state == "compiling"]
    return ", ".join(ready + busy) or "none"


def implementations(seen: Mapping[str, set[str]]) -> str:
    """The implementation line for what served in an `attention_ulysses.observing` scope."""
    names = {candidate.backend: candidate.name for candidate in known().values()}
    return attention_ulysses.describe(dict(seen), names)
