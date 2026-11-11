"""Select and install attention backends for constructed modules: Diffusers' attention modules
and the Runtime's own sites (`attention_site`), which need no Diffusers.

Every kernel is a Runtime-local forward (`attention_upstream`, `attention_sol`) that
`attention_ulysses` makes one call of, so one wrapper serves every degree: a Runtime site calls
it, and it is registered with Diffusers for a Diffusers site. A site takes
the first kernel of its hardware class's cell (`TABLE`, by `arch` row) at its attention
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

A kernel's row can also rule out one call: a call it refuses (`Candidate.refusal`) runs on SDPA,
whose row serves every call, and the run's record counts it (`evidence`). This is per call, for a
pin too, and selection does not learn it: what a call carries is not its site's (Qwen-Image 2.1
masks only its prefill's text, so its other steps keep their kernel).
"""

from __future__ import annotations

import hashlib
import math
import shutil
import site
import sys
from collections import deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import msgspec

from cozy_runtime.accel.attention import _EXPECTED, _PRODUCERS, AttentionProducer
from cozy_runtime.accel.kernels import KernelsUnavailable, Launch, quantize_once
from cozy_runtime.author._context import device as placed_device
from cozy_runtime.internal import (
    arch,
    attention_int4,
    attention_kitchen_sol,
    attention_sage2_fixed,
    attention_sol,
    attention_table,
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
from cozy_runtime.internal.attention_site import Site
from cozy_runtime.internal.attention_ulysses import BackendCall, Local
from cozy_runtime.internal.encoding.formats import Tensor
from cozy_runtime.internal.encoding.selection import DeviceFacts
from cozy_runtime.internal.parallel.commands import Choice

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ParallelConfig
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
    from diffusers.models.attention_processor import Attention
    from torch import nn

    from cozy_runtime.internal.producers import SiteModule

    class _Managed(Protocol):
        """A processor with Diffusers' per-site backend selection."""

        _attention_backend: object

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
#: A self-check's long keys: past Kitchen's and `_attn4`'s 1024, where they change tile.
LONG_KEYS = 1536


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
    #: what a call may carry besides one shape of Q/K/V: a mask, causal attention, causal
    #: attention at a query length other than the keys' (aligned bottom-right: the queries are
    #: the keys' last, as flash-attention aligns it), fewer K/V heads than Q heads, a query
    #: length other than the keys', several batch rows, and a softmax scale other than D^-1/2
    masks: bool = False
    causal: bool = True
    causal_cross: bool = False
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
    #: the lowest cell level it may be listed at (`TABLE`)
    level: int = 16
    #: a sparse kernel: the model's layout routes each call's rows (`attention_sol`)
    sparse: bool = False
    #: the dense kernels a sparse one computes rows with, opened beside it: its dense steps',
    #: then its protected prefix's; none: the table's 16-bit dense ladder (`_bind_dense`), so
    #: only a 16-bit sparse kernel may declare none
    dense: tuple[str, ...] = ()
    #: the dense kernel a sparse row's exact paths take (H3: the token refiner, block 1 and the
    #: blocks whose key gain has outlier channels); empty: the 16-bit dense ladder
    exact_rows: str = ""
    #: the Runtime module whose `KERNELS` (a `TritonKernels` set) it launches: it serves once
    #: they are built here (`triton_kernels`)
    kernels: str = ""
    #: the most tokens its compiled kernels take; zero is any
    max_tokens: int = 0

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
            for case in (
                (d, n, batch, query, 257, False),
                (d, n, batch, 257, 257, True),
                (d, n, 1, LONG_KEYS, LONG_KEYS, False),
            )
            if self.causal or not case[5]
        ]

    def refusal(self, query: Tensor, key: Tensor, options: attention_ulysses.Options) -> str:
        """Why it does not serve this call, or empty, read on every call: `_local`."""
        dtype, dim = str(query.dtype).removeprefix("torch."), int(query.shape[-1])
        scale = options.get("scale")
        limits = (
            (not self.admits(dtype, dim), f"{dtype} activations at head dimension {dim}"),
            (options.get("attn_mask") is not None and not self.masks, "an attention mask"),
            (bool(options.get("is_causal")) and not self.causal, "causal attention"),
            (
                bool(options.get("is_causal"))
                and query.shape[1] != key.shape[1]
                and not self.causal_cross,
                "causal attention at a query length other than K's",
            ),
            (query.shape[2] != key.shape[2] and not self.gqa, "fewer K/V heads than Q heads"),
            (query.shape[1] != key.shape[1] and not self.cross, "a query length other than K's"),
            (query.shape[0] != 1 and not self.batches, "more than one batch row"),
            (
                scale is not None
                and not self.scales
                and not math.isclose(scale, dim**-0.5, rel_tol=1e-6),
                "a softmax scale other than D^-1/2",
            ),
            (
                bool(self.max_tokens) and int(query.shape[1]) > self.max_tokens,
                f"{query.shape[1]} tokens: past its {self.max_tokens}",
            ),
        )
        return next((f"{self.name} does not serve {what}" for off, what in limits if off), "")


_GPU = frozenset({arch.MMA_SYNC, arch.WGMMA, arch.TCGEN05})
_MMA, _WGMMA, _TCGEN05 = (frozenset({f}) for f in (arch.MMA_SYNC, arch.WGMMA, arch.TCGEN05))
_FP8, _FP4, _INT4 = frozenset({"fp8"}), frozenset({"fp4"}), frozenset({"int4"})
#: Kitchen's INT8 MMA: every row but Blackwell Ultra's (sm103 has no int8 tensor cores).
_INT8 = frozenset({"int8"})
_HALF, _BF16, _D128 = frozenset({"bfloat16", "float16"}), frozenset({"bfloat16"}), frozenset({128})
_D64_128 = frozenset({64, 128})
#: What the wheel's FA3 serves (`prebuilt.FlashAttention3`); `flash-attn3-full` serves more.
_FA3 = frozenset((d, dtype) for d in (64, 128) for dtype in ("bfloat16", "float16"))
_PY, _KITCHEN, _SAGE3 = kernel_sources.PYTHON, "comfy-kitchen", "sageattn3"
#: SageAttention3's API over the wheel's `_sage3` (its group mean is the per-block-mean
#: candidate's Triton set); the source build's is `sageattn3.api`.
_SAGE3_WHEEL = "cozy_runtime.internal.sage3"
_E4M3 = "cozy_runtime.internal.attention_fp8"
#: A kernel the Runtime wheel alone carries (`prebuilt.ATTENTION`): no source recipe builds it.
PREBUILT = "prebuilt"
#: The fixed K means' compiled reduction (`attention_sage2_fixed_mean.REDUCE_MAX` tiles).
_MEAN_TOKENS = 2048 * 256


class _Admission(TypedDict):
    """A row's admission fields (`Candidate`), as every sparse kernel's states them."""

    sparse: bool
    dtypes: frozenset[str]
    max_head_dim: int
    head_dims: frozenset[int]
    causal: bool
    gqa: bool
    cross: bool
    batches: bool


#: Kitchen Sol's exact paths: Kitchen INT8, whose per-block scales carry those keys (H3 on an
#: RTX 5090: 0.008-0.030 rel L2, FP4 0.10-0.28).
_KITCHEN_EXACT = "kitchen-int8"
#: Every sparse kernel's admission: one BF16 document at head dimension 128, equal Q/K/V.
_SOL = _Admission(
    sparse=True, dtypes=_BF16, max_head_dim=128, head_dims=_D128, causal=False, gqa=False,
    cross=False, batches=False,
)  # fmt: skip

KERNELS: tuple[Candidate, ...] = (
    Candidate("sdpa", "_cozy_sdpa", upstream.SDPA, masks=True, causal_cross=True),
    Candidate(
        "cudnn", "_cozy_cudnn", upstream.CUDNN, _GPU, dtypes=_HALF, max_head_dim=128, masks=True,
    ),
    Candidate(
        "flash-attn3", "_cozy_flash_attn3", upstream.FA3, _WGMMA, dtypes=_HALF, max_head_dim=256,
        shapes=_FA3, causal_cross=True, artifact="flash-attn3-full",
        import_name="flash_attn_interface",
    ),
    Candidate(
        "flash-attn4", "_cozy_flash_attn4", upstream.FA4, _WGMMA | _TCGEN05, dtypes=_HALF,
        max_head_dim=256, causal_cross=True, artifact=_PY, import_name="flash_attn.cute",
    ),
    Candidate(
        "flash-attn4-fp8", "_cozy_flash_attn4_fp8", upstream.FA4_FP8, _TCGEN05, level=8,
        dtypes=_BF16, head_dims=_D64_128, max_head_dim=128, causal_cross=True, gqa=False,
        artifact=_PY, import_name="flash_attn.cute", kernels=_E4M3,
    ),
    Candidate(
        "sageattention", "_cozy_sage_int8_fp8", upstream.SAGE, _MMA | _WGMMA, _FP8, level=8,
        dtypes=_HALF, max_head_dim=128, artifact="sageattention", import_name="sageattention",
    ),
    Candidate(
        "kitchen-int8", "_cozy_kitchen_int8", upstream.KITCHEN, _GPU, _INT8, level=8, dtypes=_HALF,
        max_head_dim=256, masks=True, causal=False, artifact=_KITCHEN,
        import_name="comfy_kitchen",
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
        max_tokens=_MEAN_TOKENS,
    ),
    Candidate(
        "kitchen-int4qk", "_cozy_kitchen_int4qk", attention_int4.KITCHEN_INT4QK, _MMA, _INT4,
        level=4, dtypes=_HALF, head_dims=_D128, max_head_dim=128, causal=False,
        artifact=PREBUILT, import_name="cozy_runtime._kernels._attn4",
    ),
    Candidate(
        attention_sage2_fixed.DENSE_NAME, "_cozy_sage2_fixed_kmean", attention_sage2_fixed.DENSE,
        _WGMMA, level=8, dtypes=_BF16, max_head_dim=128, head_dims=_D128, causal=False,
        gqa=False, batches=False, artifact="sageattention", import_name="sageattention",
        kernels="cozy_runtime.internal.attention_sage2_fixed_mean", max_tokens=_MEAN_TOKENS,
    ),
    Candidate(
        "sol-attn", "_cozy_sol_bf16", attention_sol.SOL, _GPU, _FP8, artifact=_PY,
        import_name="sol_attn", **_SOL,
    ),
    Candidate(
        "sol-attn-fa4-fp8", "_cozy_sol_fa4_fp8", attention_sol.SOL_FP8, _TCGEN05, level=8,
        artifact=_PY, import_name="sol_attn", dense=("flash-attn4-fp8",), **_SOL,
    ),
    Candidate(
        attention_sage2_fixed.SOL_NAME, "_cozy_sol_sage2_fixed_kmean", attention_sage2_fixed.SOL,
        _WGMMA, level=8, artifact=_PY, import_name="sol_attn",
        dense=(attention_sage2_fixed.DENSE_NAME,), **_SOL,
    ),
    Candidate(
        "kitchen-sol", "_cozy_kitchen_sol_int8", attention_kitchen_sol.KITCHEN_SOL, _GPU, _INT8,
        level=8, artifact=_KITCHEN, import_name="comfy_kitchen", dense=("kitchen-int8",),
        exact_rows=_KITCHEN_EXACT, **_SOL,
    ),
    Candidate(
        "kitchen-sol-fp4-lowmem", "_cozy_kitchen_sol_fp4_lowmem",
        attention_kitchen_sol.KITCHEN_SOL_FP4_LOWMEM, _MMA, _FP4 | _INT8, level=4,
        artifact=_KITCHEN, import_name="comfy_kitchen",
        dense=("sageattention3-fp4-global-lowmem", "kitchen-int8"), exact_rows=_KITCHEN_EXACT,
        **_SOL,
    ),
    Candidate(
        "kitchen-sol-int4qk", "_cozy_kitchen_sol_int4qk", attention_kitchen_sol.KITCHEN_SOL_INT4QK,
        _MMA, _INT4 | _INT8, level=4, artifact=_KITCHEN, import_name="comfy_kitchen",
        dense=("kitchen-int4qk", "kitchen-int8"), exact_rows=_KITCHEN_EXACT, **_SOL,
    ),
    Candidate(
        "kitchen-sol-fp4-fixed-means", "_cozy_kitchen_sol_fp4_fixed_means",
        attention_kitchen_sol.KITCHEN_SOL_FP4_FIXED_MEANS, _MMA, _FP4 | _INT8, level=4,
        artifact=_KITCHEN, import_name="comfy_kitchen",
        dense=("sageattention3-fp4-fixed-means", "kitchen-int8"), exact_rows=_KITCHEN_EXACT,
        **_SOL,
    ),
)  # fmt: skip
BY_NAME: dict[str, Candidate] = {c.name: c for c in KERNELS}
#: A sparse kernel's undeclared rows run on the 16-bit dense ladder: only a 16-bit one may.
if _UNDECLARED := [c.name for c in KERNELS if c.sparse and not c.dense and not c.exact]:
    raise ValueError(f"quantizing sparse kernels without their dense rows: {_UNDECLARED}")
#: Kernels no longer here (Paul, 2026-11-04): a pin naming one runs the normal choice and warns,
#: where an unknown name refuses.
RETIRED = frozenset({"flash-attn3-fp8", "sageattention-fp16pv", "flashinfer-bf16-fp8"})


def retired(pin: str) -> str:
    """The removed kernel PIN names, else nothing."""
    name = pin.rpartition("=")[2]
    return name if name in RETIRED else ""



#: Each class's kernels per layout, attention level, head dimension and token bucket, fastest
#: first (`attention_table`): a site walks its level's cell, each level above, then SDPA. A
#: hand-made table lists no 4-bit kernel (Paul, 11-11: in H3 it loses small details, the
#: seagulls, for little speed), so a site whose model tolerates 4 bits serves 8-bit kernels;
#: only a measured table, past the 4-bit quality bar Paul signs, lists them (xs-171).
TABLE = attention_table.load(BY_NAME)
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
        local=Local(
            base.local.forward,
            base.local.impl,
            producer.context_parallel,
            narrow=base.local.narrow,
        ),
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


def _sol_objects(candidate: Candidate) -> bool:
    """A sparse kernel whose sparse steps run Sol's compiled CuTe objects (`attention_sol`)."""
    return candidate.sparse and candidate.import_name == "sol_attn"


def dense_identity(name: str) -> str:
    """The dense rows a sparse kernel recomputes with: its row's declared kernels and the exact
    one beside them, where undeclared the 16-bit dense ladder bound now."""
    candidate = known()[name]
    ladder, dense = attention_sol.dense_identity(), candidate.dense
    if not dense:
        return ladder
    prefix = f"; sparse-prefix={dense[-1]}" if dense[-1] != dense[0] else ""
    return f"{dense[0]}{prefix}; exact={candidate.exact_rows or ladder}"


def device_row(device: DeviceFacts | None) -> arch.Row:
    """DEVICE's row, with its half-rate accumulation (`arch.klass`); none off CUDA."""
    if device is None or device.kind != "cuda":
        return arch.NONE
    return replace(arch.row(device.sm), half_accumulate=device.half_accumulate)


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
    component-qualified prefix that names the site; unlisted sites 16. A level of 4 walks the
    8-bit cells, since no cell lists a 4-bit kernel (`DENSE`)."""
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


def walk(
    row: arch.Row, at: int, layout: bool = False, head_dim: int = 0, bucket: int = -1
) -> list[str]:
    """A site's kernels on ROW for one token BUCKET (-1: the last), best first: a LAYOUT site's
    sparse cells at level AT and each level above (its model asked for sparse attention), then
    the dense cells the same way, then SDPA, each producer standing in before its base; none the
    row cannot run."""
    levels = attention_table.LEVELS[attention_table.LEVELS.index(at) :]
    found, names, klass = known(), list[str](), arch.klass(row)
    for cells in ("sparse", "dense") if layout else ("dense",):
        for step in levels:
            names += TABLE.ranking(klass, cells, step, head_dim, bucket)
    stand_ins: dict[str, list[str]] = {}
    for name, producer in _PRODUCERS.items():
        stand_ins.setdefault(producer.base, []).append(name)
    names = [n for name in (*names, "sdpa") for n in (*stand_ins.get(name, ()), name)]
    return [n for n in dict.fromkeys(names) if n in found and not _off_row(found[n], row)]


#: An attention site: (component, module path, module, processor).
type _Record = tuple[str, str, SiteModule, object]


@dataclass(frozen=True, slots=True)
class Ready:
    """One candidate compiled, imported and launched on this device, with its backend's call
    (`attention_ulysses.function`), which Diffusers' member and a Runtime site both reach."""

    candidate: Candidate
    function: BackendCall
    #: The compiled artifact serving it: flat string fields, as an observation row carries them.
    artifact: dict[str, str] | None = None

    @property
    def name(self) -> str:
        return self.candidate.name


def _registry() -> type[_AttentionBackendRegistry]:
    """Diffusers' attention backend registry, or a typed absence."""
    try:
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    except ImportError as exc:
        raise AttentionRefusal(
            "attention_kernel_absent",
            f"this environment's diffusers exposes no attention backend registry: {exc}",
        ) from exc
    return _AttentionBackendRegistry


def _member(candidate: Candidate) -> AttentionBackendName:
    """The Diffusers member serving this candidate, registered on first use."""
    return attention_ulysses.member(candidate.backend, _local(candidate))


def _function(candidate: Candidate) -> BackendCall:
    """This candidate's backend call, with no Diffusers."""
    return attention_ulysses.function(candidate.backend, _local(candidate))


#: One site's selection by name (`Pick.spec`): per token band, its upper bound on `min(Lq, Lk)`
#: (0: none) and the kernels a call tries in turn, SDPA after them.
type Spec = tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class Pick:
    """One site's selection: per token band, its upper bound (0: none, the last) and the ready
    kernels a call tries in turn, the first unless its row refuses the call, SDPA after them.
    One band of one kernel dispatches to that kernel's own backend; any other is a ladder, a
    backend of its own (`_ladder`), whose call takes its band in `attention_ulysses._call`."""

    rungs: tuple[tuple[int, tuple[Ready, ...]], ...]

    @staticmethod
    def of(entry: Ready) -> Pick:
        return Pick(((0, (entry,)),))

    @property
    def spec(self) -> Spec:
        return tuple((upper, tuple(r.name for r in chain)) for upper, chain in self.rungs)

    @property
    def served(self) -> tuple[Ready, ...]:
        """The kernel each band serves, once each."""
        return tuple({chain[0].name: chain[0] for _, chain in self.rungs}.values())

    @property
    def kernels(self) -> tuple[Ready, ...]:
        """Every kernel a call may reach, once each."""
        return tuple({r.name: r for _, chain in self.rungs for r in chain}.values())

    @property
    def backend(self) -> str:
        """The backend its site dispatches to."""
        if len(self.rungs) == 1 and len(self.rungs[0][1]) == 1:
            return self.rungs[0][1][0].candidate.backend
        return _ladder(self)[0]


#: ladder backend -> its selection by name, for reading back what a site serves
_LADDERS: dict[str, Spec] = {}


def _ladder(pick: Pick) -> tuple[str, Local]:
    """PICK's backend and kernel: a ladder named by its spec, made once per process."""
    spec = pick.spec
    name = "_cozy_ladder_" + hashlib.sha256(repr(spec).encode()).hexdigest()[:12]
    if name not in _LADDERS:
        rungs = tuple((upper, *_chained(chain)) for upper, chain in pick.rungs)
        local = Local(
            _unbanded,
            lambda *_: name,
            head_local=all(_local(r.candidate).head_local for r in pick.kernels),
            rungs=rungs,
            band=_band if len(rungs) > 1 else None,
        )
        attention_ulysses.function(name, local)
        _LADDERS[name] = spec
    return name, attention_ulysses._LOCALS[name]


def _band(tokens: int) -> str:
    """A ladder call's band: its token bucket's label (`attention_table.Table.band`)."""
    return TABLE.band(TABLE.bucket(tokens))


def _unbanded(*_: object, **__: object) -> Tensor:
    raise RuntimeError("a ladder's call takes a band before its forward (`attention_ulysses`)")


def _chained(chain: tuple[Ready, ...]) -> tuple[str, Local]:
    """CHAIN's first kernel, whose refused call goes to the next, the last's to SDPA."""
    first = chain[0].candidate
    if len(chain) == 1:
        return first.backend, _local(first)
    return first.backend, replace(_local(first), instead=_chained(chain[1:]))


def _handle(processor: object, pick: Pick) -> str | AttentionBackendName:
    """What a site's processor dispatches by: a Runtime site its backend's name, a Diffusers
    processor Diffusers' member."""
    if len(pick.rungs) == 1 and len(pick.rungs[0][1]) == 1:
        candidate = pick.rungs[0][1][0].candidate
        return candidate.backend if isinstance(processor, Site) else _member(candidate)
    name, local = _ladder(pick)
    return name if isinstance(processor, Site) else attention_ulysses.member(name, local)


#: backend -> the kernel its Diffusers member calls (`_local`)
_LOCALS: dict[str, Local] = {}
#: The kernel a call another kernel's row refuses runs on, after the band's next ones: SDPA's
#: row refuses none.
_SDPA = "sdpa"
_INSTEAD = (BY_NAME[_SDPA].backend, BY_NAME[_SDPA].local)


def _local(candidate: Candidate) -> Local:
    """CANDIDATE's forward as every call reaches it: a call its row refuses (`refusal`) on SDPA
    instead, a sparse one's dense rows through its declared kernels' own."""
    found = _LOCALS.get(candidate.backend)
    if found is not None:
        return found
    forward = candidate.local.forward
    if candidate.dense:
        rows = [_dense_rows(BY_NAME[name]) for name in candidate.dense]
        named = candidate.exact_rows
        exact = _dense_rows(BY_NAME[named]) if named else attention_sol._dense
        forward = partial(forward, dense=rows[0], prefix_dense=rows[-1], exact=exact)
    instead = None if candidate.name == _SDPA else _INSTEAD
    found = _LOCALS[candidate.backend] = replace(
        candidate.local, forward=forward, refusal=candidate.refusal, instead=instead
    )
    return found


def _dense_rows(candidate: Candidate) -> attention_sol.DenseRows:
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
#: A site and what it is given: (component, module path, module, processor, selection).
type _Site = tuple[str, str, SiteModule, _Managed, Pick]
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
    if candidate.artifact == PREBUILT:
        why = (
            "its prebuilt kernel failed its check on this card"
            if candidate.name in _WHEEL_REFUSED
            else prebuilt.missing(candidate.name, device.sm)
        )
        raise AttentionRefusal("attention_kernel_absent", f"{candidate.name}: {why}")
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

    def call(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        out: Tensor = ready.function(q, k, v, is_causal=causal)
        return out

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
    if (
        prebuilt.ATTENTION.get(candidate.name) != "_fa3"
        or device.kind != "cuda"
        or not prebuilt.serves(candidate.name, device.sm)
    ):
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
    return Ready(candidate, _function(candidate), {"artifact": "prebuilt", "key": digest})


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
    if candidate.producer is not None:
        base = (_waited if wait else _opened)(BY_NAME[candidate.producer.base], device)
        kernel = _import(base.candidate)
        missing = [n for n in candidate.producer.requires if not callable(getattr(kernel, n, None))]
        if missing:
            raise AttentionRefusal(
                "attention_kernel_absent",
                f"{candidate.name}: {base.candidate.import_name} lacks {', '.join(missing)}",
            )
        return Ready(candidate, _function(candidate), base.artifact)
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
        if candidate.exact_rows not in ("", *candidate.dense):
            (_waited if wait else _opened)(BY_NAME[candidate.exact_rows], device)
    else:
        artifact = _compiled(candidate, device, wait) if candidate.artifact else None
    wheel = (artifact or {}).get("artifact") == "prebuilt"
    if wheel:
        if candidate.import_name == _SAGE3:
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
        if candidate.local is upstream.FA3:
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
    ready = Ready(candidate, _function(candidate), artifact)
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


def _expected(live_tokens: int, module: nn.Module) -> None:
    """A model's early length statement reaches Sol when a Sol site serves `module`."""
    found = known()
    if not any(_sol_objects(found[n]) for n in observed({"": module}).totals() if n in found):
        return
    import torch

    attention_sol.expect(live_tokens, torch.device(str(placed_device())))


_EXPECTED.append(_expected)


def _called(seen: attention_ulysses.Seen) -> dict[str, set[str]]:
    """Each kernel this rank's calls reached in SEEN, and the bands they fell in."""
    names = _names()
    called: dict[str, set[str]] = {names.get(b, b): set() for b in seen}
    for (_site, band), backends in seen.bands.items():
        for backend in backends:
            called.setdefault(names.get(backend, backend), set()).add(band)
    return called


def evidence(seen: attention_ulysses.Seen | None) -> list[dict[str, object]]:
    """This rank's kernels for a run's execution record: every kernel its calls reached, with
    the bands they fell in, and every kernel of its chains it did not serve and why (compiling,
    failed, absent, unsupported). A kernel that took calls another's row refused counts them,
    per reason (`attention_ulysses.Seen`)."""
    called = _called(seen) if seen is not None else {}
    if not called:
        return []  # no attention site ran here
    states = {name: kernel_compile.State(name, "ready") for name in called}
    states.update(_STATES)
    names, took = _names(), dict[str, list[str]]()
    for (reason, backend), calls in sorted((seen.refused if seen else {}).items()):
        took.setdefault(names.get(backend, backend), []).append(f"{reason} x{calls}")
    for name, reasons in took.items():
        states[name] = kernel_compile.State(name, "ready", detail="; ".join(reasons))
    rows: list[dict[str, object]] = []
    for name, state in sorted(states.items()):
        row: dict[str, object] = {**state.document(), "kernel": name, "served": name in called}
        if any(called.get(name, ())):
            row["bands"] = sorted(called[name] - {""}) or [""]
        rows.append(row)
    return rows


def served_sites(
    applied: Applied | None, seen: attention_ulysses.Seen | None
) -> dict[str, list[execution_evidence.SiteGroup]]:
    """A run's sites by kind (`Applied.groups`), with the kernel each band was served by in its
    calls (SEEN); a group none of whose calls ran here has no bands."""
    if applied is None or seen is None:
        return {}
    names = _names()
    banded: dict[str, dict[str, set[str]]] = {}
    for (dispatched, band), backends in seen.bands.items():
        held = banded.setdefault(dispatched, {}).setdefault(band, set())
        held.update(names.get(b, b) for b in backends)
    return _site_document(
        applied.groups,
        lambda group: {
            band: "+".join(sorted(kernels))
            for band, kernels in sorted(banded.get(group.backend, {}).items())
        },
    )


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


def _dense_pick(
    device: DeviceFacts, row: arch.Row, resolved: dict[str, Ready], refused: dict[str, str]
) -> Pick:
    """The 16-bit dense ladder a sparse kernel's undeclared rows run on (`Candidate.dense`,
    `exact_rows`: BF16 Sol's dense steps, protected prefix and dense paths, every sparse kernel's
    exact paths): per token bucket, the first kernel of the table's 16-bit dense cell for ROW's
    class at head dimension 128 that serves BF16 rows of any query length here, else SDPA. It
    runs locally after the Ulysses exchange, so it needs no context parallelism. One still
    compiling is passed over and selected again once ready."""
    found, rungs = known(), list[tuple[int, tuple[Ready, ...]]]()
    for bucket in TABLE.buckets():
        chosen: Ready | None = None
        for name in walk(row, 16, False, 128, bucket):
            candidate = found.get(name)
            if (
                candidate is None
                or candidate.sparse
                or _off_row(candidate, row)
                or not candidate.admits("bfloat16", 128)
                or not candidate.cross
                or name in refused
            ):
                continue
            if name not in resolved:
                try:
                    resolved[name] = _opened(candidate, device)
                except Exception as exc:
                    refused.setdefault(name, f"{type(exc).__name__}: {exc}"[:300])
                    continue
            chosen = resolved[name]
            break
        if chosen is None:
            raise AttentionRefusal("attention_kernel_absent", "Sol has no ready dense reference")
        if rungs and rungs[-1][1][0].name == chosen.name:
            rungs[-1] = (bucket, (chosen,))
        else:
            rungs.append((bucket, (chosen,)))
    edges = TABLE.tokens
    return Pick(tuple((edges[b] if b < len(edges) else 0, chain) for b, chain in rungs))


#: A bound dense ladder's identity -> its selection by name, for a group's choice
_DENSE_SPECS: dict[str, Spec] = {}


def _bind_dense(pick: Pick) -> None:
    """PICK as the 16-bit dense rows (`attention_sol.bind_dense`): a call picks its band by
    `min(Lq, Lk)`, so a protected prefix and a refiner's short document take their own."""
    backend = pick.backend
    local = attention_ulysses._LOCALS[backend] if backend in _LADDERS else _local(
        pick.rungs[0][1][0].candidate
    )

    def rows(*, query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
        options = attention_ulysses.Options(
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False
        )
        return attention_ulysses._call(backend, local, query, key, value, options)

    identity = spelled(pick.spec)
    _DENSE_SPECS[identity] = pick.spec
    attention_sol.bind_dense(rows, identity)


def bind_dense(device: DeviceFacts) -> str:
    """Bind the 16-bit dense rows a sparse kernel's undeclared rows run on here, as a selection
    with a sparse site does, and say which: what calls a sparse kernel outside a selection (the
    attention bench) binds before calling it."""
    _bind_dense(_dense_pick(device, device_row(device), {}, {}))
    return attention_sol.dense_identity()


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


def _trees(roots: Mapping[str, object]) -> Iterator[tuple[str, nn.Module]]:
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
            yield label, cast("nn.Module", value)  # a module tree, as it walks
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
    """Every Runtime site (`attention_site.Site`) and Diffusers attention processor, including
    Diffusers' with no backend dispatch. An environment without Diffusers has only the first."""
    trees = list(_trees(roots))
    if not trees:  # a weightless call has none, and imports no Diffusers (or torch) to say so
        return
    classes: tuple[type, ...] = ()
    with suppress(ImportError):
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_processor import Attention, MochiAttention

        classes = (Attention, MochiAttention, AttentionModuleMixin)
    seen: set[int] = set()
    for component, tree in trees:
        for path, module in tree.named_modules():
            processor = getattr(module, "processor", None)
            if id(module) in seen or processor is None:
                continue
            if isinstance(processor, Site) or isinstance(module, classes):
                seen.add(id(module))
                yield component, path, cast("SiteModule", module), processor  # it holds one


class ClassicSdpa:
    """Diffusers' `AttnProcessor2_0`, the same math, through the per-site backend contract.

    Classic `Attention` sites (SDXL's UNet, AutoencoderKL) call torch SDPA directly, so no
    kernel could be selected, pinned or observed on them. `adopt` gives each one this
    processor. Auto selection keeps SDPA there, bit-identical to before; a pin may choose
    another kernel.
    """

    _attention_backend: AttentionBackendName | None = None
    _parallel_config: ParallelConfig | None = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        temb: Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
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


def _site_records(roots: Mapping[str, object]) -> Iterator[tuple[str, str, SiteModule, _Managed]]:
    """Only processors implementing Diffusers' per-site backend selection contract."""
    for component, path, module, processor in _attention_records(roots):
        if hasattr(processor, "_attention_backend"):
            yield component, path, module, processor


def _unmanaged_reason(component: str, path: str, processor: object) -> str:
    return (
        f"{component}/{path} uses {type(processor).__name__}, which has no per-site "
        "attention backend selection"
    )


def _sites(roots: Mapping[str, object]) -> Iterator[tuple[str, SiteModule, _Managed]]:
    for component, _path, module, processor in _site_records(roots):
        yield component, module, processor


def sites(roots: Mapping[str, object]) -> list[tuple[str, object, object]]:
    """Every runtime-selectable attention site: (component, module, processor)."""
    return list(_sites(roots))


def _head_dim(module: object) -> int:
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


def _projection_dtype(projection: object) -> str:
    """Encoded output contract, or an ordinary floating Linear's result dtype."""
    import torch

    # PEFT returns the base result's dtype, even with differently typed factors.
    projection = _base(projection)
    declared = getattr(projection, "out_dtype", None)
    if declared is None and type(projection) is torch.nn.Linear:
        declared = projection.weight.dtype
    name = str(declared).removeprefix("torch.")
    return name if name in _FLOATS else ""


def _dtype(module: object) -> str:
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
    #: each component's sites by kind and the backend they dispatch to (`_groups`)
    groups: tuple[Group, ...] = ()

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
        groups = self.groups + tuple(g for g in later.groups if g not in self.groups)
        return replace(
            self, hosts=hosts, artifacts=artifacts or None, fallbacks=fallbacks, groups=groups
        )

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for kernels in self.hosts.values():
            for name, sites in kernels.items():
                out[name] = out.get(name, 0) + sites
        return out

    def document(self) -> execution_evidence.Boot:
        names = _names()
        return _boot(
            self.totals(),
            {name: dict(kernels) for name, kernels in self.hosts.items()},
            list(self.offered),
            self.pin,
            self.artifacts or {},
            self.skipped or {},
            self.fallbacks,
            _site_document(self.groups, lambda group: _labelled(_spec(group.backend, names))),
        )

    def line(self) -> str:
        """The ONE line a pod prints at boot. It is the whole per-run attention record."""
        return self.document()["line"]


@dataclass(frozen=True, slots=True)
class Group:
    """A component's sites that are alike (layout, level, head dimension: their table cell) and
    dispatch to one backend, and how many."""

    component: str
    layout: str
    level: int
    head_dim: int
    backend: str
    sites: int = 0


def _labelled(spec: Spec) -> dict[str, str]:
    """SPEC's band labels (`attention_table.Table.band`, over its merged bands) and the kernel
    each serves; one band is ""."""
    if len(spec) == 1:
        return {"": spec[0][1][0]}
    out, lower = {}, 0
    for upper, chain in spec:
        if not lower:
            out[f"<{upper}"] = chain[0]
        else:
            out[f">={lower}" if not upper else f"{lower}-{upper - 1}"] = chain[0]
        lower = upper
    return out


def _site_document(
    groups: Sequence[Group], bands: Callable[[Group], dict[str, str]]
) -> dict[str, list[execution_evidence.SiteGroup]]:
    out: dict[str, list[execution_evidence.SiteGroup]] = {}
    found = known()
    for group in sorted(groups, key=lambda g: (g.component, g.layout, g.level, g.head_dim)):
        served = bands(group)
        row = execution_evidence.SiteGroup(
            layout=group.layout,
            level=group.level,
            head_dim=group.head_dim,
            sites=group.sites,
            bands=served,
        )
        producers = {
            name: producer.base
            for kernels in served.values()
            for name in kernels.split("+")
            if name in found and (producer := found[name].producer) is not None
        }
        if producers:
            row["producers"] = producers
        out.setdefault(group.component, []).append(row)
    return out


def line(
    totals: Mapping[str, int],
    hosts: Mapping[str, Mapping[str, int]],
    offered: Sequence[str],
    pin: str,
    skipped: Collection[str] = (),
    sites: Mapping[str, Sequence[execution_evidence.SiteGroup]] | None = None,
) -> str:
    """The boot line, from the facts alone — so a GROUP prepare renders the same sentence from
    its merged facts as a single one does from its own `Applied` (cr-124). A component whose
    sites serve by token band reads `flash-attn3 <12288, sageattention >=12288`."""
    ignored = f"; skipped {sorted(skipped)}" if skipped else ""
    if not totals:
        none = "NONE — this construction holds no backend-selectable diffusers attention site"
        return none + ignored
    chosen = ", ".join(f"{name} x{sites}" for name, sites in sorted(totals.items()))

    def held(component: str, kernels: Mapping[str, int]) -> str:
        banded = [
            ", ".join(f"{kernel} {band}" for band, kernel in group["bands"].items())
            for group in (sites or {}).get(component, ())
            if len(group["bands"]) > 1
        ]
        return " | ".join(dict.fromkeys(banded)) if banded else "+".join(sorted(kernels))

    where = "; ".join(
        f"{component}: {held(component, kernels)}" for component, kernels in sorted(hosts.items())
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
    sites: dict[str, list[execution_evidence.SiteGroup]] | None = None,
) -> execution_evidence.Boot:
    # RENDERED HERE, not by the worker: one renderer for the one line, in the module that
    # knows what it means.
    boot = execution_evidence.Boot(
        kernels=kernels,
        hosts=hosts,
        offered=offered,
        pin=pin,
        line=line(kernels, hosts, offered, pin, skipped, sites),
    )
    if kernels:
        boot["table"] = TABLE.identity
    if sites:
        boot["sites"] = sites
    if fallbacks:
        boot["fallbacks"] = list(fallbacks)
        boot["line"] += "; " + "; ".join(
            f"{row['component']}: selected {row['selected']} instead of "
            f"{row['preferred']}{f' at {band}' if (band := row.get('band')) else ''}: "
            f"{row['reason']}"
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
    sites: dict[str, list[execution_evidence.SiteGroup]] = {}
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
        for component, groups in row.get("sites", {}).items():
            sites[f"{parameter}/{component}"] = groups
        for name, count in row["kernels"].items():
            kernels[name] = kernels.get(name, 0) + count
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
    return _boot(kernels, hosts, offered, pin, artifacts, skipped, fallbacks, sites)


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
    if degree > 1 and not _local(candidate).head_local:
        return f"{candidate.name} cannot shard sequence-parallel degree {degree}"
    if isinstance(getattr(module, "processor", None), Site):
        return ""
    try:
        if _member(candidate) not in _registry()._backends:
            return f"{candidate.name} is not registered with Diffusers"
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


def _tail(
    order: Sequence[str],
    first: Ready,
    row: arch.Row | None,
    device: DeviceFacts | None,
    module: object,
    degree: int,
    resolved: dict[str, Ready],
    refused: dict[str, str],
) -> tuple[Ready, ...]:
    """The dense kernels after FIRST in its band's ORDER that a call FIRST's row refuses tries in
    turn, SDPA after them: only ones ready here now (torch's own, or opened already), never one
    this would start building."""
    found, dtype, head_dim = known(), _dtype(module), _head_dim(module)
    tail: list[Ready] = []
    for name in order[order.index(first.name) + 1 :]:
        candidate = found.get(name)
        if (
            name == _SDPA
            or candidate is None
            or candidate.sparse
            or name in refused
            or _unfit(candidate, row, module, dtype, head_dim, degree)
        ):
            continue
        if name not in resolved:
            if device is None:
                continue
            if candidate.artifact or candidate.import_name:
                opened: Ready | None = kernel_ready.ready(_slot(name, device))
                if opened is None:
                    continue
                resolved[name] = opened
            else:
                try:
                    resolved[name] = _opened(candidate, device)
                except AttentionRefusal:
                    continue
        tail.append(resolved[name])
    return tuple(tail)


def _site_pick(
    orders: Sequence[Sequence[str]],
    row: arch.Row | None,
    device: DeviceFacts | None,
    component: str,
    module: object,
    degree: int,
    resolved: dict[str, Ready],
    refused: dict[str, str],
    fallbacks: list[execution_evidence.AttentionFallback],
) -> Pick | None:
    """A site's selection from its ORDERS, one per token bucket (`walk`): each bucket's first
    kernel that serves here (`_first`) and the ready kernels after it (`_tail`), adjacent equal
    buckets merged. At most one package producer per site (it replaces the site's processor): a
    bucket whose first is another producer takes that producer's base."""
    rungs: list[tuple[int, tuple[Ready, ...]]] = []
    passed: dict[tuple[str, str, str, str], list[int]] = {}
    producer = ""
    for bucket, order in enumerate(orders):
        held: list[execution_evidence.AttentionFallback] = []
        names = list(order)
        while True:
            held.clear()
            first = _first(names, row, device, component, module, degree, resolved, refused, held)
            if first is None:
                return None
            if first.candidate.producer is None or producer in ("", first.name):
                break
            names.remove(first.name)
        producer = producer or (first.name if first.candidate.producer is not None else "")
        chain = (first, *_tail(names, first, row, device, module, degree, resolved, refused))
        for row_ in held:
            key = (row_["component"], row_["preferred"], row_["selected"], row_["reason"])
            passed.setdefault(key, []).append(bucket)
        if rungs and [r.name for r in rungs[-1][1]] == [r.name for r in chain]:
            rungs[-1] = (bucket, chain)
        else:
            rungs.append((bucket, chain))
    edges = TABLE.tokens
    pick = Pick(tuple((edges[b] if b < len(edges) else 0, chain) for b, chain in rungs))
    for (component_, preferred, selected, reason), buckets in passed.items():
        fallback = execution_evidence.AttentionFallback(
            component=component_, preferred=preferred, selected=selected, reason=reason
        )
        if len(buckets) < len(orders):
            fallback["band"] = ",".join(TABLE.band(b) for b in buckets)
        if fallback not in fallbacks:
            fallbacks.append(fallback)
    return pick


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
    selection that GPU serves there, and sparse kernels its 16-bit dense rows, each kernel
    resolved on DEVICE as a pin is, or refuses. Every other site takes, per token bucket, the
    first kernel of its walk on ROW (else DEVICE's) at the model's `bits` for it (`walk`,
    `level`; `_site_pick`); a classic site keeps SDPA. Without a DEVICE, sites take the READY
    kernels in order.
    """
    scope, name = _pin_parts(pin)
    if device is not None:
        row = row or device_row(device)
    hosts: dict[str, dict[str, int]] = {}
    assignments: list[_Site] = []
    resolved = {entry.name: entry for entry in ready}
    taken = {} if choice is None else {(c, path): spec for c, path, spec in choice.sites}
    if choice is not None:
        if device is None:
            raise AttentionRefusal("attention_kernel_unsupported", "a choice installs on a device")
        lacking: dict[str, str] = {}
        specs = (*taken.values(), choice.dense)
        named = {k for spec in specs for _, chain in spec for k in chain}
        for kernel in sorted(named):
            try:
                resolved[kernel] = pinned(kernel, device, wait=False)[0]
            except AttentionRefusal as exc:
                lacking[kernel] = exc.code
        if lacking:
            raise Lacking(lacking)
    refused: dict[str, str] = dict(skipped or {})
    matched = False
    fallbacks: list[execution_evidence.AttentionFallback] = []
    for record in _attention_records(roots):
        component, path, module, processor = record
        forced = bool(name and _matches(component, scope))
        matched |= forced
        pick: Pick | None = None
        first = "no available attention kernel"
        if choice is None and hasattr(processor, "_attention_backend"):
            # What the site is, for the records (`_groups`): its layout, level and head dim.
            at = 16 if forced and name == EXACT else level(bits, component, path)
            vars(module)["_cozy_attention_site"] = (sparse_site(module), at, _head_dim(module))
        if forced and name != EXACT:
            reason = _pin_reason(resolved.get(name), f"{name} is not open here", record, degree)
            if reason:
                raise AttentionRefusal(
                    "attention_kernel_unsupported",
                    f"{reason}. A pinned kernel is never quietly replaced",
                )
            pick = Pick.of(resolved[name])
        elif not hasattr(processor, "_attention_backend"):
            if forced:
                reason = _unmanaged_reason(component, path, processor)
                raise AttentionRefusal("attention_kernel_unsupported", reason)
            continue
        elif choice is not None:
            spec = taken.get((component, path), ())
            if not spec:
                raise AttentionRefusal(
                    "attention_kernel_unsupported",
                    f"the choice names no kernel for {component}/{path}",
                )
            first = spec[0][1][0]
            if not any(_pin_reason(resolved[k], "", record, degree) for _, c in spec for k in c):
                pick = Pick(tuple((upper, tuple(resolved[k] for k in c)) for upper, c in spec))
        else:
            at = 16 if forced else level(bits, component, path)
            orders = (
                [_walk_order(row, at, module, processor, b) for b in TABLE.buckets()]
                if row is not None
                else [[entry.name for entry in ready]]
            )
            pick = _site_pick(
                orders, row, device, component, module, degree, resolved, refused, fallbacks
            )
        if pick is None:
            detail = _site_refusal(first, component, _dtype(module), _head_dim(module))
            if degree > 1:
                detail = AttentionRefusal(
                    detail.code,
                    f"{detail}; sequence-parallel degree {degree} requires context parallelism",
                )
            raise detail
        # Managed: a pin refuses an unmanaged processor (`_pin_reason`), the walk passes it by.
        assignments.append((component, path, module, cast("_Managed", processor), pick))
        for entry in pick.served:
            hosts.setdefault(component, {})
            hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
    if name and not matched:
        raise AttentionRefusal(
            "attention_kernel_unsupported", f"{pin!r} matches no attention component"
        )
    sparse = any(e.candidate.sparse for *_, pick in assignments for e in pick.served)
    if choice is not None and sparse:
        if not choice.dense:
            raise AttentionRefusal("attention_kernel_absent", "the choice names no dense reference")
        _bind_dense(Pick(tuple((u, tuple(resolved[k] for k in c)) for u, c in choice.dense)))
    elif device is not None and row is not None and sparse:
        _bind_dense(_dense_pick(device, row, resolved, refused))
    _install_sites(
        [
            (pick, component, path, module)
            for component, path, module, _processor, pick in assignments
        ]
    )
    for _component, _path, _module, processor, pick in assignments:
        processor._attention_backend = _handle(processor, pick)
    if skipped is not None:
        skipped.update(refused)
    artifacts = {name: entry.artifact for name, entry in resolved.items() if entry.artifact}
    waiting = tuple(
        n for n in sorted(refused) if device is not None and kernel_ready.opening(_slot(n, device))
    )
    return Applied(
        hosts,
        tuple(resolved),
        pin,
        artifacts or None,
        refused or None,
        waiting,
        tuple(fallbacks),
        _groups(roots, pin),
    )


def _walk_order(
    row: arch.Row, at: int, module: object, processor: object, bucket: int = -1
) -> list[str]:
    """A site's kernels at level AT in one token BUCKET: a classic site keeps SDPA, every
    other one walks."""
    if isinstance(processor, ClassicSdpa):
        return ["sdpa"]
    return walk(row, at, sparse_site(module), _head_dim(module), bucket)


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


def _install_sites(records: Sequence[tuple[Pick, str, str, SiteModule]]) -> None:
    """Install metadata hooks atomically, preserving hooks owned by preparation: Sol's site
    where any band serves a sparse kernel, a package producer's processor where one serves."""
    # Validate all producer topology/hooks before any metadata or processor mutation.
    for pick, _component, _path, module in records:
        for entry in pick.served:
            if entry.candidate.producer is not None:
                entry.candidate.producer.validate(module)
    added: list[tuple[str, str, SiteModule]] = []
    processors: list[tuple[SiteModule, object]] = []
    try:
        for pick, component, path, module in records:
            sparse = [e.candidate for e in pick.served if e.candidate.sparse]
            if not sparse:
                continue
            local = component.rsplit("/", 1)[-1]
            if getattr(module, "_cozy_sol_site", None) is None:
                added.append((local, path, module))
            attention_sol.install_site(module, local, path)
            for candidate in sparse:
                if candidate.producer is not None:
                    processors.append((module, module.processor))
                    producers.install(
                        module,
                        candidate.producer,
                        candidate.backend,
                        candidate.import_name,
                        local,
                        path,
                    )
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


def _backend(processor: object) -> str:
    """The backend a site's processor dispatches to now: a kernel's or a ladder's."""
    backend = getattr(processor, "_attention_backend", None)
    return str(getattr(backend, "value", backend))


def _spec(backend: str, names: Mapping[str, str]) -> Spec:
    """What a site dispatching to BACKEND serves, by name (`Pick.spec`)."""
    return _LADDERS.get(backend) or ((0, (names.get(backend, backend),)),)


def _served(spec: Spec) -> tuple[str, ...]:
    """The kernel each band of SPEC serves, once each."""
    return tuple(dict.fromkeys(chain[0] for _, chain in spec))


def spelled(spec: Spec) -> str:
    """SPEC in a line: `flash-attn3 <12288, sageattention`, a refused call's next ones after
    `>`; one kernel alone."""
    return ", ".join(
        f"{'>'.join(chain)}{f' <{upper}' if upper else ''}" for upper, chain in spec
    )


def _groups(roots: Mapping[str, object], pin: str = "") -> tuple[Group, ...]:
    """Each component's sites by what they are (layout, level, head dim) and the backend they
    dispatch to, counted. A site `install` never selected (a follower's) is not here; a request
    at 16 bits (`EXACT`) is level 16 where it reaches."""
    scope, name = _pin_parts(pin)
    counted: dict[Group, int] = {}
    for component, _path, module, processor in _site_records(roots):
        facts = vars(module).get("_cozy_attention_site")
        if facts is None:
            continue
        sparse, at, head_dim = facts
        if name == EXACT and _matches(component, scope):
            at = 16
        group = Group(component, "sparse" if sparse else "dense", at, head_dim, _backend(processor))
        counted[group] = counted.get(group, 0) + 1
    return tuple(replace(group, sites=sites) for group, sites in counted.items())


def observed(roots: Mapping[str, object], pin: str = "") -> Applied:
    """Read actual processor choices; never substitute the prepared recommendation."""
    names = _names()
    hosts: dict[str, dict[str, int]] = {}
    for component, _module, processor in _sites(roots):
        for name in _served(_spec(_backend(processor), names)):
            hosts.setdefault(component, {})
            hosts[component][name] = hosts[component].get(name, 0) + 1
    served = {name for kernels in hosts.values() for name in kernels}
    artifacts = {
        ready.name: ready.artifact
        for ready in cast("list[Ready]", kernel_ready.served(_SLOTS))
        if ready.name in served and ready.artifact
    }
    return Applied(hosts, (), pin, artifacts=artifacts or None, groups=_groups(roots, pin))


def chosen(roots: Mapping[str, object]) -> Choice:
    """What this GPU serves now: every site's selection, and the 16-bit dense rows where a
    sparse kernel serves (`attention_sol.binding`)."""
    names = _names()
    sites = tuple(
        (component, path, _spec(_backend(processor), names))
        for component, path, _module, processor in _site_records(roots)
    )
    found = known()
    if not any(
        kernel in found and found[kernel].sparse for *_, spec in sites for kernel in _served(spec)
    ):
        return Choice(sites=sites)
    return Choice(sites=sites, dense=_DENSE_SPECS.get(attention_sol.dense_identity(), ()))


def difference(ours: Choice, theirs: Choice) -> str:
    """The first way THEIRS serves other than OURS; "" when it serves the same."""
    if ours.dense != theirs.dense:
        return (
            f"the 16-bit dense rows {spelled(ours.dense) if ours.dense else 'none'} "
            f"against {spelled(theirs.dense) if theirs.dense else 'none'}"
        )
    held = {(component, path): spec for component, path, spec in theirs.sites}
    for component, path, spec in ours.sites:
        other = held.pop((component, path), None)
        if other != spec:
            theirs_ = spelled(other) if other else "none"
            return f"{component}/{path}: {spelled(spec)} against {theirs_}"
    if held:
        (component, path), spec = sorted(held.items())[0]
        return f"{component}/{path}: none against {spelled(spec)}"
    return ""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Prepared processor choices, restored without resolution or policy re-execution, with
    the 16-bit dense rows bound when one of them is sparse."""

    choices: tuple[tuple[_Managed, object], ...]
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
        kernel in found and found[kernel].sparse
        for processor in held
        for kernel in _served(_spec(_backend(processor), names))
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
        for component, path, module, _, pick in changed
        if any(e.candidate.sparse for e in pick.served)
        and getattr(module, "_cozy_sol_site", None) is None
    ]
    temporary_processors = [
        (module, module.processor)
        for _, _, module, _, pick in changed
        if any(e.candidate.producer for e in pick.served)
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
    shape = f"{dtype or 'unknown'} activations at head dimension {head_dim or 'unknown'}"
    if name == "no available attention kernel":
        return AttentionRefusal(
            "attention_kernel_unsupported",
            f"no attention kernel on this machine serves {component}'s attention ({shape}): that "
            "is a bug in the package or in cozy, not in your request; try another lane with `--lane`",
        )
    return AttentionRefusal(
        "attention_kernel_unsupported",
        f"{name} does not serve {component}'s attention: {shape}. A pinned kernel is never "
        "quietly replaced: pin another, or none",
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

    #: (component, path, module, processor, selection) per site it changes
    sites: tuple[_Site, ...]
    pin: str = ""
    #: sites in scope that keep their prepared kernel, and why the pin could not run there
    fallbacks: tuple[execution_evidence.AttentionFallback, ...] = ()
    #: the 16-bit dense rows for this request (a sparse kernel serves), restored after it
    dense: Pick | None = None

    def apply(self) -> Applied:
        _install_sites([(pick, c, path, module) for c, path, module, _, pick in self.sites])
        if self.dense is not None:
            _bind_dense(self.dense)
        hosts: dict[str, dict[str, int]] = {}
        for component, _, _, processor, pick in self.sites:
            processor._attention_backend = _handle(processor, pick)
            for entry in pick.served:
                hosts.setdefault(component, {})
                hosts[component][entry.name] = hosts[component].get(entry.name, 0) + 1
        served = {e.name: e for *_, pick in self.sites for e in pick.kernels}
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
    if degree > 1 and not _local(candidate).head_local:
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
            "attention_kernel_unsupported", f"{name} matches no attention site"
        )
    if backend == EXACT:  # a compiled module keeps the kernel it was compiled with
        return exact(device, [] if _baked_in(selected) else records, name, degree)
    candidate = known()[backend]
    held = observed(selected).hosts
    why = _baked_in(selected)
    why = why and f"{why}; a compiled module keeps the kernel it was compiled with"
    entry = None
    if not why and degree > 1 and not _local(candidate).head_local:
        why = f"{backend} cannot shard sequence-parallel degree {degree}"
    if not why:
        try:
            entry = pinned(backend, device)[0]
        except AttentionRefusal as exc:
            if exc.code in ("attention_kernel_unknown", "attention_kernel_needs_executor"):
                raise
            why = str(exc)
    # A pinned sparse kernel's undeclared rows are the table's 16-bit dense ladder, resolved for
    # this request (route by rule), never whatever an earlier selection here bound.
    sparse = entry is not None and entry.candidate.sparse
    dense = _dense_pick(device, device_row(device), {}, {}) if sparse else None
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
        # Managed: `_pin_reason` refuses an unmanaged processor.
        sites.append((component, path, module, cast("_Managed", processor), Pick.of(entry)))
    return Swap(tuple(sites), name, tuple(fallbacks), dense)


def exact(
    device: DeviceFacts,
    records: Sequence[_Record],
    pin: str = EXACT,
    degree: int = 1,
) -> Swap:
    """A request at 16 bits: each site its walk at level 16, per token bucket; a sparse site
    keeps its sparse kernel, whose dense rows go exact too."""
    row = device_row(device)
    resolved: dict[str, Ready] = {}
    refused: dict[str, str] = {}
    sites: list[_Site] = []
    fallbacks: list[execution_evidence.AttentionFallback] = []
    for component, path, module, processor in records:
        if not hasattr(processor, "_attention_backend"):
            continue
        orders = [_walk_order(row, 16, module, processor, b) for b in TABLE.buckets()]
        pick = _site_pick(
            orders, row, device, component, module, degree, resolved, refused, fallbacks
        )
        if pick is not None:
            # Managed: the loop passes an unmanaged processor by, above.
            sites.append((component, path, module, cast("_Managed", processor), pick))
    sparse = any(e.candidate.sparse for *_, pick in sites for e in pick.served)
    dense = _dense_pick(device, row, resolved, refused) if sparse else None
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
