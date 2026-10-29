"""Upstream BF16 Sol-Attn with explicit document and conditioning semantics.

No sparse kernel is implemented here. Sparse block selection is approximate;
protected query rows are recomputed with the selected dense reference. The
model supplies its live layout, step policy and dense module paths explicitly.

The kernel compiles once per object per machine, out of process (`kernel_compile`): one per
card, and on SM90 one per 64-token block layout (`variant`), which only the request knows. So
the first sight of a length (a model's intake statement, or the first dense step) submits its
compile, and conditioning and the dense steps cover it. A sparse step never waits for it: while
the object builds, the step runs the dense reference (`dense_building`), and the object serves
from the step boundary after it is ready (`kernel_ready`); the run's kernel rows say from which
step. Upstream's preprocess kernels take the length, the head count and the block count as
runtime integers here (`kernel_sources`' Python recipe fixes their source): one build per card
(`preprocess`, a `triton_kernels.warm_job`, at machine start with the card's object) serves every
length and head window; a step runs the dense reference only while that one build runs, and for
good when it failed. Their autotuning is pinned to one configuration, because its candidates differ
bitwise and a timing race must not choose the output.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from importlib import import_module
from inspect import Parameter, Signature, signature
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as Device
    from torch.nn import Module

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._attention import _request
from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.author._observations import Scalar
from cozy_runtime.internal import accel, arch, config, kernel_cache, kernel_compile, kernel_ready
from cozy_runtime.internal.attention_ulysses import Local

# Tensor and module imports remain type-only, preserving the Torch-free base surface.


@dataclass(frozen=True, slots=True)
class Site:
    component: str
    path: str
    #: The module's key RMSNorm (`norm_k`), when it has one (`outlier_keys`).
    keys: Any = field(default=None, compare=False, repr=False)


#: A key RMSNorm gain that peaks at this multiple of its RMS or more puts outlier channels in
#: every key, which no quantized scale carries: its layer's dense rows take the exact kernel,
#: as a model's exact paths do. H3: 5.1-7.0 at blocks 45, 48, 49 (FP4 0.25-0.56 rel L2 there),
#: 1.06-1.55 at every other layer.
KEY_GAIN_CREST = 3.0


def outlier_keys(norm: Any) -> bool:
    """Whether a key RMSNorm's gain has an outlier channel (`KEY_GAIN_CREST`). Read on its
    module's first call, where its weight is resident, and kept on it: weights alone decide,
    so every rank and every step agree."""
    if getattr(norm, "weight", None) is None:
        return False
    if not hasattr(norm, "_cozy_outlier_keys"):
        gain = norm.weight.detach().float().abs()
        norm._cozy_outlier_keys = bool(gain.amax() >= KEY_GAIN_CREST * gain.square().mean().sqrt())
    return bool(norm._cozy_outlier_keys)


_SITE: ContextVar[Site | None] = ContextVar("cozy_sol_site", default=None)
_DENSE: Callable[..., Tensor] | None = None
_DENSE_IDENTITY = ""
#: The BF16 kernel the model's exact paths take when the dense one quantizes; else `_DENSE`.
_EXACT: Callable[..., Tensor] | None = None
_COUNTS: ContextVar[dict[str, int] | None] = ContextVar("cozy_sol_counts", default=None)
_RANK_COUNTS: ContextVar[dict[int, dict[str, int]] | None] = ContextVar(
    "cozy_sol_rank_counts", default=None
)
_KERNEL_ROWS: ContextVar[list[dict[str, Scalar]] | None] = ContextVar(
    "cozy_sol_kernel_rows", default=None
)
#: This request's word on each object it asked for, by key digest: building, served or failed.
_SAID: ContextVar[dict[str, str] | None] = ContextVar("cozy_sol_said", default=None)
#: What a step serves in place of a sparse one, by count.
COUNTS = ("sparse", "dense_step", "dense_building", "dense_path", "dense_prefix")


@contextmanager
def observing(
    *,
    ranks: dict[int, dict[str, int]] | None = None,
    kernels: list[dict[str, Scalar]] | None = None,
    held: dict[str, object] | None = None,
) -> Iterator[dict[str, int]]:
    """Count successful calls in this request, restoring any outer observation. `kernels`
    receives one row per kernel this request started building or resolved. The request's
    producer states live here too: fresh ones, or a follower attempt's `held` store."""
    counts = dict.fromkeys(COUNTS, 0)
    token = _COUNTS.set(counts)
    rank_token = _RANK_COUNTS.set(ranks)
    kernel_token = _KERNEL_ROWS.set(kernels)
    said_token = _SAID.set({})
    try:
        with _request(held):
            yield counts
    finally:
        _COUNTS.reset(token)
        _RANK_COUNTS.reset(rank_token)
        _KERNEL_ROWS.reset(kernel_token)
        _SAID.reset(said_token)


def _kernel_row(row: dict[str, Scalar]) -> None:
    rows = _KERNEL_ROWS.get()
    if rows is not None:
        rows.append(row)


def _count(kind: str) -> None:
    counts = _COUNTS.get()
    if counts is not None:
        counts[kind] += 1


def record_rank(rank: int, counts: dict[str, int]) -> None:
    """Retain counts acknowledged by a follower on the existing mirrored-call reply."""
    ranks = _RANK_COUNTS.get()
    if ranks is None:
        return
    if (
        type(rank) is not int
        or rank < 1
        or set(counts) != set(COUNTS)
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError("invalid follower Sol attention counts")
    total = ranks.setdefault(rank, dict.fromkeys(counts, 0))
    for name, value in counts.items():
        total[name] += value


def bind_dense(
    call: Callable[..., Tensor], identity: str, exact: Callable[..., Tensor] | None = None
) -> None:
    """Install the dense reference a selection resolved, and the BF16 kernel the model's exact
    paths take when that one quantizes. Selection runs at preparation and at a run's step and
    stage boundaries, so a rebind never changes a running step."""
    if not callable(call) or not identity or not (exact is None or callable(exact)):
        raise ValueError("Sol requires a resolved dense attention reference and its identity")
    global _DENSE, _DENSE_IDENTITY, _EXACT
    _DENSE, _DENSE_IDENTITY, _EXACT = call, identity, exact


#: A bound dense reference: (dense, its identity, exact).
type Binding = tuple[Callable[..., Tensor] | None, str, Callable[..., Tensor] | None]


def binding() -> Binding:
    """The dense reference bound now, for `rebind` after a request changes it."""
    return _DENSE, _DENSE_IDENTITY, _EXACT


def rebind(held: Binding) -> None:
    global _DENSE, _DENSE_IDENTITY, _EXACT
    _DENSE, _DENSE_IDENTITY, _EXACT = held


def dense_identity() -> str:
    """The dense reference bound for a sparse kernel whose site's walk supplies one."""
    return _DENSE_IDENTITY


@dataclass(frozen=True, slots=True)
class _InstalledSite:
    site: Site
    forward: Callable[..., Tensor]
    original: Callable[..., Tensor]
    bind_original: bool
    had_instance_forward: bool
    prior_instance_forward: object


def install_site(module: Module, component: str, path: str) -> None:
    """Scope the actual forward, after pre-hooks and before post-hooks.

    Paired hooks cannot identify an invocation whose pre-hook never ran: an
    always-call post-hook can consume an outer recursive call's token. A local
    try/finally around the real forward has exactly one token per entered call.
    """
    site = Site(component, path, getattr(module, "norm_k", None))
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is not None:
        if existing.site != site:
            raise ValueError("attention module was already assigned another Sol site")
        if module.forward is not existing.forward:
            raise ValueError("attention forward was replaced after installing its Sol site")
        return
    original = module.forward
    bound = isinstance(original, MethodType) and original.__self__ is module
    call = original.__func__ if bound else original
    had_forward = "forward" in vars(module)
    prior = vars(module).get("forward")
    original_signature = signature(original)
    self_name = "_cozy_module"
    while self_name in original_signature.parameters:
        self_name += "_"

    @wraps(call, updated=())
    def scoped_forward(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        # Read the copied module's record rather than closing over a bound method
        # on the original module. deepcopy must execute the copied weights.
        installed = self._cozy_sol_site
        token = _SITE.set(installed.site)
        try:
            if installed.bind_original:
                return installed.original(self, *args, **kwargs)
            return installed.original(*args, **kwargs)
        finally:
            _SITE.reset(token)

    if not bound:
        # A callable object's metadata must not retain its original mutable state.
        # Its explicit signature below still preserves the public call contract.
        del scoped_forward.__wrapped__
    scoped_forward.__dict__["__signature__"] = Signature(
        [Parameter(self_name, Parameter.POSITIONAL_ONLY), *original_signature.parameters.values()],
        return_annotation=original_signature.return_annotation,
    )
    wrapper = MethodType(scoped_forward, module)
    module.forward = wrapper
    try:
        module._cozy_sol_site = _InstalledSite(site, wrapper, call, bound, had_forward, prior)
    except BaseException:
        if had_forward:
            module.forward = prior
        else:
            del module.forward
        raise


def remove_site(module: Module, component: str, path: str) -> None:
    """Restore the previous forward only while our exact wrapper still owns it."""
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is None:
        return
    if existing.site != Site(component, path):
        raise ValueError("attention module is assigned another Sol site")
    if module.forward is not existing.forward:
        raise ValueError("attention forward was replaced after installing its Sol site")
    if existing.had_instance_forward:
        module.forward = existing.prior_instance_forward
    else:
        del module.forward
    del module._cozy_sol_site


def _dense(query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
    if _DENSE is None:
        raise RuntimeError("Sol dense reference was not resolved before execution")
    output = _DENSE(query=query, key=key, value=value, scale=scale)
    _validate_result(output, query)
    return output


def _exact(query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
    if _EXACT is None:
        return _dense(query=query, key=key, value=value, scale=scale)
    output = _EXACT(query=query, key=key, value=value, scale=scale)
    _validate_result(output, query)
    return output


def _validate_result(output: Tensor, query: Tensor) -> None:
    import torch

    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("attention backend returned incompatible output")


#: tcgen05's route: Sol on FA4's block-sparse forward (`attention_sol_fa4`). Upstream's SM100
#: kernel ran its exact blocks at 0.55 PF on a B200, FA4 at 1.2 PF in BF16 and 1.72 PF in FP8
#: (h3a-093).
FA4 = "fa4"
#: Upstream's CuTe routes, and the capability each one's calling convention names. Only sm90's
#: kernel is specialized on the live length, and only through `make_kernel`'s static route
#: layout: 64-token blocks, grouped 64 at a time, the tail group's block count, and whether the
#: last block is full. Every operand layout is dynamic, so its object is a function of
#: (blocks, full), never of the exact count. The other routes' `make_kernel()` takes nothing:
#: one object per card serves every length.
ROUTES = {"sm89": (8, 9), "sm90": (9, 0), "sm120": (12, 0)}
_BLOCK = 64
_PINNED: list[bool] = []
KERNEL = "sol-attn"
OPTIONS = "--enable-tvm-ffi"
FUNCTION = "func"
_DLPACK_CUDA = 2
_SCALE = 128**-0.5


def route(capability: tuple[int, int]) -> str:
    """Sol's route on this card's row: FA4's on tcgen05, upstream's sm90 on wgmma, sm120 on
    block-scaled mma.sync (fp4 rows), sm89 on the other fp8 mma.sync rows; else none."""
    row = arch.row(capability[0] * 10 + capability[1])
    if row.family == arch.TCGEN05:
        return FA4
    if row.family == arch.WGMMA:
        return "sm90"
    if row.family == arch.MMA_SYNC and row.fp8:
        return "sm120" if row.fp4 else "sm89"
    return ""


def variant(tokens: int, capability: tuple[int, int]) -> dict[str, Any]:
    """What of this length the card's kernel is compiled for: nothing, except on sm90's."""
    if type(tokens) is not int or tokens < 1:
        raise ValueError("a Sol variant needs a positive token count")
    if route(capability) != "sm90":
        return {}
    return {"blocks": -(-tokens // _BLOCK), "full": tokens % _BLOCK == 0}


def _card(device: Device) -> tuple[int, int] | None:
    """A card with a Sol route."""
    if device.type != "cuda":
        return None
    found = _arch(device)
    return found if route(found) else None


def _capability(device: Device) -> tuple[int, int] | None:
    """A card whose sparse steps run upstream's own CuTe kernel, which compiles here."""
    found = _card(device)
    return found if found is not None and route(found) in ROUTES else None


def target(capability: tuple[int, int]) -> str:
    """CuTe-DSL's own arch spelling (`env_manager.detect_gpu_arch`) unless its config names one."""
    named = config.cute_dsl_environment().get("CUTE_DSL_ARCH", "")
    if named:
        return named
    major, minor = capability
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


@functools.cache
def sol_library() -> dict[str, Any]:
    """Every library input to the object: sources, DSL and toolkit, FFI ABI, Python tag."""
    cutlass = import_module("cutlass")
    tvm_ffi = import_module("tvm_ffi")
    toolkit = import_module("cutlass.base_dsl.version_info").CUDA_VERSION
    root = Path(str(import_module("sol_attn").__file__)).resolve().parent
    sources = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        data = path.read_bytes()
        sources.update(path.relative_to(root).as_posix().encode())
        sources.update(len(data).to_bytes(8, "little"))
        sources.update(data)
    return {
        "sol_attn_sources": sources.hexdigest(),
        "cutlass_dsl": str(cutlass.__version__),
        "cuda_toolkit": f"{toolkit.major}.{toolkit.minor}",
        "tvm_ffi": str(tvm_ffi.__version__),
        "python": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "cute_dsl_env": {
            name: value
            for name, value in config.cute_dsl_environment().items()
            if name != "CUTE_DSL_ARCH"
        },
    }


def job(tokens: int, capability: tuple[int, int]) -> kernel_compile.Job:
    """The compile of the object serving `tokens` on this card: GPU-less, the DSL's target
    imposed before it imports."""
    spelled = target(capability)
    shape = variant(tokens, capability)
    inputs = {
        "library": sol_library(),
        "target": spelled,
        "options": OPTIONS,
        "variant": shape,
    }
    return kernel_compile.Job(
        "sol",
        kernel_cache.Key.of(KERNEL, inputs),
        {"route": route(capability), **shape},
        {"CUDA_VISIBLE_DEVICES": "", "CUTE_DSL_ARCH": spelled},
    )


class _Placeholder:
    """A host tensor that reports CUDA device 0 through DLPack. Tracing reads only dtype, rank,
    strides and memory space; no pointer is dereferenced, so no GPU is needed."""

    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor
        self.ndim = tensor.ndim

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_DLPACK_CUDA, 0)

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        capsule = self.tensor.__dlpack__(*args, **kwargs)
        name_of = ctypes.pythonapi.PyCapsule_GetName
        name_of.restype, name_of.argtypes = ctypes.c_char_p, [ctypes.py_object]
        pointer_of = ctypes.pythonapi.PyCapsule_GetPointer
        pointer_of.restype = ctypes.c_void_p
        pointer_of.argtypes = [ctypes.py_object, ctypes.c_char_p]
        name = name_of(capsule)
        # DLManagedTensorVersioned: version(8) manager_ctx(8) deleter(8) flags(8), then the
        # DLTensor, whose device follows its data pointer. Unversioned: DLTensor first.
        offset = 40 if name == b"dltensor_versioned" else 8
        ctypes.c_int32.from_address(pointer_of(capsule, name) + offset).value = _DLPACK_CUDA
        return capsule


@request_path.forbid("kernel build: sol")
def compile_variant(route: str, blocks: int = 0, full: bool = False) -> bytes:
    """Upstream `_compile_<route>`, traced from placeholders; SM90's at a length that IS this
    variant. Every route takes q, k, v, output, kc, vc, threshold and lse."""
    import torch

    cute = import_module("cutlass.cute")
    cuda = import_module("cuda.bindings.driver")
    to_cute_tensor = import_module("sol_attn.common").to_cute_tensor
    upstream = import_module(f"sol_attn.{route}")
    operands = [((2, 2, 2, 128), "bfloat16")] * 6 + [((2, 2, 2), "float32")] * 2
    arguments = [
        to_cute_tensor(_Placeholder(torch.empty(shape, dtype=getattr(torch, dtype))))
        for shape, dtype in operands
    ]
    scalars: tuple[int, ...]
    if route == "sm90":
        tokens = blocks * _BLOCK if full else blocks * _BLOCK - 1
        operator, scalars = upstream.make_kernel(tokens, 1), (0,)  # the sink range, packed
    else:
        operator, scalars = upstream.make_kernel(), (0, 0)  # the sink's first and last block
    compiled = cute.compile(
        operator, *arguments, _SCALE, *scalars, stream=cuda.CUstream(0), options=OPTIONS
    )
    with tempfile.TemporaryDirectory(prefix="cozy-kernel-") as scratch:
        path = Path(scratch) / "kernel.o"
        compiled.export_to_c(object_file_path=str(path), function_name=FUNCTION)
        return path.read_bytes()


@request_path.forbid("kernel build: sol")
def build(spec: dict[str, Any], staging: Path, report: Callable[..., None]) -> dict[str, object]:
    """The compile manager's builder for one object (`kernel_compile.BUILDERS["sol"]`)."""
    data = compile_variant(spec["route"], int(spec.get("blocks", 0)), bool(spec.get("full")))
    (staging / "object").write_bytes(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


class Loaded:
    """Upstream's call convention over a loaded object: `stream` is positional in TVM-FFI."""

    __slots__ = ("function",)

    def __init__(self, function: Any) -> None:
        self.function = function

    def __call__(self, *arguments: Any, stream: Any = None) -> Any:
        return self.function(*arguments, stream)


def load(data: bytes) -> Loaded:
    """Link verified bytes from a file with no name, private to this process: what was hashed
    is exactly what runs. `TemporaryFile` is O_TMPFILE, else unlinked at once; it needs no
    `os.memfd_create`, which python-build-standalone's old glibc build lacks."""
    cute = import_module("cutlass.cute")
    for library in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
        if Path(library).exists():
            # As flash-attn-4's own disk cache does: a loaded object resolves the DSL
            # runtime's symbols globally.
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    with tempfile.TemporaryFile(prefix="cozy-kernel-") as anonymous:
        anonymous.write(data)
        anonymous.flush()
        path = f"/proc/self/fd/{anonymous.fileno()}"
        return Loaded(getattr(cute.runtime.load_module(path, enable_tvm_ffi=True), FUNCTION))


@functools.cache
def _job(blocks: int, full: bool, capability: tuple[int, int]) -> kernel_compile.Job:
    return job(blocks * _BLOCK if full else blocks * _BLOCK - 1, capability)


def _job_for(tokens: int, capability: tuple[int, int]) -> kernel_compile.Job:
    shape = variant(tokens, capability)
    return _job(shape.get("blocks", 1), shape.get("full", False), capability)


def card_jobs(store: kernel_cache.Store, capability: tuple[int, int]) -> list[kernel_compile.Job]:
    """What a card compiles ahead of any request: upstream's preprocess, and its one object, or
    on SM90, whose object is per length layout, every layout this machine has served."""
    found = [preprocess_job(capability[0] * 10 + capability[1])]
    if route(capability) != "sm90":
        return [*found, _job_for(1, capability)]
    return [*found, *kernel_compile.learned(store, _scope(capability))]


def submit_card(index: int) -> None:
    """As soon as Sol is ready here, for a machine whose kernel boot has not: `card_jobs`."""
    import torch

    device = torch.device("cuda", index)
    capability = _capability(device)
    store = kernel_cache.configured()
    if capability is None or store is None:
        return
    if route(capability) != "sm90":
        kernel_compile.submit(store, preprocess_job(capability[0] * 10 + capability[1]))
        _ready(1, device, -2)
        return
    for found in card_jobs(store, capability):
        kernel_compile.submit(store, found)


def _scope(capability: tuple[int, int]) -> str:
    """Learned layouts are this library's on this card and driver: another learns its own."""
    library = {**sol_library(), "driver": accel.driver_version()}
    digest = hashlib.sha256(json.dumps(library, sort_keys=True).encode()).hexdigest()
    return f"{KERNEL}-{target(capability)}-{digest[:16]}"


def expect(tokens: int, device: Device) -> None:
    """A model stated its live length at request intake: a whole conditioning ahead of the
    first dense step. Its row carries step -1."""
    _ready(tokens, device, -1)


def _row(
    tokens: int, found: kernel_compile.Job, source: str, started: float, **extra: Scalar
) -> dict[str, Scalar]:
    shape = json.loads(found.key.inputs)["variant"]
    return {
        "source": source,
        "tokens": tokens,
        **{name: value for name, value in shape.items()},
        "key": found.key.digest[:16],
        "ms": round((time.perf_counter() - started) * 1000, 1),
        **extra,
    }


def _ready(tokens: int, device: Device, step: int) -> tuple[Loaded | None, str]:
    """The object for this length once it serves here (`kernel_ready`), else None and the count
    of the dense step in its place: `dense_building` while it builds, `dense_step` without a
    store or after its build failed. The first sight of a length (a model's intake statement,
    or a dense step) starts it opening, so conditioning and the dense steps cover its build.
    One loaded object serves every head count, batch and device (h3a-087)."""
    capability = _capability(device)
    store = kernel_cache.configured()
    if capability is None or store is None:
        return None, "dense_step"
    found = _job_for(tokens, capability)
    digest, said = found.key.digest, _SAID.get()
    started = time.perf_counter()
    name = f"{KERNEL} {digest[:16]}"
    loaded = kernel_ready.ready(name, functools.partial(_open, store, found, capability))
    word = (said or {}).get(digest, "")
    if loaded is not None:
        if word != "served":
            ms = kernel_compile.status(store, found).ms
            source = "built" if word == "building" else "store"
            _kernel_row(_row(tokens, found, source, started, step=step, compile_ms=ms))
        word = "served"
    elif why := kernel_ready.failed(name):
        if word != "failed":
            state = f"{type(why).__name__}: {why}"[:300]
            _kernel_row(_row(tokens, found, "failed", started, step=step, state=state))
        word = "failed"
    elif word != "building":
        _kernel_row(_row(tokens, found, "building", started, step=step))
        word = "building"
    if said is not None:
        said[digest] = word
    return loaded, {"served": "", "failed": "dense_step"}.get(word, "dense_building")


def _open(
    store: kernel_cache.Store, found: kernel_compile.Job, card: tuple[int, int], wait: bool
) -> Loaded | None:
    """The object linked (`kernel_ready`'s opener), after its build when `wait`. An SM90 layout
    is learned, so this machine's next boot builds it ahead (`card_jobs`)."""
    if json.loads(found.key.inputs)["variant"]:
        kernel_compile.learn(store, _scope(card), found)
    data = store.get(found.key)
    if data is None and wait:
        state = kernel_compile.wait(store, found)
        data = store.get(found.key)
        if data is None:
            raise kernel_compile.CompileFailed(state.state, state.line())
    return None if data is None else load(data)


def _preprocessed(query: Tensor) -> str:
    """Whether upstream's preprocess kernels serve on this card: "" once they do, else what a
    sparse step serves instead (`dense_building` while their one build runs, `dense_step` when
    it failed). Every length and head count shares it."""
    index = query.device.index or 0
    name = f"{KERNEL} preprocess cuda:{index}"
    if kernel_ready.ready(name, functools.partial(_warm, index)) is not None:
        return ""
    return "dense_step" if kernel_ready.failed(name) is not None else "dense_building"


def preprocess_job(arch: int) -> kernel_compile.Job:
    """The one build of upstream's preprocess for a card architecture (`preprocess`)."""
    from cozy_runtime.internal import triton_kernels

    return triton_kernels.warm_job(
        f"{KERNEL}-preprocess", f"{__name__}:preprocess", [], arch, {"library": sol_library()}
    )


def _warm(index: int, wait: bool) -> bool | None:
    """`_preprocessed`'s opener: `preprocess` built by the machine's compile manager for this
    card, then loaded here (its build waited for when `wait`, else None while it runs)."""
    store = kernel_cache.configured()
    if store is None:
        raise kernel_compile.CompileFailed("absent", "no machine kernel store to compile into")
    major, minor = _arch(index)
    found = preprocess_job(major * 10 + minor)
    state = (kernel_compile.wait if wait else kernel_compile.submit)(store, found)
    if state.state in ("compiling", "pending"):
        return None
    if state.state != "ready":
        raise kernel_compile.CompileFailed(state.state, state.line())
    import torch

    with torch.cuda.device(index):
        preprocess()
    return True


def preprocess() -> None:
    """Upstream's exact-threshold preprocess (`preprocess.prepare`) on the current card,
    launching nothing: in the compile manager's builder it compiles (`_warm`), in an executor it
    loads what that built. Its operands are host stand-ins in the live ones' class: what compiles
    is the descriptors' block shapes and the constants. Lengths, head counts and block counts are
    runtime integers none of the kernels specializes on, so any stand-in sizes serve."""
    import torch

    upstream = import_module("sol_attn.preprocess")
    descriptor = import_module("triton.tools.tensor_descriptor").TensorDescriptor
    _pin_preprocess()
    tokens, heads = _BLOCK, 1
    blocks, bf16, f32 = -(-tokens // _BLOCK), torch.bfloat16, torch.float32
    rows = torch.empty(1, _BLOCK, heads, 128, dtype=bf16)
    tiles = descriptor.from_tensor(rows, [1, _BLOCK, 1, 128])
    shape = (tokens, heads, blocks, 128, _BLOCK, 128)
    upstream._reduce_kc_kernel.warmup(tiles, bf16, *shape, grid=(1,))
    upstream._reduce_vc_kernel.warmup(tiles, bf16, *shape, grid=(1,))
    upstream._pool_query_kernel.warmup(tiles, bf16, *shape, grid=(1,), num_warps=4, num_stages=1)
    upstream._exact_fused_threshold_kernel.warmup(
        bf16, f32, bf16, f32, _SCALE, heads, blocks, 128, 64, 128, 1.0,
        grid=(1,), num_warps=4, num_stages=1,
    )  # fmt: skip


def _pin_preprocess() -> None:
    """Upstream autotunes its K/V block reductions over 4 and 8 warps by timing, and the two
    reduce in different orders: measured bitwise-different summaries, which can flip a
    routing decision. Keep upstream's first candidate only, so no timing race picks it."""
    if _PINNED:
        return
    preprocess = import_module("sol_attn.preprocess")
    for tuned in (preprocess._reduce_kc_kernel, preprocess._reduce_vc_kernel):
        chosen = [c for c in tuned.configs if c.num_warps == 4 and c.num_stages == 1]
        if len(chosen) != 1:
            raise RuntimeError("Sol preprocess no longer offers its 4-warp, 1-stage candidate")
        tuned.configs = chosen
    _PINNED.append(True)


def _native(
    query: Tensor, key: Tensor, value: Tensor, scale: float | None, sink_tokens: int, loaded: Loaded
) -> Tensor:
    """Call the upstream kernel through the compiled variant; a full-document sink selects
    every KV block. This does not perform the wrapper's protected-query recomputation."""
    _pin_preprocess()
    batch, tokens, heads, _ = query.shape
    named = ROUTES[route(_arch(query.device))]  # the calling convention upstream branches on
    interface = import_module("sol_attn.interface")
    interface._compiled[(query.device.index, named, batch, tokens, heads, 1)] = loaded
    return interface._sol_attn_cute(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        arch=named,
        scale=_SCALE if scale is None else float(scale),
        tau=1.0,
        thresh_type="exact",
        kv_splits=1,
        sink_tokens=sink_tokens,
        sink_start=0,
    )


def sol_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """One BF16 NHD document; preserve its prefix and zero explicit trailing padding.

    Policy v1 uses upstream tau=1, exact threshold statistics and one KV split.
    `exact` names threshold computation, not dense-equivalent attention. Initial
    dense steps and paths are explicit model policy in the active layout. Under
    Ulysses this runs after the exchange, on the full document of a head subset;
    Sol's block routing is computed per head, so the split matches one GPU.
    """
    return _attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def _attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    sparse: Callable[..., Tensor] | None = None,
    dense: Callable[..., Tensor] = _dense,
    prefix_dense: Callable[..., Tensor] | None = None,
    exact: Callable[..., Tensor] | None = None,
) -> Tensor:
    """Shared input and model-policy boundary for the opt-in sparse implementations. What a
    sparse kernel admits (one BF16 document, head dimension 128, equal Q/K/V, no mask or
    causality) is its row's (`attention.Candidate`), asserted before this runs. A row that
    declares its dense kernels gives `exact` too: the BF16 kernel its model's exact paths take."""
    import torch

    if torch.compiler.is_compiling():
        raise ValueError("Sol's model layout and site scope require eager execution")
    layout, site = _ACTIVE_LAYOUT.get(), _SITE.get()
    if layout is None or site is None:
        raise ValueError(
            "Sol requires model-owned attention layout and a registered attention site"
        )
    if dropout_p != 0:
        raise ValueError("Sol is an inference kernel and does not support dropout")
    tensors = (query, key, value)
    if query.shape != value.shape or any(
        t.device != query.device or t.requires_grad for t in tensors
    ):
        raise ValueError("Sol requires inference inputs of one geometry on one device")
    if sparse is None and _card(query.device) is None:
        raise ValueError(f"Sol has no route on {query.device}")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    return _execute(
        query,
        key,
        value,
        scale,
        layout,
        site,
        sparse=sparse,
        dense=dense,
        prefix_dense=prefix_dense,
        exact=exact,
    )


def _arch(device: Device | int) -> tuple[int, int]:
    import torch

    found = tuple(torch.cuda.get_device_capability(device))
    return (found[0], found[1])


def backend(device: Device | int) -> str:
    """The route this card's sparse steps take: FA4's block-sparse forward, else upstream's
    `cute_<route>` where its DSL imports, else none (upstream would fall back to Triton)."""
    found = route(_arch(device))
    if found == FA4:
        return "fa4_block_sparse"
    available = found and import_module("sol_attn.interface")._cute_runtime_available()
    return f"cute_{found}" if available else "none"


SOL = Local(sol_attention, lambda query, *_: backend(query.device))


def _execute(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float | None,
    layout: AttentionLayout,
    site: Site,
    *,
    sparse: Callable[..., Tensor] | None = None,
    dense: Callable[..., Tensor] = _dense,
    prefix_dense: Callable[..., Tensor] | None = None,
    exact: Callable[..., Tensor] | None = None,
) -> Any:
    """Apply the model's explicit policy after hardware/operand validation. An exact path's
    dense rows (its dense steps, its protected prefix, a dense path) take the exact kernel on
    every route: the walked reference's (`_exact`), or the one a declaring row gives; on a
    declaring row's quantizing sparse kernel, its sparse steps run dense on it too. A layer whose
    key gain has outlier channels (`outlier_keys`) is an exact path too."""
    import torch

    def on(paths: tuple[str, ...]) -> bool:
        return any(site.path == p or site.path.startswith(p + ".") for p in paths)

    dense_path = on(layout.dense_paths)
    if on(layout.exact_paths) or outlier_keys(site.keys):
        said = _SAID.get()
        if said is not None and "exact:" + site.path not in said:
            said["exact:" + site.path] = "exact"
            _kernel_row({"source": "exact", "site": site.path})  # the receipt names each exact site
        if dense is _dense:
            dense = _exact
        elif exact is not None:
            dense = prefix_dense = exact
            # A quantizing sparse kernel (Kitchen's INT8) cannot carry these keys on its sparse
            # rows either: the whole call is dense on the exact kernel. H3 block 49 on an RTX 4090,
            # against FP32: Kitchen INT8 Sol 0.10-0.11 rel L2, BF16 Sol 0.05-0.08, this 0.016-0.022.
            dense_path = dense_path or sparse is not None
    if dense_path and query.shape[1] < layout.live_tokens:
        # A refiner has its own shorter document; its explicit dense policy wins.
        output = dense(query=query, key=key, value=value, scale=scale)
        _count("dense_path")
        return output
    if layout.live_tokens > query.shape[1]:
        raise ValueError("attention layout exceeds Q/K/V sequence length")
    q, k, v = (t[:, : layout.live_tokens] for t in (query, key, value))
    card = _card(query.device)
    fa4 = sparse is None and card is not None and route(card) == FA4
    loaded: object = None
    instead = "dense_path" if dense_path else "dense_step"
    if not dense_path and sparse is None:
        # The first sight of this length starts its kernels opening off the request (a CuTe
        # route's object, upstream's preprocess), and the dense steps run while they build.
        unbuilt = "" if card is None else _preprocessed(query)
        ready, waiting = (FA4, "") if fa4 else _ready(layout.live_tokens, query.device, layout.step)
        if ready is not None and unbuilt:
            ready, waiting = None, unbuilt
        if layout.step >= layout.dense_until_step:
            loaded, instead = ready, waiting
    if dense_path or layout.step < layout.dense_until_step or (sparse is None and loaded is None):
        # A dense step, a dense path, or a sparse step whose variant is not ready: the dense
        # reference serves it.
        live_output = dense(query=q, key=k, value=v, scale=scale)
        _count(instead)
    elif fa4:
        # The exact tiles take the dense rows' precision; the model's exact paths stay BF16.
        # The protected prefix's query tiles attend every key inside the same call.
        from cozy_runtime.internal import attention_sol_fa4

        live_output = attention_sol_fa4.sparse(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale,
            layout.protected_prefix,
            fp8=_EXACT is not None and dense is _dense,
        )
        _validate_result(live_output, q)
        _count("sparse")
    else:
        if sparse is not None:
            live_output = sparse(q, k, v, scale, layout.protected_prefix)
        else:
            assert loaded is not None
            live_output = _native(q, k, v, scale, layout.protected_prefix, cast("Loaded", loaded))
        _validate_result(live_output, q)
        if any(
            live_output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (query, key, value)
        ):
            raise ValueError("Sol returned output sharing input storage")
        _count("sparse")
        if layout.protected_prefix:
            prefix = dense if prefix_dense is None else prefix_dense
            live_output[:, : layout.protected_prefix] = prefix(
                query=q[:, : layout.protected_prefix], key=k, value=v, scale=scale
            )
            _count("dense_prefix")
    _validate_result(live_output, q)
    if layout.live_tokens == query.shape[1]:
        return live_output
    output = torch.zeros_like(query)
    output[:, : layout.live_tokens].copy_(live_output)
    return output
