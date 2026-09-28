"""Upstream BF16 Sol-Attn with explicit document and conditioning semantics.

No sparse kernel is implemented here. Sparse block selection is approximate;
protected query rows are recomputed with the selected dense reference. The
model supplies its live layout, step policy and dense module paths explicitly.

The kernel compiles once per object per machine, out of process (`kernel_compile`): one per
card, and on SM90 one per 64-token block layout (`variant`), which only the request knows. So
the first sight of a length (a model's intake statement, or the first dense step) submits its
compile, and conditioning and the dense steps cover it. A sparse step whose object is not
compiled yet runs the dense reference instead: nothing waits on a compile. Upstream's
preprocess autotuning is pinned to one configuration, because its candidates differ bitwise
and a timing race must not choose the output.
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
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from inspect import Parameter, Signature, signature
from pathlib import Path
from types import MethodType
from typing import Any

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.author._observations import Scalar
from cozy_runtime.internal import config, kernel_cache, kernel_compile
from cozy_runtime.internal.attention_ulysses import Local

# Tensors, devices, modules and the dense reference are `Any`: torch is not in the check venv.


@dataclass(frozen=True, slots=True)
class Site:
    component: str
    path: str


_SITE: ContextVar[Site | None] = ContextVar("cozy_sol_site", default=None)
_DENSE: Callable[..., Any] | None = None
_DENSE_IDENTITY = ""
_COUNTS: ContextVar[dict[str, int] | None] = ContextVar("cozy_sol_counts", default=None)
_RANK_COUNTS: ContextVar[dict[int, dict[str, int]] | None] = ContextVar(
    "cozy_sol_rank_counts", default=None
)
_KERNEL_ROWS: ContextVar[list[dict[str, Scalar]] | None] = ContextVar(
    "cozy_sol_kernel_rows", default=None
)


@contextmanager
def observing(
    *,
    ranks: dict[int, dict[str, int]] | None = None,
    kernels: list[dict[str, Scalar]] | None = None,
) -> Iterator[dict[str, int]]:
    """Count successful calls in this request, restoring any outer observation. `kernels`
    receives one row per kernel this request started building or resolved."""
    counts = dict.fromkeys(("sparse", "dense_step", "dense_path", "dense_prefix"), 0)
    token = _COUNTS.set(counts)
    rank_token = _RANK_COUNTS.set(ranks)
    kernel_token = _KERNEL_ROWS.set(kernels)
    try:
        yield counts
    finally:
        _COUNTS.reset(token)
        _RANK_COUNTS.reset(rank_token)
        _KERNEL_ROWS.reset(kernel_token)


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
        or set(counts) != {"sparse", "dense_step", "dense_path", "dense_prefix"}
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError("invalid follower Sol attention counts")
    total = ranks.setdefault(rank, dict.fromkeys(counts, 0))
    for name, value in counts.items():
        total[name] += value


def bind_dense(call: Callable[..., Any], identity: str) -> None:
    """Install the dense reference a selection resolved. Selection runs at preparation and at
    request boundaries, never inside a request, so a rebind never changes a running one."""
    if not callable(call) or not identity:
        raise ValueError("Sol requires a resolved dense attention reference and its identity")
    global _DENSE, _DENSE_IDENTITY
    _DENSE, _DENSE_IDENTITY = call, identity


def dense_identity() -> str:
    return _DENSE_IDENTITY


@dataclass(frozen=True, slots=True)
class _InstalledSite:
    site: Site
    forward: Callable[..., Any]
    original: Callable[..., Any]
    bind_original: bool
    had_instance_forward: bool
    prior_instance_forward: Any


def install_site(module: Any, component: str, path: str) -> None:
    """Scope the actual forward, after pre-hooks and before post-hooks.

    Paired hooks cannot identify an invocation whose pre-hook never ran: an
    always-call post-hook can consume an outer recursive call's token. A local
    try/finally around the real forward has exactly one token per entered call.
    """
    site = Site(component, path)
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


def remove_site(module: Any, component: str, path: str) -> None:
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


def _dense(query: Any, key: Any, value: Any, scale: float | None) -> Any:
    if _DENSE is None:
        raise RuntimeError("Sol dense reference was not resolved before execution")
    output = _DENSE(query=query, key=key, value=value, scale=scale)
    _validate_result(output, query)
    return output


def _validate_result(output: Any, query: Any) -> None:
    import torch

    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("attention backend returned incompatible output")


#: Upstream's CuTe routes. Only SM90's kernel is specialized on the live length, and only
#: through `make_kernel`'s static route layout: 64-token blocks, grouped 64 at a time, the tail
#: group's block count, and whether the last block is full. Every operand layout is dynamic,
#: so its object is a function of (blocks, full), never of the exact count. The other routes'
#: `make_kernel()` takes nothing: one object per card serves every length.
ROUTES = {(8, 9): "sm89", (9, 0): "sm90", (10, 0): "sm100", (12, 0): "sm120"}
_BLOCK = 64
#: Loaded objects by key digest; one serves every head count, batch and device (h3a-087).
_KERNELS: dict[str, Loaded] = {}
#: Keys this process already submitted, so a dense step submits each once.
_SUBMITTED: set[str] = set()
_PINNED: list[bool] = []
KERNEL = "sol-attn"
OPTIONS = "--enable-tvm-ffi"
FUNCTION = "func"
_DLPACK_CUDA = 2
_SCALE = 128**-0.5


def variant(tokens: int, capability: tuple[int, int]) -> dict[str, Any]:
    """What of this length the card's kernel is compiled for: nothing, except on SM90."""
    if type(tokens) is not int or tokens < 1:
        raise ValueError("a Sol variant needs a positive token count")
    if capability != (9, 0):
        return {}
    return {"blocks": -(-tokens // _BLOCK), "full": tokens % _BLOCK == 0}


def _capability(device: Any) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    found = tuple(import_module("torch").cuda.get_device_capability(device))
    return (found[0], found[1]) if found in ROUTES else None


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
    arch = target(capability)
    shape = variant(tokens, capability)
    inputs = {"library": sol_library(), "target": arch, "options": OPTIONS, "variant": shape}
    return kernel_compile.Job(
        "sol",
        kernel_cache.Key.of(KERNEL, inputs),
        {"route": ROUTES[capability], **shape},
        {"CUDA_VISIBLE_DEVICES": "", "CUTE_DSL_ARCH": arch},
    )


class _Placeholder:
    """A host tensor that reports CUDA device 0 through DLPack. Tracing reads only dtype, rank,
    strides and memory space; no pointer is dereferenced, so no GPU is needed."""

    def __init__(self, tensor: Any) -> None:
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


def compile_variant(route: str, blocks: int = 0, full: bool = False) -> bytes:
    """Upstream `_compile_<route>`, traced from placeholders; SM90's at a length that IS this
    variant. Every route takes q, k, v, output, kc, vc, threshold and lse."""
    torch = import_module("torch")
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
        operator = upstream.forward if route == "sm100" else upstream.make_kernel()
        scalars = (0, 0)  # the sink's first and last block
    compiled = cute.compile(
        operator, *arguments, _SCALE, *scalars, stream=cuda.CUstream(0), options=OPTIONS
    )
    with tempfile.TemporaryDirectory(prefix="cozy-kernel-") as scratch:
        path = Path(scratch) / "kernel.o"
        compiled.export_to_c(object_file_path=str(path), function_name=FUNCTION)
        return path.read_bytes()


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


def specialize(tokens: int, device: Any, step: int) -> None:
    """The first sight of a length (a model's intake statement, or a dense step): submit its
    object's compile to the machine's compile manager, so dense steps cover it."""
    capability = _capability(device)
    store = kernel_cache.configured()
    if capability is None or store is None:
        return
    started = time.perf_counter()
    try:
        found = _job_for(tokens, capability)
        if found.key.digest in _SUBMITTED or found.key.digest in _KERNELS:
            return
        _SUBMITTED.add(found.key.digest)
        state = kernel_compile.submit(store, found)
        if variant(tokens, capability):
            kernel_compile.learn(store, _scope(capability), found)
        _kernel_row(_row(tokens, found, state.state, started, step=step))
    except Exception as exc:
        # Nothing needs the kernel yet; the sparse step asks again and falls back THERE.
        _kernel_row(
            {
                "source": "submit_failed",
                "tokens": tokens,
                "step": step,
                "detail": f"{type(exc).__name__}: {exc}"[:300],
            }
        )


def submit_card(index: int) -> None:
    """At the startup warm, as soon as Sol is ready here: a card whose kernel takes no length
    compiles its one object; SM90 compiles every length layout this machine has served."""
    device = import_module("torch").device("cuda", index)
    capability = _capability(device)
    store = kernel_cache.configured()
    if capability is None or store is None:
        return
    if capability != (9, 0):
        specialize(1, device, -2)
        return
    for found in kernel_compile.learned(store, _scope(capability)):
        if found.key.digest not in _SUBMITTED:
            _SUBMITTED.add(found.key.digest)
            kernel_compile.submit(store, found)


def _scope(capability: tuple[int, int]) -> str:
    """Learned layouts are this library's on this card: another build learns its own."""
    digest = hashlib.sha256(json.dumps(sol_library(), sort_keys=True).encode()).hexdigest()
    return f"{KERNEL}-{target(capability)}-{digest[:16]}"


def expect(tokens: int, device: Any) -> None:
    """A model stated its live length at request intake: a whole conditioning ahead of the
    first dense step. Its row carries step -1."""
    specialize(tokens, device, -1)


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


def _ready(tokens: int, device: Any, step: int) -> Loaded | None:
    """The object for this length when it is compiled, else None and its compile submitted:
    the step then runs dense. Nothing here compiles or waits."""
    capability = _capability(device)
    store = kernel_cache.configured()
    if capability is None or store is None:
        return None
    found = _job_for(tokens, capability)
    loaded = _KERNELS.get(found.key.digest)
    if loaded is not None:
        return loaded
    started = time.perf_counter()
    data = store.get(found.key)
    if data is None:
        state = kernel_compile.submit(store, found)
        _SUBMITTED.add(found.key.digest)
        _kernel_row(_row(tokens, found, "cold_compile", started, step=step, state=state.line()))
        return None
    loaded = _KERNELS[found.key.digest] = load(data)
    entry = kernel_compile.status(store, found)
    _kernel_row(_row(tokens, found, "store", started, step=step, compile_ms=entry.ms))
    return loaded


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
    query: Any, key: Any, value: Any, scale: float | None, sink_tokens: int, loaded: Loaded
) -> Any:
    """Call the upstream kernel through the compiled variant; a full-document sink selects
    every KV block. This does not perform the wrapper's protected-query recomputation."""
    _pin_preprocess()
    batch, tokens, heads, _ = query.shape
    capability = tuple(import_module("torch").cuda.get_device_capability(query.device))
    import_module("sol_attn.interface")._compiled[
        (query.device.index, capability, batch, tokens, heads, 1)
    ] = loaded
    return import_module("sol_attn").sol_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        scale=scale,
        tau=1.0,
        thresh_type="exact",
        kv_splits=1,
        sink_start=0,
        sink_tokens=sink_tokens,
    )


def sol_attention(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    """One BF16 NHD document; preserve its prefix and zero explicit trailing padding.

    Policy v1 uses upstream tau=1, exact threshold statistics and one KV split.
    `exact` names threshold computation, not dense-equivalent attention. Initial
    dense steps and paths are explicit model policy in the active layout. Under
    Ulysses this runs after the exchange, on the full document of a head subset;
    Sol's block routing is computed per head, so the split matches one GPU.
    """
    import torch

    if torch.compiler.is_compiling():
        raise ValueError("Sol's model layout and site scope require eager execution")
    layout, site = _ACTIVE_LAYOUT.get(), _SITE.get()
    if layout is None or site is None:
        raise ValueError(
            "Sol requires model-owned attention layout and a registered attention site"
        )
    if attn_mask is not None or is_causal or dropout_p != 0 or enable_gqa:
        raise ValueError("Sol supports unmasked noncausal inference without dropout or GQA")
    tensors = (query, key, value)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in tensors):
        raise ValueError("Sol requires batch/sequence/head/dimension tensors")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Sol requires equal Q/K/V geometry")
    if query.shape[0] != 1 or query.shape[-1] != 128 or not query.shape[2]:
        raise ValueError("Sol requires one document with head dimension 128")
    if any(
        t.dtype != torch.bfloat16 or t.device != query.device or t.requires_grad for t in tensors
    ):
        raise ValueError("Sol requires BF16 inference inputs on one device")
    if _capability(query.device) is None:
        raise ValueError(f"Sol serves upstream's CuTe routes only: {sorted(ROUTES.values())}")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    return _execute(query, key, value, scale, layout, site)


def backend(device: Any) -> str:
    """The route upstream selects on this device: `cute_<sm>`, or its silent Triton fallback."""
    return str(import_module("sol_attn").get_sol_attn_backend(device))


SOL = Local(sol_attention, lambda query, *_: backend(query.device))


def _execute(
    query: Any, key: Any, value: Any, scale: float | None, layout: AttentionLayout, site: Site
) -> Any:
    """Apply the model's explicit policy after hardware/operand validation."""
    import torch

    dense_path = any(site.path == p or site.path.startswith(p + ".") for p in layout.dense_paths)
    if dense_path and query.shape[1] < layout.live_tokens:
        # A refiner has its own shorter document; its explicit dense policy wins.
        output = _dense(query, key, value, scale)
        _count("dense_path")
        return output
    if layout.live_tokens > query.shape[1]:
        raise ValueError("attention layout exceeds Q/K/V sequence length")
    q, k, v = (t[:, : layout.live_tokens] for t in (query, key, value))
    loaded = None
    if not dense_path:
        # The dense steps run on the GPU while this length's variant compiles on the CPU.
        specialize(layout.live_tokens, query.device, layout.step)
        if layout.step >= layout.dense_until_step:
            loaded = _ready(layout.live_tokens, query.device, layout.step)
    if loaded is None:
        # A dense step, a dense path, or a sparse step whose variant is still compiling: the
        # dense reference serves it, and nothing waits on the compile.
        live_output = _dense(q, k, v, scale)
        _count("dense_path" if dense_path else "dense_step")
    else:
        live_output = _native(q, k, v, scale, layout.protected_prefix, loaded)
        _validate_result(live_output, q)
        if any(
            live_output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (query, key, value)
        ):
            raise ValueError("Sol returned output sharing input storage")
        _count("sparse")
        if layout.protected_prefix:
            live_output[:, : layout.protected_prefix] = _dense(
                q[:, : layout.protected_prefix], k, v, scale
            )
            _count("dense_prefix")
    _validate_result(live_output, q)
    if layout.live_tokens == query.shape[1]:
        return live_output
    output = torch.zeros_like(query)
    output[:, : layout.live_tokens].copy_(live_output)
    return output
