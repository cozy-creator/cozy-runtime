"""The construction substrate: the author's real `load()` runs on meta under guards, and
every parameter it registers is observed for the fill.

**Substrate: meta device, not FakeTensorMode.** Recorded divergence from the issue's
wording, with the measurement behind it: `torch.nn.Module._apply` hard-codes
`isinstance(param, FakeTensor)` into `should_use_swap_tensors`, and `torch.utils.
swap_tensors` refuses any tensor `FakeTensorMode` holds a weakref to — which is every
tensor it made. So under an ambient FakeTensorMode, `.to(torch.bfloat16)`, `.float()` and
`.cuda()` raise `_apply(): Couldn't swap Linear.weight` on ANY module. That is the single
most common construction idiom in diffusers and transformers, so a fake-tensor substrate
cannot execute real ecosystem `load()` at all. v1 hit the same wall from the other side and
shimmed around it (`graphs/hollow.py::_hollow_module_moves`); meta has no wall to shim.
Meta keeps every property the derivation needs — zero allocation, exact shape/dtype
propagation, `.item()`/`nonzero`/`numpy` refusing loudly — and preserves tie identity
across `.to()`, which is what the alias census reads.

**Gap doctrine.** A construct the harness cannot derive correctly ends as harness code or
as a TYPED REFUSAL that names itself (`HarnessGap` / `ConstructionFault`). It never ends as
a new author-declared fact — that is how v1's surface grew.
"""

from __future__ import annotations

import sys
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

from cozy_runtime.author._errors import ConformanceError
from cozy_runtime.author._loader import (
    BUILT,
    CONSTRUCTION,
    FILLED,
    _members,
)


class DeriveError(Exception):
    """Base of the harness's typed outcomes. Always names the construct at fault."""

    def __init__(self, construct: str, message: str, *, site: str = "") -> None:
        super().__init__(f"{construct}: {message}" + (f"\n  at {site}" if site else ""))
        self.construct = construct
        self.site = site


class ConstructionFault(DeriveError, ConformanceError):
    """The author's construction is not derivable, and the fix is in the package."""

    default_code = "construction_fault"


class HarnessGap(DeriveError):
    """The HARNESS cannot derive this construct correctly, and says so rather than
    guessing. The fix lane is this file — never a new author declaration."""


# --------------------------------------------------------------------------- torch


def torch_module() -> ModuleType:
    """torch, imported LAZILY. `import cozy_runtime` stays torch-free (§1.0): torch is an
    optional extra that only the derivation harness and the serving executor need."""
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise HarnessGap(
            "torch",
            "the derivation harness executes the author's real load() under meta "
            "parameters, which needs torch in the release environment. Base cozy_runtime "
            "stays torch-free; install the `derive` extra",
        ) from exc
    return cast("ModuleType", torch)


def is_virtual(tensor: object) -> bool:
    """Does this tensor have NO storage? The ONE producer of that fact.

    v1's pgw#1661 is the whole reason this is a function and not an `isinstance`: h3's 300
    torchao `Float8Tensor` denoiser weights wrapped fake data and answered `isinstance(t,
    FakeTensor)` FALSE, so a weights-free guard priced a config-only tree at 23 GB on an
    8 GiB card. The fix was one predicate every consumer imports — "two spellings of one
    fact is how this drifted in the first place". Recursion through
    `is_traceable_wrapper_subclass` is what makes a wrapper answer for its contents.
    """
    return _is_virtual(tensor, 0)


def _is_virtual(tensor: object, depth: int) -> bool:
    if depth < 4:
        inner = _wrapped(tensor)
        if inner:
            return all(_is_virtual(held, depth + 1) for held in inner)
    device = getattr(tensor, "device", None)
    if device is None:
        return True
    return str(getattr(device, "type", device)) == "meta"


def _wrapped(tensor: object) -> tuple[object, ...]:
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if not is_traceable_wrapper_subclass(tensor):
        return ()
    try:
        names, _ = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
    except Exception:
        return ()
    return tuple(getattr(tensor, n) for n in names if hasattr(tensor, n))


# ----------------------------------------------------------------------- substrate


@contextmanager
def refuse_compile() -> Iterator[None]:
    """`torch.compile(module)` and `nn.Module.compile` REFUSE during construction, naming
    themselves.

    The wrapping spelling returns an `OptimizedModule` whose every `state_dict` key gains an
    `_orig_mod.` prefix, so the derived tensor requirements would demand `blk._orig_mod.weight`
    from checkpoints that carry `blk.weight` — a construction no artifact satisfies. The lazy
    `nn.Module.compile` is refused for a different reason: derive is derive-by-execution, so
    the callable it installs would trace under fake tensors and launch real kernels on fake
    pointers (pgw#1659, the day v1's neutered-to-eager compile killed a CUDA context
    process-wide). Neither refusal reaches the invocation path, and neither is a ruling that
    `torch.compile` cannot ship.
    """
    torch = torch_module()
    original = torch.compile
    original_method = torch.nn.Module.compile
    why = (
        "compiling during construction renames every destination with an `_orig_mod.` "
        "prefix, so the derived tensor requirements matches no checkpoint. Compilation is a "
        "runtime plan decision, never a construction fact — delete the call from load()"
    )

    def refuse(*args: object, **kwargs: object) -> object:
        raise ConstructionFault("torch.compile", why, site=_author_site())

    torch.compile = refuse  # type: ignore[assignment,attr-defined]
    torch.nn.Module.compile = refuse  # type: ignore[assignment,method-assign]
    try:
        yield
    finally:
        torch.compile = original  # type: ignore[assignment,attr-defined]
        torch.nn.Module.compile = original_method  # type: ignore[method-assign]


@dataclass(slots=True)
class _Capability:
    """The hardware variant answered instead of this box's card, and whether it was asked.

    Under derive the answer is never meaningful: a construction that asks WHICH card is
    refused (D3), and `read` is how the harness knows. Under serve the executor answers its
    own card. Availability (`is_available`, `device_count`) is answered and NOT marked: a
    library's import-time environment probe asks it (diffusers' `torch_utils.get_device()`
    on first import, which a factory's lazy `from diffusers import …` triggers), it names no
    card, and a construction that differed by availability would still be judged on its
    real census at serve (D8) — while marking it refused every diffusers package whose
    first diffusers import sits in the factory.
    """

    variant: str
    read: bool = False

    @property
    def major_minor(self) -> tuple[int, int]:
        if self.variant in _NO_CUDA:
            return (0, 0)
        digits = self.variant.removeprefix("sm")
        if not digits.isdigit() or len(digits) < 2:
            raise ConstructionFault(
                "hardware_variant",
                f"{self.variant!r} is not an sm<major><minor> variant name",
            )
        return int(digits[:-1]), int(digits[-1])


#: Hardware variants with no CUDA card: constructions there see none.
_NO_CUDA = ("cpu", "mps")


@contextmanager
def _capability(cap: _Capability) -> Iterator[None]:
    torch = torch_module()
    saved = {
        name: getattr(torch.cuda, name)
        for name in ("get_device_capability", "get_device_name", "is_available", "device_count")
    }

    def mark(value: object, *, read: bool = True) -> Callable[..., object]:
        def answer(*args: object, **kwargs: object) -> object:
            if read:
                cap.read = True
            return value

        return answer

    cuda = cap.variant not in _NO_CUDA
    torch.cuda.get_device_capability = mark(cap.major_minor)  # type: ignore[assignment]
    torch.cuda.get_device_name = mark(cap.variant)  # type: ignore[assignment]
    torch.cuda.is_available = mark(cuda, read=False)  # type: ignore[assignment]
    torch.cuda.device_count = mark(1 if cuda else 0, read=False)  # type: ignore[assignment]
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(torch.cuda, name, value)


def _author_site() -> str:
    """The AUTHOR's frame, skipping the harness and torch — v1's `_package_site`. A
    refusal that names its own file names the wrong line."""
    frame: Any = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if "/cozy_runtime/" not in name and "/torch/" not in name and "<" not in name:
            return f"{name}:{frame.f_lineno} in {frame.f_code.co_name}"
        frame = frame.f_back
    return ""


# --------------------------------------------------------------------------- guard

#: Ops that copy a tensor's contents to the host. Meta refuses them all, loudly but not
#: usefully ("Cannot copy out of meta tensor"); the guard turns each into a typed refusal
#: that names the construct. `__array__` is listed separately from `numpy` on purpose:
#: diffusers' `EulerDiscreteScheduler.set_timesteps` reaches the host through the numpy
#: PROTOCOL, and v1 shipped a shim that listed only `numpy` and missed it entirely.
_EGRESS = frozenset(
    {"item", "numpy", "__array__", "tolist", "__bool__", "__int__", "__float__", "__index__"}
)

#: Ops that MOVE a tensor, or pin it. There are no developer-facing placement knobs (§1) —
#: where work runs is platform policy derived from measurement — so naming a device during
#: construction is not a knob the harness tolerates, it is a fact the author does not own.
_MOVE = frozenset({"cpu", "cuda", "pin_memory", "to_empty"})


def _named_device(args: tuple[object, ...], kwargs: Mapping[str, object]) -> str | None:
    """The device this call NAMES, if any. `.to(dtype)` names none, and must stay legal —
    it is the single most common construction idiom in the ecosystem."""
    candidates = [kwargs.get("device"), *args[1:3]]
    for value in candidates:
        if not isinstance(value, str) and type(value).__name__ != "device":
            continue  # str() of anything else (a tensor, say) costs and names no device
        text = str(value)
        if text.startswith(("cpu", "cuda", "mps", "xpu", "hpu", "npu")):
            return text
    return None


#: In-place ops that WRITE a value without reading the one already there. Every leaf torch
#: ships runs one of these in its own `__init__`, and every fill overwrites the result, so
#: they are invisible to the tensor schema and must not be refused.
_PURE_WRITE = frozenset(
    {
        "zero_",
        "fill_",
        "normal_",
        "uniform_",
        "random_",
        "constant_",
        "ones_",
        "zeros_",
        "eye_",
        "dirac_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "trunc_normal_",
        "orthogonal_",
        "sparse_",
        "requires_grad_",
        "detach_",
        "share_memory_",
    }
)


def _check_inplace(name: str, args: tuple[object, ...], seen: Observations) -> None:
    """The one-fill-transform rule (§1.1), decided by VALUE DEPENDENCE rather than by clock.

    An initialization writes values from nothing; a TRANSFORM reads the values already
    there. Under meta parameters there are no values there, so a transform is provably
    computing on nothing — which is the mechanical reason a load()-time weight transform
    cannot be derived and must not exist. Weight-layout work is STORAGE territory: a
    registered encoding with a runtime provider (the Marlin-class case), never author
    repacking, which would also desynchronize derive from serve.

    `copy_` into a checkpoint destination is refused: a storage-identity test is too
    weak (`w.copy_(w.t().contiguous())` breaks the link while still reading every value).
    Non-persistent buffers are derived from config, never filled from a checkpoint; their
    initialization may finish after registration, as upstream rotary embeddings do.
    """
    import torch

    target = args[0]
    if not isinstance(target, torch.Tensor):
        return
    seen.mutations.setdefault(id(target), []).append(name)
    seen.holds.append(target)
    if name in _PURE_WRITE:
        return
    # A value transform is initialization until the fill can have written the destination:
    # on a constructor temporary (the audio VAE's Kaiser-sinc filter normalizes before it
    # becomes a buffer), or inside the factory call that registered it (timm's
    # `_trunc_normal_` is `uniform_`, `erfinv_`, `mul_`, `add_`, `clamp_`). Once that call
    # returns, the same op transforms checkpoint values and remains storage territory, as
    # it does on every weight-plane tensor the fill installed in a placeholder's stead.
    held, construction = seen.registered.get(id(target), (None, None))
    if seen.filled.get(id(target)) is target:
        construction = None
    elif held is not target:
        return
    if construction is not None and construction is CONSTRUCTION.get():
        return
    raise ConstructionFault(
        f"in-place weight transform `{name}`",
        "this op reads the values already in the destination, and under meta parameters "
        "there are none — a load()-time weight transform is a conformance failure (the "
        "one-fill-transform rule, §1.1). Pre-transposed and tile-permuted layouts are "
        "encodings with runtime providers, never author repacking",
        site=_author_site(),
    )


@dataclass(slots=True)
class Observations:
    """What the guard saw. Read AFTER construction, never acted on during it — a census
    that repairs what it measures cannot name the defect it was written for (v1's rule)."""

    real_allocations: list[tuple[str, str]] = field(default_factory=list)
    mutations: dict[int, list[str]] = field(default_factory=dict)
    holds: list[object] = field(default_factory=list)
    registered: dict[int, tuple[object, object | None]] = field(default_factory=dict)
    #: The plane's tensors the fill installed, held weakly: the plane unmaps them at will.
    filled: weakref.WeakValueDictionary[int, object] = field(
        default_factory=weakref.WeakValueDictionary
    )

    def register(self, destination: object) -> None:
        """A checkpoint destination, and the factory call (if any) that registered it."""
        self.registered[id(destination)] = (destination, CONSTRUCTION.get())

    def fill(self, obj: object) -> None:
        """Every parameter and persistent buffer of the components the fill just wrote."""
        nn = torch_module().nn
        for _, member in _members(obj):
            for module in member.modules() if isinstance(member, nn.Module) else ():
                for tensor in module._parameters.values():
                    if tensor is not None:
                        self.filled[id(tensor)] = tensor
                for name, tensor in module._buffers.items():
                    if tensor is not None and name not in module._non_persistent_buffers_set:
                        self.filled[id(tensor)] = tensor


_PLACEMENT = threading.local()


@contextmanager
def placing() -> Iterator[None]:
    """The RUNTIME is placing bytes — the guard steps aside for exactly this window.

    The guard refuses a device pin because placement is platform policy derived from
    measurement (§1), and the fill plane IS that policy: cr-005 reserves ordinary CUDA
    destinations, stages through pinned host memory and copies on its own stream, all of
    which are device-naming operations by definition. So the fence is on WHO, not on what:
    author code cannot reach this context manager (it lives under `internal/`, outside the
    author surface), and every allocation inside it is ledgered rather than stray.
    """
    prior = getattr(_PLACEMENT, "on", False)
    _PLACEMENT.on = True
    try:
        yield
    finally:
        _PLACEMENT.on = prior


def _guard_mode(seen: Observations) -> Any:
    torch = torch_module()

    class DeriveGuard(torch.overrides.TorchFunctionMode):  # type: ignore[misc,name-defined]
        def __torch_function__(
            self,
            func: Any,
            types: Any,
            args: tuple[object, ...] = (),
            kwargs: Mapping[str, object] | None = None,
        ) -> object:
            if getattr(_PLACEMENT, "on", False):
                return func(*args, **(kwargs or {}))
            name = getattr(func, "__name__", "")
            if name in _MOVE:
                raise ConstructionFault(
                    f"device move via {name}()",
                    "placement is platform policy derived from measurement (§1) and there "
                    "are no developer-facing placement knobs. A construction-time move "
                    "pins the tensor where the author guessed, forever",
                    site=_author_site(),
                )
            named = _named_device(args, kwargs or {})
            if named is not None:
                raise ConstructionFault(
                    f"device pin `device={named}`",
                    "naming a device during construction is not a knob — it survives every "
                    "later placement the runtime performs. Allocate shapes and dtypes and "
                    "let the runtime decide where they live",
                    site=_author_site(),
                )
            if name in _EGRESS and args and not is_virtual(args[0]):
                pass  # a REAL tensor may legally answer; only virtual egress is a fault
            elif name in _EGRESS and args:
                raise ConstructionFault(
                    f"host egress via {name}()",
                    "`__init__` must allocate shapes and dtypes, never VALUES (§1.1): this "
                    "construction reads a parameter's contents, which do not exist until "
                    "the fill. Derived tables that are pure functions of config belong in "
                    "`register_buffer` with no device pin",
                    site=_author_site(),
                )
            if name.endswith("_") and not name.endswith("__") and args:
                _check_inplace(name, args, seen)
            # An init the fill overwrites, on no storage (`nn.init` passes it as `tensor=`): its
            # meta kernels cost the factory seconds and write nothing.
            target = args[0] if args else (kwargs or {}).get("tensor")
            if name in _PURE_WRITE and isinstance(target, torch.Tensor) and is_virtual(target):
                return target
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor) and not is_virtual(result):
                device = str(result.device)
                if device != "cpu" or result.numel() > 4096:
                    seen.real_allocations.append((f"{name} -> {device}", _author_site()))
            return result

    return DeriveGuard()


#: The largest parameter a construction may hold as a constant (`serving_substrate`).
_CONSTANT_BYTES = 1 << 20


@contextmanager
def serving_substrate(variant: str, seen: Observations) -> Iterator[_Capability]:
    """The SERVE substrate: parameters weightless, DERIVED BUFFERS REAL (cr-008b).

    Derive mode puts everything on `meta`, and for derive that is exactly right — it walks
    shapes and matches keys and must allocate nothing at all. Serving cannot use the same
    substrate, and the reason is a class of tensor derive only has to NAME: a non-persistent
    buffer is a pure function of config that NO CHECKPOINT SUPPLIES. CLIP's `position_ids`
    is `arange(77)`; created under a meta device it has a shape and no value, and after
    `to_empty` it is whatever was in that memory. Serving uninitialized memory under a
    derived name is the alternative, and it is not one — the fill plane refuses it by name.

    So this substrate does what a real serving loader does: it lets `__init__` EXECUTE, on
    the host, and swaps each PARAMETER to `meta` the moment it is registered. Parameters are
    the bytes that matter — 6.4 GiB of them here — and they are still never allocated for
    real: the transient peak is one leaf's parameters, freed as soon as the swap lands.
    Buffers are kilobytes and keep the values their own construction computed, which is what
    makes `position_ids` a derived table materialized on the target device rather than a shape
    with garbage in it. The fit decides a persistent buffer: one the checkpoint stores goes to
    meta and binds like a parameter, one it does not keeps that value (`weights.materialize`).

    A constant a module makes a parameter (`requires_grad=False`, at most `_CONSTANT_BYTES`:
    rotary_embedding_torch's frequencies) keeps its value until the factory returns (`BUILT`),
    so a buffer derived from it there has one: SeedVR2's F32 rotary frequencies were a meta
    buffer no checkpoint stores, an unbound placeholder at their first read.
    """
    torch = torch_module()
    nn = torch.nn
    cap = _Capability(variant)
    original = nn.Module.register_parameter
    original_buffer = nn.Module.register_buffer
    constants: list[tuple[Any, str, Any]] = []

    def weightless(module: Any, name: str, held: Any, made: dict[int, Any]) -> None:
        # A device move, performed BY THE RUNTIME, so the guard steps aside exactly here —
        # the fence is on WHO moves bytes, not on what (see `placing`).
        if id(held) not in made:
            with placing():
                made[id(held)] = type(held)(held.to("meta"), requires_grad=held.requires_grad)
        module._parameters[name] = made[id(held)]
        seen.register(made[id(held)])

    def built() -> None:
        made: dict[int, Any] = {}
        for module, name, held in constants:
            if module._parameters.get(name) is held:
                weightless(module, name, held, made)
        constants.clear()

    def register(module: Any, name: str, param: Any) -> None:
        original(module, name, param)
        held = module._parameters.get(name)
        if held is None:
            return
        # A placeholder registered again is a tie (`lm_head.weight = embed_tokens.weight`): it
        # keeps its one object, so the census sees one destination (cr-218).
        if is_virtual(held):
            seen.register(held)
        elif not held.requires_grad and held.nbytes <= _CONSTANT_BYTES:
            constants.append((module, name, held))
        else:
            weightless(module, name, held, {})

    def register_buffer(module: Any, name: str, value: Any, persistent: bool = True) -> None:
        original_buffer(module, name, value, persistent=persistent)
        held = module._buffers.get(name)
        if persistent and isinstance(held, torch.Tensor):
            seen.register(held)

    swap = torch.__future__.get_swap_module_params_on_conversion()
    with ExitStack() as stack:
        torch.__future__.set_swap_module_params_on_conversion(False)
        stack.callback(torch.__future__.set_swap_module_params_on_conversion, swap)
        nn.Module.register_parameter = register
        stack.callback(setattr, nn.Module, "register_parameter", original)
        nn.Module.register_buffer = register_buffer
        stack.callback(setattr, nn.Module, "register_buffer", original_buffer)
        stack.callback(FILLED.reset, FILLED.set(seen.fill))
        stack.callback(built)  # constants a construction outside `loader.construct` made
        stack.callback(BUILT.reset, BUILT.set(built))
        stack.enter_context(refuse_compile())
        stack.enter_context(_capability(cap))
        stack.enter_context(_guard_mode(seen))
        yield cap
