"""The derive-mode harness: execute the author's real `load()` and census what it built.

This is the design's single point of failure (cr-004's falsify-early mandate) and it is
built to be attacked: `scripts/derive-corpus.py` runs a hostile corpus through it and diffs
every derived census against the SAME construction performed for real, because a silent
wrong derivation is worse than any refusal.

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

import re
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, fields
from types import ModuleType
from typing import Any, Literal, cast

from cozy_runtime.author._loader import (
    Artifact,
    Census,
    Loader,
    canonical_dtype,
    census,
    loading,
)

SUBSTRATE = "meta-device"

#: The dtype-diet rule, shared with the checkpoint header's plain/1-absence rule (§1.1):
#: dtype is serialized ONLY where it deviates from bf16. Absence means bf16. Spelled in
#: TensorFS's closed namespace, like every census dtype (cr-087).
DEFAULT_DTYPE = "bf16"

TensorKind = Literal["parameter", "buffer"]

#: A non-persistent buffer this size is not a rope table — it is a hidden cache, and the
#: authoring rule says a GB-scale tensor is a NAMED saved component (§1.1).
DERIVED_TENSOR_ELEMENT_CAP = 1 << 26


class DeriveError(Exception):
    """Base of the harness's typed outcomes. Always names the construct at fault."""

    def __init__(self, construct: str, message: str, *, site: str = "") -> None:
        super().__init__(f"{construct}: {message}" + (f"\n  at {site}" if site else ""))
        self.construct = construct
        self.message = message
        self.site = site


class ConstructionFault(DeriveError):
    """The author's construction is not derivable, and the fix is in the package."""


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
def refuse_compile(*, lazy: bool) -> Iterator[None]:
    """`torch.compile(module)` REFUSES, naming itself; `lazy` refuses `nn.Module.compile` too.

    The wrapping spelling returns an `OptimizedModule` whose every `state_dict` key gains an
    `_orig_mod.` prefix. Under CONSTRUCTION (`lazy=True`) the derived tensor requirements
    would then demand `blk._orig_mod.weight` from checkpoints that carry `blk.weight` — a
    construction no artifact satisfies — and the lazy `nn.Module.compile` is refused there
    too, for a different reason: derive is derive-by-execution, so the callable it installs
    would trace under fake tensors and launch real kernels on fake pointers (pgw#1659, the
    day v1's neutered-to-eager compile killed a CUDA context process-wide).

    Under `Model.warm` (`lazy=False`, cr-110) the tensors are real and the destinations
    filled, so only the wrapping spelling is refused: the object the author's code would then
    call is not the one the construction record and the residency plane hold. The in-place
    spellings (`module.compile()`, diffusers' `compile_repeated_blocks`) move no key,
    parameter or storage and are allowed (#702). Neither refusal reaches the invocation
    path, and neither is a ruling that `torch.compile` cannot ship.
    """
    torch = torch_module()
    original = torch.compile
    original_method = torch.nn.Module.compile
    why = (
        "compiling during construction renames every destination with an `_orig_mod.` "
        "prefix, so the derived tensor requirements matches no checkpoint. Compilation is a "
        "runtime plan decision, never a construction fact — delete the call from load()"
        if lazy
        else "torch.compile(module) wraps the module and prefixes every state_dict key with "
        "`_orig_mod.`, so what warm() then calls is not the object the fill and residency "
        "planes hold — use the in-place spelling, module.compile(), which moves no key"
    )

    def refuse(*args: object, **kwargs: object) -> object:
        target = args[0] if args else kwargs.get("model")
        if not lazy and not isinstance(target, torch.nn.Module):
            # A function or a bound method: nothing is wrapped and no key moves. This is
            # the call `nn.Module.compile` itself makes, over `_call_impl`.
            return original(*args, **kwargs)
        raise ConstructionFault("torch.compile", why, site=_author_site())

    torch.compile = refuse  # type: ignore[assignment,attr-defined]
    if lazy:
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
        if self.variant == "cpu":
            return (0, 0)
        digits = self.variant.removeprefix("sm")
        if not digits.isdigit() or len(digits) < 2:
            raise ConstructionFault(
                "hardware_variant",
                f"{self.variant!r} is not an sm<major><minor> variant name",
            )
        return int(digits[:-1]), int(digits[-1])


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

    cuda = cap.variant != "cpu"
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
        if value is None or isinstance(value, (int, bool)):
            continue
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
    # A value transform on a local constructor temporary is ordinary
    # initialization: the audio VAE's Kaiser-sinc filter normalizes before it
    # becomes a persistent buffer. Once a tensor is a checkpoint destination, the
    # same op is a load-time transform and remains storage territory.
    if name in _PURE_WRITE or seen.registered.get(id(target)) is not target:
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
    registered: dict[int, object] = field(default_factory=dict)


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
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor) and not is_virtual(result):
                device = str(result.device)
                if device != "cpu" or result.numel() > 4096:
                    seen.real_allocations.append((f"{name} -> {device}", _author_site()))
            return result

    return DeriveGuard()


@contextmanager
def _track_registrations(seen: Observations) -> Iterator[None]:
    """Record the edge where a local tensor becomes a checkpoint destination."""
    torch = torch_module()
    nn = torch.nn
    parameter = nn.Module.register_parameter
    buffer = nn.Module.register_buffer

    def register_parameter(module: Any, name: str, value: Any) -> None:
        parameter(module, name, value)
        held = module._parameters.get(name)
        if isinstance(held, torch.Tensor):
            seen.registered[id(held)] = held

    def register_buffer(module: Any, name: str, value: Any, persistent: bool = True) -> None:
        buffer(module, name, value, persistent=persistent)
        held = module._buffers.get(name)
        if persistent and isinstance(held, torch.Tensor):
            seen.registered[id(held)] = held

    nn.Module.register_parameter = register_parameter
    nn.Module.register_buffer = register_buffer
    try:
        yield
    finally:
        nn.Module.register_parameter = parameter
        nn.Module.register_buffer = buffer


@contextmanager
def meta_substrate(variant: str, seen: Observations) -> Iterator[_Capability]:
    """The complete derive-mode substrate: meta parameters, no compile, declared card."""
    torch = torch_module()
    cap = _Capability(variant)
    swap = torch.__future__.get_swap_module_params_on_conversion()
    with ExitStack() as stack:
        torch.__future__.set_swap_module_params_on_conversion(False)
        stack.callback(torch.__future__.set_swap_module_params_on_conversion, swap)
        stack.enter_context(torch.device("meta"))
        stack.enter_context(refuse_compile(lazy=True))
        stack.enter_context(_capability(cap))
        stack.enter_context(_track_registrations(seen))
        stack.enter_context(_guard_mode(seen))
        yield cap


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
    Buffers are kilobytes and they keep the values their own construction computed, which is
    what makes `position_ids` a derived table materialized on the target device rather than
    a shape with garbage in it.
    """
    torch = torch_module()
    nn = torch.nn
    cap = _Capability(variant)
    original = nn.Module.register_parameter
    original_buffer = nn.Module.register_buffer

    def register(module: Any, name: str, param: Any) -> None:
        original(module, name, param)
        held = module._parameters.get(name)
        if held is None:
            return
        # A device move, performed BY THE RUNTIME, so the guard steps aside exactly here —
        # the fence is on WHO moves bytes, not on what (see `placing`).
        with placing():
            module._parameters[name] = type(held)(held.to("meta"), requires_grad=held.requires_grad)
        held = module._parameters[name]
        seen.registered[id(held)] = held

    def register_buffer(module: Any, name: str, value: Any, persistent: bool = True) -> None:
        original_buffer(module, name, value, persistent=persistent)
        held = module._buffers.get(name)
        if persistent and isinstance(held, torch.Tensor):
            seen.registered[id(held)] = held

    swap = torch.__future__.get_swap_module_params_on_conversion()
    with ExitStack() as stack:
        torch.__future__.set_swap_module_params_on_conversion(False)
        stack.callback(torch.__future__.set_swap_module_params_on_conversion, swap)
        nn.Module.register_parameter = register
        stack.callback(setattr, nn.Module, "register_parameter", original)
        nn.Module.register_buffer = register_buffer
        stack.callback(setattr, nn.Module, "register_buffer", original_buffer)
        stack.enter_context(refuse_compile(lazy=True))
        stack.enter_context(_capability(cap))
        stack.enter_context(_guard_mode(seen))
        yield cap


# ---------------------------------------------------------------------------- rows

#: Fusion, permutation, re-keying, quantization and subclass reconstruction are
#: UNREPRESENTABLE here: a `LogicalTensor` carries ONE name and no source key, part list,
#: transform, scale reference or converter. Those are AOT artifact work (§1.1).
#: `_prove_unrepresentable()` runs at import and holds each row's field set to an EQUALITY,
#: so a new fact enters only through a diff that touches this table.


@dataclass(frozen=True, slots=True)
class LogicalTensor:
    """One logical tensor the construction demands, and the ONE destination it lands in."""

    key: str
    component: str
    shape: tuple[int, ...]
    kind: TensorKind
    module_class: str
    alias_group: int | None
    dtype: str | None
    """Present only where it deviates from bf16 (the dtype-diet rule). Where present it is
    the destination's COMPUTE/interface contract and the decode target."""


@dataclass(frozen=True, slots=True)
class DerivedTensor:
    """A tensor the construction BUILDS and no checkpoint fills — a non-persistent buffer.

    It carries NO device, and neither does `LogicalTensor`: placement is platform policy
    derived from measurement (§1), so a device is never a construction fact.
    """

    key: str
    component: str
    shape: tuple[int, ...]
    dtype: str | None


@dataclass(frozen=True, slots=True)
class DestinationSet:
    """The COMPLETE destination set of one named component, in construction order."""

    component: str
    module_class: str
    training: bool
    compiled: bool
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeclaredCache:
    """A `loader.cache(...)` handle: memory-only, declared-bounded (§1.3)."""

    name: str
    max_entries: int
    max_bytes: int


_ROW_FIELDS: Mapping[type, frozenset[str]] = {
    LogicalTensor: frozenset(
        {"key", "component", "shape", "kind", "module_class", "alias_group", "dtype"}
    ),
    DerivedTensor: frozenset({"key", "component", "shape", "dtype"}),
    DestinationSet: frozenset({"component", "module_class", "training", "compiled", "keys"}),
    DeclaredCache: frozenset({"name", "max_entries", "max_bytes"}),
}


def _prove_unrepresentable() -> None:
    for row, expected in _ROW_FIELDS.items():
        actual = {f.name for f in fields(row)}
        if actual != expected:
            raise HarnessGap(
                "row shape",
                f"{row.__name__} fields are {sorted(actual)}, the reviewed set is "
                f"{sorted(expected)} — destination-side fusion, permutation, re-key, "
                "quantization and subclass reconstruction stay UNREPRESENTABLE",
            )


_prove_unrepresentable()


# -------------------------------------------------------------------------- census


@dataclass(frozen=True, slots=True)
class RichCensus:
    """The deep walk: every parameter and buffer, its owning leaf class, its alias group,
    and the non-persistent tensors no checkpoint will ever fill."""

    logical: tuple[LogicalTensor, ...]
    derived: tuple[DerivedTensor, ...]
    sets: tuple[DestinationSet, ...]


def _ordered_destination_sets(
    shallow: Sequence[Census], rich: RichCensus
) -> tuple[DestinationSet, ...]:
    """Attach deep metadata to the Loader's one authoritative destination traversal.

    `census()` is the walk `Loader._fill` executes: component mapping order, then each
    module's exact `state_dict()` order. The deep walk supplies kind/class/alias metadata,
    but its `named_parameters()`-then-`named_buffers()` order is not another traversal.
    """
    templates: dict[str, DestinationSet] = {}
    for row in rich.sets:
        if row.component in templates:
            raise ConstructionFault(
                "duplicate component",
                f"component {row.component!r} is constructed more than once; one component "
                "has one ordered DestinationSet",
            )
        templates[row.component] = row

    ordered: list[DestinationSet] = []
    for view in shallow:
        keys: dict[str, list[str]] = {name: [] for name in view.components}
        for destination in view.destinations:
            if destination.component not in keys:
                raise ConstructionFault(
                    "component census",
                    f"destination {destination.key!r} belongs to a component absent from "
                    "the Loader census",
                )
            keys[destination.component].append(destination.key)
        for name in view.components:
            template = templates.pop(name, None)
            if template is None:
                raise ConstructionFault(
                    "component census",
                    f"the Loader constructs component {name!r} and the deep census does not",
                )
            if set(keys[name]) != set(template.keys):
                raise ConstructionFault(
                    "component census",
                    f"the Loader and deep census disagree on component {name!r}",
                )
            ordered.append(
                DestinationSet(
                    component=name,
                    module_class=template.module_class,
                    training=template.training,
                    compiled=template.compiled,
                    keys=tuple(keys[name]),
                )
            )
    if templates:
        raise ConstructionFault(
            "component census",
            f"the deep census found components the Loader did not: {sorted(templates)}",
        )
    return tuple(ordered)


def _load_typed(model: Any, loader: Any) -> None:
    """Run the author's `load()`, turning torch's own untyped refusals into named ones.

    A raw `NotImplementedError` from a missing meta kernel names a torch internal and a
    line inside `_refs`; the package author needs the OP and the reason. The two causes
    are genuinely different lanes: a data-dependent op has no correct shape function and
    the package must stop asking, while a custom op merely LACKS one and the harness (or
    the package's own `torch.library` registration) can supply it.
    """
    from cozy_runtime.author._errors import ConformanceError
    from cozy_runtime.internal.sandbox import FenceViolation

    try:
        model.load(loader)
    except (DeriveError, FenceViolation, ConformanceError):
        raise
    except NotImplementedError as exc:
        raise _classify(exc) from exc
    except RuntimeError as exc:
        raise _classify(exc) from exc


#: Torch messages the harness can name. Each entry is (fragment, construct, explanation).
#: The list is short on purpose: an unrecognized escape becomes a HarnessGap that CONFESSES
#: rather than a plausible-looking guess, which is the gap doctrine's whole point.
_KNOWN: tuple[tuple[str, str, str], ...] = (
    (
        "data-independent implementation does not exist",
        "data-dependent shape",
        "`__init__` allocates shapes and dtypes, never values (§1.1), so an op whose OUTPUT "
        "SHAPE depends on tensor contents has no derivable answer",
    ),
    (
        "uninitialized parameter",
        "torch.nn.LazyModuleMixin",
        "a lazy module has NO shape until its first forward, so there is no tensor requirements "
        "to derive and no checkpoint key to match. Declare the shapes in `__init__`",
    ),
    (
        "no CUDA GPUs are available",
        "device pin `device=cuda`",
        "the derivation runs with no accelerator by construction; construction that needs "
        "a card is construction that decides placement",
    ),
    (
        "Cannot copy out of meta tensor",
        "host egress",
        "the construction read a parameter's contents, which do not exist until the fill",
    ),
)


def _classify(exc: Exception) -> DeriveError:
    text = str(exc)
    first = text.split("\n")[0][:220]
    for fragment, construct, why in _KNOWN:
        if fragment.lower() in text.lower():
            return ConstructionFault(construct, f"{first} — {why}", site=_author_site())
    operator = re.match(r"([A-Za-z_][\w.]*::[\w.]+)", text)
    if operator is not None or "Meta kernel registered" in text or "fake impl" in text:
        return HarnessGap(
            operator.group(1) if operator else "meta kernel",
            f"{first}. A meta kernel is a SHAPE FUNCTION and is the only fact a derivation "
            "needs from an operator — register one with `torch.library.register_fake` "
            "beside the op. The harness will not guess a shape and will not silently drop "
            "the tensor",
            site=_author_site(),
        )
    return HarnessGap(
        f"unclassified {type(exc).__name__}",
        f"{first}. The harness has no name for this construct, which is itself the "
        "finding: the fix lane is a named case here or in `_KNOWN`, never a new "
        "author-declared fact",
        site=_author_site(),
    )


def component_members(built: Sequence[object]) -> list[tuple[str, object]]:
    """The component roots, by the SAME rule the fill plane used (`_loader.census`).

    Two component tables that can disagree are two schemas, and only one of them gets
    filled — so this reuses the loader's own member walk rather than restating it.
    """
    from cozy_runtime.author._loader import ModuleLike, Scheduler, _members

    out: list[tuple[str, object]] = []
    for obj in built:
        for name, member in _members(obj):
            if member is None:
                raise ConstructionFault(
                    "unbuilt component",
                    f"the constructed object declares component {name!r} and left it None. "
                    "A lazy component build swallows per-component failures into a log "
                    "line, so an unbuilt component must be named here or it becomes a "
                    "missing destination set nobody notices (v1 tcg#65)",
                )
            if isinstance(member, Scheduler) or not isinstance(member, ModuleLike):
                continue
            out.append((name, member))
    return out


def walk(
    components: Sequence[tuple[str, object]],
    strays: list[str] | None = None,
    substrate: str = "meta",
) -> RichCensus:
    """Census the component roots.

    SUBSTRATE-AGNOSTIC on purpose: the corpus runs this same function over a real CPU
    construction to hunt a silent wrong derivation, which only works if the walk cannot
    tell the two apart. `strays` is the one substrate-aware output — pass a list to
    collect every tensor that is not on `substrate`, and `None` when censusing a real
    construction where landing anywhere is the whole idea.

    `substrate` is what lets the SERVE side use the same arm the derive side does: under
    derivation every tensor must be on meta, and after a fill every tensor must be on the
    device the fill plane placed it on. Same question, same code, two answers.
    """
    torch = torch_module()
    logical: list[LogicalTensor] = []
    derived: list[DerivedTensor] = []
    sets: list[DestinationSet] = []
    groups: dict[object, int] = {}

    for name, member in components:
        if not isinstance(member, torch.nn.Module):
            continue
        _refuse_wrapper(name, member)
        classes = {prefix: type(sub).__name__ for prefix, sub in member.named_modules()}
        skip = _non_persistent(member)
        keys: list[str] = []

        for local, tensor in _tensors(member):
            key = f"{name}.{local}"
            leaf = classes.get(local.rpartition(".")[0], type(member).__name__)
            kind = "parameter" if isinstance(tensor, torch.nn.Parameter) else "buffer"
            if strays is not None and str(tensor.device) != substrate:
                strays.append(f"{key} on {tensor.device}")
            if kind == "buffer" and local in skip:
                derived.append(
                    DerivedTensor(
                        key=key,
                        component=name,
                        shape=tuple(int(d) for d in tensor.shape),
                        dtype=_dtype(tensor),
                    )
                )
                continue
            logical.append(
                LogicalTensor(
                    key=key,
                    component=name,
                    shape=tuple(int(d) for d in tensor.shape),
                    kind=cast("Any", kind),
                    module_class=leaf,
                    alias_group=_group(tensor, groups),
                    dtype=_dtype(tensor),
                )
            )
            keys.append(key)

        sets.append(
            DestinationSet(
                component=name,
                module_class=type(member).__name__,
                training=bool(member.training),
                compiled=False,
                keys=tuple(keys),
            )
        )
    return RichCensus(tuple(logical), tuple(derived), tuple(sets))


def _tensors(module: object) -> Iterator[tuple[str, Any]]:
    """Parameters then buffers, WITHOUT de-duplication — a tie must appear under every
    name that reaches it or the alias census cannot see it (v1 `census._tie_groups`)."""
    yield from module.named_parameters(remove_duplicate=False)  # type: ignore[attr-defined]
    yield from module.named_buffers(remove_duplicate=False)  # type: ignore[attr-defined]


def _non_persistent(root: Any) -> frozenset[str]:
    """Read `_non_persistent_buffers_set` directly. A `state_dict()` diff would also report
    a buffer a subclass removed in `state_dict`, and would materialize a dict of every
    tensor in the module to answer a question about a handful of names (v1's reasoning)."""
    names: set[str] = set()
    for prefix, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            names.add(f"{prefix}.{local}" if prefix else local)
    return frozenset(names)


def _group(tensor: Any, groups: dict[object, int]) -> int | None:
    """Alias identity by STORAGE, falling back to object identity.

    Never `_tied_weights_keys`: that lists names a class MIGHT tie, and whether a tie is
    live is a config question (`tie_word_embeddings`) — reading it as an answer
    false-positives real models (v1, on `Qwen2_5_VLForConditionalGeneration`).
    """
    try:
        token: object = (tensor.untyped_storage()._cdata, int(tensor.storage_offset()))
    except Exception:
        token = id(tensor)
    if token in groups:
        return groups[token]
    groups[token] = len(groups)
    return groups[token]


def _dtype(tensor: Any) -> str | None:
    """The dtype-diet rule: absence means bf16 (§1.1). Canonical TensorFS spelling."""
    name = canonical_dtype(tensor.dtype)
    return None if name == DEFAULT_DTYPE else name


def _refuse_wrapper(name: str, member: Any) -> None:
    if hasattr(member, "_orig_mod"):
        raise ConstructionFault(
            "torch.compile",
            f"component {name!r} is an OptimizedModule wrapping {type(member._orig_mod).__name__}",
        )
    if type(member).__name__ in ("LazyLinear", "LazyConv2d") or any(
        type(p).__name__ == "UninitializedParameter" for _, p in member.named_parameters()
    ):
        raise ConstructionFault(
            "torch.nn.LazyModuleMixin",
            f"component {name!r} holds an UninitializedParameter: a lazy module has NO "
            "shape until its first forward, so there are no TensorRequirements to derive and no "
            "checkpoint key to match. Declare the shapes in `__init__`",
        )


# ------------------------------------------------------------------- the derivation


@dataclass(frozen=True, slots=True)
class Derivation:
    """The TensorRequirements one `load()` derives, plus its bookkeeping.

    It is a pure function of (factory, config document): no candidate, no snapshot, no
    hardware variant and no verdict (model-code-fit §3, D3). The `factory` and
    `logical_tensors` are the rows a checkpoint header is matched against.
    """

    logical_tensors: tuple[LogicalTensor, ...]
    destination_sets: tuple[DestinationSet, ...]
    derived: tuple[DerivedTensor, ...]
    caches: tuple[DeclaredCache, ...]
    component_use: Mapping[str, tuple[str, ...]]
    observations: Observations

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(d.component for d in self.destination_sets)

    def rows(self) -> list[tuple[str, str, str | None, tuple[int, ...]]]:
        """The code destinations as `(component, key, dtype?, shape)` — the one projection
        TensorFS reads, shared with the serving census."""
        return [
            (
                t.component,
                t.key[len(t.component) + 1 :],
                _logical_dtype_constraint(t.dtype),
                t.shape,
            )
            for t in self.logical_tensors
        ]


def _logical_dtype_constraint(dtype: str | None) -> str | None:
    """Ordinary bf16/f16 weights may convert; semantic islands retain exact dtype."""

    return None if dtype in (None, "bf16", "f16") else dtype


def derive(
    model: Any,
    artifact: Artifact,
    *,
    component_use: Mapping[str, Sequence[str]] | None = None,
    mode: str = "derive",
    backend: Any = None,
    substrate_device: str = "meta",
) -> Derivation:
    """Execute `model.load(loader)` on the meta substrate and derive TensorRequirements.

    `mode="serve"` runs the IDENTICAL substrate and the identical walk, and differs only in
    that the fill path runs. That is the derive/serve fence made mechanical rather than
    argued: the two modes cannot produce different rows, because there is one factory, one
    substrate and one census, and the only branch between them is whether bytes are moved
    into destinations the census has already agreed on.

    A construction that reads the device is refused (D3): TensorRequirements is
    hardware-independent, and a factory that branches on the card has no single one.
    """
    torch = torch_module()
    if mode == "derive" and torch.cuda.is_initialized():
        raise HarnessGap(
            "cuda_context",
            "a CUDA context existed before derivation — the harness runs with no GPU, and a "
            "live context means the fence did not hold",
        )
    seen = Observations()
    with meta_substrate("cpu", seen) as cap:
        loader = Loader(artifact, owner=model, mode=cast("Any", mode), backend=backend)
        with loading(loader):
            _load_typed(model, loader)
        built = loader.constructed()

    if cap.read:
        raise ConstructionFault(
            "hardware-conditional construction",
            "the construction read the device capability — TensorRequirements is a function "
            "of the config document alone (model-code-fit D3); construction that differs "
            "per card is unrepresentable",
        )
    _refuse_real_allocations(seen)

    shallow = [census(obj) for obj in built]
    strays: list[str] = []
    rich = walk(component_members(built), strays, substrate_device)
    if strays:
        raise ConstructionFault(
            "CPU-stray tensor",
            f"{len(strays)} destination(s) are not on the {substrate_device!r} substrate: "
            f"{', '.join(strays[:4])}. Every tensor lives where the runtime places it; one "
            "that names its own device serves from there forever and the ledger never "
            "learns it exists",
        )
    _refuse_hidden_caches(rich)

    logical = tuple(sorted(rich.logical, key=lambda t: t.key))
    destination_sets = _ordered_destination_sets(shallow, rich)
    _check_agreement(logical, destination_sets, [d.key for c in shallow for d in c.destinations])
    constructed = tuple(d.component for d in destination_sets)
    return Derivation(
        logical_tensors=logical,
        destination_sets=destination_sets,
        derived=tuple(sorted(rich.derived, key=lambda d: d.key)),
        caches=tuple(
            sorted(
                (DeclaredCache(c.name, c.max_entries, c.max_bytes) for c in loader.caches()),
                key=lambda c: c.name,
            )
        ),
        component_use=_validate_component_use(constructed, component_use or {}),
        observations=seen,
    )


def _check_agreement(
    logical: Sequence[LogicalTensor],
    destination_sets: Sequence[DestinationSet],
    destinations: Sequence[str],
) -> None:
    """EXACT ordered one-to-one agreement between the deep walk and the Loader census.

    Membership proves the tensor schema; sequence proves the rows record the same construction
    traversal the Loader hands to the fill plane.
    """
    actual = list(destinations)
    keys = [t.key for t in logical]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ConstructionFault(
            "duplicate destination",
            f"{len(duplicates)} logical tensor key(s) appear twice: {', '.join(duplicates[:6])} "
            "— a destination is ONE tensor; a repeated key is a re-key by another name",
        )
    covered: list[str] = []
    for dset in destination_sets:
        covered.extend(dset.keys)
    if sorted(covered) != sorted(keys):
        only_set = sorted(set(covered) - set(keys))
        only_log = sorted(set(keys) - set(covered))
        raise ConstructionFault(
            "incomplete destination set",
            "the DestinationSets do not partition the logical tensors exactly "
            f"(+{only_set[:4]} / -{only_log[:4]})",
        )
    missing = sorted(set(keys) - set(actual))
    surplus = sorted(set(actual) - set(keys))
    if missing or surplus:
        raise ConstructionFault(
            "census disagreement",
            f"the harness census and the fill plane disagree: {len(missing)} deep-only "
            f"{missing[:4]}, {len(surplus)} fill-only {surplus[:4]}",
        )
    if covered != actual:
        first = next(
            (
                index
                for index, (declared, filled) in enumerate(zip(covered, actual, strict=False))
                if declared != filled
            ),
            min(len(covered), len(actual)),
        )
        declared = covered[first] if first < len(covered) else "<end>"
        filled = actual[first] if first < len(actual) else "<end>"
        raise ConstructionFault(
            "census order disagreement",
            f"the deep census and the Loader census contain the same destinations in a "
            f"different order at index {first}: deep {declared!r}, Loader {filled!r} — "
            "header/load order is construction identity",
        )


def _refuse_real_allocations(seen: Observations) -> None:
    if not seen.real_allocations:
        return
    what, site = seen.real_allocations[0]
    raise ConstructionFault(
        "real allocation during construction",
        f"{len(seen.real_allocations)} tensor(s) with real storage were built while "
        f"deriving — first: {what}. `__init__` must be meta-device safe: allocate shapes "
        "and dtypes, never values (§1.1). A CPU-stray buffer serves from the wrong device "
        "and a pinned one is invisible to the ledger",
        site=site,
    )


def _refuse_hidden_caches(rich: RichCensus) -> None:
    for tensor in rich.derived:
        elements = 1
        for dim in tensor.shape:
            elements *= dim
        if elements > DERIVED_TENSOR_ELEMENT_CAP:
            raise ConstructionFault(
                "hidden cache",
                f"{tensor.key} is a non-persistent buffer of {elements} elements — at that "
                "scale it is a NAMED saved component, not a derived table (§1.1). No "
                "checkpoint fills it and no ledger meters it",
            )


def _validate_component_use(
    constructed: Sequence[str], declared: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Prove every declared component NAME exists in the derived construction.

    This is the second validation clock (§1.1): `describe` checked syntax and placement and
    could not prove existence, because the artifact was not selected yet. Nothing about
    call ORDER is derived or checked here — a declaration is a possible-set, never a phase.
    """
    built = set(constructed)
    out: dict[str, tuple[str, ...]] = {}
    for method, names in sorted(declared.items()):
        unknown = sorted(set(names) - built)
        if unknown:
            raise ConstructionFault(
                "unknown component",
                f"@uses_components on {method}() names {unknown} which this construction "
                f"does not build (it builds {sorted(built)}) — the existence proof runs at "
                "exact binding, and this pair is rejected",
            )
        out[method] = tuple(sorted(names))
    return out
