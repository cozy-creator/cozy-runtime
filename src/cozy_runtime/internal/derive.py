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
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

from cozy_runtime.author._loader import Artifact, Loader, census, loading
from cozy_runtime.internal.contract import (
    DEFAULT_DTYPE,
    Candidate,
    DeclaredCache,
    DerivedTensor,
    DestinationSet,
    LogicalTensor,
    ModelConstructionAttestation,
    ModelConstructionContract,
    check_agreement,
)

SUBSTRATE = "meta-device"

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
    """The author's construction is not derivable, and the fix is in the endpoint."""


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
def _refuse_compile() -> Iterator[None]:
    """`torch.compile` during construction REFUSES, naming itself.

    It wraps the module in an `OptimizedModule` whose every `state_dict` key gains an
    `_orig_mod.` prefix, so the derived code topology would demand `blk._orig_mod.weight`
    from checkpoints that carry `blk.weight` — a construction that no artifact on earth
    satisfies. v1 could not refuse (its h3 endpoint compiled the VAE decoder for a real
    1.68x serve win) so it neutered compile to eager for the drive (pgw#1659, after a
    compiled callable handed fake tensors read a fake data pointer, launched a real kernel,
    and killed the CUDA context process-wide while the drive finished green against the
    corpse). v2 has no graph-mint consumer that needs the call to survive, so the honest
    answer is the refusal: the fix lane is the endpoint deleting the call.
    """
    torch = torch_module()
    original = torch.compile
    original_method = torch.nn.Module.compile

    def refuse(*args: object, **kwargs: object) -> object:
        raise ConstructionFault(
            "torch.compile",
            "compiling during construction renames every destination with an `_orig_mod.` "
            "prefix, so the derived code topology matches no checkpoint. Compilation is a "
            "runtime plan decision, never a construction fact — delete the call from load()",
            site=_author_site(),
        )

    torch.compile = refuse  # type: ignore[assignment,attr-defined]
    torch.nn.Module.compile = refuse  # type: ignore[assignment,method-assign]
    try:
        yield
    finally:
        torch.compile = original  # type: ignore[assignment,attr-defined]
        torch.nn.Module.compile = original_method  # type: ignore[method-assign]


@dataclass(slots=True)
class _Capability:
    """The DECLARED hardware variant, answered instead of this box's card.

    Construction that gates on `sm89` vs `sm90` derives PER VARIANT (cr-004): one contract
    each, never one contract for both. `read` records whether the construction actually
    asked — an undeclared variant on a construction that asks is a refusal, and a declared
    variant on one that never asks is simply the same contract twice.
    """

    variant: str
    read: bool = False

    @property
    def major_minor(self) -> tuple[int, int]:
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

    def mark(value: object) -> Callable[..., object]:
        def answer(*args: object, **kwargs: object) -> object:
            cap.read = True
            return value

        return answer

    torch.cuda.get_device_capability = mark(cap.major_minor)  # type: ignore[assignment]
    torch.cuda.get_device_name = mark(cap.variant)  # type: ignore[assignment]
    torch.cuda.is_available = mark(True)  # type: ignore[assignment]
    torch.cuda.device_count = mark(1)  # type: ignore[assignment]
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(torch.cuda, name, value)


def _author_site() -> str:
    """The AUTHOR's frame, skipping the harness and torch — v1's `_endpoint_site`. A
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
#: they are invisible to the topology and must not be refused.
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

    `copy_` is refused unconditionally, and the reason is the substrate rather than a
    heuristic: under meta parameters EVERY tensor in scope is empty, so a construction-time
    `copy_` either reads the destination or copies something equally empty. Neither is a
    fact. (A storage-identity test was tried first and is too weak — `w.copy_(w.t()
    .contiguous())` breaks the storage link while still reading every value.)
    """
    import torch

    target = args[0]
    if not isinstance(target, torch.Tensor):
        return
    seen.mutations.setdefault(id(target), []).append(name)
    seen.holds.append(target)
    if name in _PURE_WRITE:
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
def meta_substrate(variant: str, seen: Observations) -> Iterator[_Capability]:
    """The complete derive-mode substrate: meta parameters, no compile, declared card."""
    torch = torch_module()
    cap = _Capability(variant)
    swap = torch.__future__.get_swap_module_params_on_conversion()
    with ExitStack() as stack:
        torch.__future__.set_swap_module_params_on_conversion(False)
        stack.callback(torch.__future__.set_swap_module_params_on_conversion, swap)
        stack.enter_context(torch.device("meta"))
        stack.enter_context(_refuse_compile())
        stack.enter_context(_capability(cap))
        stack.enter_context(_guard_mode(seen))
        yield cap


# -------------------------------------------------------------------------- census


@dataclass(frozen=True, slots=True)
class RichCensus:
    """The deep walk: every parameter and buffer, its owning leaf class, its alias group,
    and the non-persistent tensors no checkpoint will ever fill."""

    logical: tuple[LogicalTensor, ...]
    derived: tuple[DerivedTensor, ...]
    sets: tuple[DestinationSet, ...]


def _load_typed(model: Any, loader: Any) -> None:
    """Run the author's `load()`, turning torch's own untyped refusals into named ones.

    A raw `NotImplementedError` from a missing meta kernel names a torch internal and a
    line inside `_refs`; the endpoint author needs the OP and the reason. The two causes
    are genuinely different lanes: a data-dependent op has no correct shape function and
    the endpoint must stop asking, while a custom op merely LACKS one and the harness (or
    the endpoint's own `torch.library` registration) can supply it.
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
        "a lazy module has NO shape until its first forward, so there is no code topology "
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

    Two component tables that can disagree are two topologies, and only one of them gets
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
    """The dtype-diet rule: absence means bf16 (§1.1)."""
    name = str(tensor.dtype).removeprefix("torch.")
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
            "shape until its first forward, so there is no code topology to derive and no "
            "checkpoint key to match. Declare the shapes in `__init__`",
        )


# ------------------------------------------------------------------- the derivation


@dataclass(frozen=True, slots=True)
class Derivation:
    contract: ModelConstructionContract
    attestation: ModelConstructionAttestation
    observations: Observations


def derive(
    model: Any,
    artifact: Artifact,
    *,
    release: str,
    application: str,
    hardware_variant: str,
    declared_variants: Sequence[str] = (),
    component_use: Mapping[str, Sequence[str]] | None = None,
    timestep_plans: Mapping[str, str] | None = None,
    mode: str = "derive",
    backend: Any = None,
    substrate_device: str = "meta",
) -> Derivation:
    """Execute `model.load(loader)` on the meta substrate and derive its exact contract.

    `mode="serve"` runs the IDENTICAL substrate and the identical walk, and differs only in
    that the fill path runs. That is the derive/serve fence made mechanical rather than
    argued: the two modes cannot produce different contract bytes, because there is one
    factory, one substrate and one census, and the only branch between them is whether
    bytes are moved into destinations the census has already agreed on.
    """
    torch = torch_module()
    if mode == "derive" and torch.cuda.is_initialized():
        raise HarnessGap(
            "cuda_context",
            "a CUDA context existed before derivation — the harness runs with no GPU, and a "
            "live context means the fence did not hold",
        )
    seen = Observations()
    with meta_substrate(hardware_variant, seen) as cap:
        loader = Loader(artifact, owner=model, mode=cast("Any", mode), backend=backend)
        with loading(loader):
            _load_typed(model, loader)
        built = loader.constructed()

    if cap.read and not declared_variants:
        raise ConstructionFault(
            "hardware-conditional construction",
            f"the construction read the device capability while deriving for "
            f"{hardware_variant!r} with no declared variant set — construction that differs "
            "per card derives PER DECLARED VARIANT, never once (cr-004)",
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

    contract = ModelConstructionContract(
        candidate=Candidate(
            release=release,
            application=application,
            model_class=f"{type(model).__module__}:{type(model).__qualname__}",
            factory="|".join(r.factory_identity for r in loader.records()),
            snapshot=artifact.snapshot,
            config_digest=artifact.config.digest(),
            hardware_variant=hardware_variant,
        ),
        logical_tensors=tuple(sorted(rich.logical, key=lambda t: t.key)),
        destination_sets=tuple(sorted(rich.sets, key=lambda d: d.component)),
        derived=tuple(sorted(rich.derived, key=lambda d: d.key)),
        caches=tuple(
            sorted(
                (DeclaredCache(c.name, c.max_entries, c.max_bytes) for c in loader.caches()),
                key=lambda c: c.name,
            )
        ),
    )
    check_agreement(contract, [d.key for c in shallow for d in c.destinations])
    validated = _validate_component_use(contract, component_use or {})
    attestation = ModelConstructionAttestation(
        contract_digest=contract.digest,
        substrate=SUBSTRATE,
        torch_version=torch.__version__,
        fences=("no_compile", "no_cuda", "no_host_egress", "no_real_allocation", "meta_parameters"),
        double_derived=False,
        component_use=validated,
        timestep_plans=dict(timestep_plans or {}),
    )
    return Derivation(contract, attestation, seen)


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
    contract: ModelConstructionContract, declared: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Prove every declared component NAME exists in the derived construction.

    This is the second validation clock (§1.1): `describe` checked syntax and placement and
    could not prove existence, because the artifact was not selected yet. Nothing about
    call ORDER is derived or checked here — a declaration is a possible-set, never a phase.
    """
    built = set(contract.components)
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
