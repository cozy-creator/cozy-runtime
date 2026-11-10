"""Install Ulysses hooks on declared sharded components.

Every GPU of the group calls the component itself (`spmd`); Diffusers' hooks split its inputs and
gather its outputs inside the call. Select the attention backend before installing CP; existing
accelerate wrappers on sharded components are refused.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from cozy_runtime.author._model import _declared_state
from cozy_runtime.internal.parallel import spmd
from cozy_runtime.internal.parallel.plan import GroupRefusal

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


class ContextParallelUnavailable(GroupRefusal):
    """This generation cannot be sharded, named exactly."""

    default_code = "context_parallel_unavailable"


@dataclass(slots=True)
class CpComms:
    """The GROUP's communication facts, passed explicitly - never the default group read
    ambiently (pgw#773's cross-group corruption). `pg` is replaced when the group forms
    again; every installed component reads it per collective."""

    pg: ProcessGroup | None
    rank: int
    device: torch.device | str


class _GroupMesh:
    """The mesh shape diffusers' `ParallelConfig.setup` reads: ONE Ulysses axis of `size`
    over the group's process group; the ring axis is degree 1 and refuses if asked for."""

    def __init__(
        self, comms: CpComms, size: int, local_rank: int, *, axis: str = "ulysses"
    ) -> None:
        self._comms = comms
        self._size = int(size)
        self._local_rank = int(local_rank)
        self._axis = axis

    def size(self) -> int:
        return self._size

    def get_group(self) -> ProcessGroup | None:
        if self._axis == "ring":
            raise ContextParallelUnavailable(
                "ring attention is not supported: the runtime installs Ulysses sequence "
                "parallelism only (ring_degree=1)"
            )
        return self._comms.pg

    def get_local_rank(self) -> int:
        return self._local_rank

    def _flatten(self) -> _GroupMesh:
        return _GroupMesh(self._comms, self._size, self._local_rank)

    def __getitem__(self, key: str | tuple[str, ...]) -> _GroupMesh:
        if isinstance(key, tuple) or key == "ulysses":
            return _GroupMesh(self._comms, self._size, self._local_rank)
        if key == "ring":
            return _GroupMesh(self._comms, 1, 0, axis="ring")
        raise KeyError(key)


_CP_PLAN_ATTR = "_cp_plan"


def _modules_of(model: object) -> Iterator[tuple[str, nn.Module]]:
    """Every torch module reachable from the model's DECLARED state, and the components
    of any pipeline among them. Declared state is the author's statement of what the
    model holds (§1.1); nothing else is walked."""
    seen: set[int] = set()
    for name in _declared_state(type(model)):
        held = getattr(model, name, None)
        if held is None:
            continue
        candidates = [(name, held)]
        components = getattr(held, "components", None)
        if isinstance(components, dict):
            candidates += [(f"{name}.{key}", value) for key, value in components.items()]
        for extra in ("transformer", "transformer_2"):
            if hasattr(held, extra):
                candidates.append((f"{name}.{extra}", getattr(held, extra)))
        for label, module in candidates:
            if module is None or id(module) in seen or not hasattr(module, "named_modules"):
                continue
            seen.add(id(module))
            yield label, module


def sharding_candidates(model: object) -> Sequence[tuple[str, object]]:
    """The components that DECLARE a context-parallel plan and can be told to shard."""
    return _candidates(model)


def _candidates(model: object) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in _modules_of(model)
        if getattr(module, _CP_PLAN_ATTR, None) and hasattr(module, "enable_parallelism")
    ]


def install_context_parallel(
    model: object, *, degree: int, comms: CpComms | None
) -> tuple[str, ...]:
    """Install Ulysses sequence parallelism at `degree` on every declaring component."""
    if int(degree) <= 1:
        return ()
    components = _candidates(model)
    if not components:
        raise ContextParallelUnavailable(
            "no declared component carries a `_cp_plan`; this model has not been adapted "
            "for context parallelism and sharding it would produce silently wrong output"
        )
    _refuse_if_forward_wrapped(components)
    if comms is None:
        raise ContextParallelUnavailable(
            "context parallelism at degree>1 requires the group's explicit process group "
            "(CpComms); installing against the default process group is exactly the "
            "pgw#773 cross-group corruption"
        )
    installed: list[str] = []
    for name, component in components:
        refuse_unless_heads_divisible(heads=_declared_heads(component), degree=int(degree))
        _install_on_component(component, degree=int(degree), comms=comms)
        spmd.install_call_check(component, name)
        installed.append(name)
    return tuple(installed)


def _install_on_component(component: nn.Module, *, degree: int, comms: CpComms) -> None:
    import torch

    try:
        from diffusers.hooks.context_parallel import apply_context_parallel
        from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
        from diffusers.models.attention import AttentionModuleMixin
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
        )
        from diffusers.models.attention_processor import Attention

        attention_modules = _sharded_modules(component)
    except Exception as exc:  # version drift must be typed
        raise ContextParallelUnavailable(
            f"diffusers CP surface unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    attention_classes: tuple[type, ...] = (Attention, AttentionModuleMixin)
    # `ulysses_anything` is unconditional at degree > 1. H3 packs one sequence whose length
    # is whatever the caller packed -- 37,763 / 73,439 / 106,932 / 109,308 rows measured --
    # and diffusers' plain sharder asserts "Tensor size along dimension to be sharded must
    # be divisible by mesh size" on every one of them. Verified both ways on the official
    # and the AdaLN-pruned H3 DiT over gloo, 2026-09-08: a 37-row packed sequence refuses
    # that assertion without this flag and shards with it, while a divisible length is
    # unaffected either way. The config validates exactly this combination -- ulysses only,
    # ring 1 -- and ring attention is never installed here.
    cp_config = ContextParallelConfig(ulysses_degree=int(degree), ulysses_anything=True)
    config = ParallelConfig(context_parallel_config=cp_config)
    for module in attention_modules:
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_attention_backend"):
            continue
        backend = processor._attention_backend
        if backend is None:
            # Diffusers leaves it unannotated: the active backend and its keyword arguments.
            active: Callable[[], tuple[AttentionBackendName, object]] = (
                _AttentionBackendRegistry.get_active_backend
            )
            backend, _ = active()
        else:
            backend = AttentionBackendName(backend)
        if not _AttentionBackendRegistry._is_context_parallel_available(backend):
            supported = sorted(_AttentionBackendRegistry._supports_context_parallel)
            raise ContextParallelUnavailable(
                f"attention backend {backend.value!r} does not support context parallelism; "
                f"set one of {supported} before arming"
            )

    device = (
        comms.device if isinstance(comms.device, torch.device) else torch.device(str(comms.device))
    )
    mesh = _GroupMesh(comms, int(degree), int(comms.rank))
    # A stand-in with the members `setup` and Diffusers' hooks read (`_GroupMesh`).
    config.setup(int(comms.rank), int(degree), device, mesh=cast("DeviceMesh", mesh))
    vars(component)["_parallel_config"] = config  # as Module.__setattr__ stores a plain value
    for module in attention_modules:
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_parallel_config"):
            processor._parallel_config = config
    apply_context_parallel(component, cp_config, getattr(component, _CP_PLAN_ATTR))


def _sharded_modules(component: nn.Module) -> list[nn.Module]:
    """The modules that run on sharded tensors: those under each `_cp_plan` entry that splits
    a module's INPUTS (`blocks.0`/`blocks.*` name the list), or the whole component when the
    plan splits its own. A module before the split sees the replicated sequence: H3 shards
    its packed sequence at `transformer_blocks.0`, after `token_refiner`, whose attention given
    a CP config would concatenate degree copies of its text."""
    from diffusers.models._modeling_parallel import ContextParallelInput

    roots: dict[int, nn.Module] = {}
    for key, inputs in getattr(component, _CP_PLAN_ATTR).items():
        if isinstance(inputs, dict) and any(
            isinstance(value, ContextParallelInput) and not value.split_output
            for value in inputs.values()
        ):
            path = re.sub(r"(^|\.)(0|\*)$", "", key)
            root = component.get_submodule(path) if path else component
            roots[id(root)] = root
    modules: dict[int, nn.Module] = {}
    for root in roots.values() or (component,):
        modules.update((id(module), module) for module in root.modules())
    return list(modules.values())


def _refuse_if_forward_wrapped(components: list[tuple[str, nn.Module]]) -> None:
    """Accelerate can restore a saved forward and remove CP's subsequently installed hooks.

    Only the sharded components matter. Runtime staging and paging keep their forwards;
    an offloaded text encoder or VAE does not participate in the collective.
    """
    for name, component in components:
        for inner_name, module in component.named_modules():
            if hasattr(module, "_hf_hook") or hasattr(module, "_old_forward"):
                raise ContextParallelUnavailable(
                    f"{name}.{inner_name}: an accelerate forward wrapper precedes context "
                    "parallelism and may remove its hooks when restored"
                )


def _declared_heads(component: nn.Module) -> int:
    config = getattr(component, "config", None)
    for attr in ("num_attention_heads", "num_heads", "attention_heads"):
        for holder in (config, component):
            if holder is None:
                continue
            try:
                heads = int(getattr(holder, attr, 0) or 0)
            except (TypeError, ValueError):
                continue
            if heads > 0:
                return heads
    return 0


def refuse_unless_heads_divisible(*, heads: int, degree: int) -> None:
    """Checked at component installation. Sequence length is NOT checked: under
    `ulysses_anything` the packed sequence is split by `tensor_split`, so any length shards
    (H3 packs 37,763 / 73,439 / 106,932 / 109,308 rows and none of them divide). The head
    count is the constraint that remains a CAPABILITY statement rather than a mechanical
    one -- it is what `@sequence_parallel(degrees=...)` asserts, and a package that declares
    a degree its heads do not divide is refused here rather than served ragged."""
    d = int(degree)
    if d <= 1:
        return
    if int(heads) % d:
        raise ContextParallelUnavailable(
            f"attention head count {heads} is not divisible by ulysses_degree {d}"
        )
