"""Install Ulysses hooks and guard collectives on declared sharded components.

Runtime mirrors each component call before torch and diffusers hooks execute.
Every rank enters the same call gate and owning component scope. Sampling, text
encoding and VAE work remain on the leader. Select the attention backend before
installing CP; existing accelerate wrappers on sharded components are refused.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal.parallel.plan import GroupRefusal

_tls = threading.local()


class gated_call:
    """Mark the dynamic extent in which collectives have participants on every rank."""

    def __enter__(self) -> None:
        _tls.active = getattr(_tls, "active", 0) + 1

    def __exit__(self, *exc: object) -> None:
        _tls.active = max(0, getattr(_tls, "active", 1) - 1)


def in_gated_call() -> bool:
    return bool(getattr(_tls, "active", 0))


class ContextParallelUnavailable(GroupRefusal):
    """This generation cannot be sharded, named exactly."""

    default_code = "context_parallel_unavailable"


class UngatedShardedForward(GroupRefusal):
    """A forward through a CP-sharded module outside the group's call gate."""

    default_code = "ungated_sharded_forward"


@dataclass(frozen=True, slots=True)
class CpComms:
    """The GROUP's communication facts, passed explicitly - never the default group read
    ambiently (pgw#773's cross-group corruption)."""

    pg: Any
    rank: int
    device: Any


class _GroupMesh:
    """The mesh shape diffusers' `ParallelConfig.setup` reads: ONE Ulysses axis of `size`
    over the group's process group; the ring axis is degree 1 and refuses if asked for."""

    def __init__(self, pg: Any, size: int, local_rank: int, *, axis: str = "ulysses") -> None:
        self._pg = pg
        self._size = int(size)
        self._local_rank = int(local_rank)
        self._axis = axis

    def size(self) -> int:
        return self._size

    def get_group(self) -> Any:
        if self._axis == "ring":
            raise ContextParallelUnavailable(
                "ring attention is not supported: the runtime installs Ulysses sequence "
                "parallelism only (ring_degree=1)"
            )
        return self._pg

    def get_local_rank(self) -> int:
        return self._local_rank

    def _flatten(self) -> _GroupMesh:
        return _GroupMesh(self._pg, self._size, self._local_rank)

    def __getitem__(self, key: str | tuple[str, ...]) -> _GroupMesh:
        if isinstance(key, tuple) or key == "ulysses":
            return _GroupMesh(self._pg, self._size, self._local_rank)
        if key == "ring":
            return _GroupMesh(self._pg, 1, 0, axis="ring")
        raise KeyError(key)


_CP_PLAN_ATTR = "_cp_plan"
_GUARD_ATTR = "_cozy_cp_gate_guard"


def _modules_of(model: Any) -> Iterator[tuple[str, Any]]:
    """Every torch module reachable from the model's DECLARED state, and the components
    of any pipeline among them. Declared state is the author's statement of what the
    model holds (§1.1); nothing else is walked."""
    from cozy_runtime.author._model import _declared_state

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


def sharding_candidates(model: Any) -> list[tuple[str, Any]]:
    """The components that DECLARE a context-parallel plan and can be told to shard."""
    return [
        (name, module)
        for name, module in _modules_of(model)
        if getattr(module, _CP_PLAN_ATTR, None) and hasattr(module, "enable_parallelism")
    ]


def install_context_parallel(model: Any, *, degree: int, comms: CpComms | None) -> tuple[str, ...]:
    """Install Ulysses sequence parallelism at `degree` on every declaring component."""
    if int(degree) <= 1:
        return ()
    components = sharding_candidates(model)
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
        _install_gate_guard(component, name)
        installed.append(name)
    return tuple(installed)


def _install_gate_guard(component: Any, name: str) -> None:
    if getattr(component, _GUARD_ATTR, None) is not None:
        return

    def guard(module: Any, args: Any, kwargs: Any = None) -> None:
        if in_gated_call():
            return
        raise UngatedShardedForward(
            f"sharded component {name!r} needs the Runtime call gate inside an active "
            "component scope, so every rank participates in its collectives"
        )

    handle = component.register_forward_pre_hook(guard, with_kwargs=True)
    with contextlib.suppress(Exception):  # a frozen module keeps the hook anyway
        setattr(component, _GUARD_ATTR, handle)


def _install_on_component(component: Any, *, degree: int, comms: CpComms) -> None:
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

        attention_root = _attention_root(component)
    except Exception as exc:  # version drift must be typed
        raise ContextParallelUnavailable(
            f"diffusers CP surface unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    attention_classes: tuple[type, ...] = (Attention, AttentionModuleMixin)
    # H3 builds its full packed sequence after token_refiner, then shards it at
    # transformer_blocks.0. Giving the replicated refiner a CP config incorrectly
    # concatenates degree copies of its text sequence inside attention. The pruned
    # and turbo components inherit this same architecture and split boundary.
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
    for module in attention_root.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_attention_backend"):
            continue
        backend = processor._attention_backend
        if backend is None:
            backend, _ = _AttentionBackendRegistry.get_active_backend()
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
    mesh = _GroupMesh(comms.pg, int(degree), int(comms.rank))
    config.setup(int(comms.rank), int(degree), device, mesh=mesh)
    component._parallel_config = config
    for module in attention_root.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_parallel_config"):
            processor._parallel_config = config
    apply_context_parallel(component, cp_config, getattr(component, _CP_PLAN_ATTR))


def _attention_root(component: Any) -> Any:
    try:
        from diffusers.models.transformers.transformer_minimax_h3 import (
            MiniMaxH3Transformer3DModel,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "diffusers.models.transformers.transformer_minimax_h3":
            raise
        # Older Diffusers packages do not define H3. Their other architectures
        # still use their existing CP path; absence of this optional model is not
        # absence of context parallelism.
        return component
    return (
        component.transformer_blocks
        if isinstance(component, MiniMaxH3Transformer3DModel)
        else component
    )


def _refuse_if_forward_wrapped(components: list[tuple[str, Any]]) -> None:
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


def _declared_heads(component: Any) -> int:
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
