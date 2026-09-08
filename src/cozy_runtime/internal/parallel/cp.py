"""Installing Ulysses sequence parallelism on a generation - the runtime's job, never the
package's.

Installing CP puts split/all-to-all/gather hooks ON THE MODULES, so from that moment EVERY
forward through them issues collectives - and the model's component-use method is the only
seam that supplies participants on the other ranks. A forward from anywhere else (a warm
compile, a mint seed, a probe, a package calling a component by hand) runs on rank 0 alone
and HANGS the whole group in NCCL. `gated_call` marks the dynamic extent in which
collectives have participants; the guard installed on every sharded component refuses a
forward outside it (`ungated_sharded_forward`) instead of hanging.

The ordering is FORCED: set the attention backend, enable parallelism, then compile -
compiling AFTER the hooks puts the split/all-to-all/gather inside the traced graph. Ring
attention is not installed (it needs the attention kernel's LSE); this is Ulysses only.

CP DOES COMPOSE WITH A DEGRADED RESIDENCY (h3a-019). It does not compose with ONE thing,
and diffusers #12533 says exactly which: `accelerate` saves `module._old_forward` before
wrapping `module.forward`, and `enable_model_cpu_offload` RESTORES that saved forward at
the end of every pipeline run. diffusers installs the context-parallel hooks the same way
- `HookRegistry.register_hook` ASSIGNS `module.forward` - so accelerate's restore silently
strips them on the second call and the sequence stops being split. The maintainer's own
diagnosis on that issue, and the reported symptom is a shape error (4096 against 2048) on
the second call, per rank and deterministic: not a hang, not a race, not a numerical drift.

So the hazard is a FORWARD-WRAPPING offload on a SHARDED component, and that is what
`_refuse_if_forward_wrapped` refuses. It is not "weights that do not all fit". This
runtime's own staging parks a component with `to_empty(device="meta")` and re-fills it
from the CAS (`internal/fill.py`): it never touches `forward`, and the module OBJECT
survives, so the CP hooks survive with it. A text encoder or a VAE that is offloaded,
parked or paged is not a participant in any collective and is not this guard's business.
`refuse_if_cp_hooks_stripped` is the loud half: if something does strip the hooks anyway,
the next forward refuses by name instead of producing #12533's shape error.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast

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

    def __getitem__(self, key: Any) -> _GroupMesh:
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
    refuse_unless_shard_invariant_quant(model, degree=int(degree))
    installed: list[str] = []
    for name, component in components:
        refuse_unless_heads_divisible(heads=_declared_heads(component), degree=int(degree))
        _install_on_component(component, degree=int(degree), comms=comms)
        _record_cp_hooks(component, name)
        _install_gate_guard(component, name)
        installed.append(name)
    return tuple(installed)


def _install_gate_guard(component: Any, name: str) -> None:
    if getattr(component, _GUARD_ATTR, None) is not None:
        return

    def guard(module: Any, args: Any, kwargs: Any = None) -> None:
        if in_gated_call():
            # THE ONE CHECK THAT COSTS A FORWARD (h3a-019). Once per outer forward -- ~30
            # times an attempt, over a list of a few hundred identity comparisons -- and it
            # is what turns #12533 from a shape error deep in `apply_rotary_emb` into a
            # refusal that names the cause. It fires INSIDE the gate because that is the
            # only place a stripped hook can still be caught before the collective.
            refuse_if_cp_hooks_stripped(module, name)
            return
        raise UngatedShardedForward(
            f"forward through the context-parallel component {name!r} outside the group's "
            "call gate: this rank would issue collectives no other rank is in. Every forward "
            "at degree>1 goes through a mirrored component-use method (compile warmups, mint "
            "seeds, probes and direct component calls are refused under context parallelism)"
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
    for module in component.modules():
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
        break

    device = (
        comms.device if isinstance(comms.device, torch.device) else torch.device(str(comms.device))
    )
    mesh = _GroupMesh(comms.pg, int(degree), int(comms.rank))
    config.setup(int(comms.rank), int(degree), device, mesh=cast(Any, mesh))
    component._parallel_config = config
    for module in component.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is not None and hasattr(processor, "_parallel_config"):
            processor._parallel_config = config
    # `sharding_candidates` filtered on this same attribute of this same object moments
    # ago; there is no seam between the two for it to change across.
    apply_context_parallel(component, cp_config, getattr(component, _CP_PLAN_ATTR))


#: What `accelerate` stamps on a module whose `forward` it has WRAPPED. `_old_forward` is
#: the saved original and `_hf_hook` the wrapper's own handle; `enable_model_cpu_offload`,
#: `enable_sequential_cpu_offload` and `cpu_offload_with_hook` all leave both. These are the
#: collision surface of diffusers #12533, because the same offload RESTORES `_old_forward`
#: after each pipeline run and a CP hook installed on top of it disappears with that
#: restore. `_all_hooks` / `_offload_gpu_id` are the pipeline-level spelling of the same
#: thing and were the previous guard's markers -- they are attributes of `DiffusionPipeline`,
#: which is not an `nn.Module`, so `_modules_of` never reached them and they never fired.
#: The per-module names below do, and they fire for pipeline-level offload too, because
#: that is where its hooks actually land.
_FORWARD_WRAPPING_OFFLOAD = ("_hf_hook", "_old_forward")

#: The names diffusers gives a context-parallel hook in a module's `HookRegistry`
#: (`hooks/context_parallel.py`: `cp_input---{}` / `cp_output---{}`).
_CP_HOOK_PREFIXES = ("cp_input---", "cp_output---")

#: diffusers' own group offloading, which lives in the SAME `HookRegistry` CP does
#: (`hooks/group_offloading.py`). It is not accelerate: it chains through the registry
#: instead of saving and restoring `forward`, so it composes with CP and is ALLOWED. It is
#: listed here so `_diffusers_hook`'s mere presence is never read as an offload signal --
#: `apply_context_parallel` sets that attribute itself.
_GROUP_OFFLOAD_HOOKS = ("group_offloading", "lazy_prefetch_group_offloading")

_HOOKED_ATTR = "_cozy_cp_hooked"


def _registry_hooks(module: Any) -> tuple[str, ...]:
    registry = getattr(module, "_diffusers_hook", None)
    return tuple(getattr(registry, "hooks", None) or ())


def _forward_wrapping_offload(module: Any) -> str:
    """The name of an accelerate wrapper on this module's `forward`, or "" for none."""
    for marker in _FORWARD_WRAPPING_OFFLOAD:
        if getattr(module, marker, None) is not None:
            return marker
    return ""


def _refuse_if_forward_wrapped(components: list[tuple[str, Any]]) -> None:
    """Refuse the offload that actually breaks Ulysses, on the components that are sharded.

    Scoped to the SHARDING CANDIDATES on purpose (h3a-019). The old guard walked every
    declared component of the model, so a text encoder or a VAE with an offload marker
    refused a generation whose DiT was the only thing being sharded -- and those components
    are never in a collective. What matters is a forward-wrapping offload on a component
    that carries CP hooks, plus a device map that SCATTERS one such component, since an
    all-to-all cannot span a module whose layers sit on different cards.

    Weights that merely do not all fit at once are not this guard's business: staging parks
    to `meta` and re-fills from the CAS without touching `forward`, and the sharded module
    object survives the round trip with its hooks attached.
    """
    for name, component in components:
        for module_name, module in component.named_modules():
            marker = _forward_wrapping_offload(module)
            if not marker:
                continue
            where = f"{name}.{module_name}" if module_name else name
            raise ContextParallelUnavailable(
                f"{where} carries {marker!r}: an accelerate offload is already wrapped "
                "around this component's `forward`, which is diffusers #12533's BROKEN "
                "ORDER -- the wrapper restores the forward it saved at the end of every "
                "run, and the context-parallel hooks installed on top of it go with it, so "
                "the second call runs unsharded. The documented order is context "
                "parallelism first, offload second, and the runtime installs CP at prepare: "
                "a package that offloads in `load` has already taken that order away. "
                "diffusers' own group offloading composes and is admitted, and so is this "
                "runtime's staging and block paging, which never touch `forward`"
            )
        scatter = _scattered_devices(getattr(component, "hf_device_map", None))
        if scatter:
            raise ContextParallelUnavailable(
                f"{name} carries an `hf_device_map` over {sorted(scatter)}: a sharded "
                "component's all-to-all cannot span a module whose layers live on "
                "different devices, and a map that differs by rank makes the ranks "
                "disagree about which layer collects. One device per sharded component"
            )


def _scattered_devices(device_map: Any) -> set[str]:
    """The distinct targets of a device map that would break a sharded component, or none.

    One device is fine however it is spelled; more than one, or a host/disk target, is not.
    """
    if not isinstance(device_map, dict) or not device_map:
        return set()
    targets = {str(value) for value in device_map.values()}
    if len(targets) > 1 or targets & {"cpu", "disk", "meta"}:
        return targets
    return set()


def _record_cp_hooks(component: Any, name: str) -> None:
    """Remember every module `apply_context_parallel` hooked, and the forward it installed.

    diffusers installs a CP hook by ASSIGNING `module.forward`, so the installed callable's
    identity is the evidence that the hook is still in force. Nothing in this runtime
    reassigns a forward -- `to_empty` replaces parameter storage and leaves methods alone --
    so an identity that changed means something outside took the forward back.
    """
    hooked: list[list[Any]] = []
    for module in component.modules():
        if any(entry.startswith(_CP_HOOK_PREFIXES) for entry in _registry_hooks(module)):
            hooked.append([module, module.forward])
    if not hooked:
        raise ContextParallelUnavailable(
            f"{name}: `apply_context_parallel` installed no hook this runtime can find. The "
            "component would run UNSHARDED while its rank issues all-to-alls, which is "
            "silently wrong output rather than a failure"
        )
    setattr(component, _HOOKED_ATTR, hooked)


def refuse_if_cp_hooks_stripped(component: Any, name: str) -> None:
    """The loud half of #12533: a CP hook that was taken back refuses BEFORE the collective.

    Two facts per hooked module, both O(1) and neither touching the device: the CP hook is
    still REGISTERED, and nothing that saves and restores `forward` has wrapped it since.
    A forward whose identity merely changed is not by itself a fault -- diffusers' group
    offloading registers through the same `HookRegistry` and legitimately re-wraps, CHAINING
    beneath what is already there -- so a changed identity with the hook still registered
    and no accelerate wrapper present is adopted rather than refused.

    Runs inside the call gate, where the group is still coherent enough to fail all K ranks
    with the same typed refusal instead of one rank shipping a full-length sequence into an
    all-to-all its peers sharded.
    """
    for entry in getattr(component, _HOOKED_ATTR, ()) or ():
        module, installed = entry[0], entry[1]
        if not any(hook.startswith(_CP_HOOK_PREFIXES) for hook in _registry_hooks(module)):
            raise ContextParallelUnavailable(
                f"{name}: the context-parallel hook on {type(module).__name__} was REMOVED "
                "from its registry after install. This rank would run the full sequence "
                "while its peers all-to-all over shards - silently wrong output, refused "
                "here instead"
            )
        if module.forward is installed:
            continue
        marker = _forward_wrapping_offload(module)
        if marker:
            if getattr(module, "_old_forward", None) is installed:
                # THE ORDER THAT WORKS, and #12533's own documented workaround: context
                # parallelism first, offload second. The wrapper saved OUR forward, so its
                # end-of-run restore puts the sharded forward back rather than removing it,
                # and the module oscillates between the two for the rest of the generation.
                # The RECORDED forward stays the sharded one on purpose: adopting the
                # wrapper's would make the very next restore look like a strip.
                continue
            raise ContextParallelUnavailable(
                f"{name}: {type(module).__name__} acquired {marker!r} after the "
                "context-parallel hooks were installed and its `forward` has been taken "
                "back. That is diffusers #12533 exactly: the wrapper restores the forward "
                "it saved at the end of every run, so THIS call is sharded and the next one "
                "is not. Refused before the collective rather than served as the shape "
                "error the issue reports"
            )
        entry[1] = module.forward


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
    """The mechanical half of `@sequence_parallel`, checked at install. Sequence length is
    NOT checked: under
    `ulysses_anything` the packed sequence is split by `tensor_split`, so any length shards
    (H3 packs 37,763 / 73,439 / 106,932 / 109,308 rows and none of them divide). The head
    count is the constraint that remains a CAPABILITY statement rather than a mechanical
    one -- it is what `@sequence_parallel(degrees=...)` asserts, and a package that declares
    a degree its heads do not divide is refused here rather than served ragged.

    It runs at PREPARE, on a pod already bought: nothing upstream reads a head count, and
    the pre-rental narrowing (`cozy-creator internal/rental`) reads only the author's
    DECLARED degrees. So this buys no rental decision -- it buys the name of the fault. An
    all-to-all over unequal head splits fails loudly either way; it fails as a tensor shape
    deep in the attention rather than as the author's declaration being wrong, which is
    what it actually is. For H3 it cannot fire: 56 heads divides 2, 4 and 8."""
    d = int(degree)
    if d <= 1:
        return
    if int(heads) % d:
        raise ContextParallelUnavailable(
            f"attention head count {heads} is not divisible by ulysses_degree {d}"
        )


def refuse_unless_shard_invariant_quant(model: Any, *, degree: int) -> None:
    """W8A8 composes with CP with one typed exception: `rowwise` derives a scale per TOKEN,
    a function of that token alone and identical whichever rank holds it; `pertensor`
    derives one scalar from the LOCAL shard, so two ranks quantize the same logical tensor
    differently and the group produces silently wrong output. That is the quiet failure
    mode, which is why this refuses rather than warns."""
    if int(degree) <= 1:
        return
    bad: list[str] = []
    for name, module in _modules_of(model):
        for inner_name, inner in module.named_modules():
            mode = getattr(inner, "gemm_mode", "")
            if mode and mode != "rowwise":
                bad.append(f"{name}.{inner_name}={mode}")
                if len(bad) >= 4:
                    break
        if len(bad) >= 4:
            break
    if bad:
        raise ContextParallelUnavailable(
            "w8a8 gemm_mode must be `rowwise` under context parallelism: a per-tensor "
            "activation scale is derived from the local sequence shard, so each rank "
            f"quantizes differently and the output is silently wrong. Found {', '.join(bad)}"
        )
