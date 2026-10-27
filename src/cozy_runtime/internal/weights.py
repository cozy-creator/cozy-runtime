"""Model weights: Runtime builds each model on meta, and its local residency module
fills ordinary Torch tensors from TensorFS at declared component/block boundaries.

`Backend` is the loader's backend. `fit` judges, `materialize` keeps weights on meta and derived
buffers real, `fill` checks each destination against its stored row, and `commit` installs
encoded leaves, loads each component (`Memory.load`) and puts the execution layer's catch on it.
A stored tensor becomes the module's through its route (`_Routes`: as stored, decoded, or a
native leaf's buffers); a LoRA'd linear bakes on page-in (`_Bakes`), named by its fill identity.
`Placement` is what Runtime mechanisms reach a placed component through: graph pins, KV caches
and notes.
"""

from __future__ import annotations

import functools
import time
import zlib
from collections import Counter
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from cozy_runtime import graphs
from cozy_runtime.author._loader import (
    TENSORFS_NAMES,
    Census,
    TensorLike,
    TensorSpec,
    _members,
    canonical_dtype,
)
from cozy_runtime.author._model import PLACED, LoRAServing
from cozy_runtime.internal import execution, lora_bake
from cozy_runtime.internal.derive import placing
from cozy_runtime.internal.encoding import (
    TORCH_DTYPES,
    DeviceFacts,
    LeafProvider,
    Selection,
    admit,
    device_line,
    implementation_digest,
    launch_providers,
)
from cozy_runtime.internal.encoding.leaves import sites as leaf_sites
from cozy_runtime.internal.fill import (
    Checkpoint,
    PlanRow,
    TensorRequirement,
    _refuse,
    dtype_name,
    servability,
    tensorfs_requirement_dtype,
)
from cozy_runtime.internal.fill import fit as fill_fit
from cozy_runtime.internal.parallel import ulysses  # noqa: F401 (Ulysses' torch side, served)
from cozy_runtime.internal.residency import (
    Binding,
    Declared,
    Hold,
    Level,
    Memory,
    Model,
    Region,
    Request,
    ResidencyError,
    StaleBinding,
    TensorLayout,
    Weights,
    blocks_of,
)
from cozy_runtime.internal.resolution import ResolvedModelPlan, TensorResolution

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.nn import Module


# ---------------------------------------------------------------------------- the backend


class Backend:
    """The loader's `Backend` (§1.1's one fill path): construction on meta, then each component
    registered with the executor's `Memory`."""

    def __init__(
        self,
        checkpoints: Mapping[str, Checkpoint],
        rows: Sequence[PlanRow],
        *,
        torch: ModuleType,
        plan: ResolvedModelPlan,
        memory: Memory,
        device: torch.device,
        construction: str,
        device_facts: DeviceFacts,
        release: str,
        custody: str = "canonical",
    ) -> None:
        self.memory = memory
        self.device = device
        self.release = release
        self.torch = torch
        self.construction = construction
        self.checkpoints = dict(checkpoints)
        self.plan = {row.key: row for row in rows}
        self.custody = custody
        self.state = "open"
        self.expected: dict[str, frozenset[str]] = {}
        #: component -> stored tensors the code does not build: never read or mapped
        self.skipped: dict[str, list[str]] = {}
        self.enqueued: dict[str, TensorLike] = {}
        self.roots: dict[str, Module] = {}
        #: each component's loaded model, and its LoRA bakes
        self.models: dict[str, Model] = {}
        self.bakes: dict[str, _Bakes] = {}
        #: id(module) -> the loaded model (of any construction) and path holding it, and the module
        self.owned: dict[int, tuple[Model, str, object]] = {}
        self.indexed: tuple[int, ...] = ()
        self.derived: dict[str, dict[tuple[Module, str], Tensor]] = {}
        #: Persistent buffers the checkpoint does not store (full keys): they keep their values.
        self.unstored: set[str] = set()
        self.stats: dict[str, float] = {}
        #: the registration's own legs (leaves installed, each component's load and catch), in
        #: execution order: what the construction's `fill` leg spent beside validation
        self.legs: list[tuple[str, float]] = []
        self.device_facts = device_facts
        self.providers = launch_providers()
        self.execution = plan
        self.resolved: dict[str, TensorResolution] = plan.by_key()
        planned, held = set(self.resolved), set(self.plan)
        if planned != held:
            raise _refuse(
                "plan_divergence",
                f"the resolved plan [{plan.digest()[:19]}…] names {len(planned)} destinations "
                f"and this checkpoint set supplies {len(held)}: missing from the plan "
                f"{sorted(held - planned)[:3]}, absent from the checkpoint "
                f"{sorted(planned - held)[:3]}",
                sorted(planned ^ held)[:6],
            )
        self.servability = servability(rows, plan)
        self.selected: dict[str, Selection] = {
            key: self._select(key, row) for key, row in self.plan.items()
        }
        self.encoded_leaves = "refuse"
        #: contract key -> (installed leaf, replaced module, derived roles)
        self.leaves: dict[str, tuple[Any, Any, tuple[str, ...]]] = {}
        #: (component, provider) -> installed leaves, counted when they install (`gemm.selected`)
        self.gemm_sites: Counter[tuple[str, str]] = Counter()

    def _select(self, key: str, row: PlanRow) -> Selection:
        """The plan's provider for one destination, matched by digest against this build and
        admitted on this device. Anything else is `plan_divergence`, never a re-selection."""
        resolved = self.resolved.get(key)
        if resolved is None:
            raise _refuse("plan_divergence", f"{key}: the plan names no such destination", [key])
        actual = (row.encoded.scheme, tuple(row.shape), dtype_name(row.dtype))
        planned = (resolved.encoding, tuple(resolved.geometry), resolved.output_dtype)
        if actual != planned:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan priced encoding/geometry/dtype {planned} and the checkpoint "
                f"holds {actual}",
                [key],
            )
        provider = next(
            (
                c
                for c in self.providers.get(row.encoded.scheme, ())
                if implementation_digest(c) == resolved.implementation_digest
            ),
            None,
        )
        if provider is None or str(provider.route) != resolved.route:
            raise _refuse(
                "plan_divergence",
                f"{key}: the plan executes {resolved.implementation} "
                f"[{resolved.implementation_digest[:19]}…, {resolved.route}] and this build "
                + (
                    "carries no implementation with that digest"
                    if provider is None
                    else f"has it on route {provider.route!r}"
                ),
                [key],
            )
        why = admit(provider, self.device_facts, resolved.output_dtype, row.encoded)
        if why:
            raise _refuse(
                "plan_divergence",
                f"{key}: {device_line(self.device_facts)} does not serve "
                f"{resolved.implementation}: {why}",
                [key],
            )
        return Selection(provider, row.encoded, resolved.reason)

    # -- the Backend protocol

    @property
    def bound(self) -> frozenset[str]:
        """The components the binding selects; the construction's others stay on meta."""
        return frozenset(self.checkpoints)

    def expect(self, sets: Mapping[str, Sequence[str]]) -> None:
        """Completeness sets: a destination the census never reaches refuses the commit."""
        self.expected = {name: frozenset(keys) for name, keys in sets.items()}

    def fit(self, walked: Census, *, encoded_leaves: str) -> Mapping[str, object] | None:
        """`fill.fit` over the serving census, per checkpoint the census reads."""
        by_checkpoint: dict[str, tuple[Checkpoint, list[str]]] = {}
        for component in walked.components:
            checkpoint = self.checkpoints.get(component)
            if checkpoint is None:
                raise _refuse(
                    "destination_absent",
                    f"the construction built component {component!r} and the binding names "
                    f"no checkpoint for it (bound: {', '.join(sorted(self.checkpoints))})",
                    [component],
                )
            by_checkpoint.setdefault(checkpoint.id, (checkpoint, []))[1].append(component)
        verdict: dict[str, object] | None = None
        ignored: list[str] = []
        unstored: list[str] = []
        warnings: list[str] = []
        routes = {"verbatim": 0, "decoded_float": 0, "encoded_gemm": 0}
        for checkpoint, constructed in by_checkpoint.values():
            members = set(constructed)
            requirements = [
                TensorRequirement(
                    component=d.component,
                    key=d.key[len(d.component) + 1 :],
                    shape=list(d.spec.shape),
                    logical_dtype=tensorfs_requirement_dtype(d.spec.dtype),
                    buffer=d.buffer,
                )
                for d in walked.destinations
                if d.component in members
            ]
            fit = fill_fit(
                requirements,
                checkpoint,
                encoded_leaves=encoded_leaves == "accept",
                custody=self.custody,
                device=self.device_facts.predicate,
            )
            if not fit.get("ok"):
                return fit
            verdict = fit
            extras = [str(k) for k in fit.get("ignored", ())]  # type: ignore[attr-defined]
            ignored.extend(extras)
            if extras:
                warnings.append(str(fit.get("warning") or self._skipped_line(extras)))
            unstored.extend(str(k) for k in fit.get("unstored", ()))  # type: ignore[attr-defined]
            counted: Mapping[str, int] = fit.get("routes", {})  # type: ignore[assignment]
            for route in routes:
                routes[route] += int(counted.get(route, 0))
        if verdict is None:
            return None
        for full in ignored:
            component, _, key = full.partition("/")
            self.skipped.setdefault(component, []).append(key)
            if component in self.expected:
                self.expected[component] -= {f"{component}.{key}"}
        self.unstored = {full.replace("/", ".", 1) for full in unstored}
        if unstored:
            more = f", and {len(unstored) - 5} more" if len(unstored) > 5 else ""
            warnings.append(
                f"{len(unstored)} persistent buffer(s) the checkpoint does not store keep their "
                f"constructed values: {', '.join(unstored[:5])}{more}"
            )
        if len(by_checkpoint) > 1:
            verdict = {**verdict, "ignored": ignored, "routes": routes}
        return {**verdict, "unstored": unstored, "warnings": warnings}

    def _skipped_line(self, extras: Sequence[str]) -> str:
        """TensorFS 0.3.90's warning for the stored tensors fit skipped: how many, how many
        bytes, and the first five."""
        rows = (self.plan.get(full.replace("/", ".", 1)) for full in extras)
        nbytes = sum(row.stored_nbytes for row in rows if row is not None)
        more = f", and {len(extras) - 5} more" if len(extras) > 5 else ""
        return (
            f"{len(extras)} stored tensor(s) the code does not build were skipped, not "
            f"loaded: {nbytes} B ({', '.join(extras[:5])}{more})"
        )

    def materialize(
        self, obj: object, walked: Census, *, encoded_leaves: str = "refuse"
    ) -> Mapping[str, TensorLike]:
        """Weights go to meta, persistent buffers the checkpoint stores included, so varena binds
        every stored tensor at every name and a read anywhere gets the checkpoint's value.
        Derived (non-persistent) buffers, and persistent ones the checkpoint does not store, move
        to the device with the values their construction computed."""
        self.encoded_leaves = encoded_leaves
        native = sorted(k for k, t in self.resolved.items() if t.route == "encoded_gemm")
        if native and encoded_leaves != "accept":
            raise _refuse(
                "leaf_unconsented",
                f"{len(native)} of this artifact's tensors ({', '.join(native[:3])}…) serve "
                f"through a native encoded leaf on {device_line(self.device_facts)}, and this "
                f"package declares encoded_leaves={encoded_leaves!r}: the leaf replaces the "
                'linear op, so the package must state `encoded_leaves="accept"`',
                native[:6],
            )
        roots = dict(_members(obj))
        live = dict(walked.live)
        stored = {d.key for d in walked.destinations if d.buffer}
        with placing():
            for name in walked.components:
                member = roots.get(name)
                if not isinstance(member, self.torch.nn.Module):
                    continue
                self.roots[name] = member
                live.update(_weightless_buffers(member, name, stored))
                self.derived[name] = _derived_values(member, name, self.unstored)
                _restore_derived(self.derived[name], self.device)
        return live

    def fill(self, key: str, spec: TensorSpec, destination: TensorLike) -> None:
        """Validate one destination against its stored row. Nothing moves."""
        row = self.plan.get(key)
        if row is None:
            raise _refuse(
                "destination_absent",
                f"{key!r} is not a destination this checkpoint supplies",
                [key],
            )
        shape = tuple(destination.shape)
        if shape != tuple(row.shape) or (spec.nbytes and spec.nbytes != row.nbytes):
            raise _refuse(
                "length_mismatch",
                f"{key}: the destination is {shape} and the checkpoint supplies {row.shape} "
                f"({row.nbytes} B)",
                [key],
            )
        if dtype_name(destination.dtype) != dtype_name(row.dtype):
            raise _refuse(
                "dtype_mismatch",
                f"{key}: the checkpoint's logical dtype is {row.dtype} and the destination is "
                f"{dtype_name(destination.dtype)}",
                [key],
            )
        self.enqueued[key] = destination
        self.state = "staging"

    def commit(self) -> None:
        """Install encoded leaves on meta, then register every component with the memory."""
        missing = sorted(
            key
            for name, keys in self.expected.items()
            if name in self.roots
            for key in keys
            if key not in self.enqueued
        )
        if missing:
            raise _refuse(
                "incomplete_fill",
                f"{len(missing)} destination(s) never reached the fill plane "
                f"({', '.join(missing[:4])}): a component's destinations all fill or none commits",
                missing[:6],
            )
        started = leg = time.perf_counter()
        order = list(self.enqueued)
        self.leaves = {
            key: self._install_leaf(key) for key in order if self._route(key) == "encoded_gemm"
        }
        self.gemm_sites = leaf_sites(self.leaves)
        leg = self._leg("leaves", leg)
        for name, root in self.roots.items():
            keys = [key for key in order if self.plan[key].component == name]
            if keys:
                self.models[name] = self._register(name, root, keys)
        if self.device.type == "cuda" and any(b.bakes for b in self.bakes.values()):
            lora_bake.warm(self.device.index or 0)  # its kernels build at load, in a child
            self._leg("lora_warm", leg)
        self.stats["register_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.state = "ready"

    def poison(self, keys: Sequence[str]) -> None:
        """Abort: every registered model goes."""
        self.close()
        self.state = "poisoned"

    def close(self) -> None:
        for model in self.models.values():
            for _, release in graphs.DECODERS:
                release(model.module)
            graphs.release(self.torch, model.module)
            model.close()
        self.models.clear()
        self.owned, self.indexed = {}, ()

    def sizes(self) -> dict[str, int]:
        """Each loaded component's stored bytes."""
        held: dict[str, int] = {}
        for row in self.plan.values():
            if row.component in self.models:
                held[row.component] = held.get(row.component, 0) + row.stored_nbytes
        return held

    # -- construction internals

    def _owner(self, module: object) -> tuple[Model, str] | None:
        """The loaded model and module path that hold `module`, when a component this process
        loaded does, of any construction (H3's turbo overlay is its own). Indexed once per set
        of loaded models: a fill asks for every hooked linear of every block it fills."""
        models = list(self.memory.models.values())
        loaded = tuple(map(id, models))
        if loaded != self.indexed:  # a model loaded, closed or reloaded since
            self.indexed = loaded
            self.owned = {
                id(sub): (model, path, sub)
                for model in models
                for path, sub in model.module.named_modules()
            }
        held = self.owned.get(id(module))
        return None if held is None or held[2] is not module else (held[0], held[1])

    @contextmanager
    def _reading(self, module: object) -> Iterator[None]:
        """`module`'s stored tensors bound while a fill reads them outside its model's forward
        (a LoRA source's factors): only the piece of that model which holds `module` (its
        block, with the model's common), released once read."""
        owned = self._owner(module)
        if owned is None:  # not a loaded model's: its tensors are its own
            yield
            return
        binding = owned[0].retain((owned[1],))
        try:
            with binding.prepare():
                yield
        finally:
            binding.close()

    def _route(self, key: str) -> str:
        return str(self.selected[key].provider.route)

    def _register(self, name: str, root: Module, keys: list[str]) -> Model:
        """One component loaded into the memory: its stored tensors, bound through their
        routes, its LoRAs baked on page-in, under the execution layer's catch. Served in eval
        mode: a model built in training mode would apply its dropouts to every request."""
        leg = time.perf_counter()
        root.eval()
        rows = {self.plan[key].name: self.plan[key] for key in keys}
        bakes = self.bakes[name] = _Bakes(self, root, rows, set(blocks_of(root)))
        checkpoint = self.checkpoints[name]
        routes = _Routes(self, rows)
        leg = self._leg(f"{name}.routes", leg)
        bound = {stored: row.key.partition(".")[2] for stored, row in rows.items()}
        vars(root)["_cozy_weights"] = Declared(
            Weights(
                checkpoint.ck,
                name,
                release=self.release,
                keys=tuple(rows),
                bound={stored: path for stored, path in bound.items() if path != stored},
            ),
            fill=bakes if bakes.bakes else None,
            bind=routes,
            finals=routes,
        )
        # At RAM: pages reach the device through placement (passes, the plan, an explicit warm),
        # never at construction (v6 13633 filled the card with H3's transformer before its TE).
        model = self.memory.load(root, Level.RAM)
        if self.device.type == "cpu":
            _float32(self.torch, root)
        self._leg(f"{name}.load", leg)
        return model

    def install_execution(self) -> None:
        """Runtime's pure cuts and block names on every registered component, after the
        construction's execution plan (its fused blocks and glue stand). Arbitrary package code
        is never replayed."""
        for name in self.models:
            execution.install(self.roots[name])

    def _leg(self, name: str, since: float) -> float:
        now = time.perf_counter()
        self.legs.append((name, round((now - since) * 1000, 1)))
        return now

    def _install_leaf(self, key: str) -> tuple[Any, Any, tuple[str, ...]]:
        """Replace the linear op that owns `key` with its encoded leaf, on meta. Returns the
        leaf, the replaced module, and the roles the provider derives from stored bytes."""
        torch = self.torch
        row = self.plan[key]
        component, _, local = key.partition(".")
        owner_path, _, attribute = local.removesuffix(".weight").rpartition(".")
        root = self.roots.get(component)
        owner = root.get_submodule(owner_path) if root is not None and owner_path else root
        replaced = getattr(owner, attribute, None) if attribute else None
        weight = getattr(replaced, "weight", None)
        if (
            not local.endswith(".weight")
            or len(row.shape) != 2
            or weight is None
            or tuple(weight.shape) != tuple(row.shape)
        ):
            raise _refuse(
                "leaf_schema",
                f"{key}: the native route replaces a rank-2 linear op named by its `.weight`; "
                "this destination has no such module",
                [key],
            )
        provider = self.selected[key].provider
        assert isinstance(provider, LeafProvider)
        parts = {
            part.role: torch.empty(
                part.shape, dtype=getattr(torch, TORCH_DTYPES[part.dtype]), device="meta"
            )
            for part in row.encoded.parts
        }
        leaf = provider.leaf(torch, parts, replaced, getattr(torch, TORCH_DTYPES[row.dtype]))
        for hooks in (
            "_forward_pre_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
        ):
            setattr(leaf, hooks, getattr(replaced, hooks))
        with placing():
            for name in leaf._non_persistent_buffers_set:
                held = leaf._buffers.get(name)
                if held is not None:
                    leaf._buffers[name] = torch.empty_like(held, device=self.device)
        setattr(owner, attribute, leaf)
        return leaf, replaced, _derived(leaf, parts)


# ----------------------------------------------------------------- routes and LoRA bakes


class _Routes:
    """Bind stored parts as ordinary weights, decoded floats, or native leaf role buffers.

    Residency owns and caches the returned tensors with their block. Dropping that block
    releases its decoded storage too; active scopes and graph bindings keep their own owners.
    """

    def __init__(self, backend: Backend, rows: Mapping[str, PlanRow]) -> None:
        self.torch = backend.torch
        self.rows = dict(rows)
        self.routes = {stored: backend._route(row.key) for stored, row in rows.items()}
        self.providers = {
            stored: backend.selected[row.key].provider for stored, row in rows.items()
        }
        self.leaves = {
            stored: backend.leaves[row.key]
            for stored, row in rows.items()
            if row.key in backend.leaves
        }
        self._layouts = {key: self._layout(key) for key in self.rows}
        self._identity = repr(
            tuple(
                (
                    key,
                    implementation_digest(self.providers[key]),
                    self.providers[key].params,
                    None if layout is None else tuple(layout.items()),
                )
                for key, layout in sorted(self._layouts.items())
            )
        )

    @property
    def identity(self) -> str:
        return self._identity

    def layout(self, key: str) -> Mapping[str, TensorLayout] | None:
        return self._layouts[key]

    def _layout(self, key: str) -> Mapping[str, TensorLayout] | None:
        row = self.rows[key]
        name = row.key.partition(".")[2]
        route = self.routes[key]
        if route in ("verbatim", "decoded_float"):
            return MappingProxyType(
                {
                    name: TensorLayout(
                        TENSORFS_NAMES[row.dtype],
                        row.shape,
                        source="value" if route == "verbatim" else None,
                    )
                }
            )
        leaf, _, derived = self.leaves[key]
        targets = {}
        for part in row.encoded.parts:
            role = part.role
            if role not in leaf._cozy_roles:
                continue
            buffer = leaf.BUFFERS.get(role, role)
            value = leaf._buffers[buffer]
            if value is None:
                continue
            targets[f"{name.removesuffix('.weight')}.{buffer}"] = TensorLayout(
                TENSORFS_NAMES[canonical_dtype(value.dtype)],
                tuple(value.shape),
                tuple(value.stride()),
                role if role not in derived else None,
            )
        try:
            for target in targets.values():
                target.elements()
        except ResidencyError:
            return None  # e.g. a broadcast/overlapping view needs an explicit native recipe.
        return MappingProxyType(targets)

    def write(
        self, key: str, parts: Mapping[str, Tensor], destinations: Mapping[str, Tensor]
    ) -> None:
        """Cold final-byte fill. Warm residency never invokes provider code here.

        This Python writer is not an asynchronous/native-prefetch descriptor. A native
        handoff must name an implemented TensorFS read/decode operation explicitly.
        """
        route = self.routes[key]
        if route == "decoded_float":
            (destination,) = destinations.values()
            self.providers[key].decode(self.torch, parts, destination)
        elif route == "verbatim":
            for destination in destinations.values():
                destination.copy_(parts["value"])
        else:
            produced = self(key, parts)
            for name, destination in destinations.items():
                destination.copy_(produced[name])

    def __call__(self, key: str, parts: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        torch, row = self.torch, self.rows[key]
        name = row.key.partition(".")[2]
        route = self.routes[key]
        if route == "verbatim":
            return {name: parts["value"]}
        if route == "decoded_float":
            dtype = getattr(torch, TORCH_DTYPES[row.dtype])
            device = next(iter(parts.values())).device
            out = torch.empty(row.shape, dtype=dtype, device=device)
            self.providers[key].decode(torch, parts, out)
            return {name: out}
        leaf, replaced, derived = self.leaves[key]
        provider = cast(LeafProvider, self.providers[key])
        made = _bound(
            leaf, derived, parts, lambda: provider.leaf(torch, parts, replaced, leaf.out_dtype)
        )
        return {f"{name.removesuffix('.weight')}.{buffer}": t for buffer, t in made.items()}


class _PlainDtype:
    """An immutable cast of ordinary stored values into managed final buffers."""

    def __init__(self, base: _Routes, dtype: str) -> None:
        if any(route != "verbatim" for route in base.routes.values()):
            raise StaleBinding("prepared dtype overrides require plain stored weights")
        self.base = base
        layouts: dict[str, Mapping[str, TensorLayout]] = {}
        for key in base.rows:
            original = base.layout(key)
            if original is None or any(value.source != "value" for value in original.values()):
                raise StaleBinding("prepared dtype overrides require ordinary value layouts")
            if any(
                value.dtype.startswith("F") and value.dtype not in {"F16", "F32"}
                for value in original.values()
            ):
                raise StaleBinding("prepared dtype overrides support F16, BF16 and F32 sources")
            layouts[key] = MappingProxyType(
                {
                    name: TensorLayout(
                        dtype if value.dtype in {"F16", "BF16", "F32"} else value.dtype,
                        value.shape,
                        value.strides,
                        value.source,
                    )
                    for name, value in original.items()
                }
            )
        self.layouts = MappingProxyType(layouts)
        unchanged = all(dict(row) == dict(base.layout(key) or {}) for key, row in layouts.items())
        self.identity = base.identity if unchanged else repr((base.identity, "plain-cast/1", dtype))

    def layout(self, key: str) -> Mapping[str, TensorLayout]:
        return self.layouts[key]

    def write(
        self, key: str, parts: Mapping[str, Tensor], destinations: Mapping[str, Tensor]
    ) -> None:
        # The original source remains in its stored dtype. copy_ casts directly into the
        # final region; no unowned full-weight temporary or rounded-source round trip.
        self.base.write(key, parts, destinations)


def _float32(torch: ModuleType, root: Module) -> None:
    """A CPU computes 16-bit floats some two hundred times slower than float32 (torch's own CPU
    kernels: a 1024² matmul takes 2.0 s in bf16 and 9.5 ms in f32 on AVX2), so a CPU executor's
    module is float32: its placeholders convert, and each bound weight is converted while its
    block runs (varena's `module.to(dtype)`); the stored bytes stay as they are. Swapped, so a
    weight read outside its block's forward reads converted too. A package that casts its inputs
    to the 16-bit dtype it built in runs unchanged: each module owning float tensors widens its
    16-bit float inputs."""
    future = torch.__future__
    swapping = future.get_swap_module_params_on_conversion()
    future.set_swap_module_params_on_conversion(True)
    try:
        root.float()
    finally:
        future.set_swap_module_params_on_conversion(swapping)
    narrow = (torch.float16, torch.bfloat16)

    def widen(_: Module, args: Any, kwargs: Any) -> Any:
        def up(value: Any) -> Any:
            narrowed = isinstance(value, torch.Tensor) and value.dtype in narrow
            return value.float() if narrowed else value

        return torch.utils._pytree.tree_map(up, (args, kwargs))

    for sub in root.modules():
        owned = (*sub._parameters.values(), *sub._buffers.values())
        if any(t is not None and t.is_floating_point() for t in owned):
            sub.register_forward_pre_hook(widen, with_kwargs=True)


def _derived(leaf: Any, parts: Mapping[str, Any]) -> tuple[str, ...]:
    """The stored roles `leaf` keeps as other tensors than the stored ones (cast, swizzled): a
    bind makes them again from the stored bytes. A stored role it does not read is neither."""
    kept = leaf.roles()
    return tuple(sorted(role for role in kept if role in parts and kept[role] is not parts[role]))


def _bound(
    leaf: Any, derived: Sequence[str], parts: Mapping[str, Tensor], fresh: Callable[[], Any]
) -> dict[str, Tensor]:
    """`leaf`'s role buffers, by buffer name, from one bind's stored `parts`: the roles it keeps
    as stored, and the `derived` ones from a fresh leaf of the same provider."""
    buffers = leaf.BUFFERS
    made = {
        buffers.get(r, r): t for r, t in parts.items() if r in leaf._cozy_roles and r not in derived
    }
    if derived:
        built = fresh()
        made.update({buffers.get(r, r): built._buffers[buffers.get(r, r)] for r in derived})
    return made


@runtime_checkable
class LoRASource(Protocol):
    """A model's forward hook on a linear that adds a LoRA update the model owns (H3's turbo
    overlay). Its `lora_factors()` (identity, down, up, scale; None when it adds nothing now)
    are baked into the linear's bytes, and `_cozy_baked` is set while they hold it. The identity
    names the factors' content the same in every process (the package's declared source and the
    site), so pages baked with them are kept and found again."""

    _cozy_baked: bool

    def lora_factors(self) -> tuple[str, Tensor, Tensor, float] | None: ...


class LoRALayer(Protocol):
    active_adapters: Sequence[str]
    scaling: Mapping[str, float]
    _cozy_baked: bool


@dataclass(slots=True)
class _Bake:
    """One LoRA'd linear: its stored base weight, its PEFT layer with each adapter's factors
    (stored keys and scale), and the model's own hooks."""

    weight: str
    #: the stored weight's encoding: what the bake writes into
    encoding: str = ""
    layer: LoRALayer | None = None
    factors: list[tuple[str, str, float]] = field(default_factory=list)
    hooks: list[LoRASource] = field(default_factory=list)
    #: An svdq.nvfp4 leaf: its LoRAs join its low-rank branch as more ranks when it runs
    #: (`SvdqLinear.adopt`), never its bytes, so no page refills for them.
    leaf: Any = None


class _Bakes:
    """A component's LoRA'd linears, baked into their weights' bytes as the pages arrive
    (`Declared.fill`), so a resident page carries its delta and an evicted one bakes again on
    its refault. `identity` and a block's `overlay` name what a bake writes the same in every
    process, and each weight's stochastic rounding is seeded from them: a re-bake is the kept
    page's bytes."""

    def __init__(
        self, backend: Backend, root: Module, rows: Mapping[str, PlanRow], blocks: set[str]
    ) -> None:
        self.torch, self.owner, self.reading = backend.torch, backend._owner, backend._reading
        device = backend.device
        self.gpu = (device.index or 0) if device.type == "cuda" else None
        self.bakes: dict[str, _Bake] = {}
        #: Each attempt chooses once; a kernel landing at a later step cannot change its fill.
        #: Concurrent attempts keep their own choice. Idle capture uses the last served one.
        self._choice: ContextVar[tuple[lora_bake.Strategy, lora_bake.How | None] | None] = (
            ContextVar("cozy_lora_fill", default=None)
        )
        self._last: tuple[lora_bake.Strategy, lora_bake.How | None] = ("per_step", None)
        self.baked = 0
        #: block -> the package hooks its fill bakes (`overlay`)
        self._hooked: dict[str, tuple[LoRASource, ...]] = {}
        names = {row.key.partition(".")[2]: stored for stored, row in rows.items()}
        for local, stored in names.items():
            encoded = rows[stored].encoded
            site = local.removesuffix(".weight")
            roles = {p.role: p.dtype for p in encoded.parts}
            try:
                module = root.get_submodule(site)
            except AttributeError:  # a tensor named `.weight` on no module of its own
                module = None
            leaf = module if hasattr(module, "adopt") else None
            if site == local or not (
                leaf is not None or lora_bake.bakeable(encoded.scheme, dict(encoded.params), roles)
            ):
                continue
            bake = _Bake(stored, repr((encoded.scheme, encoded.params)), leaf=leaf)
            hooks = list(module._forward_hooks.values()) if module is not None else []
            if any(site.startswith(block + ".") for block in blocks):
                bake.hooks = [hook for hook in hooks if isinstance(hook, LoRASource)]
            target = site.removesuffix(".base_layer")
            if target != site:
                layer = cast(LoRALayer, root.get_submodule(target))
                for adapter in getattr(layer, "active_adapters", ()):
                    a = names.get(f"{target}.lora_A.{adapter}.weight")
                    b = names.get(f"{target}.lora_B.{adapter}.weight")
                    if a is None or b is None:
                        break
                    bake.factors.append((a, b, float(layer.scaling[adapter])))
                else:
                    if bake.factors:
                        bake.layer = layer
            if bake.layer is not None or bake.hooks:
                self.bakes[stored] = bake
        if self.bakes:
            self.choose("bake")  # A lazily constructed model's first attempt uses the default.

    def choose(self, strategy: lora_bake.Strategy) -> None:
        """Choose the attempt's serving recipe once, before its first component scope."""
        if not self.bakes:
            return
        gpu = self.gpu if strategy == "bake" else None
        # Rank-local compilation can finish at different times, and the cards may differ.
        # A group uses the reference implementation on every rank until a common fused
        # readiness contract exists. This choice is fixed before any scope binds its weights.
        group = (
            self.torch.distributed.is_initialized() and self.torch.distributed.get_world_size() > 1
        )
        how: lora_bake.How | None = "eager" if gpu is None or group else lora_bake.serving(gpu)
        self._choice.set((strategy if how is not None else "per_step", how))

    @property
    def serving(self) -> lora_bake.Strategy:
        return (self._choice.get() or self._last)[0]

    @property
    def how(self) -> lora_bake.How | None:
        return (self._choice.get() or self._last)[1]

    @property
    def identity(self) -> str | None:
        """What every page's bake shares, the same in every process (`Filling.identity`): the
        strategy and recipe (`lora_bake.recipe`), torch and the GPU's architecture, and each
        baked weight's stored name and encoding with its stored factor pairs and scales. What a
        package hook arms is each page's own (`overlay`). The attempt's serving recipe is fixed
        by `choose`: a kernel becoming ready affects the next attempt, never a refault halfway
        through this one."""
        self._last = self._choice.get() or self._last
        baked = self.serving == "bake"
        for bake in self.bakes.values():  # each LoRA'd linear told whether its bytes carry it
            if bake.layer is not None:
                bake.layer._cozy_baked = baked
            for hook in bake.hooks:
                hook._cozy_baked = baked
            if bake.leaf is not None:
                bake.leaf.adopt(functools.partial(_deltas, bake) if baked else None)
        return self._described(armed=False)

    def overlay(self, block: str, keys: Collection[str]) -> Hashable:
        """What a fill of `block` (its stored `keys`) bakes beyond `identity` now: each of its
        hooks' armed factors, named as `identity` names a delta (None where none is armed).
        Residency serves a page only to a use that wants the overlay it was filled with, so a
        block filled outside a LoRA'd forward never stands in for one, nor the reverse."""
        if self.serving != "bake":
            return ()
        hooks = self._hooked.get(block)
        if hooks is None:
            hooks = self._hooked[block] = tuple(
                hook
                for stored, bake in sorted(self.bakes.items())
                if stored in keys and bake.leaf is None
                for hook in bake.hooks
            )
        return tuple(self._delta(hook) for hook in hooks)

    def _delta(self, hook: LoRASource) -> object:
        """One hook's armed delta by its content: its factors' model and path, dtypes and scale.
        Factors no loaded model holds have no name: a fresh object, equal to no other."""
        got = hook.lora_factors()
        if got is None:
            return None
        named = self.owner(got[0])
        if named is None:
            return object()
        return (named[0].name, named[1], str(got[1].dtype), str(got[2].dtype), repr(float(got[3])))

    def _described(self, *, armed: bool) -> str | None:
        """`identity`, or with `armed` each hook's armed delta too: what seeds a bake's rounding."""
        torch = self.torch
        if self.serving != "bake":
            return repr((self.serving,))
        held: list[object] = [lora_bake.recipe(self.how == "fused"), torch.__version__]
        held.append(torch.cuda.get_device_capability(self.gpu) if self.gpu is not None else "cpu")
        for stored, bake in sorted(self.bakes.items()):
            if bake.leaf is not None:
                continue
            deltas: list[object] = [(a, b, repr(scale)) for a, b, scale in bake.factors]
            for hook in bake.hooks if armed else ():
                delta = self._delta(hook)
                if delta is None:
                    continue
                if not isinstance(delta, tuple):
                    return None
                deltas.append(delta)
            held.append((stored, bake.encoding, deltas))
        return repr(held)

    def __call__(self, block: object, parts: Mapping[str, Mapping[str, Tensor]]) -> None:
        if self.serving != "bake":
            return
        identity = self._described(armed=True)
        for stored, bake in self.bakes.items():
            weight = parts.get(stored)
            if bake.leaf is not None or weight is None:
                continue
            if any(a not in parts or b not in parts for a, b, _ in bake.factors):
                continue
            downs = [parts[a]["value"] for a, _, _ in bake.factors]
            ups = [parts[b]["value"] * scale for _, b, scale in bake.factors]
            torch, named, got = self.torch, None, None
            # A hook's factors are another model's tensors, read here outside its forward: each
            # bound as its own piece only while copied (a few MB each, not the source whole),
            # then asked for again, as the bound tensors.
            with ExitStack() as reading:
                for hook in bake.hooks:
                    named = hook.lora_factors()
                    if named is None:
                        continue
                    reading.enter_context(self.reading(named[0]))
                    got = hook.lora_factors()
                    assert got is not None
                    downs.append(got[1])
                    ups.append(got[2] * got[3])
                delta = bool(downs)
                if delta:
                    kind = downs[0].dtype
                    down = torch.cat([d.to(kind) for d in downs])
                    up = torch.cat([u.to(kind) for u in ups], dim=1)
                del named, got, downs, ups  # no view of a source outlives its piece's bind
            if not delta:
                continue
            lora_bake.bake(
                torch,
                dict(weight),
                down,
                up,
                seed=zlib.crc32(f"{identity}\0{stored}".encode()),
                how=self.how,
            )
            self.baked += 1


def _deltas(bake: _Bake) -> list[tuple[object, Tensor, Tensor, float]]:
    """An svdq leaf's LoRAs as it runs: each `(identity, down, up, scale)`, the bound factors of
    its PEFT layer's active adapters, then its model's hooks'."""
    out: list[tuple[object, Tensor, Tensor, float]] = []
    layer = cast(Any, bake.layer)
    for adapter in getattr(layer, "active_adapters", ()):
        down, up = layer.lora_A[adapter], layer.lora_B[adapter]
        out.append((down, down.weight, up.weight, float(layer.scaling[adapter])))
    for hook in bake.hooks:
        got = hook.lora_factors()
        if got is not None:
            out.append((got[0], got[1], got[2], float(got[3])))
    return out


# ------------------------------------------------------------------------- the placement


class Placement:
    """A placed component's Runtime-owned residency, graph/cache lifetime, and LoRA choice."""

    def __init__(self, memory: Memory, backend: Backend) -> None:
        self.memory = memory
        self.torch = backend.torch
        self.models = dict(backend.models)
        self.bakes = dict(backend.bakes)
        # Retain the requested mode even for a component hosted entirely on a follower.
        self._lora_mode: ContextVar[lora_bake.Strategy] = ContextVar(
            "cozy_lora_mode", default="bake"
        )
        self.notes: dict[str, str] = {}
        self.attempts = 0
        #: the request's plan (`ctx.plan`): the components each step runs together
        self.steps: list[frozenset[str]] = []
        #: components whose blocks are held bound, with the stream their binds queued on, and
        #: those this step found no free room to hold
        self.holds: dict[str, tuple[Hold, Any]] = {}
        self.unheld: set[str] = set()
        for name, model in self.models.items():
            PLACED[model.module] = (self, name)
            # A cached graph owns its exact weight tensors. Residency may ask this owner
            # to retire it only when the model is idle and a later scope actually needs room.
            model.retire_graphs = functools.partial(self._retire_graphs, name)

    def _retire_graphs(self, component: str) -> None:
        root = self.models[component].module
        for _, release in graphs.DECODERS:
            release(root)
        graphs.retire(self.torch, root)

    def open_attempt(self) -> None:
        self.attempts += 1
        self.notes.clear()
        self.steps = []
        self.release()
        self.unheld.clear()  # A previous request's failed whole-model hold is not a future limit.
        self._lora_mode.set("bake")
        for bakes in self.bakes.values():
            bakes.choose("bake")  # Followers open here; the invoke may then select per_step.
        for model in self.models.values():
            graphs.mark(model.module)

    def plan(self, steps: Sequence[str | Collection[str]], request: Request | None = None) -> None:
        self.steps = [frozenset([step] if isinstance(step, str) else step) for step in steps]
        if request is not None:
            request.resolve(self.models)

    @contextmanager
    def using(self, method: str, components: tuple[str, ...]) -> Iterator[None]:
        """A local scope (`_ComponentScope`): one residency pass per component for the call, its
        forwards nested in it, and its blocks held bound from free bytes while its plan step
        lasts. A hold never outlives the step: a later step's activations cannot evict pinned
        pages (S3). A component with no free room binds per block."""
        entering = {name for name in components if name in self.models}
        self.release(keep=entering.union(*(s for s in self.steps if s & entering)))
        for name in sorted(entering):  # graphs of another fill go before this one binds
            graphs.retire_stale(self.torch, self.models[name].module)
        stream = self.torch.cuda.current_stream() if self.memory.cuda else None
        for name in sorted(entering - self.holds.keys() - self.unheld):
            shed = functools.partial(self._shed, name)
            hold = self.models[name].hold(shed, free_only=True)
            if hold is None:
                self.unheld.add(name)
            else:
                self.holds[name] = (hold, stream)
        with ExitStack() as passes:
            for name in sorted(entering):
                passes.enter_context(self.models[name].passing(method))
            yield

    def release(self, keep: Collection[str] = ()) -> None:
        """The holds of components outside `keep`: all of them at the request's end."""
        for name in [name for name in self.holds if name not in keep]:
            self.holds.pop(name)[0].release()
            self.unheld.clear()  # room came back

    def _shed(self, component: str) -> None:
        self.holds.pop(component, None)

    def graph_held(self, component: str) -> Any:
        """The stream `component`'s held blocks bound on, or None when they are not held."""
        held = self.holds.get(component)
        return None if held is None else held[1]

    def serve_lora(self, strategy: lora_bake.Strategy) -> None:
        """Fix the request's LoRA serving (`lora_mode`) and implementation before its scopes."""
        self._lora_mode.set(strategy)
        for bakes in self.bakes.values():
            bakes.choose(strategy)
        if (
            strategy == "bake"
            and self.torch.distributed.is_initialized()
            and self.torch.distributed.get_world_size() > 1
        ):
            self.note("lora.bake", "eager reference on every executor rank")

    def lora_choices(self, components: tuple[str, ...]) -> dict[str, LoRAServing]:
        """The choice in this attempt's context, including entirely remote components."""
        return {
            name: (bakes._choice.get() or bakes._last)
            if (bakes := self.bakes.get(name)) is not None and bakes.bakes
            else ("bake", "eager")
            if bakes is not None
            else (self._lora_mode.get(), "eager")
            for name in components
        }

    @contextmanager
    def mirrored_lora(self, choices: Mapping[str, LoRAServing]) -> Iterator[None]:
        """A follower uses the transmitted recipe, never its own kernel-readiness timing."""
        with ExitStack() as stack:
            for name, choice in choices.items():
                if choice[1] == "fused":
                    raise ValueError("a mirrored LoRA scope requires the common eager recipe")
                if (bakes := self.bakes.get(name)) is not None:
                    token = bakes._choice.set(choice)
                    stack.callback(bakes._choice.reset, token)
            yield

    def retain(self, component: str, paths: tuple[str, ...]) -> Any:
        """The region's address ownership, independent of its physical replay lease."""
        return self.models[component].retain(paths)

    def represent(
        self, component: str, dtype: Any, *, resident_only: bool = False
    ) -> Callable[[], None]:
        """Preflight every requested cast before capture applies any representation change."""
        if dtype not in (self.torch.float16, self.torch.bfloat16, self.torch.float32):
            raise StaleBinding("prepared weight dtype must be float16, bfloat16 or float32")
        model = self.models[component]
        base = model.declared_finals
        if not isinstance(base, _Routes):
            raise StaleBinding("prepared dtype overrides require the ordinary weight routes")
        layout = _PlainDtype(base, TENSORFS_NAMES[canonical_dtype(dtype)])
        return self.memory._configure_binding(model, layout, resident_only=resident_only)

    def graph_accounted(self) -> bool:
        return bool(self.memory.cuda)

    def graph_domain(self, component: str) -> Any:
        return graphs.capture_device(self.torch, self.memory)

    def region(
        self,
        owner: str,
        kind: str,
        sets: Module | Sequence[Module],
        *,
        release: Callable[[], None] | None = None,
    ) -> Region:
        """An explicitly owned cache. Its owner releases entries at safe scope boundaries."""
        return self.memory.region(owner, kind, sets=sets, release=release)

    def graph_pool(self, component: str) -> graphs.domain.Pool:
        """Stock Torch owns the graph pool; the recording retains its binding and tensors."""
        return graphs.domain.Pool(self.torch.cuda.graph_pool_handle())

    def graph_pass(self, component: str, entry: str) -> Any:
        return self.models[component].passing(entry, executing=True)

    def graph_idle(self) -> AbstractContextManager[object]:
        """An idle capture's request: it fills free bytes only and never evicts."""
        return self.memory.request(graphs.IDLE_REQUEST, idle=True)

    def graph_resident(self, binding: Binding) -> bool:
        """Every page `binding` covers is on the device now (an idle capture never fills one)."""
        return binding.resident()

    def note(self, key: str, value: str) -> None:
        self.notes[key] = value

    def document(self) -> dict[str, Any]:
        """Aggregate at the request boundary; replay updates only constant-size counters."""
        return {
            "notes": dict(self.notes),
            "graphs": {name: graphs.describe(model.module) for name, model in self.models.items()},
        }

    def baked(self) -> int:
        return sum(bakes.baked for bakes in self.bakes.values())


def _derived_values(
    root: Module, component: str, unstored: Collection[str]
) -> dict[tuple[Module, str], Tensor]:
    """Buffers with their constructed values: non-persistent ones, which no checkpoint supplies,
    and the persistent ones this checkpoint does not store (`unstored`, full keys)."""
    held: dict[tuple[Module, str], Tensor] = {}
    for path, sub in root.named_modules(remove_duplicate=False):
        for local, value in sub._buffers.items():
            key = f"{component}.{path}.{local}" if path else f"{component}.{local}"
            derived = local in getattr(sub, "_non_persistent_buffers_set", ()) or key in unstored
            if derived and value is not None and str(value.device) != "meta":
                held[(sub, local)] = value.detach()
    return held


def _weightless_buffers(root: Module, component: str, stored: Collection[str]) -> dict[str, Tensor]:
    """Every persistent buffer the checkpoint stores goes to meta at each of its names, as a
    parameter is built: varena binds a meta tensor only, and a construction value left in place
    would be what a read outside the module's forward sees (Klein's VAE batch norm, se-074).
    Returns the destinations' new meta tensors by key."""
    made: dict[int, Tensor] = {}
    live: dict[str, Tensor] = {}
    for path, sub in root.named_modules(remove_duplicate=False):
        for local, value in list(sub._buffers.items()):
            key = f"{component}.{path}.{local}" if path else f"{component}.{local}"
            if value is None or value.device.type == "meta":
                continue
            if key not in stored and id(value) not in made:
                continue
            meta = made.setdefault(id(value), value.detach().to("meta"))
            sub._buffers[local] = meta
            if key in stored:
                live[key] = meta
    return live


def _restore_derived(held: Mapping[tuple[Module, str], Tensor], device: torch.device) -> None:
    for (sub, local), value in held.items():
        sub._buffers[local] = value.to(device, copy=True)
