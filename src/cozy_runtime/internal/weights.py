"""Model weights (varena-slim §3): Runtime builds each model on meta, and one
`tensorfs.varena.torch.Memory` per executor decides where every weight byte lives.

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
import threading
import time
import zlib
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from tensorfs.varena.torch import (
    Binding,
    Cache,
    Declared,
    Hold,
    Level,
    Memory,
    Model,
    Weights,
    blocks_of,
)

from cozy_runtime import graphs
from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, _members
from cozy_runtime.author._model import PLACED
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
    measure_device,
    measure_runtime,
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
from cozy_runtime.internal.resolution import ResolvedModelPlan, TensorResolution, script_plan

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
        #: each component's bind routes, which keep its decoded tensors for the generation
        self.routes: dict[str, _Routes] = {}
        self.derived: dict[str, dict[tuple[Module, str], Tensor]] = {}
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

    @classmethod
    def for_script(
        cls,
        checkpoints: Mapping[str, Checkpoint],
        rows: Sequence[PlanRow],
        *,
        torch: ModuleType,
        memory: Memory,
        device: torch.device,
        construction: str,
        release: str,
        encoded_leaves: str = "accept",
    ) -> Backend:
        """A backend for a driver script with no model class: resolve a `script_plan` over
        these rows, then construct, through the executor's own authority."""
        facts = measure_device(torch, device.index or 0)
        providers = launch_providers()
        plan = script_plan(
            rows=rows,
            components=sorted({row.component for row in rows}),
            release=release,
            store="script",
            snapshot=next(iter(checkpoints.values())).id,
            snapshots={name: book.id for name, book in checkpoints.items()},
            providers=providers,
            device=facts,
            runtime=measure_runtime(torch, release),
            encoded_leaves=encoded_leaves,
            dtype_name=dtype_name,
        )
        return cls(
            checkpoints,
            rows,
            torch=torch,
            plan=plan,
            memory=memory,
            device=device,
            construction=construction,
            device_facts=facts,
            release=release,
        )

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
        if len(by_checkpoint) > 1:
            verdict = {**verdict, "ignored": ignored, "routes": routes}
        return {**verdict, "warnings": warnings}

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
        """Weights stay on meta. Derived (non-persistent) buffers, which no checkpoint
        supplies, move to the device with the values their construction computed."""
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
        with placing():
            for name in walked.components:
                member = roots.get(name)
                if not isinstance(member, self.torch.nn.Module):
                    continue
                self.roots[name] = member
                self.derived[name] = _derived_values(member)
                _restore_derived(self.derived[name], self.device)
        return walked.live

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
                f"({', '.join(missing[:4])}): a DestinationSet is complete or nothing commits",
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
        for routes in self.routes.values():
            routes.close()
        self.routes.clear()

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
        routes = self.routes[name] = _Routes(self, name, rows, root)
        leg = self._leg(f"{name}.routes", leg)
        vars(root)["_varena"] = Declared(
            Weights(checkpoint.ck, name, release=self.release, keys=tuple(rows)),
            fill=bakes if bakes.bakes else None,
            bind=routes,
        )
        # At RAM: pages reach the device through placement (passes, the plan, an explicit warm),
        # never at construction (v6 13633 filled the card with H3's transformer before its TE).
        model = self.memory.load(root, Level.RAM)
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
        kept = leaf.roles()
        derived = tuple(sorted(role for role in parts if kept.get(role) is not parts[role]))
        return leaf, replaced, derived


# ----------------------------------------------------------------- routes and LoRA bakes


class _Routes:
    """A stored tensor's parts as the module's tensors (`Declared.bind`): as stored, decoded
    into a float tensor, or a native leaf's role buffers (derived roles made from the stored
    ones). A decoded tensor is decoded once a generation (`delivery.cost_class`), not once a
    bind: it is kept under the fill identity it was decoded from, as the memory's cache, which
    varena sheds on its ask (the next bind decodes again). Leaf buffers live for that bind."""

    def __init__(
        self, backend: Backend, component: str, rows: Mapping[str, PlanRow], root: Module
    ) -> None:
        self.backend, self.component, self.root = backend, component, root
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
        self.lock = threading.Lock()
        #: stored key -> (the fill identity it was decoded under, the decoded tensor)
        self.decoded: dict[str, tuple[str, Tensor]] = {}
        self.cache: Cache | None = None

    def __call__(self, key: str, parts: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        torch, row = self.torch, self.rows[key]
        name = row.key.partition(".")[2]
        route = self.routes[key]
        if route == "verbatim":
            return {name: parts["value"]}
        if route == "decoded_float":
            identity = self._identity()
            with self.lock:
                held = self.decoded.get(key)
                if held is not None and held[0] == identity:
                    return {name: held[1]}
            dtype = getattr(torch, TORCH_DTYPES[row.dtype])
            out = torch.empty(row.shape, dtype=dtype, device=next(iter(parts.values())).device)
            self.providers[key].decode(torch, parts, out)
            if identity:
                self._keep(key, identity, out)
            return {name: out}
        leaf, replaced, derived = self.leaves[key]
        path, buffers = name.removesuffix(".weight"), leaf.BUFFERS
        made = {f"{path}.{buffers.get(r, r)}": t for r, t in parts.items() if r not in derived}
        if derived:
            provider = cast(LeafProvider, self.providers[key])
            fresh = provider.leaf(torch, parts, replaced, leaf.out_dtype)
            for role in derived:
                made[f"{path}.{buffers.get(role, role)}"] = fresh._buffers[buffers.get(role, role)]
        return made

    def _identity(self) -> str:
        """What the stored bytes are now: as stored, or the fill (LoRA bake) identity they were
        baked under; "" where the fill declares none, whose decode is never kept."""
        model = self.backend.models.get(self.component)
        if model is None or model.fill is None:
            return "stored"
        return model.identity or ""

    def _keep(self, key: str, identity: str, out: Tensor) -> None:
        with self.lock:
            self.decoded[key] = (identity, out)
            if self.cache is None:
                self.cache = self.backend.memory.cache(
                    f"decoded:{self.component}", self.shed, sets=self.root
                )
            self.cache.cached(self._held())

    def _held(self) -> int:
        return sum(t.numel() * t.element_size() for _, t in self.decoded.values())

    def shed(self, want: int) -> int:
        """Varena's ask: decoded tensors go until `want` bytes have (every one for `want` <= 0)."""
        with self.lock:
            freed = 0
            for key in list(self.decoded):
                if want > 0 and freed >= want:
                    break
                tensor = self.decoded.pop(key)[1]
                freed += tensor.numel() * tensor.element_size()
            if self.cache is not None:
                self.cache.cached(self._held())
            return freed

    def close(self) -> None:
        """The generation ends: every decoded tensor, and the cache that accounted for them."""
        with self.lock:
            self.decoded.clear()
            if self.cache is not None:
                self.cache.close()
                self.cache = None


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


class _Bakes:
    """A component's LoRA'd linears, baked into their weights' bytes as the pages arrive
    (`Declared.fill`), so a resident page carries its delta and an evicted one bakes again on
    its refault. `identity` names what a bake writes the same in every process, and each
    weight's stochastic rounding is seeded from it: a re-bake is the kept page's bytes."""

    def __init__(
        self, backend: Backend, root: Module, rows: Mapping[str, PlanRow], blocks: set[str]
    ) -> None:
        self.torch, self.owner, self.reading = backend.torch, backend._owner, backend._reading
        self.bakes: dict[str, _Bake] = {}
        self.strategy: lora_bake.Strategy = "bake"
        self.baked = 0
        names = {row.key.partition(".")[2]: stored for stored, row in rows.items()}
        for local, stored in names.items():
            encoded = rows[stored].encoded
            site = local.removesuffix(".weight")
            roles = {p.role: p.dtype for p in encoded.parts}
            if site == local or not lora_bake.bakeable(encoded.scheme, dict(encoded.params), roles):
                continue
            bake = _Bake(stored, repr((encoded.scheme, encoded.params)))
            try:
                hooks = list(root.get_submodule(site)._forward_hooks.values())
            except AttributeError:  # a tensor named `.weight` on no module of its own
                hooks = []
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

    @property
    def identity(self) -> str | None:
        """What a bake writes now, the same in every process (`Filling.identity`): the strategy
        and recipe (`lora_bake.recipe`), torch and the GPU's architecture, and each baked
        weight's stored name and encoding with each delta's content (a stored factor pair of
        this checkpoint, or the identity a hook's package declares), dtypes and scale."""
        torch = self.torch
        baked = self.strategy == "bake"
        for bake in self.bakes.values():  # each LoRA'd linear told whether its bytes carry it
            if bake.layer is not None:
                bake.layer._cozy_baked = baked
            for hook in bake.hooks:
                hook._cozy_baked = baked
        if not baked:
            return repr((self.strategy,))
        device = torch.cuda.current_device() if torch.cuda.is_initialized() else -1
        held: list[object] = [lora_bake.recipe(device), torch.__version__]
        held.append(torch.cuda.get_device_capability(device) if device >= 0 else "cpu")
        for stored, bake in sorted(self.bakes.items()):
            deltas: list[object] = [(a, b, repr(scale)) for a, b, scale in bake.factors]
            for hook in bake.hooks:
                got = hook.lora_factors()
                if got is None:
                    continue
                named = self.owner(got[0])
                if named is None:
                    return None
                dtypes = (str(got[1].dtype), str(got[2].dtype))
                deltas.append((named[0].name, named[1], *dtypes, repr(float(got[3]))))
            held.append((stored, bake.encoding, deltas))
        return repr(held)

    def __call__(self, block: object, parts: Mapping[str, Mapping[str, Tensor]]) -> None:
        identity = self.identity
        if self.strategy != "bake":
            return
        for stored, bake in self.bakes.items():
            weight = parts.get(stored)
            if weight is None or any(a not in parts or b not in parts for a, b, _ in bake.factors):
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
                torch, dict(weight), down, up, seed=zlib.crc32(f"{identity}\0{stored}".encode())
            )
            self.baked += 1


# ------------------------------------------------------------------------- the placement


class Placement:
    """What a Runtime mechanism handed a placed component's module reaches (`PLACED`): graph
    pins and KV caches as the memory's caches, a request's LoRA serving, notes, and each
    scope's varena pass and step hold (`using`)."""

    def __init__(self, memory: Memory, backend: Backend) -> None:
        self.memory = memory
        self.torch = backend.torch
        self.models = dict(backend.models)
        self.bakes = dict(backend.bakes)
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

    def open_attempt(self) -> None:
        self.attempts += 1
        self.notes.clear()
        self.steps = []
        self.release()
        for model in self.models.values():
            graphs.mark(model.module)

    def plan(self, steps: Sequence[str | Collection[str]]) -> None:
        self.steps = [frozenset([step] if isinstance(step, str) else step) for step in steps]

    @contextmanager
    def using(self, method: str, components: tuple[str, ...]) -> Iterator[None]:
        """A local scope (`_ComponentScope`): one varena pass per component for the call, its
        forwards nested in it, and its blocks held bound from free bytes while its plan step
        lasts. A hold never outlives the step: a later step's activations cannot evict pinned
        pages (S3). A component with no free room binds per block."""
        entering = {name for name in components if name in self.models}
        self.release(keep=entering.union(*(s for s in self.steps if s & entering)))
        stream = self.torch.cuda.current_stream() if self.memory.cuda else None
        for name in sorted(entering - self.holds.keys() - self.unheld):
            shed = functools.partial(self.holds.pop, name, None)
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

    def graph_held(self, component: str) -> Any:
        """The stream `component`'s held blocks bound on, or None when they are not held."""
        held = self.holds.get(component)
        return None if held is None else held[1]

    def serve_lora(self, strategy: lora_bake.Strategy) -> None:
        """A request's LoRA serving (`lora_mode`): baked, or per step."""
        for bakes in self.bakes.values():
            bakes.strategy = strategy

    def retain(self, component: str, paths: tuple[str, ...]) -> Any:
        """The region's address ownership, independent of its physical replay lease."""
        return self.models[component].retain(paths)

    def graph_pool(self) -> Any:
        return self.memory.new_pool()

    def graph_accounted(self) -> bool:
        return self.memory.cuda and self.memory.native.context_measurable()

    def graph_domain(self, component: str) -> Any:
        return graphs.capture_device(self.torch, self.memory)

    def graph_allocating(self, pool: Any, *, route: bool = True) -> Any:
        return self.memory.allocating(pool, route=route)

    def graph_levels(self) -> tuple[tuple[tuple[str, str], int], ...]:
        """The process's cut levels: a graph recorded under them replays only under them."""
        return execution.catch.levels()

    def graph_segments(self, component: str) -> tuple[str, ...]:
        """The model's declared block topology, which graph regions align with."""
        return tuple(self.models[component].blocks)

    def graph_pass(self, component: str, entry: str) -> Any:
        return self.models[component].passing(entry)

    def graph_idle(self) -> AbstractContextManager[object]:
        """An idle capture's request: it fills free bytes only and never evicts."""
        return self.memory.request("graphs-idle", idle=True)

    def graph_resident(self, binding: Binding) -> bool:
        """Every page `binding` covers is on the device now (an idle capture never fills one)."""
        return binding.resident()

    def cache(
        self, owner: str, shed: Callable[[int], int], served: Sequence[Module]
    ) -> Cache | None:
        """Device bytes `owner` keeps for the components `served` (their roots), counted and
        shed on ask in the order of their next use; None off a GPU."""
        return self.memory.cache(owner, shed, sets=list(served)) if self.memory.cuda else None

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


def _derived_values(root: Module) -> dict[tuple[Module, str], Tensor]:
    """Non-persistent buffers with their constructed values: no checkpoint supplies them."""
    held: dict[tuple[Module, str], Tensor] = {}
    for _, sub in root.named_modules():
        for local in getattr(sub, "_non_persistent_buffers_set", ()):
            value = sub._buffers.get(local)
            if value is not None and str(value.device) != "meta":
                held[(sub, local)] = value.detach()
    return held


def _restore_derived(held: Mapping[tuple[Module, str], Tensor], device: torch.device) -> None:
    for (sub, local), value in held.items():
        sub._buffers[local] = value.to(device, copy=True)
