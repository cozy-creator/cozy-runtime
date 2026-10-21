"""Model weights (memory-varena.md): Runtime builds each model on meta, and one
`tensorfs.varena.torch.Memory` per executor decides where every weight byte lives.

`Backend` is the loader's backend. `fit` judges, `materialize` keeps weights on meta and derived
buffers real, `fill` checks each destination against its stored row, and `commit` installs
encoded leaves and loads each component with `Memory.load`. A stored tensor becomes the
module's through its route (`_Routes`: as stored, decoded, or a native leaf's buffers); a LoRA'd
linear bakes on page-in (`_Bakes`), both declared on the module (`Declared`). `Residency` is
what Runtime mechanisms reach a placed component through: graph pins, KV caches and notes.
"""

from __future__ import annotations

import inspect
import threading
import time
import weakref
import zlib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from tensorfs.varena.torch import (
    Cache,
    Declared,
    Memory,
    Model,
    Shrink,
    Source,
    Unbound,
    blocks_of,
)

from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, _members
from cozy_runtime.author._model import PLACED
from cozy_runtime.internal import canonical, lora_bake
from cozy_runtime.internal.derive import placing
from cozy_runtime.internal.derived_values import DerivedValues
from cozy_runtime.internal.encoding import (
    TORCH_DTYPES,
    Capabilities,
    DeviceFacts,
    LeafProvider,
    Selection,
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
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult
from cozy_runtime.internal.qualification import qualified
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
        qualified: tuple[Capabilities, QualificationResult],
        device_facts: DeviceFacts,
        custody: str = "canonical",
        rank: int = 0,
        world: int = 1,
    ) -> None:
        self.memory = memory
        self.device = device
        self.torch = torch
        self.construction = construction
        self.checkpoints = dict(checkpoints)
        self.plan = {row.key: row for row in rows}
        self.custody = custody
        self.rank, self.world = rank, world
        self.state = "open"
        self.expected: dict[str, frozenset[str]] = {}
        #: component -> stored tensors the code does not build: never read or mapped
        self.skipped: dict[str, list[str]] = {}
        self.enqueued: dict[str, TensorLike] = {}
        self.roots: dict[str, Module] = {}
        #: each component's registered model, and its LoRA bakes
        self.models: dict[str, Model] = {}
        self.bakes: dict[str, _Bakes] = {}
        self.derived: dict[str, dict[tuple[Module, str], Tensor]] = {}
        self.derived_owners: list[DerivedValues] = []
        self.stats: dict[str, float] = {}
        self.device_facts = device_facts
        self.providers = launch_providers()
        self.capabilities, self.probe = qualified
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
        objective: str = "latency",
        encoded_leaves: str = "accept",
    ) -> Backend:
        """A backend for a driver script with no model class: measure, qualify, resolve a
        `script_plan` over these rows, then construct, through the executor's own authority."""
        facts = measure_device(torch, device.index or 0)
        providers = launch_providers()
        qualification = qualified(
            torch,
            providers,
            facts,
            sorted({dtype_name(row.dtype) for row in rows}),
            release=release,
        )
        plan = script_plan(
            rows=rows,
            components=sorted({row.component for row in rows}),
            release=release,
            store="script",
            snapshot=next(iter(checkpoints.values())).id,
            snapshots={name: book.id for name, book in checkpoints.items()},
            providers=providers,
            capabilities=qualification.capabilities,
            device=facts,
            runtime=measure_runtime(torch, release),
            facts=PlanFacts(),
            objective=objective,
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
            qualified=(qualification.capabilities, qualification.result),
            device_facts=facts,
        )

    def _select(self, key: str, row: PlanRow) -> Selection:
        """The plan's provider for one destination, matched by digest against this build and
        this device's capability records. Anything else is `plan_divergence`, never a
        re-selection."""
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
        records = self.capabilities.qualified(
            row.encoded.scheme,
            self.device_facts,
            resolved.output_dtype,
            {resolved.implementation_digest: provider},
        )
        if not records:
            raise _refuse(
                "plan_divergence",
                f"{key}: {device_line(self.device_facts)} holds no capability record for "
                f"{resolved.implementation}",
                [key],
            )
        return Selection(provider, records[0], row.encoded, resolved.reason)

    # -- the Backend protocol

    @property
    def bound(self) -> frozenset[str]:
        """The components the binding selects; the construction's others stay on meta."""
        return frozenset(self.checkpoints)

    def ahead(self) -> None:
        """The loader calls this before construction; source reads wait for registration."""
        self.stats["ahead_ms"] = 0.0

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
        started = time.perf_counter()
        order = list(self.enqueued)
        self.leaves = {
            key: self._install_leaf(key) for key in order if self._route(key) == "encoded_gemm"
        }
        self.gemm_sites = leaf_sites(self.leaves)
        for name, root in self.roots.items():
            keys = [key for key in order if self.plan[key].component == name]
            if keys:
                self.models[name] = self._register(name, root, keys)
        if self.device.type == "cuda" and any(b.bakes for b in self.bakes.values()):
            lora_bake.warm(self.device.index or 0)  # its kernels build at load, in a child
        self.stats["register_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.state = "ready"

    def poison(self, keys: Sequence[str]) -> None:
        """Abort: every registered model goes."""
        self.close()
        self.state = "poisoned"

    def close(self) -> None:
        for owner in self.derived_owners:
            owner.close()
        self.derived_owners.clear()
        for model in self.models.values():
            model.close()
        self.models.clear()

    # -- construction internals

    def _route(self, key: str) -> str:
        return str(self.selected[key].provider.route)

    def _register(self, name: str, root: Module, keys: list[str]) -> Model:
        """Declare the component's source and transforms; varena owns its byte placement."""
        rows = {self.plan[key].name: self.plan[key] for key in keys}
        bakes = self.bakes[name] = _Bakes(self.torch, root, rows, set(blocks_of(root)))
        checkpoint = self.checkpoints[name]
        source = Source(
            checkpoint.ck, name, checkpoint.id, keys=tuple(rows), fingerprint=self._layout(name)
        )
        vars(root)["_varena"] = Declared(
            source,
            state=tuple(getattr(root, "_cozy_state", ())),
            fill=bakes.pending() if bakes.bakes else None,
            bind=_Routes(self, rows),
        )
        return self.memory.load(root, level="ram")

    def _layout(self, name: str) -> str:
        """The source and transform identity used to reject incompatible retained pages."""
        return canonical.digest(
            {
                "manifest": self.checkpoints[name].id,
                "construction": self.execution.construction.document(),
                "component": name,
                "rank": self.rank,
                "world": self.world,
            }
        )

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
    """A stored tensor's parts as the module's tensors (`Declared.bind`): as stored,
    decoded into a fresh float tensor, or a native leaf's role buffers (derived roles made from
    the stored ones). What it makes lives for that bind only."""

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

    def __call__(self, key: str, parts: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        torch, row = self.torch, self.rows[key]
        name = row.key.partition(".")[2]
        route = self.routes[key]
        if route == "verbatim":
            return {name: parts["value"]}
        if route == "decoded_float":
            dtype = getattr(torch, TORCH_DTYPES[row.dtype])
            out = torch.empty(row.shape, dtype=dtype, device=next(iter(parts.values())).device)
            self.providers[key].decode(torch, parts, out)
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


@runtime_checkable
class LoRASource(Protocol):
    """A model's forward hook on a linear that adds a LoRA update the model owns (H3's turbo
    overlay). Its `lora_for_call()` (identity, down, up, scale; None for the base call)
    are baked into the linear's bytes, and `_cozy_baked` is set while they hold it."""

    _cozy_baked: bool

    def lora_for_call(
        self, arguments: Mapping[str, object]
    ) -> tuple[str, Tensor, Tensor, float] | None:
        """Immutable source/component/site identity and lazy factors for this root call.

        Selection reads metadata only, before placement or graph replay. The identity
        names actual source bytes; module repr or object identity is not provenance.
        """
        ...


class LoRALayer(Protocol):
    active_adapters: Sequence[str]
    scaling: Mapping[str, float]
    _cozy_baked: bool


def _forward_arguments(
    module: Module, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> dict[str, object]:
    """The declared class method, never Varena's instance-level `*args/**kwargs` wrapper."""
    signature = inspect.signature(inspect.unwrap(type(module).forward))
    parameters = tuple(signature.parameters.values())
    if not parameters or not any(
        value.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for value in parameters[1:]
    ):
        raise Unbound("LoRA selection requires a declared forward signature")
    arguments = dict(signature.bind(module, *args, **kwargs).arguments)
    arguments.pop(parameters[0].name)
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL and arguments.get(parameter.name):
            raise Unbound("LoRA selection cannot infer unnamed positional forward arguments")
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            extra = arguments.pop(parameter.name, {})
            if isinstance(extra, Mapping):
                arguments.update(extra)
    return arguments


@dataclass(slots=True)
class _Bake:
    """One LoRA'd linear: its stored base weight, its PEFT layer with each adapter's factors
    (stored keys and scale), and the model's own hooks."""

    weight: str
    layer: LoRALayer | None = None
    factors: list[tuple[str, str, float]] = field(default_factory=list)
    hooks: list[LoRASource] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _BakeUse:
    weight: str
    factors: tuple[tuple[str, str, float], ...]
    external: tuple[tuple[str, Tensor, Tensor, float], ...]


@dataclass(frozen=True, slots=True)
class _BakeSnapshot:
    """One registration's immutable transform: no call inputs or live hook selection."""

    version: str
    strategy: lora_bake.Strategy
    uses: tuple[_BakeUse, ...]
    owner: _Bakes
    prefetch_ready: bool = True

    def prepare(self) -> str:
        return self.version

    def __call__(self, block: object, parts: Mapping[str, Mapping[str, Tensor]]) -> None:
        if self.strategy != "bake":
            return
        torch = self.owner.torch
        for use in self.uses:
            weight = parts.get(use.weight)
            if weight is None or any(a not in parts or b not in parts for a, b, _ in use.factors):
                continue
            downs = [parts[a]["value"] for a, _, _ in use.factors]
            ups = [parts[b]["value"] * scale for _, b, scale in use.factors]
            for _, down, up, scale in use.external:
                downs.append(down)
                ups.append(up * scale)
            if not downs:
                continue
            kind = downs[0].dtype
            lora_bake.bake(
                torch,
                dict(weight),
                torch.cat([d.to(kind) for d in downs]),
                torch.cat([u.to(kind) for u in ups], dim=1),
                seed=zlib.crc32(use.weight.encode()),
            )
            self.owner.baked += 1


class _Bakes:
    """A component's LoRA'd linears, baked into their weights' bytes as the pages arrive
    (`Declared.fill`), so a resident page carries its delta and an evicted one bakes again on
    its refault. Each weight's own seed makes every bake of it the same bytes."""

    def __init__(
        self, torch: ModuleType, root: Module, rows: Mapping[str, PlanRow], blocks: set[str]
    ) -> None:
        self.torch = torch
        self.bakes: dict[str, _Bake] = {}
        self.strategy: lora_bake.Strategy = "bake"
        self.baked = 0
        self.lock = threading.RLock()
        self.active: _BakeSnapshot | None = None
        names = {row.key.partition(".")[2]: stored for stored, row in rows.items()}
        for local, stored in names.items():
            encoded = rows[stored].encoded
            site = local.removesuffix(".weight")
            roles = {p.role: p.dtype for p in encoded.parts}
            if site == local or not lora_bake.bakeable(encoded.scheme, dict(encoded.params), roles):
                continue
            bake = _Bake(stored)
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
        self.call_selected = any(bake.hooks for bake in self.bakes.values())

    def pending(self) -> _BakeSnapshot:
        """Fixed transforms are already known; external selections await their real call."""
        return self.snapshot({}, prefetch_ready=not self.call_selected)

    def snapshot(
        self, arguments: Mapping[str, object], *, prefetch_ready: bool = True
    ) -> _BakeSnapshot:
        """Resolve metadata once; later fills never ask a mutable hook which overlay is armed."""
        strategy = self.strategy
        uses, identities = [], []
        for bake in self.bakes.values():
            external = tuple(
                value for hook in bake.hooks if (value := hook.lora_for_call(arguments)) is not None
            )
            if any(not isinstance(value[0], str) or not value[0] for value in external):
                raise ValueError("LoRA source identities must be nonempty immutable source keys")
            uses.append(_BakeUse(bake.weight, tuple(bake.factors), external))
            identities.append(
                {
                    "weight": bake.weight,
                    "stored": list(bake.factors),
                    "external": [(identity, scale) for identity, _, _, scale in external],
                }
            )
        version = canonical.digest(
            {
                "bake": lora_bake.VARIANT,
                "strategy": strategy,
                "weights": identities,
            }
        )
        return _BakeSnapshot(version, strategy, tuple(uses), self, prefetch_ready)

    def _flags(self) -> list[LoRALayer | LoRASource]:
        flags: list[LoRALayer | LoRASource] = []
        for bake in self.bakes.values():
            if bake.layer is not None:
                flags.append(bake.layer)
            flags.extend(bake.hooks)
        return flags

    @contextmanager
    def call(self, model: Model, arguments: Mapping[str, object]) -> Iterator[None]:
        # Other threads serialize. Same-thread nesting may borrow the exact selection,
        # but may never replace bytes/flags an outer forward is still consuming.
        with self.lock:
            selected = self.snapshot(arguments)
            prior, active = model.fill, self.active
            if active is not None and selected.version != active.version:
                raise Unbound("a nested call cannot change an active LoRA selection")
            flags = [(item, item._cozy_baked) for item in self._flags()]
            model.fill = self.active = active or selected
            for item, _ in flags:
                item._cozy_baked = selected.strategy == "bake"
            try:
                yield
            except BaseException:
                model.fill = prior
                for item, value in flags:
                    item._cozy_baked = value
                raise
            finally:
                self.active = active

    def install_call(self, model: Model) -> None:
        """Outside graph __call__, which itself precedes nn.Module's forward prehooks."""
        if not self.bakes or getattr(type(model.module), "_cozy_fill_call", False):
            return
        root, owner = model.module, self
        cls = type(root)
        call = cls.__call__

        def __call__(module: Any, *args: Any, **kwargs: Any) -> Any:
            arguments = (
                _forward_arguments(module, args, kwargs)
                if any(bake.hooks for bake in owner.bakes.values())
                else {}
            )
            with owner.call(model, arguments):
                return call(module, *args, **kwargs)

        root.__class__ = type(
            cls.__name__,
            (cls,),
            {
                "__call__": __call__,
                "_cozy_fill_call": True,
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
            },
        )


# ------------------------------------------------------------------------- the residency


class Residency:
    """What a Runtime mechanism handed a placed component's module reaches (`PLACED`): graph
    pins and KV caches as the memory's caches, a request's LoRA serving, and notes."""

    def __init__(self, memory: Memory, backend: Backend) -> None:
        self.memory = memory
        self.models = dict(backend.models)
        self.bakes = dict(backend.bakes)
        self.notes: dict[str, str] = {}
        self.attempts = 0
        self.pins: dict[tuple[str, str], _Pin] = {}
        self.derived: list[DerivedValues] = []
        for name, model in self.models.items():
            PLACED[model.module] = (self, name)
            owner = DerivedValues.install(
                memory, model.module, name, backend.torch, partial(self._retire, name)
            )
            if owner is not None:
                self.derived.append(owner)
                backend.derived_owners.append(owner)

    def _retire(self, component: str) -> None:
        for (name, _), token in tuple(self.pins.items()):
            if name == component:
                token.cache.shrink(token.nbytes)

    def open_attempt(self) -> None:
        self.attempts += 1
        self.notes.clear()
        for name, bakes in self.bakes.items():
            if bakes.call_selected:
                self.models[name].fill = bakes.pending()

    def prepare_calls(self) -> None:
        """Install only after graph wrappers, so replays also select their transform first."""
        for name, bakes in self.bakes.items():
            model = self.models.get(name)
            if model is not None:
                bakes.install_call(model)

    def serve_lora(self, strategy: lora_bake.Strategy) -> None:
        """A request's LoRA serving (`lora_mode`): baked, or per step."""
        for bakes in self.bakes.values():
            bakes.strategy = strategy

    def pin(self, component: str, owner: str, shed: Callable[[], int]) -> _Pin | None:
        """Every weight of `component` held bound for `owner` (a CUDA graph replays at their
        addresses, decoded copies included) by a cache of the memory's. When memory asks for
        room or the component's LoRAs change, `shed` drops what read them there; the weights
        stay bound until the last reference to the pin goes, so a replay in flight keeps them.
        None when part of the component streams (a slot's address moves)."""
        key = (component, owner)
        model = self.models.get(component)
        if model is None:
            return None
        version = model.fill.prepare() if model.fill is not None else None
        if version != model.version:
            # Replays bypass the module's prehook. Shed old recordings without
            # borrowing their bindings; the eager forward refreshes the content.
            for (name, _), token in tuple(self.pins.items()):
                if name == component:
                    token.cache.shrink(token.nbytes)
            return None
        # A graph replay bypasses the eager prehook. The selected bytes may be
        # unchanged warm content whose request-local edges are still pending.
        self.memory._fresh(model)
        held = self.pins.get(key)
        if held is not None:
            return held

        token_ref: weakref.ReferenceType[_Pin] | None = None

        def give(_: int) -> int:
            token = token_ref() if token_ref is not None else None
            if token is None or token.retired:
                return 0
            token.retired = True
            if self.pins.get(key) is token:
                self.pins.pop(key)
            # An in-flight recording may still own the token. Its last reference,
            # after the consumer's borrowed hold exits, closes the persistent hold.
            try:
                return shed()
            finally:
                # Retired external recordings are live Torch allocations, not
                # cached bytes available to this owner. Publish before remeasure.
                cache.nbytes = 0

        cache = self.memory.cache(owner, give)
        try:
            with cache.serving():
                cache.hold(model)
                token = _Pin(
                    model, cache, [owner for owner in self.derived if owner.component == component]
                )
                token_ref = weakref.ref(token)
                self.pins[key] = token
        except Unbound:
            cache.close()
            return None
        except BaseException:
            cache.close()
            raise
        return token

    def cache(self, owner: str, shed: Shrink) -> Cache | None:
        """Device bytes `owner` keeps for speed, counted and shed on ask; None off a GPU."""
        return self.memory.cache(owner, shed) if self.memory.cuda else None

    def note(self, key: str, value: str) -> None:
        self.notes[key] = value

    def baked(self) -> int:
        return sum(bakes.baked for bakes in self.bakes.values())


class _Pin:
    """A pinned component's token: its weights stay bound while a reference lives."""

    __slots__ = ("__weakref__", "cache", "derived", "model", "retired")

    def __init__(self, model: Model, cache: Cache, derived: Sequence[DerivedValues] = ()) -> None:
        self.model, self.cache, self.retired = model, cache, False
        self.derived = tuple(derived)
        weakref.finalize(self, cache.close)

    @property
    def nbytes(self) -> int:
        return self.cache.nbytes

    @nbytes.setter
    def nbytes(self, value: int) -> None:
        if not self.retired:
            self.cache.nbytes = value

    @contextmanager
    def serving(self) -> Iterator[bool]:
        retired = self.retired
        if retired:
            yield False
            return
        with ExitStack() as stack:
            # Reclamation may have retired this token before the derived lease is acquired.
            # Take those leases first: their callbacks retire graph pins, never the reverse.
            for owner in self.derived:
                stack.enter_context(owner.serving())
            stack.enter_context(self.cache.serving())
            # cache_active may have waited for a reclamation callback to finish.
            if self.retired:
                yield False
                return
            hold = self.cache.hold(self.model)
            try:
                yield True
            finally:
                # Public Hold.close joins this consumer into the binding stream.
                hold.close()


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
