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

import os
import time
import weakref
import zlib
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from tensorfs.varena.torch import (
    Cache,
    Declared,
    Memory,
    Model,
    Shrink,
    Unbound,
    WeightGrant,
    blocks_of,
)

from cozy_runtime.author._executor_requests import HostGrant
from cozy_runtime.author._loader import Census, TensorLike, TensorSpec, _members
from cozy_runtime.author._model import PLACED
from cozy_runtime.internal import canonical, lora_bake
from cozy_runtime.internal.derive import placing
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
from cozy_runtime.internal.parallel import ulysses  # noqa: F401 (Ulysses' torch side, served)
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import QualificationResult
from cozy_runtime.internal.qualification import qualified
from cozy_runtime.internal.resolution import ResolvedModelPlan, TensorResolution, script_plan

if TYPE_CHECKING:
    import torch
    from torch import Tensor
    from torch.nn import Module

#: `ReadGrant` by manifest and layout digest.
type Grant = Callable[[str, str], HostGrant]
#: Whether this TensorFS seals a host layout only where TensorD admits it (`WeightGrant.seal`,
#: 0.13.3); an older one seals on every unheld load, as it always did.
ADMITS = "seal" in {f.name for f in fields(WeightGrant)}
#: GPU pages varena kept, given to TensorD (`KeepPages`): layout digest, fingerprint, and
#: `(region, span, fd)` each (the callee's to close).
type Keep = Callable[[str, str, list[tuple[int, int, int]]], None]
#: A host layout varena sealed, deposited with TensorD (`Deposit`): manifest, name, layout
#: digest, the memfd (the callee's to close).
type Deposit = Callable[[str, str, str, int], None]


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
        grant: Grant | None = None,
        deposit: Deposit | None = None,
        keeper: Keep | None = None,
    ) -> None:
        self.memory = memory
        self.device = device
        self.grant, self.deposit, self.keeper = grant, deposit, keeper
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
        #: each component's registered model, and its LoRA bakes
        self.models: dict[str, Model] = {}
        self.bakes: dict[str, _Bakes] = {}
        self.derived: dict[str, dict[tuple[Module, str], Tensor]] = {}
        #: each bound component's grant from TensorD, asked once (`ahead`); its fds are ours
        self.grants: dict[str, WeightGrant] = {}
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
        """Before the factory runs, each bound component's grant, asked once. The load registers
        it, held or sealed, the one way it loads any time."""
        started = time.perf_counter()
        for name in self.checkpoints:
            self._granted(name)
        self.stats["ahead_ms"] = round((time.perf_counter() - started) * 1000, 2)

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
        self._release()
        if self.device.type == "cuda" and any(b.bakes for b in self.bakes.values()):
            lora_bake.warm(self.device.index or 0)  # its kernels build at load, in a child
        self.stats["register_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self.state = "ready"

    def poison(self, keys: Sequence[str]) -> None:
        """Abort: every registered model goes."""
        self.close()
        self.state = "poisoned"

    def close(self) -> None:
        self._release()
        for model in self.models.values():
            model.close()
        self.models.clear()

    # -- construction internals

    def _owner(self, module: object) -> tuple[str, str] | None:
        """The checkpoint component (its content id) and module path that hold `module`, when
        a component this process loaded does, of any construction (H3's turbo overlay is its
        own)."""
        for model in list(self.memory.models.values()):
            for path, sub in model.module.named_modules():
                if sub is module:
                    return model.key, path
        return None

    def _route(self, key: str) -> str:
        return str(self.selected[key].provider.route)

    def _register(self, name: str, root: Module, keys: list[str]) -> Model:
        """One component loaded into the memory: its stored tensors, bound through their
        routes, its LoRAs baked on page-in, its bytes the host layout TensorD holds for it, else
        read from the checkpoint by varena, which seals them and deposits the layout with TensorD
        for the next load."""
        rows = {self.plan[key].name: self.plan[key] for key in keys}
        bakes = self.bakes[name] = _Bakes(self.torch, root, rows, set(blocks_of(root)), self._owner)
        grant = replace(self._granted(name), keys=tuple(rows))
        vars(root)["_varena"] = Declared(  # `_cozy_state`: what the package's forward changes
            grant,
            state=tuple(getattr(root, "_cozy_state", ())),
            fill=bakes if bakes.bakes else None,
            bind=_Routes(self, rows),
        )
        try:
            model = self.memory.load(root)
        finally:
            remaining = self.grants.pop(name, None)
            if remaining is not None:
                _close(remaining)
        for _, sealed in self.memory.sealed():
            if self.deposit is not None:
                label = f"{self.construction}/{name}"
                self.deposit(grant.manifest, label, self._layout(name), sealed)
            else:
                os.close(sealed)
        return model

    def _layout(self, name: str) -> str:
        """What a component's host tier is deposited and granted under: its manifest, the
        construction (release, model class, components, custody) and its name. Known before the
        module exists; a held tier of other regions is a miss (`Memory.load`)."""
        return canonical.digest(
            {
                "manifest": self.checkpoints[name].id,
                "construction": self.execution.construction.document(),
                "component": name,
            }
        )

    def _granted(self, name: str) -> WeightGrant:
        """A component's grant, asked of TensorD once: the sealed host tier it holds for the
        layout, registered by the load; without one, varena seals one where TensorD admits it,
        else reads the checkpoint."""
        grant = self.grants.get(name)
        if grant is None:
            checkpoint = self.checkpoints[name]
            grant = WeightGrant(checkpoint.ck, name, checkpoint.id)
            if self.grant is not None:
                held = self.grant(grant.manifest, self._layout(name))
                admitted: dict[str, Any] = {"seal": held.seal} if ADMITS else {}
                grant = replace(
                    grant,
                    host_fd=held.fd,
                    pages=held.pages,
                    fingerprint=held.fingerprint,
                    **admitted,
                )
            self.grants[name] = grant
        return grant

    def keep(self, *, quiescent: bool = False) -> None:
        """Its models' GPU pages that landed since the last call, kept past this process:
        TensorD holds them for the next executor of the same layout. `quiescent` is the
        executor's fenced, drained end boundary; cached graph pins keep their addresses."""
        if self.keeper is None:
            return
        for name, model in self.models.items():
            fingerprint, pages = model.kept(quiescent=quiescent)
            if pages:
                self.keeper(self._layout(name), fingerprint, pages)

    def _release(self) -> None:
        """Grants no registration took."""
        for name in list(self.grants):
            _close(self.grants.pop(name))

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


def _close(grant: WeightGrant) -> None:
    held = [] if grant.host_fd is None else [grant.host_fd]
    for fd in held + [f for _, f in grant.pages]:
        os.close(fd)


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
    overlay). Its `lora_factors()` (identity, down, up, scale; None when it adds nothing now)
    are baked into the linear's bytes, and `_cozy_baked` is set while they hold it."""

    _cozy_baked: bool

    def lora_factors(self) -> tuple[object, Tensor, Tensor, float] | None: ...


class LoRALayer(Protocol):
    active_adapters: Sequence[str]
    scaling: Mapping[str, float]
    _cozy_baked: bool


@dataclass(slots=True)
class _Bake:
    """One LoRA'd linear: its stored base weight, its PEFT layer with each adapter's factors
    (stored keys and scale), and the model's own hooks."""

    weight: str
    encoding: str = ""
    layer: LoRALayer | None = None
    factors: list[tuple[str, str, float]] = field(default_factory=list)
    hooks: list[LoRASource] = field(default_factory=list)


class _Bakes:
    """A component's LoRA'd linears, baked into their weights' bytes as the pages arrive
    (`Declared.fill`), so a resident page carries its delta and an evicted one bakes again on
    its refault. Each weight's own seed makes every bake of it the same bytes."""

    def __init__(
        self,
        torch: ModuleType,
        root: Module,
        rows: Mapping[str, PlanRow],
        blocks: set[str],
        owner: Callable[[object], tuple[str, str] | None],
    ) -> None:
        self.torch, self.owner = torch, owner
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

    def prepare(self) -> tuple[object, ...]:
        """As a forward starts (`Fill.prepare`): each LoRA'd linear told whether its bytes
        carry its delta, and what baking now adds (the strategy, each linear's graph and hook
        factors); pages baked for other deltas refill."""
        baked, rows = self.strategy == "bake", []
        for bake in self.bakes.values():
            if bake.layer is not None:
                bake.layer._cozy_baked = baked
            for hook in bake.hooks:
                hook._cozy_baked = baked
            hooks = (hook.lora_factors() for hook in bake.hooks)
            rows.append((bake.layer is not None, *(g and (g[0], g[3]) for g in hooks)))
        return (self.strategy, *rows)

    def identity(self) -> str | None:
        """What `prepare()` just named, the same in every process (`Fill.identity`): the
        recipe, and each baked weight's stored name and encoding with each delta's content (a
        stored factor pair of this checkpoint, or an overlay checkpoint's component and module
        path), dtypes and scale. None when an overlay's checkpoint cannot be named."""
        torch = self.torch
        if self.strategy != "bake":
            return repr((self.strategy,))
        device = torch.cuda.current_device()
        held: list[object] = [lora_bake.recipe(device), torch.__version__]
        held.append(torch.cuda.get_device_capability(device))
        for stored, bake in sorted(self.bakes.items()):
            deltas: list[object] = [(a, b, repr(scale)) for a, b, scale in bake.factors]
            for hook in bake.hooks:
                got = hook.lora_factors()
                if got is None:
                    continue
                named = self.owner(got[0])
                if named is None:
                    return None
                deltas.append((*named, str(got[1].dtype), str(got[2].dtype), repr(float(got[3]))))
            held.append((stored, bake.encoding, deltas))
        return repr(held)

    def __call__(self, block: object, parts: Mapping[str, Mapping[str, Tensor]]) -> None:
        if self.strategy != "bake":
            return
        for stored, bake in self.bakes.items():
            weight = parts.get(stored)
            if weight is None or any(a not in parts or b not in parts for a, b, _ in bake.factors):
                continue
            downs = [parts[a]["value"] for a, _, _ in bake.factors]
            ups = [parts[b]["value"] * scale for _, b, scale in bake.factors]
            for hook in bake.hooks:
                got = hook.lora_factors()
                if got is not None:
                    downs.append(got[1])
                    ups.append(got[2] * got[3])
            if not downs:
                continue
            kind, torch = downs[0].dtype, self.torch
            lora_bake.bake(
                torch,
                dict(weight),
                torch.cat([d.to(kind) for d in downs]),
                torch.cat([u.to(kind) for u in ups], dim=1),
                seed=zlib.crc32(stored.encode()),
            )
            self.baked += 1


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
        #: (component, owner) -> its pin and the cache that holds it
        self.pins: dict[tuple[str, str], tuple[_Pin, Cache]] = {}
        for name, model in self.models.items():
            PLACED[model.module] = (self, name)

    def open_attempt(self) -> None:
        self.attempts += 1
        self.notes.clear()

    def plan(self, request_id: str, steps: Sequence[str | Collection[str]]) -> None:
        """The attempt's declared steps (`Context.plan`): varena evicts and restores by them."""
        self.memory.plan(request_id, list(steps))

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
        held = self.pins.get((component, owner))
        if held is not None:
            return held[0]
        model = self.models.get(component)
        if model is None:
            return None

        def give(_: int) -> int:
            self.pins.pop((component, owner), None)
            cache.close()
            return shed()

        cache = self.memory.cache(owner, give)
        try:
            hold = cache.hold(model)
        except Unbound:
            cache.close()
            return None
        token = _Pin()
        weakref.finalize(token, hold.close)
        cache.nbytes = model.home_bytes()
        self.pins[(component, owner)] = (token, cache)
        return token

    def unpin(self, component: str, owner: str, token: object) -> None:
        """Drop this owner's cache; outstanding tokens keep its event-ordered hold alive."""
        held = self.pins.get((component, owner))
        if held is not None and held[0] is token:
            self.pins.pop((component, owner))
            held[1].close()

    def cache(self, owner: str, shed: Shrink) -> Cache | None:
        """Device bytes `owner` keeps for speed, counted and shed on ask; None off a GPU."""
        return self.memory.cache(owner, shed) if self.memory.cuda else None

    def note(self, key: str, value: str) -> None:
        self.notes[key] = value

    def baked(self) -> int:
        return sum(bakes.baked for bakes in self.bakes.values())


class _Pin:
    """A pinned component's token: its weights stay bound while a reference lives."""

    __slots__ = ("__weakref__",)


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
