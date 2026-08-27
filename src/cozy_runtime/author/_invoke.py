"""The ONE invocation kernel, in two halves that always run in this order:

    prepare   decode -> resolve -> normalize -> preflight     (nothing is admitted yet)
    invoke    hydrate -> inject -> call -> validate -> finalize

The split is the seam a PLAN is chosen across. A serving attempt prepares BEFORE the
worker admits it, so the normalized features and the preflight digests the planner
prices against are the exact ones the handler then runs on; a job prepares inside its own
accepted attempt, because a job has no ladder to price (cr-009). Both call these two
functions, in this order, and neither resolves a request twice.

Every door runs this. `run`, `job` and a socket-served attempt are user experiences over
these functions, never second execution implementations (§8) — and the per-service fakes are
capability adapters over THIS kernel, never a parallel behavioral runtime (§1.6). Where the
worker/executor process boundary, the journal and the device leases go is cr-008a's;
what a call MEANS is here.

Package placement is deliberate: the kernel is pure surface semantics (no device, config,
environment, protocol or network), so it lives beside the types it enforces and
`cozy_runtime.internal` depends on `author`, never the reverse.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import msgspec

from cozy_runtime.author._app import Registration
from cozy_runtime.author._assets import Asset, GrantedInput, Tree, bind, bind_tree
from cozy_runtime.author._context import AdapterRef, Context, Device
from cozy_runtime.author._decode import MediaDecoder
from cozy_runtime.author._defaults import Clamp, Overlay, Recipe, resolve
from cozy_runtime.author._demand import RequestFeatures, normalize
from cozy_runtime.author._describe import check
from cozy_runtime.author._errors import (
    SUCCEEDED,
    AuthorError,
    ConformanceError,
    InvalidRequest,
    Outcome,
    OutputError,
    classify,
)
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author._media import KIND_MEDIA, admits
from cozy_runtime.author._model import Model
from cozy_runtime.author._observations import Observation
from cozy_runtime.author._services import (
    MAX_OUTPUT_BYTES,
    AdjustmentRow,
    Adjustments,
    Attempt,
    Budget,
    BudgetFacts,
    CheckpointDeclaration,
    Checkpoints,
    CheckpointSave,
    Egress,
    Outputs,
    ProgressFrame,
    Scratch,
    Secrets,
    Settings,
    Telemetry,
)
from cozy_runtime.author._signature import Param, Surface
from cozy_runtime.author._walker import field_values, strip, unwrap_optional, walk


@dataclass(frozen=True, slots=True)
class Invocation:
    """Everything the RUNTIME supplies for one attempt. Author code sees none of it."""

    request_id: str
    spool: Path
    deadline: float
    boot_warmup: bool = False
    device: Device = field(default_factory=Device)
    cancel: Callable[[], bool] = field(default=lambda: False)
    models: Mapping[str, Model[Any]] = field(default_factory=dict)
    recipes: Sequence[Recipe] = ()
    clamps: Sequence[Clamp] = ()
    settings: object = None
    secrets: object = None
    adapters: tuple[AdapterRef, ...] = ()
    egress_broker: Callable[..., Any] | None = None
    trees: Mapping[str, tuple[Path, str]] = field(default_factory=dict)
    """ref -> (materialized root, digest) for every input tree THIS attempt was granted.

    A `Tree` field naming a ref absent from this mapping never hydrates and refuses typed:
    the grant IS the read capability, and there is no ambient store handle anywhere on the
    author surface for a job to reach around it (§2, jobs.md §1).
    """
    assets: Mapping[str, GrantedInput] = field(default_factory=dict)
    """input_id -> the verified, spooled input asset (cr-012). The worker fetched it
    under the grant, held it to its declared length through the one bounded reader, checked
    its content digest and sniffed its bytes, all BEFORE acceptance. An `Asset` field whose
    input_id is absent here never hydrates and refuses typed — same law as `trees`, one
    modality over: the grant IS the read capability, and a ref is never a location."""
    max_input_bytes: int = 64 << 20
    """The DEPLOYMENT's per-input ceiling, the floor under every field's own `AssetBound`.
    A field with no bound takes this one, so "unbounded" is not spellable."""
    scratch: Path | None = None
    """The RUN's scratch tree — the same directory across attempts of one run, which is what
    makes `Scratch.checkpoint_dir(key=)` resumable rather than merely named (§2)."""
    checkpoints: Callable[[CheckpointSave], CheckpointDeclaration] | None = None
    """The checkpoint DECLARATION exchange. Distinct from the lossy progress lane on
    purpose: the lossy lane has no authority over a journaled fact. It carries no bytes —
    see `Checkpoints` (#553a)."""
    budget: BudgetFacts | None = None
    """Cost-governor facts from the typed config channel. Never read from environment."""
    max_output_bytes: int = MAX_OUTPUT_BYTES
    progress: Callable[[ProgressFrame | Observation], None] | None = None
    """The runtime's lossy LIVE sink (cr-007/cr-011): progress frames and retained
    observations alike. Best-effort, never authoritative."""


@dataclass(frozen=True, slots=True)
class InvocationResult:
    """What the wire carries (worker-protocol/01). cr-003 pins the serialized shape; the
    facts this kernel is authoritative for are these."""

    result: Any
    outputs: tuple[Mapping[str, Any], ...]
    adjustments: tuple[AdjustmentRow, ...]
    features: RequestFeatures
    overlay_digest: str
    capabilities: frozenset[str]
    facts_digest: str | None = None


def surface_of(registration: Registration) -> Surface:
    """The checked surface. Build refusals are not skipped because a request arrived."""
    surface = registration.surface
    if not registration.checked:
        check(surface)
        registration.checked = True
    return surface


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """ONE request, resolved EXACTLY ONCE: decode -> resolve -> normalize -> preflight.

    The half of the kernel that runs before anything is admitted, hoisted so it can run
    before admission and its answers can be PRICED and JOURNALED rather than recomputed
    after the fact (codex audit, adopted 2026-08-25). The planner used to be handed an ad
    hoc scalar projection of the wire document while the real normalized features and the
    preflight facts were derived later, inside invocation — two derivations of one thing,
    and the one the plan was chosen against was the one nothing else ever saw again.

    Immutable, and it is what `invoke` runs. There is no second resolve.
    """

    surface: Surface
    overlay: Overlay
    features: RequestFeatures
    facts: object
    facts_digest: str | None

    def digests(self) -> dict[str, str]:
        """What a worker can hold, journal and price against — never the payload."""
        return {
            "overlay_digest": self.overlay.digest,
            "features_digest": self.features.digest,
            "facts_digest": self.facts_digest or "",
        }


def prepare(
    registration: Registration,
    wire: Mapping[str, object],
    *,
    recipes: Sequence[Recipe] = (),
    clamps: Sequence[Clamp] = (),
    settings: object = None,
) -> PreparedRequest:
    """The pre-admission half. Raises typed refusals; hydrates nothing; enters nothing."""
    surface = surface_of(registration)
    overlay = resolve(
        surface.payload_type,
        wire,
        recipes=recipes,
        clamps=clamps,
        model_params=surface.model_params,
    )
    facts, facts_digest = _preflight(surface, overlay, settings)
    return PreparedRequest(
        surface=surface,
        overlay=overlay,
        features=normalize(overlay.payload),
        facts=facts,
        facts_digest=facts_digest,
    )


def invoke(prepared: PreparedRequest, inv: Invocation, record: Attempt) -> InvocationResult:
    """Execute one PREPARED attempt end to end and return its envelope.

    `record` is built by the BOUNDARY below and passed in, so an attempt that dies before it
    produces an envelope still hands back its bounded observation ring — the failing
    attempt's log tail is the one triage most needs, and building the record inside here
    would drop it on exactly that path.
    """
    surface, overlay = prepared.surface, prepared.overlay
    inv.spool.mkdir(parents=True, exist_ok=True)

    # Hydration is EAGER and happens only now: a refused request downloads nothing (§1.3).
    _hydrate(overlay.payload, inv)

    attempt = record
    attempt.rows.extend(overlay.rows)
    ctx = Context(
        inv.request_id, inv.deadline, inv.boot_warmup, inv.device, inv.cancel, inv.adapters
    )
    kwargs = _inject(surface, attempt, ctx, overlay, prepared.facts, inv)

    result = _call(surface, kwargs)

    _validate_result(surface, result, attempt)
    attempt.closed = True
    return InvocationResult(
        result=result,
        outputs=tuple(a.row() for a in attempt.pending.values()),
        adjustments=tuple(attempt.rows),
        features=prepared.features,
        overlay_digest=overlay.digest,
        capabilities=surface.capabilities,
        facts_digest=prepared.facts_digest,
    )


def run_prepared(
    prepared: PreparedRequest, inv: Invocation
) -> tuple[InvocationResult | None, Outcome, Attempt]:
    """The invocation BOUNDARY: one PREPARED attempt in, one neutral terminal out, always.

    `invoke` raises typed refusals because author-facing code should; the boundary the
    worker speaks across cannot, because the terminal must still ship when the body
    fails (§2). What crosses is FACTS — terminal, origin, code — and never a retry command:
    RETRYABLE and FATAL are coordinator projections over these, never observations here.
    """
    record = Attempt(inv.request_id, inv.spool, inv.max_output_bytes, sink=inv.progress)
    try:
        return invoke(prepared, inv, record), SUCCEEDED, record
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return None, classify(exc), record


def attempt(
    registration: Registration, wire: Mapping[str, object], inv: Invocation
) -> tuple[InvocationResult | None, Outcome, Attempt]:
    """PREPARE then RUN, for the doors that do both at once — a job, and the warm pass.

    A serving attempt prepares BEFORE the worker admits it, which is the whole point of
    the split; a job is accept-then-prepare by design (cr-009) and has nothing to price a
    ladder against. Both call the same two functions, in the same order.
    """
    try:
        prepared = prepare(
            registration, wire, recipes=inv.recipes, clamps=inv.clamps, settings=inv.settings
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return (
            None,
            classify(exc),
            Attempt(inv.request_id, inv.spool, inv.max_output_bytes, sink=inv.progress),
        )
    return run_prepared(prepared, inv)


# ------------------------------------------------------------------------- preflight


def _preflight(surface: Surface, overlay: Overlay, settings: object) -> tuple[object, str | None]:
    spec = surface.preflight
    if spec is None:
        return None, None
    args: list[object] = [overlay.payload]
    if spec.settings_type is not None:
        args.append(_settings_value(spec.settings_type, settings, "preflight settings"))
    try:
        facts = spec.fn(*args)
    except AuthorError:
        raise
    except Exception as exc:
        raise InvalidRequest(
            f"preflight {spec.identity} refused: {exc}", code="preflight_failed"
        ) from exc
    if type(facts) is not spec.result_type:
        raise ConformanceError(
            f"preflight {spec.identity} returned {type(facts).__name__}, declared "
            f"{getattr(spec.result_type, '__name__', spec.result_type)}",
            code="preflight_result",
        )
    canonical = msgspec.json.encode(facts)
    return facts, "blake2b:" + hashlib.blake2b(canonical, digest_size=16).hexdigest()


_CARRIES: dict[tuple[object, str], bool] = {}


def _reaches(annotation: object, kind: type, depth: int = 0) -> bool:
    """Whether `kind` is reachable from ONE annotation, THROUGH containers and unions.

    The type walker emits a node per FIELD and descends struct-bearing members, so a
    `list[ImageAsset]` field is emitted as a node whose annotation is the LIST — and asking
    `issubclass(list[ImageAsset], Asset)` is false. That made a list of assets declare no
    assets, so `_hydrate` was skipped and every element arrived unhydrated: `read_bytes()`
    then refused with `asset_bytes_unavailable`, which reads like a preflight violation and
    is nothing of the kind. Found live by ev-003's judge endpoint (decisions #303), worked
    around there by wrapping each asset in a one-asset struct.

    The SILENT half was the defect. It is fixed rather than refused, because a
    `list[ImageAsset]` is an ordinary declaration and `assets_in` already collects one
    correctly at the VALUE level — only this predicate could not see it.
    """
    if depth > 8:  # a bound, not a cycle guard: annotations are trees, and shallow ones
        return False
    base, _ = strip(annotation)
    base, _ = unwrap_optional(base)
    if isinstance(base, type) and issubclass(base, kind):
        return True
    return any(_reaches(arg, kind, depth + 1) for arg in get_args(base))


def _declares(tp: object, kind: type) -> bool:
    key = (tp, kind.__name__)
    carries = _CARRIES.get(key)
    if carries is None:
        carries = _CARRIES[key] = any(_reaches(node.annotation, kind) for node in walk(tp))
    return carries


def declares_assets(tp: object) -> bool:
    """Whether a schema can carry an asset at all — a describe-time fact, so a handler
    with no asset fields never walks a value looking for one."""
    return _declares(tp, Asset)


def declares_trees(tp: object) -> bool:
    """The same fact for input TREES — a job's typed model/dataset/checkpoint inputs."""
    return _declares(tp, Tree)


def trees_in(value: object) -> list[Tree]:
    """Every input tree reachable from a payload, in declaration order."""
    found: list[Tree] = []
    if isinstance(value, Tree):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [t for item in value for t in trees_in(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in trees_in(item)]
    for member in field_values(value).values():
        found += trees_in(member)
    return found


def _materialize_trees(payload: object, inv: Invocation) -> None:
    """Bind every declared input tree to the root the GRANT materialized for it (§2).

    An ungranted ref is a typed refusal here rather than a `FileNotFoundError` deep in the
    body: a job that names a read it was never granted must not reach the filesystem at all.
    """
    if not declares_trees(type(payload)):
        return
    for tree in trees_in(payload):
        granted = inv.trees.get(tree.ref)
        if granted is None:
            raise InvalidRequest(
                f"input tree {tree.ref!r} is not granted to this attempt: this run "
                f"materialized {sorted(inv.trees) or 'no trees'}. A tree is a digest-verified "
                "READ CAPABILITY minted by the coordinator, never a path a job may name",
                code="tree_ungranted",
            )
        root, digest = granted
        if not root.is_dir():
            raise InvalidRequest(
                f"input tree {tree.ref!r} was granted at a location that is not a directory: "
                "the coordinator minted this address, so this runtime substitutes none",
                code="tree_unmaterialized",
            )
        bind_tree(tree, root=root, digest=digest, attempt=inv.request_id)


def asset_inputs(payload: object) -> list[tuple[str, int, Asset]]:
    """Every input asset with its STABLE IDENTITY: `(input_id, order, asset)`.

    `input_id` is the payload FIELD PATH with the list index included, which is
    worker-protocol/01's rule stated in code: inputs are identified, ORDERED subjects and
    never a digest-sorted anonymous list. `order` is the position within the containing
    list — 0 for a scalar field — because for H3's mixed reference sets the ORDER IS MODEL
    INPUT, not a presentation fact, and two assets with equal bytes are two subjects.

    Deliberately a second walk beside `assets_in`: that one answers "is this handle mine",
    which is a question about identity and must collect a repeat, while this one answers
    "which grant entry fills this field", which is a question about position.
    """
    found: list[tuple[str, int, Asset]] = []

    def walk_value(value: object, path: str, order: int, seen: set[int]) -> None:
        if isinstance(value, Asset):
            found.append((path, order, value))
            return
        if isinstance(value, Tree) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk_value(item, f"{path}.{index}" if path else str(index), index, seen)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk_value(item, f"{path}.{key}" if path else str(key), 0, seen)
            return
        for name, member in field_values(value).items():
            walk_value(member, f"{path}.{name}" if path else name, 0, seen)

    walk_value(payload, "", 0, set())
    return found


def asset_bounds(payload_type: object) -> dict[str, AssetBound]:
    """field path (no index) -> the `AssetBound` annotating it. Describe-time knowledge."""
    out: dict[str, AssetBound] = {}
    for node in walk(payload_type):
        if not _reaches(node.annotation, Asset):
            continue
        bound = node.marker(AssetBound)
        if isinstance(bound, AssetBound):
            out[node.path] = bound
    return out


def _unindexed(input_id: str) -> str:
    """`refs.2.clip` -> `refs.clip`: a bound annotates the FIELD, not an element."""
    return ".".join(part for part in input_id.split(".") if not part.isdigit())


def _check_input(asset: Asset, granted: GrantedInput, bound: AssetBound | None, cap: int) -> None:
    """The typed gate every input passes before the entrypoint exists to it.

    MEDIA-TYPE HONESTY is the first check and it is about the BYTES: `granted.media_type`
    is what the worker sniffed, not what the caller declared, so a `.png` that is a
    zip refuses here rather than inside a decoder the endpoint did not agree to run.
    """
    kind = type(asset).kind
    if not admits(kind, granted.media_type):
        raise InvalidRequest(
            f"input {granted.input_id!r} is {granted.media_type or 'of no recognised type'} "
            f"and the field is a {type(asset).__name__}, which admits "
            f"{list(KIND_MEDIA.get(kind, ())) or 'any recognised type'}. The type comes "
            "from the BYTES, not from the declaration: a mislabelled upload refuses at the "
            "field rather than inside a decoder this endpoint never agreed to run",
            code="asset_media_type",
        )
    if bound is not None and bound.media_types and granted.media_type not in bound.media_types:
        raise InvalidRequest(
            f"input {granted.input_id!r} is {granted.media_type!r} and this field declares "
            f"{list(bound.media_types)}",
            code="asset_media_type",
        )
    limit = min(bound.max_bytes, cap) if bound is not None and bound.max_bytes else cap
    if limit and granted.length > limit:
        raise InvalidRequest(
            f"input {granted.input_id!r} is {granted.length} B and this field admits "
            f"{limit} B. A size bound is a TYPED property of the field, so it refuses "
            "before the handler is entered and not inside it",
            code="asset_too_large",
        )


def _hydrate(payload: object, inv: Invocation) -> None:
    """Bind every declared input asset to the bytes the GRANT already materialized (§1.3).

    Symmetric with `_materialize_trees` and for the same reason: an ungranted ref is a
    typed refusal here rather than a `FileNotFoundError` deep in the body. What changed at
    cr-012 is where the bytes come from — this used to read the ref as a LOCAL PATH and
    copy it, which made every asset ref a filesystem read of the caller's choosing. Now
    the worker fetches, verifies and spools before acceptance, and the only thing this
    side does is match an identified input against the FIELD it fills, which is the half
    only this side knows.
    """
    _materialize_trees(payload, inv)
    if not declares_assets(type(payload)):
        return
    bounds = asset_bounds(type(payload))
    for input_id, _order, asset in asset_inputs(payload):
        granted = inv.assets.get(input_id)
        if granted is None:
            raise InvalidRequest(
                f"input asset {input_id!r} (ref {asset.ref!r}) is not granted to this "
                f"attempt: this run materialized {sorted(inv.assets) or 'no assets'}. An "
                "asset is a digest-verified READ CAPABILITY minted by the coordinator, "
                "never a location a request may name",
                code="asset_ungranted",
            )
        bound = bounds.get(_unindexed(input_id))
        _check_input(asset, granted, bound, inv.max_input_bytes)
        bind(
            asset,
            local=granted.local,
            attempt=inv.request_id,
            media_type=granted.media_type,
            digest=granted.digest,
            length=granted.length,
            max_decoded_bytes=(bound.max_decoded_bytes or 0) if bound is not None else 0,
            input_id=input_id,
        )


# ------------------------------------------------------------------------- injection


def _inject(
    surface: Surface,
    attempt: Attempt,
    ctx: Context,
    overlay: Overlay,
    facts: object,
    inv: Invocation,
) -> dict[str, object]:
    """Build the call kwargs. A service absent from the signature is never constructed —
    which is precisely why a handler without `Outputs` structurally cannot save."""
    kwargs: dict[str, object] = {}
    for param in surface.params:
        if param.role == "context":
            kwargs[param.name] = ctx
        elif param.role == "payload":
            kwargs[param.name] = overlay.payload
        elif param.role == "preflight":
            kwargs[param.name] = facts
        elif param.role == "model":
            kwargs[param.name] = _model(param, inv)
        else:
            kwargs[param.name] = _service(param, attempt, ctx, inv)
    return kwargs


def _model(param: Param, inv: Invocation) -> object:
    model = inv.models.get(param.name)
    if model is None:
        raise InvalidRequest(
            f"no model bound for parameter {param.name!r}: bindings state selection "
            "(endpoint.toml / the deployment binding), code states capability",
            code="unbound_model",
            fields=[param.name],
        )
    return model


def _service(param: Param, attempt: Attempt, ctx: Context, inv: Invocation) -> object:
    match param.capability:
        case "save":
            return Outputs(attempt)
        case "telemetry":
            return Telemetry(attempt, ctx)
        case "adjust":
            return Adjustments(attempt)
        case "egress":
            return Egress(attempt, inv.egress_broker)
        case "settings":
            return Settings(_settings_value(param.arg, inv.settings, "settings"))
        case "secrets":
            return Secrets(_settings_value(param.arg, inv.secrets, "secrets"))
        case "scratch":
            return Scratch(attempt, inv.scratch)
        case "checkpoints":
            return Checkpoints(attempt, inv.checkpoints)
        case "budget":
            return Budget(attempt, inv.budget)
        case "media_decode":
            return MediaDecoder(
                attempt,
                active=lambda: any(model._cozy_active is not None for model in inv.models.values()),
            )
    raise ConformanceError(f"no injector for capability {param.capability!r}")


def _settings_value(schema: object, value: object, what: str) -> Any:
    """Deployment-owned values, validated against the AUTHOR-declared strict schema."""
    if isinstance(schema, type) and isinstance(value, schema):
        return value
    try:
        return msgspec.convert(value, type=schema)
    except (msgspec.ValidationError, TypeError) as exc:
        raise InvalidRequest(
            f"{what} do not validate against the declared schema "
            f"{getattr(schema, '__name__', schema)}: {exc}",
            code="settings_invalid",
        ) from exc


# ------------------------------------------------------------------------- execution


def _call(surface: Surface, kwargs: Mapping[str, object]) -> object:
    """`def` runs right here on the calling (executor) thread; `async def` runs on a
    PRIVATE event loop created for THIS attempt and closed with it.

    `await` interleaves I/O within one attempt and never admits a second — this function
    is synchronous, so law 8's one-active-GPU-attempt rule is unchanged by the spelling.
    """
    if not surface.is_async:
        return surface.fn(**kwargs)
    loop = asyncio.new_event_loop()
    try:
        coroutine = surface.fn(**kwargs)
        assert asyncio.iscoroutine(coroutine)
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


# ------------------------------------------------------------------------- result


def assets_in(value: object, _seen: set[int] | None = None) -> list[Asset]:
    """Every asset reachable from a value, in declaration order.

    Assets are collected EVERY time they appear — the cycle guard covers containers only,
    because "the same handle returned twice" is exactly what step 1 must catch.
    """
    seen = _seen if _seen is not None else set()
    if isinstance(value, Asset):
        return [value]
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        return [a for item in value for a in assets_in(item, seen)]
    if isinstance(value, dict):
        return [a for item in value.values() for a in assets_in(item, seen)]
    found: list[Asset] = []
    for member in field_values(value).values():
        found += assets_in(member, seen)
    return found


def _validate_result(surface: Surface, result: object, attempt: Attempt) -> None:
    """§3.4 step 1: the actual return value validates against the EXACT annotated result
    schema, and every pending handle reachable from it belongs to THIS attempt."""
    if type(result) is not surface.result_type:
        raise OutputError(
            f"{surface.name} returned {type(result).__name__}, but its annotated result "
            f"schema is {getattr(surface.result_type, '__name__', surface.result_type)}",
            code="result_schema",
        )
    declared = {n.name: n.annotation for n in walk(surface.result_type) if "." not in n.path}
    for name, value in field_values(result).items():
        annotation = declared.get(name, object)
        if isinstance(annotation, type) and not isinstance(value, annotation):
            raise OutputError(
                f"{surface.name}.{name} is {type(value).__name__}, declared {annotation.__name__}",
                code="result_schema",
                fields=[name],
            )

    returned: list[str] = []
    for asset in assets_in(result) if declares_assets(surface.result_type) else ():
        if asset._attempt != attempt.request_id or asset.ref not in attempt.pending:
            raise OutputError(
                f"output handle {asset.ref!r} does not belong to attempt "
                f"{attempt.request_id!r}: a foreign or escaped handle is a typed failure, "
                "never a silent drop",
                code="foreign_handle",
            )
        if asset.ref in returned:
            raise OutputError(
                f"output handle {asset.ref!r} is returned twice", code="duplicate_handle"
            )
        returned.append(asset.ref)

    unreturned = sorted(set(attempt.pending) - set(returned))
    if unreturned:
        raise OutputError(
            f"saved but never returned: {', '.join(unreturned)} — an unreturned pending "
            "output is a typed failure, never a silent drop",
            code="unreturned_handle",
        )
