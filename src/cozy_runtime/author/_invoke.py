"""The ONE invocation kernel: decode -> resolve -> preflight -> hydrate -> invoke ->
validate -> finalize.

Every door runs this. `run`, `job` and a socket-served attempt are user experiences over
this function, never second execution implementations (§8) — and the per-service fakes are
capability adapters over THIS kernel, never a parallel behavioral runtime (§1.6). Where the
supervisor/executor process boundary, the journal and the device leases go is cr-008a's;
what a call MEANS is here.

Package placement is deliberate: the kernel is pure surface semantics (no device, config,
environment, protocol or network), so it lives beside the types it enforces and
`cozy_runtime.internal` depends on `author`, never the reverse.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import msgspec

from cozy_runtime.author._app import Registration
from cozy_runtime.author._assets import Asset, Tree, bind, bind_tree
from cozy_runtime.author._context import AdapterRef, Context, Device
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
from cozy_runtime.author._model import Model
from cozy_runtime.author._observations import Observation
from cozy_runtime.author._services import (
    MAX_OUTPUT_BYTES,
    AdjustmentRow,
    Adjustments,
    Attempt,
    Budget,
    BudgetFacts,
    CheckpointReceipt,
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
    scratch: Path | None = None
    """The RUN's scratch tree — the same directory across attempts of one run, which is what
    makes `Scratch.checkpoint_dir(key=)` resumable rather than merely named (§2)."""
    checkpoints: Callable[[CheckpointSave], CheckpointReceipt] | None = None
    """The DURABLE checkpoint exchange. Distinct from the lossy progress lane on purpose:
    the lossy lane has no authority over a durable fact."""
    budget: BudgetFacts | None = None
    """Cost-governor facts from the typed config channel. Never read from environment."""
    max_output_bytes: int = MAX_OUTPUT_BYTES
    progress: Callable[[ProgressFrame | Observation], None] | None = None
    """The runtime's lossy LIVE sink (cr-007/cr-011): progress frames and retained
    observations alike. Best-effort, never authoritative."""


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
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


def invoke(
    registration: Registration, wire: Mapping[str, object], inv: Invocation, record: Attempt
) -> ResultEnvelope:
    """Execute one attempt end to end and return its envelope.

    `record` is built by the BOUNDARY below and passed in, so an attempt that dies before it
    produces an envelope still hands back its bounded observation ring — the failing
    attempt's log tail is the one triage most needs, and building the record inside here
    would drop it on exactly that path.
    """
    surface = surface_of(registration)
    inv.spool.mkdir(parents=True, exist_ok=True)

    overlay = resolve(
        surface.payload_type,
        wire,
        recipes=inv.recipes,
        clamps=inv.clamps,
        model_params=surface.model_params,
    )
    features = normalize(overlay.payload)

    facts, facts_digest = _preflight(surface, overlay, inv)

    # Hydration is EAGER and happens only now: a refused request downloads nothing (§1.3).
    _hydrate(overlay.payload, inv)

    attempt = record
    attempt.rows.extend(overlay.rows)
    ctx = Context(
        inv.request_id, inv.deadline, inv.boot_warmup, inv.device, inv.cancel, inv.adapters
    )
    kwargs = _inject(surface, attempt, ctx, overlay, facts, inv)

    result = _call(surface, kwargs)

    _validate_result(surface, result, attempt)
    attempt.closed = True
    return ResultEnvelope(
        result=result,
        outputs=tuple(a.row() for a in attempt.pending.values()),
        adjustments=tuple(attempt.rows),
        features=features,
        overlay_digest=overlay.digest,
        capabilities=surface.capabilities,
        facts_digest=facts_digest,
    )


def attempt(
    registration: Registration, wire: Mapping[str, object], inv: Invocation
) -> tuple[ResultEnvelope | None, Outcome, Attempt]:
    """The invocation BOUNDARY: one attempt in, one neutral terminal out, on every path.

    `invoke` raises typed refusals because author-facing code should; the boundary the
    supervisor speaks across cannot, because the terminal must still ship when the body
    fails (§2). What crosses is FACTS — terminal, origin, code — and never a retry command:
    RETRYABLE and FATAL are coordinator projections over these, never observations here.
    """
    record = Attempt(inv.request_id, inv.spool, inv.max_output_bytes, sink=inv.progress)
    try:
        return invoke(registration, wire, inv, record), SUCCEEDED, record
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return None, classify(exc), record


# ------------------------------------------------------------------------- preflight


def _preflight(surface: Surface, overlay: Overlay, inv: Invocation) -> tuple[object, str | None]:
    spec = surface.preflight
    if spec is None:
        return None, None
    args: list[object] = [overlay.payload]
    if spec.settings_type is not None:
        args.append(_settings_value(spec.settings_type, inv.settings, "preflight settings"))
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


def _hydrate(payload: object, inv: Invocation) -> None:
    """Materialize EVERY present asset field into the immutable attempt spool."""
    _materialize_trees(payload, inv)
    if not declares_assets(type(payload)):
        return
    inputs = inv.spool / "inputs"
    for index, asset in enumerate(assets_in(payload)):
        source = Path(asset.ref.removeprefix("file://"))
        if not source.is_file():
            raise InvalidRequest(
                f"asset {asset.ref!r} is not materializable by this runtime: the supervisor "
                "fetch path (allowlisted egress, SSRF fail-closed) is cr-012's",
                code="asset_unresolvable",
            )
        inputs.mkdir(parents=True, exist_ok=True)
        local = inputs / f"{index:03d}-{source.name}"
        shutil.copyfile(source, local)
        bind(asset, local=local, attempt=inv.request_id)


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
