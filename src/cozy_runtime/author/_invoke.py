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
from typing import Any

import msgspec

from cozy_runtime.author._app import Registration
from cozy_runtime.author._assets import Asset, bind
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
from cozy_runtime.author._services import (
    MAX_OUTPUT_BYTES,
    AdjustmentRow,
    Adjustments,
    Attempt,
    Budget,
    Checkpoints,
    Egress,
    Outputs,
    Scratch,
    Secrets,
    Settings,
    Telemetry,
    TelemetryEvent,
)
from cozy_runtime.author._signature import Param, Surface
from cozy_runtime.author._walker import field_values, walk


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
    max_output_bytes: int = MAX_OUTPUT_BYTES
    progress: Callable[[TelemetryEvent], None] | None = None
    """The runtime's lossy progress sink (cr-007). Best-effort, never authoritative."""


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    """What the wire carries (worker-protocol/01). cr-003 pins the serialized shape; the
    facts this kernel is authoritative for are these."""

    result: Any
    outputs: tuple[Mapping[str, Any], ...]
    adjustments: tuple[AdjustmentRow, ...]
    telemetry: tuple[TelemetryEvent, ...]
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
    registration: Registration, wire: Mapping[str, object], inv: Invocation
) -> ResultEnvelope:
    """Execute one attempt end to end and return its envelope."""
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

    attempt = Attempt(inv.request_id, inv.spool, inv.max_output_bytes, sink=inv.progress)
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
        telemetry=tuple(attempt.events),
        features=features,
        overlay_digest=overlay.digest,
        capabilities=surface.capabilities,
        facts_digest=facts_digest,
    )


def attempt(
    registration: Registration, wire: Mapping[str, object], inv: Invocation
) -> tuple[ResultEnvelope | None, Outcome]:
    """The invocation BOUNDARY: one attempt in, one neutral terminal out, on every path.

    `invoke` raises typed refusals because author-facing code should; the boundary the
    supervisor speaks across cannot, because the terminal must still ship when the body
    fails (§2). What crosses is FACTS — terminal, origin, code — and never a retry command:
    RETRYABLE and FATAL are coordinator projections over these, never observations here.
    """
    try:
        return invoke(registration, wire, inv), SUCCEEDED
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return None, classify(exc)


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


_CARRIES_ASSETS: dict[object, bool] = {}


def declares_assets(tp: object) -> bool:
    """Whether a schema can carry an asset at all — a describe-time fact, so a handler
    with no asset fields never walks a value looking for one."""
    carries = _CARRIES_ASSETS.get(tp)
    if carries is None:
        carries = _CARRIES_ASSETS[tp] = any(
            isinstance(node.annotation, type) and issubclass(node.annotation, Asset)
            for node in walk(tp)
        )
    return carries


def _hydrate(payload: object, inv: Invocation) -> None:
    """Materialize EVERY present asset field into the immutable attempt spool."""
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
            return Scratch(attempt)
        case "checkpoints":
            return Checkpoints(attempt)
        case "budget":
            return Budget(attempt)
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
                f"{surface.name}.{name} is {type(value).__name__}, declared "
                f"{annotation.__name__}",
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


