"""The ONE invocation kernel, in two halves that always run in this order:

    prepare   decode -> resolve -> normalize -> preflight     (nothing is admitted yet)
    invoke    hydrate -> inject -> call -> validate -> finalize

The split is the seam a PLAN is chosen across. An attempt prepares BEFORE the worker
admits it, so the normalized features and the preflight digests the planner prices
against are the exact ones the handler then runs on, and no request resolves twice.

Every door runs this. `run` and a socket-served attempt are user experiences over
these functions, never second execution implementations (§8) — and the per-service fakes are
capability adapters over THIS kernel, never a parallel behavioral runtime (§1.6). Where the
worker/executor process boundary, process records and device leases go is cr-008a's;
what a call MEANS is here.

Package placement is deliberate: the kernel is pure surface semantics (no device, config,
environment, protocol or network), so it lives beside the types it enforces and
`cozy_runtime.internal` depends on `author`, never the reverse.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

import msgspec
from tensorfs.executor import Answer, Publish

from cozy_runtime.author._activity import asides
from cozy_runtime.author._app import Registration
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._calls import _Broker, _current
from cozy_runtime.author._context import CURRENT, Context, Device
from cozy_runtime.author._decode import (
    AudioProbe,
    VideoProbe,
    image_decoded_bytes,
)
from cozy_runtime.author._defaults import Overlay, resolve
from cozy_runtime.author._demand import RequestFeatures, normalize
from cozy_runtime.author._describe import check
from cozy_runtime.author._errors import (
    SUCCEEDED,
    AuthorError,
    CapabilityError,
    ConformanceError,
    InvalidRequest,
    Outcome,
    OutputError,
    classify,
    unwind,
)
from cozy_runtime.author._files import (
    File,
    Files,
    GrantedInput,
    ImageFile,
    InputMetadata,
    Tree,
    asset_inputs,
    bind,
    bind_tree,
)
from cozy_runtime.author._markers import FileBound, ImagePreparation
from cozy_runtime.author._media import KIND_MEDIA, admits, slot_kind
from cozy_runtime.author._model import Model
from cozy_runtime.author._observations import Observation, Track
from cozy_runtime.author._preview import PreviewFrame, Previews
from cozy_runtime.author._recipes import DeclaredRecipe
from cozy_runtime.author._services import (
    MAX_OUTPUT_BYTES,
    AdjustmentRow,
    Adjustments,
    Attempt,
    Outputs,
    ProgressFrame,
    Telemetry,
)
from cozy_runtime.author._signature import Param, Surface
from cozy_runtime.author._walker import field_values, strip, unwrap_optional, walk
from cozy_runtime.author._weights import ReadBytes, ReadValues, Weights
from cozy_runtime.canonical_json import JsonValue

if TYPE_CHECKING:
    from cozy_runtime.weights_store import Checkpoint, Derivation, Output


@dataclass(frozen=True, slots=True)
class Invocation:
    """Everything the RUNTIME supplies for one attempt. Author code sees none of it."""

    request_id: str
    spool: Path
    deadline: float
    device: Device = field(default_factory=Device)
    cancel: Callable[[], bool] = field(default=lambda: False)
    models: Mapping[str, Model[object]] = field(default_factory=dict)
    #: The package's declared recipes, which its `quantize` runs.
    recipes: tuple[DeclaredRecipe, ...] = ()
    trees: Mapping[str, tuple[Path, str]] = field(default_factory=dict)
    """ref -> (materialized root, digest) for every input tree THIS attempt was granted.

    A `Tree` field naming a ref absent from this mapping never hydrates and refuses typed:
    the grant IS the read capability, and there is no ambient store handle anywhere on the
    author surface to reach around it (§2).
    """
    assets: Mapping[str, GrantedInput] = field(default_factory=dict)
    """input_id -> the verified, spooled input file (cr-012). The worker fetched it
    under the grant, held it to its declared length through the one bounded reader, checked
    its content digest and sniffed its bytes, all BEFORE acceptance. An `File` field whose
    input_id is absent here never hydrates and refuses typed — same law as `trees`, one
    modality over: the grant IS the read capability, and a ref is never a location."""
    max_input_bytes: int | None = None
    """An explicit deployment ceiling narrows every field's own `FileBound`. None honors each
    installed field's declared bound; a field that declares none has no size limit."""
    sources: Mapping[str, Weights[Model[object]]] = field(default_factory=dict)
    """`Weights[M]` parameter -> its raw checkpoint, read through this attempt's source."""
    weights_bytes: ReadBytes | None = None
    weights_values: ReadValues | None = None
    tensorfs_source: Callable[[str], Checkpoint] | None = None
    tensorfs_output: Callable[[str, Derivation], Output] | None = None
    tensorfs_adopt: Callable[[Output], ModelArtifact] | None = None
    max_output_bytes: int = MAX_OUTPUT_BYTES
    progress: Callable[[ProgressFrame | Observation], None] | None = None
    """The runtime's lossy LIVE sink (cr-007/cr-011): progress frames and retained
    observations alike. Best-effort, never authoritative."""
    calls: _Broker | None = None
    publish: Callable[[Publish], Answer] | None = None
    """`Outputs.publish`'s exchange with the machine, which journals the run's output log."""
    previewing: Callable[[], bool] | None = None
    """Whether a client watches this attempt's previews now (`preview`), and `preview` where
    each encoded one goes."""
    preview: Callable[[PreviewFrame], None] | None = None
    #: No worker settles this attempt's frames (the fakes), so each encodes at its save.
    settle_at_save: bool = False
    #: False on a group's other GPUs: a save records its asset and writes nothing (`Attempt`).
    writes: bool = True


@dataclass(frozen=True, slots=True)
class InvocationResult:
    """The invocation result facts supplied by the author executor."""

    result: object
    outputs: tuple[Mapping[str, JsonValue], ...]
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
    asset_bounds: Mapping[str, FileBound | None] = field(default_factory=dict)
    #: how this request's LoRAs are served (`LORA_MODE`): baked unless it asks per step
    lora: str = "bake"
    #: what preparing it waited for (its attention kernel's compile): its attempt's first stages
    stages: Mapping[str, Track] = field(default_factory=dict)
    #: in a group, the command it was prepared from: every other GPU prepares it again
    asked: object = None

    def digests(self) -> dict[str, str]:
        """What a worker can hold, record and price against — never the payload."""
        return {
            "overlay_digest": self.overlay.digest,
            "features_digest": self.features.digest,
            "facts_digest": self.facts_digest or "",
        }


def prepare(
    registration: Registration,
    wire: Mapping[str, object],
    *,
    input_metadata: Mapping[str, InputMetadata] = {},
) -> PreparedRequest:
    """The pre-admission half. Raises typed refusals; hydrates nothing; enters nothing."""
    surface = surface_of(registration)
    wire, lora = lora_mode(surface.payload_type, wire)
    overlay = resolve(surface.payload_type, wire, model_params=surface.model_params)
    asset_bounds = _prepare_assets(surface, overlay.payload, input_metadata)
    facts, facts_digest = _preflight(surface, overlay)
    return PreparedRequest(
        surface=surface,
        overlay=overlay,
        features=normalize(overlay.payload),
        facts=facts,
        facts_digest=facts_digest,
        asset_bounds=asset_bounds,
        lora=lora,
    )


#: The one request field every function takes beside its own payload: `"per_step"` computes
#: each LoRA's update every forward instead of baking it into the weights (owner, 2026-10-04).
#: A payload that declares a field of this name keeps it; any other value is an ignored field.
LORA_MODE = "lora_mode"


def lora_mode(payload_type: object, wire: Mapping[str, object]) -> tuple[Mapping[str, object], str]:
    """The request without its `lora_mode` field, and how its LoRAs are served."""
    declared = getattr(payload_type, "__struct_fields__", ())
    if wire.get(LORA_MODE) not in ("per_step", "bake") or LORA_MODE in declared:
        return wire, "bake"
    rest = dict(wire)
    return rest, str(rest.pop(LORA_MODE))


def _bind_tensorfs(
    ctx: Context, record: Attempt, inv: Invocation, surface: Surface, handles: ExitStack
) -> None:
    """Bind the attempt's weights sources and outputs, read and written in the store."""
    if inv.tensorfs_source is None and inv.tensorfs_output is None:
        return
    source_reader, output_opener = inv.tensorfs_source, inv.tensorfs_output
    receipt_adopter = inv.tensorfs_adopt
    outputs: dict[str, Output] = {}
    declared = {output.name for output in surface.weights_outputs}

    def current() -> None:
        record.check_open("Context weights capability")
        ctx.raise_if_cancelled()

    def source(weights: Weights[Model[object]]) -> Checkpoint:
        current()
        if source_reader is None or not any(
            weights is candidate for candidate in inv.sources.values()
        ):
            raise CapabilityError(
                "source is not an admitted Weights input", code="weights_source_ungranted"
            )
        return source_reader(weights.checkpoint_ref)

    def output(slot: str, derivation: Derivation) -> Output:
        current()
        if output_opener is None or slot not in declared:
            raise CapabilityError(
                "output slot is not interface-declared", code="weights_output_ungranted"
            )
        if slot in outputs:
            raise CapabilityError("output was already opened", code="weights_output_ungranted")
        outputs[slot] = output_opener(slot, derivation)
        handles.callback(outputs[slot].close)
        return outputs[slot]

    def adopt(opened: Output) -> ModelArtifact:
        current()
        if receipt_adopter is None or not any(opened is mine for mine in outputs.values()):
            raise CapabilityError("adoption is unavailable", code="weights_receipt_mismatch")
        return receipt_adopter(opened)

    ctx._tensorfs_source, ctx._tensorfs_output, ctx._tensorfs_adopt = source, output, adopt
    for weights in inv.sources.values():
        weights._bind(functools.partial(source, weights), inv.weights_bytes, inv.weights_values)
        handles.callback(weights._bind, None, None, None)


def invoke(prepared: PreparedRequest, inv: Invocation, record: Attempt) -> InvocationResult:
    """Execute one PREPARED attempt end to end and return its envelope.

    `record` is built by the BOUNDARY below and passed in, so an attempt that dies before it
    produces an envelope still hands back its bounded observation ring — the failing
    attempt's log tail is the one triage most needs, and building the record inside here
    would drop it on exactly that path.
    """
    surface, overlay = prepared.surface, prepared.overlay
    inv.spool.mkdir(parents=True, exist_ok=True)
    record.rows.extend(overlay.rows)
    models = tuple(inv.models.values())
    record.components_active = lambda: any(model._cozy_active is not None for model in models)

    # Hydration is EAGER and happens only now: a refused request downloads nothing (§1.3).
    _hydrate(overlay.payload, inv, prepared.asset_bounds, record)

    attempt = record
    attempt.outputs = surface.outputs
    ctx = Context(inv.request_id, inv.deadline, inv.device, inv.cancel)
    ctx._recipes = inv.recipes
    ctx._progress, ctx._called = Telemetry(record, ctx).loop_progress()
    if inv.previewing is not None and inv.preview is not None:
        ctx._preview = Previews(
            inv.previewing, inv.preview, lambda why: record.emit("log", why, "warning")
        )
    handles = ExitStack()
    _bind_tensorfs(ctx, record, inv, surface, handles)
    kwargs = _inject(surface, attempt, ctx, overlay, prepared.facts, inv)
    for model in inv.models.values():
        serve = getattr(model._cozy_residency, "serve_lora", None)
        if serve is not None:
            serve(prepared.lora)

    attempt.entered = True
    if inv.calls is not None:
        inv.calls.bind(ctx, Telemetry(record, ctx), record.media, record)
    token, context = _current.set(inv.calls), CURRENT.set(ctx)
    try:
        result = _call(surface, kwargs)
        if inv.calls is not None:
            inv.calls.finish()
    finally:
        try:
            if inv.calls is not None:
                inv.calls.close()
        finally:
            CURRENT.reset(context)
            _current.reset(token)
            handles.__exit__(*sys.exc_info())

    _validate_result(
        surface, result, attempt, grant_token=inv.calls.grant_token if inv.calls else None
    )
    attempt.closed = True
    returned_handles: list[File | Tree] = [*assets_in(result), *trees_in(result)]
    return InvocationResult(
        result=result,
        outputs=tuple(
            asset.row()
            for ref, asset in attempt.pending.items()
            if ref not in attempt.committed_files
        )
        + tuple(tree.row() for tree in attempt.pending_trees.values())
        + tuple(
            asset.row()
            for asset in returned_handles
            if inv.calls is not None and asset._grant_token is inv.calls.grant_token
        ),
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
    RETRYABLE and FATAL are RecordOwner projections over these, never observations here.
    """
    record = Attempt(
        inv.request_id,
        inv.spool,
        inv.max_output_bytes,
        sink=inv.progress,
        publish=inv.publish,
        ignored=prepared.overlay.ignored,
        settle_at_save=inv.settle_at_save,
        writes=inv.writes,
    )
    record.attribution.stages.update(prepared.stages)

    def aside(name: str, ms: float) -> None:
        record.attribution.stage(name, ms)
        record.emit("stage", name, round(ms, 3))

    try:
        with asides(aside):
            return invoke(prepared, inv, record), SUCCEEDED, record
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        outcome = classify(exc)
        unwind(exc)
        return None, outcome, record
    finally:
        if record._stall_watch is not None:
            record._stall_watch.stop()
        record.closed = True


# ------------------------------------------------------------------------- preflight


def _preflight(surface: Surface, overlay: Overlay) -> tuple[object, str | None]:
    spec = surface.preflight
    if spec is None:
        return None, None
    args: list[object] = [_payload_value(surface, overlay.payload)]
    if spec.assets_parameter is not None:
        args.append(
            Files._from_bound(row.asset for row in getattr(overlay.payload, spec.assets_parameter))
        )
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


def _payload_value(surface: Surface, payload: object) -> object:
    if surface.assets is None or surface.parameters is not None:
        return payload
    annotation = next(p.annotation for p in surface.params if p.role == "payload")
    cls = cast("type[msgspec.Struct]", annotation)  # an asset-bearing payload is a Struct
    return cls(**{f.name: getattr(payload, f.name) for f in msgspec.structs.fields(cls)})


def _prepare_assets(
    surface: Surface, payload: object, metadata: Mapping[str, InputMetadata]
) -> dict[str, FileBound | None]:
    binding = surface.assets
    if binding is None:
        return {}
    values = getattr(payload, binding.parameter)
    bounds: dict[str, FileBound | None] = {}
    counts: dict[str, int] = {}
    limits = dict(binding.counts)
    for position, value in enumerate(values):
        input_id = f"{binding.parameter}.{position}.asset"
        row = metadata.get(input_id)
        if row is None or row.input_id != input_id or value.asset.ref != row.digest:
            raise InvalidRequest(
                "Files occurrence has no matching input binding",
                code="asset_ungranted",
                fields=[input_id],
            )
        taken = slot_kind((cls.kind for cls, _ in binding.kinds), row.media_type) or "file"
        options = [(cls, bound) for cls, bound in binding.kinds if cls.kind == taken]
        if not options:
            raise InvalidRequest(
                f"{input_id} is {row.media_type or 'of no recognised type'}, and "
                f"{binding.parameter} takes {[cls.kind for cls, _ in binding.kinds]} files: "
                "convert it and run again",
                code="asset_media_type",
                fields=[input_id],
            )
        cls, bound = options[0]
        counts[cls.kind] = counts.get(cls.kind, 0) + 1
        if cls.kind in limits and counts[cls.kind] > limits[cls.kind]:
            raise InvalidRequest(
                f"{binding.parameter} has more than {limits[cls.kind]} {cls.kind} files, the "
                "most this function takes: send fewer",
                code="asset_count",
                fields=[binding.parameter],
            )
        asset = cls(
            value.asset.ref, media_type=row.media_type, size_bytes=row.length, digest=row.digest
        )
        _check_input(asset, row, bound, bound.max_bytes or 0 if bound else 0, position)
        asset._input_id, asset._position = input_id, position
        asset._label = value.label
        asset._fidelity = value.fidelity
        asset._preparation = binding.preparation(cls.kind)
        value.asset = asset
        if bound is not None and cls.kind in binding.undeclared:  # the machine's budget, not 2 GiB
            declares = bound.max_bytes is not None or bound.media_types
            bound = replace(bound, max_decoded_bytes=None) if declares else None
        bounds[input_id] = bound
    try:
        Files._from_bound(value.asset for value in values)
    except ValueError as exc:
        raise InvalidRequest(str(exc), code="asset_labels", fields=[binding.parameter]) from exc
    return bounds


_CARRIES: dict[tuple[object, str], bool] = {}


def _reaches(annotation: object, kind: type, depth: int = 0) -> bool:
    """Whether `kind` is reachable from ONE annotation, THROUGH containers and unions.

    The type walker emits a node per FIELD and descends struct-bearing members, so a
    `list[ImageFile]` field is emitted as a node whose annotation is the LIST — and asking
    `issubclass(list[ImageFile], File)` is false. That made a list of files declare no
    files, so `_hydrate` was skipped and every element arrived unhydrated: `read_bytes()`
    then refused with `asset_bytes_unavailable`, which reads like a preflight violation and
    is nothing of the kind. Found live by ev-003's judge package (decisions #303), worked
    around there by wrapping each file in a one-asset struct.

    The SILENT half was the defect. It is fixed rather than refused, because a
    `list[ImageFile]` is an ordinary declaration and `assets_in` already collects one
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
    """Whether a schema can carry a file at all — a describe-time fact, so a handler
    with no file fields never walks a value looking for one."""
    return _declares(tp, File)


def declares_trees(tp: object) -> bool:
    """The same fact for input TREES: typed model, dataset and checkpoint inputs."""
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
    body: a function that names a read it was never granted must not reach the filesystem.
    """
    if not declares_trees(type(payload)):
        return
    for tree in trees_in(payload):
        granted = inv.trees.get(tree.ref)
        if granted is None:
            raise InvalidRequest(
                f"input tree {tree.ref!r} did not reach this run, which holds "
                f"{sorted(inv.trees) or 'no trees'}: send the request again, and if it recurs "
                "that is a bug in cozy, not in the request",
                code="tree_ungranted",
            )
        root, digest = granted
        if not root.is_dir():
            raise InvalidRequest(
                f"input tree {tree.ref!r} is not a directory on this machine: send the request "
                "again, and if it recurs that is a bug in cozy, not in the request",
                code="tree_unmaterialized",
            )
        bind_tree(tree, root=root, digest=digest, attempt=inv.request_id)


def _check_input(
    asset: File,
    granted: InputMetadata,
    bound: FileBound | None,
    cap: int | None,
    expected_order: int,
) -> None:
    """The typed gate every input passes before the entrypoint exists to it.

    Before hydration these are the binding's declarations; the same check runs
    again on the worker's byte-verified metadata before handler entry.
    """
    if granted.order != expected_order:
        raise InvalidRequest(
            f"input {granted.input_id!r} carries order {granted.order}, but the decoded "
            f"request places it at {expected_order}",
            code="asset_order",
            fields=[granted.input_id],
        )
    kind = type(asset).kind
    if not admits(kind, granted.media_type):
        raise InvalidRequest(
            f"{granted.input_id} is {granted.media_type or 'of no recognised type'}, and this "
            f"field takes a {kind} file ({', '.join(KIND_MEDIA.get(kind, ()))}): convert it and "
            "run again",
            code="asset_media_type",
            fields=[granted.input_id],
        )
    if bound is not None and bound.media_types and granted.media_type not in bound.media_types:
        raise InvalidRequest(
            f"{granted.input_id} is {granted.media_type}, and this field takes "
            f"{', '.join(bound.media_types)}: convert it and run again",
            code="asset_media_type",
            fields=[granted.input_id],
        )
    limits = [limit for limit in (bound.max_bytes if bound is not None else None, cap) if limit]
    if limits and granted.length > min(limits):
        raise InvalidRequest(
            f"{granted.input_id} is {granted.length} bytes, and this field takes at most "
            f"{min(limits)}: send a smaller file",
            code="asset_too_large",
            fields=[granted.input_id],
        )


def _hydrate(
    payload: object,
    inv: Invocation,
    assets_bounds: Mapping[str, FileBound | None] = {},
    record: Attempt | None = None,
) -> None:
    """Bind every declared input file to the bytes the GRANT already materialized (§1.3).

    Symmetric with `_materialize_trees` and for the same reason: an ungranted ref is a
    typed refusal here rather than a `FileNotFoundError` deep in the body. What changed at
    cr-012 is where the bytes come from — this used to read the ref as a LOCAL PATH and
    copy it, which made every file ref a filesystem read of the caller's choosing. Now
    the worker fetches, verifies and spools before acceptance, and the only thing this
    side does is match an identified input against the FIELD it fills, which is the half
    only this side knows.
    """
    _materialize_trees(payload, inv)
    if not declares_assets(type(payload)):
        return
    for item in asset_inputs(payload):
        granted = inv.assets.get(item.input_id)
        if granted is None:
            raise InvalidRequest(
                f"{item.input_id} (ref {item.asset.ref!r}) did not reach this run, which holds "
                f"{sorted(inv.assets) or 'no files'}: a file field takes a file you attach, not "
                "a path on the machine",
                code="asset_ungranted",
                fields=[item.input_id],
            )
        bound = assets_bounds.get(item.input_id, item.bound)
        if item.input_id in assets_bounds and (
            item.asset.digest,
            item.asset.media_type,
            item.asset.size_bytes,
        ) != (granted.digest, granted.media_type, granted.length):
            raise InvalidRequest(
                "Files metadata changed during hydration",
                code="asset_binding_changed",
                fields=[item.input_id],
            )
        _check_input(item.asset, granted, bound, inv.max_input_bytes, item.order)
        bind(
            item.asset,
            local=granted.local,
            source_state=granted.file_state,
            attempt=inv.request_id,
            media_type=granted.media_type,
            digest=granted.digest,
            length=granted.length,
            max_decoded_bytes=(bound.max_decoded_bytes or 0) if bound is not None else 0,
            input_id=item.input_id,
            order=item.order,
            read_guard=(lambda: record.check_open("File.read_bytes")) if record else None,
            media=record.media if record else None,
        )
        if item.input_id not in assets_bounds:
            item.asset._preparation = item.preparation
        if record is not None:
            prepare_duration(item.asset, record)


def prepare_duration(asset: File, record: Attempt) -> None:
    """Hold a hydrated input to its declared duration policy before the handler runs.

    Shorter than `min_seconds`, or longer than `max_seconds` under `excess="refuse"`,
    refuses naming the field. `excess="truncate"` ends every read at `max_seconds` and,
    when the source is longer, records a runtime-owned `adjusted` row and a warning.
    """
    preparation = asset._preparation
    if preparation is None or preparation.kind == "image" or preparation.kind != asset.kind:
        return
    assert not isinstance(preparation, ImagePreparation)
    probe = asset._reads().probe(asset)
    assert isinstance(probe, (VideoProbe, AudioProbe))
    source = probe.source_duration
    field_id = asset._input_id
    minimum, maximum = preparation.minimum, preparation.maximum
    if minimum is not None and source < minimum:
        raise InvalidRequest(
            f"{field_id} is {_seconds(source)} s of {asset.kind}, and this field needs at least "
            f"{preparation.min_seconds} s: give a longer one",
            code="media_too_short",
            fields=[field_id],
        )
    if maximum is None:
        return
    if preparation.excess == "refuse":
        if source > maximum:
            raise InvalidRequest(
                f"{field_id} is {_seconds(source)} s of {asset.kind}, and this field takes at "
                f"most {preparation.max_seconds} s: cut it and run again",
                code="media_too_long",
                fields=[field_id],
            )
        return
    asset._end = maximum
    if source > maximum:
        reason = f"{asset.kind} truncated to {preparation.max_seconds:g} s"
        record.rows.append(
            AdjustmentRow(
                "adjusted",
                field_id,
                _seconds(source),
                preparation.max_seconds,
                reason,
                source="policy",
            )
        )
        record.emit(
            "log",
            f"input {asset.kind} truncated",
            "warn",
            field=field_id,
            source_seconds=_seconds(source),
            applied_seconds=float(maximum),
        )


def _seconds(value: Fraction) -> float:
    return round(float(value), 3)


# ------------------------------------------------------------------------- injection


def _inject(
    surface: Surface,
    attempt: Attempt,
    ctx: Context,
    overlay: Overlay,
    facts: object,
    inv: Invocation,
    *,
    params: Sequence[Param] | None = None,
) -> dict[str, object]:
    """Build the call kwargs. A service absent from the signature is never constructed —
    which is precisely why a handler without `Outputs` structurally cannot save."""
    kwargs: dict[str, object] = {}
    for param in surface.params if params is None else params:
        if param.role == "context":
            kwargs[param.name] = ctx
        elif param.role == "payload":
            kwargs[param.name] = _payload_value(surface, overlay.payload)
        elif param.role == "assets":
            kwargs[param.name] = Files._from_bound(
                row.asset for row in getattr(overlay.payload, param.name)
            )
        elif param.role == "wire":
            kwargs[param.name] = getattr(overlay.payload, param.name)
        elif param.role == "preflight":
            kwargs[param.name] = facts
        elif param.role in ("model", "weights"):
            model = _model(param, inv, param.name in surface.model_on_demand)
            keyword = surface.parameters is not None
            artifact = getattr(overlay.payload, param.name, None) if keyword else None
            if artifact is not None and (
                not isinstance(artifact, ModelArtifact)
                or (
                    getattr(model, "_cozy_selection_ref", "")
                    or getattr(model, "checkpoint_ref", "")
                )
                != artifact.checkpoint
            ):
                raise InvalidRequest(
                    "bound model differs from the exact call artifact",
                    code="model_artifact_binding",
                    fields=[param.name],
                )
            kwargs[param.name] = model
        else:
            kwargs[param.name] = _service(param, attempt, ctx)
    return kwargs


def _model(param: Param, inv: Invocation, on_demand: bool) -> object:
    model = (inv.sources if param.role == "weights" else inv.models).get(param.name)
    if model is None and on_demand:
        return _Unbound(param.name)  # a request may leave it unused: refused only when read
    if model is None:
        raise InvalidRequest(
            f"this function needs a model for {param.name!r} and has no default: name one with "
            f"`model.{param.name}=<org/model>`",
            code="unbound_model",
            fields=[param.name],
        )
    return model


class _Unbound:
    """An on-demand model slot the run names no model for: reading it refuses the request."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attribute: str) -> object:
        raise InvalidRequest(
            f"this request needs a model for {self._name!r} and has no default: name one with "
            f"`model.{self._name}=<org/model>`",
            code="unbound_model",
            fields=[self._name],
        )


def _service(param: Param, attempt: Attempt, ctx: Context) -> object:
    match param.capability:
        case "save":
            return Outputs(attempt)
        case "telemetry":
            return Telemetry(attempt, ctx)
        case "adjust":
            return Adjustments(attempt)
    # `UNGRANTABLE` capabilities (egress, settings, secrets) refuse at build and never get here.
    raise ConformanceError(f"no injector for capability {param.capability!r}")


# ------------------------------------------------------------------------- execution


def _call(
    surface: Surface,
    kwargs: Mapping[str, object],
) -> object:
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
        if (broker := _current.get()) is not None:
            broker.closed = True
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


# ------------------------------------------------------------------------- result


def assets_in(value: object, _seen: set[int] | None = None) -> list[File]:
    """Every file reachable from a value, in declaration order.

    Files are collected EVERY time they appear — the cycle guard covers containers only,
    because "the same handle returned twice" is exactly what step 1 must catch.
    """
    seen = _seen if _seen is not None else set()
    if isinstance(value, File):
        return [value]
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        return [a for item in value for a in assets_in(item, seen)]
    if isinstance(value, dict):
        return [a for item in value.values() for a in assets_in(item, seen)]
    found: list[File] = []
    for member in field_values(value).values():
        found += assets_in(member, seen)
    return found


def _check_result_bounds(surface: Surface, result: object, attempt: Attempt) -> None:
    """Every returned file meets its result field's FileBound, whether this attempt
    saved it or forwarded a child's grant: the child's own bound proves nothing here."""
    for item in asset_inputs(result):
        asset, bound = item.asset, item.bound
        if bound is None:
            continue
        where = f"{surface.name}.{item.input_id}"
        if bound.media_types and asset.media_type not in bound.media_types:
            raise OutputError(
                f"{where} is {asset.media_type}, outside its declared media types "
                f"{list(bound.media_types)}",
                code="result_bound",
                fields=[item.input_id],
            )
        if bound.max_bytes is not None and asset.size_bytes > bound.max_bytes:
            raise OutputError(
                f"{where} is {asset.size_bytes} B, over its declared {bound.max_bytes}-byte bound",
                code="result_bound",
                fields=[item.input_id],
            )
        if bound.max_decoded_bytes is None:
            continue
        frame = attempt.frames.get(asset.ref)
        if frame is not None:
            decoded = frame.raw_bytes
        elif isinstance(asset, ImageFile) and asset._local is not None:
            try:
                decoded = image_decoded_bytes(asset._local)
            except Exception as exc:
                raise OutputError(
                    f"{where} image header cannot be read to check its decoded bound",
                    code="result_decoded_bound",
                    fields=[item.input_id],
                ) from exc
        else:
            # Encoded audio/video carry no cheap decoded-size probe; their producer's
            # decode path still enforces the field bound at consumption.
            continue
        if decoded > bound.max_decoded_bytes:
            raise OutputError(
                f"{where} decodes to {decoded} B, over its captured decoded bound of "
                f"{bound.max_decoded_bytes} B",
                code="result_decoded_bound",
                fields=[item.input_id],
            )


def _fills(value: object, annotation: type) -> bool:
    """A value of the declared type, or a file of its kind (a saved file in a `Latents` field)."""
    if isinstance(value, annotation):
        return True
    return (
        isinstance(value, File) and issubclass(annotation, File) and value.kind == annotation.kind
    )


def _validate_result(
    surface: Surface, result: object, attempt: Attempt, *, grant_token: object | None = None
) -> None:
    """§3.4 step 1: the actual return value validates against the EXACT annotated result
    schema, and every pending handle reachable from it belongs to THIS attempt."""
    if type(result) is not surface.result_type:
        raise OutputError(
            f"{surface.name} returned {type(result).__name__}, but its annotated result "
            f"schema is {getattr(surface.result_type, '__name__', surface.result_type)}",
            code="result_schema",
        )
    declared = {n.name: n for n in walk(surface.result_type) if "." not in n.path}
    for name, value in field_values(result).items():
        node = declared.get(name)
        if node is None:
            continue
        # The walker unwraps Optional into (annotation, optional); a None in an
        # optional field is the declared shape, not a violation.
        if value is None and node.optional:
            continue
        annotation = node.annotation
        if isinstance(annotation, type) and not _fills(value, annotation):
            raise OutputError(
                f"{surface.name}.{name} is {type(value).__name__}, declared {annotation.__name__}",
                code="result_schema",
                fields=[name],
            )

    returned: list[str] = []
    output_handles: list[File | Tree] = [*assets_in(result), *trees_in(result)]
    pending_handles = (set(attempt.pending) - set(attempt.committed_files)) | set(
        attempt.pending_trees
    )
    if sum(asset.size_bytes for asset in output_handles) > attempt.max_output_bytes:
        raise OutputError("returned files exceed the output byte budget", code="output_too_large")
    for asset in output_handles:
        received = grant_token is not None and asset._grant_token is grant_token and asset.hydrated
        if asset._attempt != attempt.request_id or (
            asset.ref not in pending_handles and not received
        ):
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

    _check_result_bounds(surface, result, attempt)

    # A published file was delivered as a product already; returning it is optional.
    unreturned = sorted(pending_handles - set(returned) - attempt.shown)
    if unreturned:
        raise OutputError(
            f"saved but never returned: {', '.join(unreturned)} — an unreturned pending "
            "output is a typed failure, never a silent drop",
            code="unreturned_handle",
        )
