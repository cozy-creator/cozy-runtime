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
worker/executor process boundary, process records and device leases go is cr-008a's;
what a call MEANS is here.

Package placement is deliberate: the kernel is pure surface semantics (no device, config,
environment, protocol or network), so it lives beside the types it enforces and
`cozy_runtime.internal` depends on `author`, never the reverse.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Coroutine, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

import msgspec

from cozy_runtime.author._app import Registration
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._assets import (
    Asset,
    Assets,
    GrantedInput,
    ImageAsset,
    InputMetadata,
    Tree,
    bind,
    bind_tree,
)
from cozy_runtime.author._calls import ChildCallError, _Broker, _current
from cozy_runtime.author._context import AdapterRef, Context, Device
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS, MediaDecoder, image_decoded_bytes
from cozy_runtime.author._defaults import Clamp, Overlay, Recipe, resolve
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
)
from cozy_runtime.author._executor_requests import Publish, Published
from cozy_runtime.author._markers import AssetBound
from cozy_runtime.author._media import KIND_MEDIA, admits
from cozy_runtime.author._model import Model
from cozy_runtime.author._model_reader import ReadValues, WeightsReader
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
from cozy_runtime.author.publication import upload_checkpoint

if TYPE_CHECKING:
    from tensorfs.derived import Derivation, DerivedTransaction, SourceCapability


@dataclass(frozen=True, slots=True)
class Invocation:
    """Everything the RUNTIME supplies for one attempt. Author code sees none of it."""

    request_id: str
    spool: Path
    deadline: float
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
    purpose: the lossy lane has no authority over a recorded fact. It carries no bytes —
    see `Checkpoints` (#553a)."""
    budget: BudgetFacts | None = None
    """Cost-governor facts from the typed config channel. Never read from environment."""
    weights_values: ReadValues | None = None
    tensorfs_source: Callable[[str], SourceCapability] | None = None
    tensorfs_output: Callable[[str, Derivation], DerivedTransaction] | None = None
    tensorfs_adopt: Callable[[Mapping[str, Any]], ModelArtifact] | None = None
    max_output_bytes: int = MAX_OUTPUT_BYTES
    progress: Callable[[ProgressFrame | Observation], None] | None = None
    """The runtime's lossy LIVE sink (cr-007/cr-011): progress frames and retained
    observations alike. Best-effort, never authoritative."""
    activity: Callable[[int, bool, bool, bool], None] | None = None
    calls: _Broker | None = None
    publish_to: str = ""
    """The repository the owner granted this root: each Model it returns is uploaded there."""
    publish: Callable[[Publish], Published] | None = None
    """`Outputs.publish`'s exchange with the worker, which journals the run's output log."""


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
    asset_bounds: Mapping[str, AssetBound | None] = field(default_factory=dict)

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
    recipes: Sequence[Recipe] = (),
    clamps: Sequence[Clamp] = (),
    settings: object = None,
    input_metadata: Mapping[str, InputMetadata] = {},
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
    asset_bounds = _prepare_assets(surface, overlay.payload, input_metadata)
    facts, facts_digest = _preflight(surface, overlay, settings)
    return PreparedRequest(
        surface=surface,
        overlay=overlay,
        features=normalize(overlay.payload),
        facts=facts,
        facts_digest=facts_digest,
        asset_bounds=asset_bounds,
    )


def _bind_tensorfs(
    ctx: Context, record: Attempt, inv: Invocation, surface: Surface, handles: ExitStack
) -> None:
    """Bind execution authority; TensorFS implements the source and output handles."""
    if inv.tensorfs_source is None and inv.tensorfs_output is None:
        return
    from tensorfs.derived import OutputCapability, SourceCapability

    source_reader = inv.tensorfs_source
    output_opener = inv.tensorfs_output
    receipt_adopter = inv.tensorfs_adopt
    outputs: dict[str, OutputCapability] = {}
    declared = {output.name for output in surface.weights_outputs}

    def current() -> None:
        record.check_open("Context TensorFS capability")
        ctx.raise_if_cancelled()

    def source(model: Model[object]) -> SourceCapability:
        current()
        if source_reader is None or not any(
            model is candidate for candidate in inv.models.values()
        ):
            raise CapabilityError(
                "source is not an admitted model input", code="weights_source_ungranted"
            )
        capability = source_reader(model.checkpoint_ref)
        handles.callback(capability.close)

        def inspect(components: Sequence[str], configs: Sequence[str]) -> Any:
            current()
            return capability.inspect(components=components, configs=configs)

        def read_part(component: str, key: str, role: str, offset: int, into: object) -> None:
            current()
            try:
                capability.read_part_into(component, key, role, offset, into)
            finally:
                current()

        def check() -> None:
            current()
            try:
                capability.check()
            finally:
                current()

        return SourceCapability(
            capability.manifest, capability.length, inspect, capability.close, read_part, check
        )

    def output(slot: str) -> OutputCapability:
        current()
        if output_opener is None or slot not in declared:
            raise CapabilityError(
                "output slot is not interface-declared", code="weights_output_ungranted"
            )
        if slot not in outputs:

            def opened(definition: Derivation) -> DerivedTransaction:
                current()
                transaction = output_opener(slot, definition)
                handles.callback(transaction.close)
                return transaction

            outputs[slot] = OutputCapability(opened)
        return outputs[slot]

    def adopt(receipt: Mapping[str, Any]) -> ModelArtifact:
        current()
        if receipt_adopter is None:
            raise CapabilityError(
                "native receipt adoption is unavailable", code="weights_receipt_mismatch"
            )
        return receipt_adopter(receipt)

    ctx._tensorfs_source, ctx._tensorfs_output, ctx._tensorfs_adopt = source, output, adopt


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
    _hydrate(overlay.payload, inv, prepared.asset_bounds, record)

    attempt = record
    attempt.rows.extend(overlay.rows)
    ctx = Context(inv.request_id, inv.deadline, inv.device, inv.cancel, inv.adapters)
    handles = ExitStack()
    _bind_tensorfs(ctx, record, inv, surface, handles)
    kwargs = _inject(surface, attempt, ctx, overlay, prepared.facts, inv)

    attempt.entered = True
    if inv.calls is not None:
        inv.calls.bind(ctx, Telemetry(record, ctx))
    token = _current.set(inv.calls)
    try:
        result = _call(surface, kwargs, inv.publish_to, inv.activity)
        if inv.calls is not None:
            inv.calls.finish()
    finally:
        if inv.calls is not None:
            inv.calls.close()
        _current.reset(token)
        handles.close()

    _validate_result(
        surface, result, attempt, grant_token=inv.calls.grant_token if inv.calls else None
    )
    attempt.closed = True
    returned_handles: list[Asset | Tree] = [*assets_in(result), *trees_in(result)]
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
    )
    try:
        return invoke(prepared, inv, record), SUCCEEDED, record
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        record.failed_at_call = isinstance(exc, ChildCallError)
        return None, classify(exc), record
    finally:
        record.closed = True


def attempt(
    registration: Registration, wire: Mapping[str, object], inv: Invocation
) -> tuple[InvocationResult | None, Outcome, Attempt]:
    """PREPARE then RUN, for the door that does both at once — a job.

    A serving attempt prepares BEFORE the worker admits it, which is the whole point of
    the split; a job is accept-then-prepare by design (cr-009) and has nothing to price a
    ladder against. Both call the same two functions, in the same order.
    """
    try:
        prepared = prepare(
            registration,
            wire,
            recipes=inv.recipes,
            clamps=inv.clamps,
            settings=inv.settings,
            input_metadata=inv.assets,
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
    args: list[object] = [_payload_value(surface, overlay.payload)]
    for role in spec.argument_order:
        if role == "assets":
            args.append(
                Assets._from_bound(
                    (row.asset for row in getattr(overlay.payload, str(spec.assets_parameter))),
                    decoded=bool(surface.assets_binding and surface.assets_binding.decoded),
                )
            )
        else:
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


def _payload_value(surface: Surface, payload: object) -> object:
    if surface.assets_binding is None or surface.invocable:
        return payload
    annotation = next(p.annotation for p in surface.params if p.role == "payload")
    cls: Any = annotation
    return cls(**{f.name: getattr(payload, f.name) for f in msgspec.structs.fields(cls)})


def _prepare_assets(
    surface: Surface, payload: object, metadata: Mapping[str, InputMetadata]
) -> dict[str, AssetBound | None]:
    binding = surface.assets_binding
    if binding is None:
        return {}
    values = getattr(payload, binding.parameter)
    bounds: dict[str, AssetBound | None] = {}
    counts: dict[str, int] = {}
    limits = dict(binding.counts)
    for position, value in enumerate(values):
        input_id = f"{binding.parameter}.{position}.asset"
        row = metadata.get(input_id)
        if row is None or row.input_id != input_id or value.asset.ref != row.digest:
            raise InvalidRequest(
                "Assets occurrence has no matching input binding",
                code="asset_ungranted",
                fields=[input_id],
            )
        options = [
            (cls, bound)
            for cls, bound in binding.kinds
            if cls.kind != "file" and admits(cls.kind, row.media_type)
        ]
        if not options:
            options = [(cls, bound) for cls, bound in binding.kinds if cls.kind == "file"]
        if len(options) != 1:
            raise InvalidRequest(
                "Assets occurrence has an unsupported media type",
                code="asset_media_type",
                fields=[input_id],
            )
        cls, bound = options[0]
        counts[cls.kind] = counts.get(cls.kind, 0) + 1
        if cls.kind in limits and counts[cls.kind] > limits[cls.kind]:
            raise InvalidRequest(
                "Assets media kind count exceeds its declared limit",
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
        if cls is ImageAsset:
            asset._image_preparation = binding.image_preparation
        value.asset = asset
        bounds[input_id] = bound
    try:
        Assets._from_bound(value.asset for value in values)
    except ValueError as exc:
        raise InvalidRequest(str(exc), code="asset_labels", fields=[binding.parameter]) from exc
    return bounds


_CARRIES: dict[tuple[object, str], bool] = {}


def _reaches(annotation: object, kind: type, depth: int = 0) -> bool:
    """Whether `kind` is reachable from ONE annotation, THROUGH containers and unions.

    The type walker emits a node per FIELD and descends struct-bearing members, so a
    `list[ImageAsset]` field is emitted as a node whose annotation is the LIST — and asking
    `issubclass(list[ImageAsset], Asset)` is false. That made a list of assets declare no
    assets, so `_hydrate` was skipped and every element arrived unhydrated: `read_bytes()`
    then refused with `asset_bytes_unavailable`, which reads like a preflight violation and
    is nothing of the kind. Found live by ev-003's judge package (decisions #303), worked
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
                "READ CAPABILITY minted by the RecordOwner, never a path a job may name",
                code="tree_ungranted",
            )
        root, digest = granted
        if not root.is_dir():
            raise InvalidRequest(
                f"input tree {tree.ref!r} was granted at a location that is not a directory: "
                "the RecordOwner minted this address, so this runtime substitutes none",
                code="tree_unmaterialized",
            )
        bind_tree(tree, root=root, digest=digest, attempt=inv.request_id)
        tree._member_token = inv.calls.grant_token if inv.calls is not None else None


@dataclass(frozen=True, slots=True)
class AssetInput:
    """One typed Asset occurrence and its lossless request address."""

    input_id: str
    order: int
    asset: Asset
    bound: AssetBound | None
    segments: tuple[str | int, ...]


def asset_inputs(payload: object) -> list[AssetInput]:
    """Every input asset with its identity, order, and concrete typed-field bound.

    `input_id` is the payload FIELD PATH with the list index included, which is
    worker-protocol/01's rule stated in code: inputs are identified, ORDERED subjects and
    never a digest-sorted anonymous list. `order` is the position within the containing
    list — 0 for a scalar field — because for H3's mixed reference sets the ORDER IS MODEL
    INPUT, not a presentation fact, and two assets with equal bytes are two subjects.

    Deliberately a second walk beside `assets_in`: that one answers "is this handle mine",
    which is a question about identity and must collect a repeat, while this one answers
    "which grant entry fills this field", which is a question about position.
    """
    found: list[AssetInput] = []

    def walk_value(
        value: object,
        segments: tuple[str | int, ...],
        order: int,
        seen: set[int],
        bound: AssetBound | None,
    ) -> None:
        if isinstance(value, Asset):
            bad = [part for part in segments if isinstance(part, str) and (not part or "." in part)]
            if bad:
                raise InvalidRequest(
                    "asset field path segments must be non-empty and contain no '.': "
                    + ", ".join(repr(part) for part in bad),
                    code="asset_input_path_unsupported",
                )
            input_id = ".".join(str(part) for part in segments)
            found.append(AssetInput(input_id, order, value, bound, segments))
            return
        if isinstance(value, Tree) or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk_value(item, (*segments, index), index, seen, bound)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk_value(item, (*segments, str(key)), order, seen, bound)
            return
        if isinstance(value, msgspec.Struct):
            for field_info in msgspec.structs.fields(type(value)):
                _base, markers = strip(field_info.type)
                declared = next((m for m in markers if isinstance(m, AssetBound)), None)
                name = field_info.encode_name
                walk_value(
                    getattr(value, field_info.name),
                    (*segments, name),
                    order,
                    seen,
                    declared if declared is not None else bound,
                )
            return
        if is_dataclass(value) and not isinstance(value, type):
            annotations = get_type_hints(type(value), include_extras=True)
            for name, member in field_values(value).items():
                _base, markers = strip(annotations.get(name, object))
                declared = next((m for m in markers if isinstance(m, AssetBound)), None)
                walk_value(
                    member,
                    (*segments, name),
                    order,
                    seen,
                    declared if declared is not None else bound,
                )

    walk_value(payload, (), 0, set(), None)
    counts: dict[str, int] = {}
    for item in found:
        counts[item.input_id] = counts.get(item.input_id, 0) + 1
    duplicates = sorted(input_id for input_id, count in counts.items() if count > 1)
    if duplicates:
        raise InvalidRequest(
            "asset field paths are ambiguous after wire-name encoding: "
            + ", ".join(repr(input_id) for input_id in duplicates),
            code="asset_input_id_ambiguous",
            fields=duplicates,
        )
    return found


def _check_input(
    asset: Asset,
    granted: InputMetadata,
    bound: AssetBound | None,
    cap: int,
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
            f"input {granted.input_id!r} is {granted.media_type or 'of no recognised type'} "
            f"and the field is a {type(asset).__name__}, which admits "
            f"{list(KIND_MEDIA.get(kind, ())) or 'any recognised type'}. Input binding "
            "metadata must satisfy the declared type; hydration also verifies its bytes",
            code="asset_media_type",
            fields=[granted.input_id],
        )
    if bound is not None and bound.media_types and granted.media_type not in bound.media_types:
        raise InvalidRequest(
            f"input {granted.input_id!r} is {granted.media_type!r} and this field declares "
            f"{list(bound.media_types)}",
            code="asset_media_type",
            fields=[granted.input_id],
        )
    limit = min(bound.max_bytes, cap) if bound is not None and bound.max_bytes else cap
    if limit and granted.length > limit:
        raise InvalidRequest(
            f"input {granted.input_id!r} is {granted.length} B and this field admits "
            f"{limit} B. A size bound is a TYPED property of the field, so it refuses "
            "before the handler is entered and not inside it",
            code="asset_too_large",
            fields=[granted.input_id],
        )


def _hydrate(
    payload: object,
    inv: Invocation,
    assets_bounds: Mapping[str, AssetBound | None] = {},
    record: Attempt | None = None,
) -> None:
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
    for item in asset_inputs(payload):
        granted = inv.assets.get(item.input_id)
        if granted is None:
            raise InvalidRequest(
                f"input asset {item.input_id!r} (ref {item.asset.ref!r}) is not granted to this "
                f"attempt: this run materialized {sorted(inv.assets) or 'no assets'}. An "
                "asset is a digest-verified READ CAPABILITY minted by the RecordOwner, "
                "never a location a request may name",
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
                "Assets metadata changed during hydration",
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
            max_decoded_bytes=(bound.max_decoded_bytes or 0)
            if bound is not None
            else (DEFAULT_DECODE_LIMITS.max_decoded_bytes if item.input_id in assets_bounds else 0),
            input_id=item.input_id,
            order=item.order,
            read_guard=(lambda: record.check_open("Asset.read_bytes")) if record else None,
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
    # Both the implicit decoded view and an explicitly requested service share one budget.
    decoder = (
        MediaDecoder(
            attempt,
            active=lambda: any(model._cozy_active is not None for model in inv.models.values()),
        )
        if "media_decode" in surface.capabilities
        else None
    )
    for param in surface.params:
        if param.role == "context":
            kwargs[param.name] = ctx
        elif param.role == "payload":
            kwargs[param.name] = _payload_value(surface, overlay.payload)
        elif param.role == "assets":
            binding = surface.assets_binding
            assert binding is not None
            kwargs[param.name] = Assets._from_bound(
                (row.asset for row in getattr(overlay.payload, param.name)),
                decoded=binding.decoded,
                project=decoder.value if decoder is not None else None,
                guard=lambda: attempt.check_open("Assets"),
            )
        elif param.role == "wire":
            kwargs[param.name] = getattr(overlay.payload, param.name)
        elif param.role == "preflight":
            kwargs[param.name] = facts
        elif param.role == "model":
            model = _model(param, inv)
            artifact = getattr(overlay.payload, param.name, None) if surface.invocable else None
            if artifact is not None and (
                not isinstance(artifact, ModelArtifact)
                or (
                    getattr(model, "_cozy_selection_ref", "")
                    or getattr(model, "checkpoint_ref", "")
                )
                != artifact.manifest.digest
            ):
                raise InvalidRequest(
                    "bound model differs from the exact call artifact",
                    code="model_artifact_binding",
                    fields=[param.name],
                )
            kwargs[param.name] = model
        elif param.capability == "media_decode":
            assert decoder is not None
            kwargs[param.name] = decoder
        else:
            kwargs[param.name] = _service(param, attempt, ctx, inv, surface)
    return kwargs


def _model(param: Param, inv: Invocation) -> object:
    model = inv.models.get(param.name)
    if model is None:
        raise InvalidRequest(
            f"no model bound for parameter {param.name!r}: bindings state selection "
            "(package.toml / the deployment binding), code states capability",
            code="unbound_model",
            fields=[param.name],
        )
    return model


def _service(
    param: Param, attempt: Attempt, ctx: Context, inv: Invocation, surface: Surface
) -> object:
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
        case "weights_read":
            return WeightsReader(
                attempt, inv.models, ctx.tensorfs_source, inv.cancel, inv.weights_values
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


def _call(
    surface: Surface,
    kwargs: Mapping[str, object],
    publish_to: str = "",
    activity_sink: Callable[[int, bool, bool, bool], None] | None = None,
) -> object:
    """`def` runs right here on the calling (executor) thread; `async def` runs on a
    PRIVATE event loop created for THIS attempt and closed with it.

    `await` interleaves I/O within one attempt and never admits a second — this function
    is synchronous, so law 8's one-active-GPU-attempt rule is unchanged by the spelling.
    A granted destination receives the returned Models before the attempt closes.
    """
    from cozy_runtime.author._activity import event_loop, observing

    with observing(activity_sink):
        if not surface.is_async and not publish_to:
            return surface.fn(**kwargs)
        loop = event_loop()
        try:
            if surface.is_async:
                coroutine = surface.fn(**kwargs)
                assert asyncio.iscoroutine(coroutine)
            else:
                coroutine = _returned(surface.fn(**kwargs))
            return loop.run_until_complete(_published(coroutine, publish_to))
        finally:
            if (broker := _current.get()) is not None:
                broker.closed = True
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()


async def _returned(value: object) -> object:
    return value


async def _published(coroutine: Coroutine[Any, Any, object], destination: str) -> object:
    """Upload each distinct Model the root returned, in result order, then return it."""
    result = await coroutine
    if destination:
        for model in dict.fromkeys(models_in(result)):
            await upload_checkpoint(model, destination=destination)
    return result


# ------------------------------------------------------------------------- result


def models_in(value: object, _seen: set[int] | None = None) -> list[ModelArtifact]:
    """Every Model reachable from a value, in declaration order."""
    seen = _seen if _seen is not None else set()
    if isinstance(value, ModelArtifact):
        return [value]
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, (list, tuple, set, frozenset)):
        return [m for item in value for m in models_in(item, seen)]
    if isinstance(value, dict):
        return [m for item in value.values() for m in models_in(item, seen)]
    return [m for member in field_values(value).values() for m in models_in(member, seen)]


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


def _check_result_bounds(surface: Surface, result: object, attempt: Attempt) -> None:
    """Every returned asset meets its result field's AssetBound, whether this attempt
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
        elif isinstance(asset, ImageAsset) and asset._local is not None:
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
        if isinstance(annotation, type) and not isinstance(value, annotation):
            raise OutputError(
                f"{surface.name}.{name} is {type(value).__name__}, declared {annotation.__name__}",
                code="result_schema",
                fields=[name],
            )

    returned: list[str] = []
    output_handles: list[Asset | Tree] = [*assets_in(result), *trees_in(result)]
    pending_handles = (set(attempt.pending) - set(attempt.committed_files)) | set(
        attempt.pending_trees
    )
    if sum(asset.size_bytes for asset in output_handles) > attempt.max_output_bytes:
        raise OutputError("returned assets exceed the output byte budget", code="output_too_large")
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

    # A published asset was delivered as a product already; returning it is optional.
    unreturned = sorted(pending_handles - set(returned) - attempt.shown)
    if unreturned:
        raise OutputError(
            f"saved but never returned: {', '.join(unreturned)} — an unreturned pending "
            "output is a typed failure, never a silent drop",
            code="unreturned_handle",
        )
