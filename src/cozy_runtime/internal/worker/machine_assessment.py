"""Bind evaluator-validated reports to this machine's retained render facts."""

from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker import grants, machine_models, workspace_byte_outputs
from cozy_runtime.internal.worker.machine_publication import PublicationRefusal
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_byte_outputs import Blob, ByteOutput, FileMember
from cozy_runtime.internal.worker.workspace_calls import CallPlan
from cozy_runtime.internal.worker.workspace_executions import Preparation
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_BYTES = 8 << 20


class Capture(msgspec.Struct, frozen=True):
    """An activation capture's selection (`ActivationCapture`'s document)."""

    components: tuple[str, ...] = ()
    steps: tuple[int, ...] = ()


class Intent(msgspec.Struct, frozen=True):
    """A render call's intent (`execution_calls.intent`)."""

    module: str = ""
    export: str = ""
    request: dict[str, Json] = {}
    capture: Capture | None = None


class _Workload(msgspec.Struct, frozen=True):
    """One evaluator workload plan, as far as its renders bind it."""

    entrypoint: str
    payloads: tuple[Json, ...]
    capture: tuple[str, ...] = ()
    capture_steps: tuple[int, ...] = ()


class _Call(msgspec.Struct, frozen=True):
    intent: bytes
    intent_digest: bytes
    prepared: bytes


class _Render(msgspec.Struct, frozen=True):
    """A claimed render: its execution joined to its current attempt."""

    state: str
    ordinal: int
    preparation: bytes
    invocation: bytes
    spec: bytes
    outcome: bytes
    outcome_digest: bytes
    memoize: int
    worker_boot: str


def _payload(sha256: str, length: int) -> tuple[FileMember, ...]:
    return (FileMember("file", "payload", Blob(sha256, length)),)


def read_file(
    workspace: Workspace, owner: str, parent: str, recipient: str, slot: str, digest: str
) -> tuple[bytes, str]:
    source = machine_models.file_source(workspace, owner, parent, digest)
    retained = machine_models.retain_bytes(workspace, owner, recipient, slot, source.source)
    held = pb.NativeByteRetentionRequest(source=retained.source, retention_id=retained.retention_id)
    with workspace_byte_outputs.leased(workspace, owner, held) as (_, lease, members):
        length = source.source.content_bytes
        if not 0 < length <= MAX_BYTES or members != _payload(digest[7:], length):
            raise WorkspaceRefusal("assessment input changed its exact bounded file")
        raw = bytearray(length)
        lease.read_into(digest, length, 0, length, raw)
        if documents.spell(hashlib.sha256(raw).digest()) != digest:
            raise WorkspaceRefusal("assessment input bytes changed identity")
    with workspace.locked() as db:
        row = db.execute(
            "SELECT request FROM byte_outputs WHERE owner=? AND id=?",
            (owner, source.source.producer_root_id),
        ).fetchone()
        if row is None:
            raise WorkspaceRefusal("assessment file lost its recorded producer")
        return bytes(raw), str(row["request"])


def _check(condition: bool) -> None:
    if not condition:
        raise PublicationRefusal("assessment.execution_binding_mismatch")


def _as[T](value: object, into: type[T]) -> T:
    """`value` decoded once into `into`: a fact of another shape binds nothing."""
    try:
        return msgspec.convert(value, into, strict=True)
    except msgspec.ValidationError as exc:
        raise PublicationRefusal("assessment.execution_binding_mismatch") from exc


def _spelled(digest: bytes) -> str:
    return documents.spell(digest) if digest else ""


def render_arguments(
    spec: pb.InvocationSpec,
    intent: Intent,
    preparation: bytes,
    call_prepared: bytes,
    checkpoint: str,
) -> dict[str, Json]:
    """Separate the observed checkpoint binding from the workload's ordinary inputs."""
    arguments = dict(intent.request)
    _check(
        spec.payload_digest
        == documents.spell(hashlib.sha256(canonical_json.encode(arguments)).digest())
    )
    if spec.HasField("job"):
        models = [row for row in spec.inputs if row.kind_mime == grants.MODEL_MIME]
        _check(len(models) == 1 and models[0].digest == checkpoint)
        arguments.pop(models[0].input_id.removeprefix(grants.MODEL_PREFIX), None)
        return arguments

    # Serving binds Model parameters through the admitted placement rather than
    # through job InputBindings. Read the exact retained preparation, never a
    # checkpoint claim supplied by the assessment script.
    prepared = _as(canonical_json.decode(call_prepared), CallPlan)
    _check(prepared.kind == "serving")
    installations = Preparation.read(preparation).installations
    _check(prepared.installation_id in installations)
    placement = installations[prepared.installation_id].placement
    serving = spec.serving
    _check(
        serving.bindings_digest == _spelled(placement.bindings_digest)
        and spec.installation_id == placement.installation_id
    )
    entries = [
        entry
        for entry in placement.entrypoints
        if _spelled(entry.entrypoint_binding_digest) == serving.entrypoint_binding_digest
        and entry.name == prepared.entrypoint
    ]
    _check(len(entries) == 1 and bool(entries[0].slots))
    bound = {model.id: model.manifest for model in placement.models}
    for slot in entries[0].slots:
        selected = {slot.reference_model_id} | {component.model_id for component in slot.components}
        _check(
            all(name in bound and _spelled(bound[name].digest) == checkpoint for name in selected)
        )
        argument = arguments.pop(slot.slot, None)
        _check(
            isinstance(argument, dict)
            and argument.get("manifest") == documents.body(bound[slot.reference_model_id])
        )
    return arguments


def verify(
    workspace: Workspace,
    owner: str,
    producer: str,
    recipient: str,
    checkpoint: CheckpointRef,
    raw: bytes,
    workloads: bytes,
) -> str:
    """cozy-eval owns metrics and gate semantics; Runtime verifies the claimed executions."""
    try:
        # The published cozy-eval wheel has no py.typed marker; it still owns
        # runtime report validation. Only its checked association facts cross here.
        from cozy_eval import contract  # type: ignore[import-untyped]
        from cozy_eval.assessment import AssessmentReport  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PublicationRefusal("assessment.validator_unavailable") from exc
    try:
        sealed = contract.load(raw, expect=AssessmentReport)
        report = sealed.body
        report.contract_check()
    except Exception as exc:
        raise PublicationRefusal("assessment.report_invalid") from exc
    # A floor, not a pin: cozy-eval validated the body; a newer revision is additive.
    family, _, revision = sealed.schema.rpartition("@")
    _check(
        family == "cozy-eval/checkpoint-validation" and revision.isdigit() and int(revision) >= 5
    )
    subject = report.subject
    _check(subject.candidate_checkpoint == checkpoint.checkpoint)
    plans = canonical_json.decode(workloads)
    _check(isinstance(plans, list) and len(plans) == len(subject.workloads))
    expected: list[tuple[_Workload, Json]] = []
    for plan, reference in zip(plans, subject.workloads, strict=True):
        workload = _as(plan, _Workload)
        _check(
            reference.digest == documents.spell(hashlib.sha256(contract.canonical(plan)).digest())
            and reference.entrypoint == workload.entrypoint
            and bool(workload.payloads)
        )
        expected.extend((workload, payload) for payload in workload.payloads)
    _check(bool(expected))
    seen: set[str] = set()
    for arm_name in ("reference", "repeat", "candidate"):
        arm = subject.arms[arm_name]
        environment = report.environment[arm_name]
        model = (
            subject.candidate_checkpoint
            if arm_name == "candidate"
            else subject.reference_checkpoint
        )
        _check(
            arm.checkpoint == model
            and len(arm.requests) == len(expected)
            and len(arm.media) == len(expected)
        )
        for optional in (arm.audio, arm.captures):
            _check(len(optional) in (0, len(expected)))
        for index, (workload, payload) in enumerate(expected):
            request = arm.requests[index]
            _check(request not in seen)
            seen.add(request)
            with workspace.locked() as db:
                call = db.one(
                    _Call,
                    "SELECT intent,intent_digest,prepared FROM execution_calls WHERE owner=? "
                    "AND parent_request=? AND child_request=?",
                    (owner, producer, request),
                )
                execution = db.one(
                    _Render,
                    "SELECT e.state,e.ordinal,e.preparation,a.invocation,a.spec,a.outcome,"
                    "a.outcome_digest,a.memoize,a.worker_boot "
                    "FROM executions e JOIN attempts a "
                    "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
                    "WHERE e.owner=? AND e.request=?",
                    (owner, request),
                )
                if call is None or execution is None:
                    raise PublicationRefusal("assessment.execution_binding_mismatch")
                _check(
                    execution.state == "succeeded"
                    and not execution.memoize
                    and bool(execution.outcome)
                )
                intent = _as(canonical_json.decode(call.intent), Intent)
                _check(intent.module + "." + intent.export == workload.entrypoint)
                _check(hashlib.sha256(call.intent).digest() == call.intent_digest)
                _check(
                    hashlib.sha256(execution.invocation).digest() == execution.spec
                    and hashlib.sha256(execution.outcome).digest() == execution.outcome_digest
                )
                spec = documents.parse(execution.invocation, pb.InvocationSpec)
                terminal = documents.parse(execution.outcome, pb.AttemptOutcomeBody)
                _check(
                    terminal.request_id == request
                    and terminal.attempt_ordinal == execution.ordinal
                    and terminal.invocation_spec_digest == documents.spell(execution.spec)
                    and terminal.status == pb.OUTCOME_STATUS_SUCCEEDED
                    and terminal.execution_started
                )
                arguments = render_arguments(
                    spec, intent, execution.preparation, call.prepared, model
                )
                _check(canonical_json.encode(arguments) == canonical_json.encode(payload))
                observation = terminal.observation
                observed = observation.environment
                _check(
                    (
                        environment.runtime_version,
                        environment.worker_image,
                        environment.accelerator,
                        environment.driver,
                        environment.cuda,
                        environment.worker_boot_id,
                        environment.execution_lane,
                        environment.execution_contract,
                        environment.kernel_symbol,
                    )
                    == (
                        observed.runtime_version,
                        _spelled(observed.worker_image_digest),
                        observed.accelerator,
                        observed.driver,
                        observed.cuda,
                        observed.worker_boot_id,
                        observed.execution_lane,
                        _spelled(observed.execution_contract_digest),
                        observed.kernel_symbol,
                    )
                )
                _check(
                    bool(observed.worker_boot_id)
                    and observed.worker_boot_id == execution.worker_boot
                )
                wanted: list[tuple[str, tuple[str, ...]]] = [
                    (arm.media[index], ("image/", "video/"))
                ]
                if arm.audio:
                    wanted.append((arm.audio[index], ("audio/",)))
                byte_outputs: list[tuple[ByteOutput, pb.OutputEntry | None]] = []
                for digest, prefixes in wanted:
                    matched = [
                        row
                        for row in terminal.output_manifest.outputs
                        if _spelled(row.digest) == digest
                        and row.mime_type.startswith(prefixes)
                        and row.length > 0
                    ]
                    _check(len(matched) == 1)
                    output = db.one(
                        ByteOutput,
                        "SELECT * FROM byte_outputs WHERE owner=? "
                        "AND request=? AND ordinal=? AND slot=?",
                        (owner, request, execution.ordinal, matched[0].output_id),
                    )
                    if output is None:
                        raise PublicationRefusal("assessment.execution_binding_mismatch")
                    byte_outputs.append((output, matched[0]))
                if arm.captures:
                    capture = Capture(workload.capture, workload.capture_steps)
                    _check(
                        intent.capture == capture
                        and spec.HasField("capture")
                        and Capture(tuple(spec.capture.components), tuple(spec.capture.steps))
                        == capture
                        and _spelled(observation.capture.content_digest) == arm.captures[index]
                    )
                    output = db.one(
                        ByteOutput,
                        "SELECT * FROM byte_outputs WHERE owner=? "
                        "AND request=? AND ordinal=? AND slot='runtime.capture'",
                        (owner, request, execution.ordinal),
                    )
                    if output is None:
                        raise PublicationRefusal("assessment.execution_binding_mismatch")
                    byte_outputs.append((output, None))
                else:
                    _check(
                        intent.capture in (None, Capture())
                        and observation.capture == pb.ActivationCaptureResult()
                    )
            for output, claimed in byte_outputs:
                source = workspace_byte_outputs.reference(output)
                retained = machine_models.retain_bytes(
                    workspace, owner, recipient, f"render.{request}.{output.slot}", source
                )
                with workspace_byte_outputs.leased(
                    workspace,
                    owner,
                    pb.NativeByteRetentionRequest(
                        source=source, retention_id=retained.retention_id
                    ),
                ) as (_, _, members):
                    if claimed is not None:
                        _check(
                            members == _payload(claimed.digest.hex(), claimed.length)
                            and source.content_bytes == claimed.length
                        )
    return str(report.verdict)
