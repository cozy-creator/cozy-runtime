"""Bind evaluator-validated reports to this machine's retained render facts."""

from __future__ import annotations

import hashlib
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal.worker import machine_models, workspace_byte_outputs
from cozy_runtime.internal.worker.machine_publication import PublicationRefusal
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_byte_outputs import Blob, ByteOutput, FileMember
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_BYTES = 8 << 20


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


def render_arguments(
    spec: dict[str, Any],
    intent: dict[str, Any],
    preparation: bytes,
    call_prepared: bytes,
    checkpoint: str,
) -> dict[str, Any]:
    """Separate the observed checkpoint binding from the workload's ordinary inputs."""
    arguments = dict(intent["request"])
    _check(
        spec.get("payload_digest")
        == documents.spell(hashlib.sha256(canonical_json.encode(arguments)).digest())
    )
    if "job" in spec:
        models = [
            row
            for row in spec.get("inputs", [])
            if row["kind_mime"] == "application/vnd.cozy.model-manifest"
        ]
        _check(len(models) == 1 and models[0]["digest"] == checkpoint)
        arguments.pop(models[0]["input_id"].removeprefix("model:"), None)
        return arguments

    # Serving binds Model parameters through the admitted placement rather than
    # through job InputBindings. Read the exact retained preparation, never a
    # checkpoint claim supplied by the assessment script.
    prepared = canonical_json.decode(call_prepared)
    _check(prepared.get("kind") == "serving")
    installations = canonical_json.decode(preparation)["installations"]
    _check(prepared["installation_id"] in installations)
    placement = installations[prepared["installation_id"]]["placement"]
    serving = spec["serving"]
    _check(
        serving["bindings_digest"] == placement["bindings_digest"]
        and spec["installation_id"] == placement["installation_id"]
    )
    entries = [
        entry
        for entry in placement["entrypoints"]
        if entry["entrypoint_binding_digest"] == serving["entrypoint_binding_digest"]
        and entry["name"] == prepared["entrypoint"]
    ]
    _check(len(entries) == 1 and bool(entries[0]["slots"]))
    bound = {model["id"]: model["manifest"] for model in placement["models"]}
    for slot in entries[0]["slots"]:
        selected = {slot["reference_model_id"]} | {
            component["model_id"] for component in slot["components"]
        }
        _check(all(name in bound and bound[name]["digest"] == checkpoint for name in selected))
        argument = arguments.pop(slot["slot"], None)
        _check(
            isinstance(argument, dict)
            and argument.get("manifest") == bound[slot["reference_model_id"]]
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
        facts: dict[str, Any] = msgspec.to_builtins(report)
    except Exception as exc:
        raise PublicationRefusal("assessment.report_invalid") from exc
    _check(sealed.schema == "cozy-eval/checkpoint-validation@5")
    subject = facts["subject"]
    _check(subject["candidate_checkpoint"] == checkpoint.checkpoint)
    plans = canonical_json.decode(workloads)
    _check(isinstance(plans, list) and len(plans) == len(subject["workloads"]))
    expected: list[tuple[dict[str, Any], Any]] = []
    for plan, reference in zip(plans, subject["workloads"], strict=True):
        _check(
            isinstance(plan, dict)
            and reference["digest"]
            == documents.spell(hashlib.sha256(contract.canonical(plan)).digest())
            and reference["entrypoint"] == plan.get("entrypoint")
        )
        _check(isinstance(plan.get("payloads"), list) and bool(plan["payloads"]))
        expected.extend((plan, payload) for payload in plan["payloads"])
    _check(bool(expected))
    seen: set[str] = set()
    for arm_name in ("reference", "repeat", "candidate"):
        arm = subject["arms"][arm_name]
        model = (
            subject["candidate_checkpoint"]
            if arm_name == "candidate"
            else subject["reference_checkpoint"]
        )
        _check(
            arm["checkpoint"] == model
            and len(arm["requests"]) == len(expected)
            and len(arm["media"]) == len(expected)
        )
        for optional in ("audio", "captures"):
            _check(len(arm[optional]) in (0, len(expected)))
        for index, (plan, payload) in enumerate(expected):
            request = arm["requests"][index]
            _check(request not in seen)
            seen.add(request)
            with workspace.locked() as db:
                call = db.execute(
                    "SELECT * FROM execution_calls WHERE owner=? "
                    "AND parent_request=? AND child_request=?",
                    (owner, producer, request),
                ).fetchone()
                execution = db.execute(
                    "SELECT e.*,a.invocation,a.spec,a.outcome,a.outcome_digest,"
                    "a.memoize,a.worker_boot "
                    "FROM executions e JOIN attempts a "
                    "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
                    "WHERE e.owner=? AND e.request=?",
                    (owner, request),
                ).fetchone()
                _check(
                    call is not None
                    and execution is not None
                    and execution["state"] == "succeeded"
                    and not execution["memoize"]
                    and bool(execution["outcome"])
                )
                intent = canonical_json.decode(call["intent"])
                _check(intent["module"] + "." + intent["export"] == plan["entrypoint"])
                _check(hashlib.sha256(call["intent"]).digest() == call["intent_digest"])
                spec = documents.read(execution["invocation"], pb.InvocationSpec)
                terminal = documents.read(execution["outcome"], pb.AttemptOutcomeBody)
                _check(
                    hashlib.sha256(execution["invocation"]).digest() == execution["spec"]
                    and hashlib.sha256(execution["outcome"]).digest() == execution["outcome_digest"]
                )
                _check(
                    terminal["request_id"] == request
                    and terminal["attempt_ordinal"] == execution["ordinal"]
                    and terminal["invocation_spec_digest"] == documents.spell(execution["spec"])
                    and terminal["status"] == pb.OUTCOME_STATUS_SUCCEEDED
                    and terminal.get("execution_started") is True
                )
                arguments = render_arguments(
                    spec, intent, execution["preparation"], call["prepared"], model
                )
                _check(canonical_json.encode(arguments) == canonical_json.encode(payload))
                observation = terminal.get("observation", {})
                observed = observation.get("environment", {})
                environment = facts["environment"][arm_name]
                names = {
                    "runtime_version": "runtime_version",
                    "worker_image": "worker_image_digest",
                    "accelerator": "accelerator",
                    "driver": "driver",
                    "cuda": "cuda",
                    "worker_boot_id": "worker_boot_id",
                    "execution_lane": "execution_lane",
                    "execution_contract": "execution_contract_digest",
                    "kernel_symbol": "kernel_symbol",
                }
                _check(
                    all(
                        environment.get(name, "") == observed.get(field, "")
                        for name, field in names.items()
                    )
                )
                _check(observed.get("worker_boot_id") == execution["worker_boot"])
                outputs = terminal.get("output_manifest", {}).get("outputs", [])
                wanted: list[tuple[str, tuple[str, ...]]] = [
                    (arm["media"][index], ("image/", "video/"))
                ]
                if arm["audio"]:
                    wanted.append((arm["audio"][index], ("audio/",)))
                byte_outputs = []
                for digest, prefixes in wanted:
                    matched = [
                        row
                        for row in outputs
                        if row.get("digest") == digest
                        and row.get("mime_type", "").startswith(prefixes)
                        and row.get("length", 0) > 0
                    ]
                    _check(len(matched) == 1)
                    output = db.one(
                        ByteOutput,
                        "SELECT * FROM byte_outputs WHERE owner=? "
                        "AND request=? AND ordinal=? AND slot=?",
                        (owner, request, execution["ordinal"], matched[0]["output_id"]),
                    )
                    if output is None:
                        raise PublicationRefusal("assessment.execution_binding_mismatch")
                    byte_outputs.append((output, matched[0]))
                expected_capture = {
                    "components": plan.get("capture", []),
                    "steps": plan.get("capture_steps", []),
                }
                if arm["captures"]:
                    _check(
                        intent.get("capture") == expected_capture
                        and spec.get("capture") == expected_capture
                        and observation.get("capture", {}).get("content_digest")
                        == arm["captures"][index]
                    )
                    output = db.one(
                        ByteOutput,
                        "SELECT * FROM byte_outputs WHERE owner=? "
                        "AND request=? AND ordinal=? AND slot='runtime.capture'",
                        (owner, request, execution["ordinal"]),
                    )
                    if output is None:
                        raise PublicationRefusal("assessment.execution_binding_mismatch")
                    byte_outputs.append((output, None))
                else:
                    _check(not intent.get("capture") and not observation.get("capture"))
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
                            members == _payload(claimed["digest"][7:], claimed["length"])
                            and source.content_bytes == claimed["length"]
                        )
    return str(facts["verdict"])
