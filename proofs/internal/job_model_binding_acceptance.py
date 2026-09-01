#!/usr/bin/env python3
"""Exact job Model binding -> attempt hold -> executor -> ArtifactSink handoff proof."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import msgspec
import tensorfs
from modeled_development_store import create as create_store

from cozy.worker.v1.wire_version import WIRE_MINOR
from cozy_runtime import canonical_json
from cozy_runtime.author import (
    App,
    ArtifactOutput,
    ArtifactSink,
    ArtifactTarget,
    CapabilityError,
    Model,
)
from cozy_runtime.author._describe import describe
from cozy_runtime.internal import canonical, descriptor, package_environment
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel
from cozy_runtime.internal.worker import grants, package_prepare
from cozy_runtime.internal.worker.attempts import AttemptEngine, AttemptRecord, AttemptRefusal
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class Request(msgspec.Struct, forbid_unknown_fields=True):
    left_component: str
    right_component: str
    prefix: str


class Result(msgspec.Struct):
    transaction_id: str
    load_refusal: str
    component_refusal: str


class LeftModel(Model[object]):
    pipe: object

    def load(self, loader: Any) -> None:
        del loader
        raise RuntimeError("derive-only job called package load")


class RightModel(Model[object]):
    pipe: object

    def load(self, loader: Any) -> None:
        del loader
        raise RuntimeError("derive-only job called package load")


app = App()
PRIVATE_REVISION = "sha256:" + "a" * 64


@app.job(artifacts=(ArtifactOutput("model", max_new_bytes=4096),))
def join(
    payload: Request,
    left: LeftModel,
    right: RightModel,
    artifacts: ArtifactSink,
) -> Result:
    try:
        left.load(None)
        load_refusal = "load-ran"
    except CapabilityError as exc:
        load_refusal = exc.code
    try:
        _ = left.pipe
        component_refusal = "component-visible"
    except CapabilityError as exc:
        component_refusal = exc.code
    receipt = artifacts.derive(
        "model",
        sources={"left": left, "right": right},
        targets={
            payload.prefix + "left": ArtifactTarget(
                source="left", source_component=payload.left_component
            ),
            payload.prefix + "right": ArtifactTarget(
                source="right", source_component=payload.right_component
            ),
        },
        order=(
            (payload.prefix + "left", "weight"),
            (payload.prefix + "right", "weight"),
        ),
    )
    return Result(receipt.artifact_transaction_id, load_refusal, component_refusal)


class HostChannel:
    """The supervisor host answers durable ArtifactSink frames, never source reads itself."""

    def __init__(self, *, owner_scope: str, request_id: str, invocation_digest: str) -> None:
        self.owner_scope = owner_scope
        self.request_id = request_id
        self.invocation_digest = invocation_digest
        self.answers: list[dict[str, Any]] = []

    def send(self, frame: dict[str, Any]) -> None:
        if frame.get("event") != "request":
            return
        kind = str(frame.get("kind", ""))
        answer: dict[str, Any] = {"event": "answer", "seq": int(frame["seq"]), "ok": True}
        if kind == "artifact_intent":
            subject = canonical_json.encode(
                {
                    "owner_authority_scope": self.owner_scope,
                    "request_id": self.request_id,
                    "invocation_spec_digest": self.invocation_digest,
                    "output_slot": str(frame["output_slot"]),
                }
            )
            answer.update(
                artifact_transaction_id="sha256:"
                + hashlib.sha256(b"cozy.runtime.artifact-transaction\0" + subject).hexdigest(),
                writer_generation=1,
                artifact_receipt_digest="",
            )
        elif kind == "artifact_receipt":
            answer["artifact_receipt_digest"] = str(frame["artifact_receipt_digest"])
        else:
            raise RuntimeError(f"unexpected durable exchange {kind!r}")
        self.answers.append(answer)

    def recv(self) -> dict[str, Any] | None:
        return self.answers.pop(0) if self.answers else None


class ProofRecords:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []

    def append(self, kind: str, body: dict[str, Any]) -> None:
        self.rows.append((kind, body))


def discovered() -> tuple[Discovered, dict[str, Any], str]:
    found = Discovered(
        app=app,
        application="job_model_binding_acceptance:app",
        project=Path(__file__).parent,
        module=sys.modules[__name__],
        surfaces=describe(app),
        bindings={},
    )
    body = descriptor.build(found)
    return found, body, descriptor.job_descriptor_id(body, "join")


def binding(descriptor_id: str, descriptor_path: Path) -> JobBinding:
    return JobBinding(
        job_descriptor_id=descriptor_id,
        build_id=PRIVATE_REVISION,
        application="job_model_binding_acceptance:app",
        package_descriptor=str(descriptor_path),
        python=sys.executable,
        environment_content_digest="sha256:" + "a" * 64,
        job="join",
    )


def offered(
    descriptor_id: str,
    left: tuple[str, int],
    right: tuple[str, int],
    *,
    serving: bool = False,
    access: str = "model",
    mime: str = grants.MODEL_MIME,
) -> tuple[dict[str, Any], bytes, grants.BoundGrant]:
    payload = canonical.write(
        {"left_component": "core", "right_component": "core", "prefix": "n1_"}
    )
    rows = [
        pb.InputBinding(
            input_id="payload",
            digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            length=len(payload),
            kind_mime="application/json",
        ),
        pb.InputBinding(input_id="model:left", digest=left[0], length=left[1], kind_mime=mime),
        pb.InputBinding(input_id="model:right", digest=right[0], length=right[1], kind_mime=mime),
    ]
    spec = pb.InvocationSpec(
        package_revision_digest=PRIVATE_REVISION,
        payload_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        inputs=rows,
        outputs=[pb.OutputBinding(output_id="model", max_bytes=4096)],
    )
    if serving:
        spec.serving.entrypoint_binding_digest = "sha256:" + "b" * 64
    else:
        spec.job.CopyFrom(
            pb.JobInvocationSpec(
                build_id=PRIVATE_REVISION,
                job_descriptor_id=descriptor_id,
                publication_contract=pb.PublicationContract(
                    outputs=[pb.OutputBinding(output_id="model", max_bytes=4096)]
                ),
            )
        )
    raw, digest = documents.identity(spec)
    document = documents.read(raw, pb.InvocationSpec)
    grant = pb.DeliveryGrant(
        invocation_spec_digest=digest,
        inputs=[
            pb.InputAccess(input_id="payload", url="file:///unused-payload"),
            pb.InputAccess(input_id="model:left", url=f"{access}://{left[0]}"),
            pb.InputAccess(input_id="model:right", url=f"model://{right[0]}"),
        ],
        outputs=[pb.OutputAccess(output_id="model", url="file:///unused-output")],
    )
    return document, digest, grants.bind(document, grant, digest)


def attempt_command(
    *,
    store_root: Path,
    descriptor_id: str,
    descriptor_path: Path,
    left: tuple[str, int],
    right: tuple[str, int],
    request_id: str,
    payload: dict[str, str],
) -> tuple[AttemptEngine, AttemptRecord, dict[str, Any]]:
    document, digest, bound = offered(descriptor_id, left, right)
    attempt = AttemptRecord(
        request_id=request_id, attempt=1, digest=digest, spec=document, grant=bound
    )
    selected = binding(descriptor_id, descriptor_path)
    attempt.job = selected
    attempt.kind = "job"
    attempt.payload = payload
    attempt.spool = store_root.parent / (request_id + "-spool")
    attempt.spool.mkdir()
    attempt.scratch = store_root.parent / (request_id + "-scratch")
    attempt.scratch.mkdir()
    engine = object.__new__(AttemptEngine)
    engine.artifact_store = store_root
    engine.owner_scope = lambda: "record-owner/proof"
    engine.records = cast(Any, ProofRecords())
    engine.boot_id = "proof-boot"
    engine.instance_id = "proof-instance"
    engine._hold_job_models(attempt, selected, descriptor_id)
    command = engine._command(attempt, f"{request_id}#1", attempt.spool, None)
    return engine, attempt, command


def run_node(
    found: Discovered,
    command: dict[str, Any],
    *,
    request_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    channel = HostChannel(
        owner_scope="record-owner/proof",
        request_id=str(command["artifact_request_id"]),
        invocation_digest=str(command["invocation_spec_digest"]),
    )
    executor = Executor(
        cast(Channel, channel), Path(command["spool"]).parent / (request_id + "-executor")
    )
    executor.discovered = found
    executor.surface_names = {surface.name: surface for surface in found.surfaces}
    reply = executor.run_job(command)
    if not reply.get("ok") or reply.get("outcome", {}).get("terminal") != "succeeded":
        raise RuntimeError(f"executor job failed: {reply}")
    ref = reply.get("result_ref")
    if not isinstance(ref, dict):
        raise RuntimeError("executor returned no result ref")
    result_bytes = (Path(command["spool"]) / RESULT_DOCUMENT).read_bytes()
    result = canonical.parse_canonical(result_bytes)
    if not isinstance(result, dict):
        raise RuntimeError("job result is not one object")
    return reply, result


def receipt_manifest(store: tensorfs.Store, transaction_id: str) -> tuple[str, int]:
    state = store.derived_lookup(transaction_id)
    receipt = state.get("receipt") if state.get("state") == "committed" else None
    if not isinstance(receipt, dict) or not isinstance(receipt.get("manifest"), dict):
        raise RuntimeError("TensorFS committed no produced Manifest")
    manifest = receipt["manifest"]
    return "sha256:" + str(manifest["sha256"]), int(manifest["length"])


def adopt_handoff(store: tensorfs.Store, manifest: tuple[str, int]) -> None:
    selected = store.manifest(manifest[0])
    closure = store.walk(manifest[0])
    operation = store.begin_operation("job-node-handoff", "proof", "node-one")
    operation.hold_manifest(manifest[0], manifest[1])
    for reached in closure:
        operation.hold_blob(str(reached["id"]), int(reached["length"]))
    evidence = json.dumps(
        {"classification_digest": tensorfs.topology_digest(bytes(selected["header"]))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    operation.commit_release(None, "2.0.0", "node-one", manifest[0], manifest[1], evidence)


def refusal_arms(
    descriptor_id: str,
    descriptor_path: Path,
    store_root: Path,
    manifest: tuple[str, int],
) -> None:
    for kwargs in (
        {"serving": True},
        {"access": "file"},
        {"mime": "application/octet-stream"},
    ):
        try:
            offered(descriptor_id, manifest, manifest, **kwargs)
        except grants.GrantRefusal:
            pass
        else:
            raise RuntimeError(f"grant red arm was admitted: {kwargs}")

    selected = binding(descriptor_id, descriptor_path)
    engine = object.__new__(AttemptEngine)
    engine.artifact_store = store_root
    document, digest, bound = offered(descriptor_id, manifest, manifest)
    missing = AttemptRecord("missing-param", 1, digest, document, grant=bound)
    missing.grant.inputs.pop("model:right")
    try:
        engine._hold_job_models(missing, selected, descriptor_id)
    except AttemptRefusal:
        pass
    else:
        raise RuntimeError("missing descriptor Model parameter was admitted")

    document, digest, bound = offered(descriptor_id, manifest, manifest)
    extra = AttemptRecord("extra-param", 1, digest, document, grant=bound)
    right = extra.grant.inputs.pop("model:right")
    extra.grant.inputs["model:extra"] = dataclasses.replace(right, input_id="model:extra")
    try:
        engine._hold_job_models(extra, selected, descriptor_id)
    except AttemptRefusal:
        pass
    else:
        raise RuntimeError("extra job Model parameter was admitted")

    no_store = object.__new__(AttemptEngine)
    no_store.artifact_store = store_root.parent / "absent-store"
    document, digest, bound = offered(descriptor_id, manifest, manifest)
    unavailable = AttemptRecord("missing-store", 1, digest, document, grant=bound)
    try:
        no_store._hold_job_models(unavailable, selected, descriptor_id)
    except AttemptRefusal as exc:
        if exc.code != "job_model_store_unavailable":
            raise RuntimeError(f"missing store returned {exc.code}") from exc
    else:
        raise RuntimeError("missing TensorFS store was admitted")

    for left in (
        ("sha256:" + "f" * 64, manifest[1]),
        (manifest[0], manifest[1] + 1),
    ):
        document, digest, bound = offered(descriptor_id, left, manifest)
        candidate = AttemptRecord("refusal", 1, digest, document, grant=bound)
        try:
            engine._hold_job_models(candidate, selected, descriptor_id)
        except AttemptRefusal as exc:
            if candidate.model_leases:
                raise RuntimeError("refused model binding leaked a TensorFS lease") from exc
        else:
            raise RuntimeError(f"absent/stale Manifest was admitted: {left}")


def main() -> None:
    found, body, descriptor_id = discovered()
    with tempfile.TemporaryDirectory(prefix="cozy-job-model-binding-") as temporary:
        root = Path(temporary)
        store_facts = create_store(root / "store")
        source = (str(store_facts["manifest_digest"]), int(store_facts["manifest_length"]))
        descriptor_path = root / "package.descriptor.json"
        descriptor_path.write_bytes(descriptor.canonical_bytes(body))
        store = tensorfs.Store.open(str(root / "store"))
        staged = package_prepare._stage_job_plans(
            body,
            root=root / "job-plans",
            build_id=PRIVATE_REVISION,
            descriptor_path=descriptor_path,
            installed=package_environment.InstalledEnvironment(
                generation=root / "private-generation",
                site_packages=root,
                python=Path(sys.executable),
                receipt_bytes=b"{}",
                receipt_digest="sha256:" + "b" * 64,
                content_digest="sha256:" + "c" * 64,
                reused=False,
            ),
            include_modeled=True,
        )
        if staged != {descriptor_id}:
            raise RuntimeError("private revision did not stage its modeled job")
        staged_binding = JobBinding.read(
            json.loads(
                (root / "job-plans" / f"{descriptor_id.removeprefix('sha256:')}.json").read_text()
            )
        )
        if staged_binding.build_id != PRIVATE_REVISION:
            raise RuntimeError("private modeled job lost its exact revision identity")

        engine1, attempt1, command1 = attempt_command(
            store_root=root / "store",
            descriptor_id=descriptor_id,
            descriptor_path=descriptor_path,
            left=source,
            right=source,
            request_id="node-one",
            payload={"left_component": "core", "right_component": "core", "prefix": "n1_"},
        )
        if len(attempt1.model_leases) != 1 or set(attempt1.job_models) != {"left", "right"}:
            raise RuntimeError("attempt did not deduplicate and hold exact input Manifests")
        _reply1, result1 = run_node(found, command1, request_id="node-one#1")
        produced1 = receipt_manifest(store, str(result1["transaction_id"]))
        if (
            result1.get("load_refusal") != "derive_model_read_only"
            or result1.get("component_refusal") != "derive_model_component_unavailable"
        ):
            raise RuntimeError(f"derive-only Model guard failed: {result1}")
        engine1.last_resort(attempt1, "proof terminal after node one")
        if attempt1.model_leases:
            raise RuntimeError("terminal release retained node-one Manifest lease")
        adopt_handoff(store, produced1)

        engine2, attempt2, command2 = attempt_command(
            store_root=root / "store",
            descriptor_id=descriptor_id,
            descriptor_path=descriptor_path,
            left=produced1,
            right=source,
            request_id="node-two",
            payload={"left_component": "n1_left", "right_component": "core", "prefix": "n2_"},
        )
        if len(attempt2.model_leases) != 2:
            raise RuntimeError("node-two did not hold both handoff Manifests")
        _reply2, result2 = run_node(found, command2, request_id="node-two#1")
        produced2 = receipt_manifest(store, str(result2["transaction_id"]))
        if produced2 in {source, produced1}:
            raise RuntimeError("multi-node ArtifactSink handoff collapsed a Manifest identity")
        engine2.last_resort(attempt2, "proof terminal after node two")

        wrong_class = dict(command1)
        wrong_class["models"] = {key: dict(value) for key, value in command1["models"].items()}
        wrong_class["models"]["left"]["class"] = "WrongModel"
        channel = HostChannel(
            owner_scope="record-owner/proof",
            request_id="wrong-class",
            invocation_digest=str(command1["invocation_spec_digest"]),
        )
        executor = Executor(cast(Channel, channel), root / "wrong-class-executor")
        executor.discovered = found
        executor.surface_names = {surface.name: surface for surface in found.surfaces}
        wrong_reply = executor.run_job(wrong_class)
        if wrong_reply.get("code") != "job_model_class_mismatch":
            raise RuntimeError(f"wrong executor Model class was admitted: {wrong_reply}")

        refusal_arms(descriptor_id, descriptor_path, root / "store", source)
        print(
            f"JOB MODEL BINDING PASS protocol_minor={WIRE_MINOR} exact_descriptor_params=2 "
            "host_capability=model:// attempt_manifest_holds=dedup terminal_release=1 "
            "derive_only=1 artifact_sink_nodes=2 manifest_handoff=1 serving=refused "
            "private_revision=1 param_class_digest_length_scheme_store_absence=refused"
        )


if __name__ == "__main__":
    main()
