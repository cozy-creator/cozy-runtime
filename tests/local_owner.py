"""A machine submission built in-process: the offer, grant and prepared state a RecordOwner
submits through SubmitMachineExecution, with local files standing in for its grants."""

from __future__ import annotations

import dataclasses
import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import signed_claims
from cozy_runtime.author._capture import ActivationCapture
from cozy_runtime.author._media import SNIFF_BYTES, sniff
from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import grants as grants_module
from cozy_runtime.internal.worker.session import WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

LOCAL_PLACEMENT = "plc-local"


def claimable(config: RuntimeConfig, options: WorkerOptions) -> tuple[RuntimeConfig, WorkerOptions]:
    """A worker config and options that admit the signed test Claim."""
    return (
        dataclasses.replace(config, record_owner_public_key=signed_claims.PUBLIC_KEY),
        dataclasses.replace(options, **signed_claims.IDENTITY),
    )


class LocalBindingError(Exception):
    """The request names an entrypoint this run staged no binding for."""


@dataclass(frozen=True, slots=True)
class LocalInput:
    """One owner-held file plus its typed position in the request."""

    source: str
    order: int


@dataclass(frozen=True, slots=True)
class LocalRequest:
    """One request, as a RecordOwner submits it."""

    entrypoint: str
    payload: Mapping[str, object]
    #: the result field paths the caller grants a destination for (`image`, `detail.thumb`)
    outputs: tuple[str, ...] = ("image",)
    capture: ActivationCapture | None = None
    attention_kernel: str = ""
    request_id: str = "run-1"
    attempt: int = 1
    #: the caller's bound on this attempt as a duration; 0 is none. The absolute deadline
    #: is minted when the offer is built.
    timeout_ms: int = 0
    #: grant the payload from this location instead of a file staged beside the run
    payload_url: str = ""
    #: `entrypoint` (serving) or `job`
    kind: str = "entrypoint"
    job_descriptor_id: str = ""
    #: job trees this run grants: `ref -> local directory`
    trees: Mapping[str, str] = field(default_factory=dict)
    #: input assets this run grants: `input_id -> (source, typed order)`
    inputs: Mapping[str, LocalInput] = field(default_factory=dict)
    #: job models: descriptor parameter -> exact Manifest (digest, length)
    models: Mapping[str, tuple[str, int]] = field(default_factory=dict)
    org: str = "local"


class LocalRecordOwner:
    """Builds one request's offer and prepared state, as a RecordOwner does."""

    def __init__(
        self,
        request: LocalRequest,
        bindings: dict[str, str],
        workspace: Path,
        *,
        package_installation_id: str,
        placement: pb.Placement | None = None,
    ) -> None:
        self.request = request
        self.bindings = bindings  # entrypoint -> entrypoint_binding_digest
        self.package_installation_id = package_installation_id
        self.placement = placement
        self.grant_dir = workspace / "grants"
        self.output_dir = workspace / "outputs"

    def destination(self, output_id: str) -> Path:
        """The exact granted destination for one output field path."""
        return self.output_dir / self._filename(output_id)

    def desired_state(self) -> pb.DesiredWorkerState:
        directive = pb.DesiredWorkerState(
            revision=1,
            posture=pb.Posture.POSTURE_ACCEPTING,
            wire_minor=WIRE_MINOR,
            record_owner_epoch=1,
        )
        if self.request.kind == "job":
            directive.job.CopyFrom(
                pb.JobDirective(
                    installation_id=self.package_installation_id,
                    job_descriptor_id=self.request.job_descriptor_id,
                    resource_caps=pb.ResourceCaps(
                        device_required=False, max_rss_bytes=0, max_disk_bytes=0
                    ),
                    publication_contract=self._publication(),
                )
            )
            return directive
        if self.placement is None:
            raise LocalBindingError("a serving run has no selected PlacementSet/1 placement")
        canonical_bytes, set_digest = documents.identity(
            pb.PlacementSet(placements=[self.placement])
        )
        directive.placement_set.CopyFrom(
            pb.DesiredPlacementSet(
                placement_set_digest=set_digest, placement_set_canonical_bytes=canonical_bytes
            )
        )
        return directive

    def offer(self) -> pb.AttemptOffer:
        self.grant_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Payload identity is the exact canonical typed-argument document.
        body = canonical.write(self.request.payload)
        spec = pb.InvocationSpec(
            installation_id=self.placement.installation_id
            if self.placement is not None
            else self.package_installation_id,
            payload_digest="sha256:" + hashlib.sha256(body).hexdigest(),
            inputs=self._input_bindings(body),
            outputs=[
                pb.OutputBinding(output_id=output_id, max_bytes=64 << 20)
                for output_id in sorted(self.request.outputs)
            ],
            attention_kernel=self.request.attention_kernel,
            capture=pb.ActivationCapture(
                components=self.request.capture.components, steps=self.request.capture.steps
            )
            if self.request.capture is not None
            else None,
            deadline_unix_ms=(
                int(time.time() * 1000) + self.request.timeout_ms if self.request.timeout_ms else 0
            ),
        )
        if self.request.kind == "job":
            spec.job.CopyFrom(
                pb.JobInvocationSpec(
                    installation_id=self.package_installation_id,
                    job_descriptor_id=self.request.job_descriptor_id,
                    publication_contract=self._publication(),
                )
            )
        else:
            if self.placement is None:
                raise RuntimeError("serving dispatch has no exact prepared placement")
            binding_digest = self._binding_digest()
            spec.serving.entrypoint_binding_digest = binding_digest
            spec.serving.attempt_binding_id = binding_digest
            spec.serving.bindings_digest = documents.spell(self.placement.bindings_digest)
        canonical_bytes, digest = documents.identity(spec)
        return pb.AttemptOffer(
            request_id=self.request.request_id,
            attempt_ordinal=self.request.attempt,
            invocation_spec_digest=digest,
            invocation_spec_canonical_bytes=canonical_bytes,
            grant=self._grant(body, digest),
            placement_id="" if self.request.kind == "job" else LOCAL_PLACEMENT,
            record_owner_epoch=1,
        )

    def _binding_digest(self) -> str:
        binding_digest = self.bindings.get(self.request.entrypoint)
        if binding_digest is None:
            known = ", ".join(sorted(self.bindings)) or "no entrypoints"
            raise LocalBindingError(
                f"no binding covers {self.request.entrypoint!r}: this run selected {known}"
            )
        return binding_digest

    def scratch_model(self) -> str:
        """The opaque scratch model a job's publishes are rewritten into."""
        return f"{self.request.org}/_job-{opaque_key('job-repository', self.request.request_id)}"

    def _publication(self) -> pb.PublicationContract:
        return pb.PublicationContract(
            grant_id=self.scratch_model(),
            outputs=[
                pb.OutputBinding(output_id=output_id, max_bytes=64 << 20)
                for output_id in self.request.outputs
            ],
        )

    def _input_bindings(self, body: bytes) -> list[pb.InputBinding]:
        """The ordered input identities, inside the spec digest."""
        rows = [
            pb.InputBinding(
                input_id="payload",
                digest="sha256:" + hashlib.sha256(body).hexdigest(),
                length=len(body),
                kind_mime="application/json",
                order=0,
            )
        ]
        rows += [
            pb.InputBinding(
                input_id=f"{grants_module.TREE_PREFIX}{ref}", kind_mime="inode/directory", order=0
            )
            for ref in sorted(self.request.trees)
        ]
        for input_id, local_input in sorted(self.request.inputs.items()):
            digest, length, media_type = _input_facts(Path(local_input.source))
            rows.append(
                pb.InputBinding(
                    input_id=input_id,
                    digest=digest,
                    length=length,
                    kind_mime=media_type,
                    order=local_input.order,
                )
            )
        if self.request.kind == "job":
            rows += [
                pb.InputBinding(
                    input_id=f"{grants_module.MODEL_PREFIX}{parameter}",
                    digest=manifest,
                    length=length,
                    kind_mime=grants_module.MODEL_MIME,
                    order=0,
                )
                for parameter, (manifest, length) in sorted(self.request.models.items())
            ]
        return rows

    def _grant(self, body: bytes, invocation_digest: bytes) -> pb.DeliveryGrant:
        """Refreshable access only; identity lives in the spec's bindings. A local grant
        never expires: every destination is a file this process owns."""
        if self.request.payload_url:
            payload_url = self.request.payload_url
        else:
            path = self.grant_dir / (
                opaque_key("local-payload-grant", self.request.request_id, self.request.attempt)
                + ".json"
            )
            path.write_bytes(body)
            payload_url = f"file://{path}"
        inputs = [pb.InputAccess(input_id="payload", url=payload_url)]
        inputs += [
            pb.InputAccess(
                input_id=f"{grants_module.TREE_PREFIX}{ref}", url=f"file://{Path(root).resolve()}"
            )
            for ref, root in sorted(self.request.trees.items())
        ]
        inputs += [
            pb.InputAccess(input_id=input_id, url=f"file://{Path(item.source).resolve()}")
            for input_id, item in sorted(self.request.inputs.items())
        ]
        if self.request.kind == "job":
            inputs += [
                pb.InputAccess(
                    input_id=f"{grants_module.MODEL_PREFIX}{parameter}", url=f"model://{manifest}"
                )
                for parameter, (manifest, _length) in sorted(self.request.models.items())
            ]
        return pb.DeliveryGrant(
            invocation_spec_digest=invocation_digest,
            credential=pb.DeliveryAccessCredential(
                issuer="cozy-runtime-local",
                key_id="local",
                credential_epoch=1,
                expires_at_unix=0,
                token=b"local",
            ),
            file_base_url=f"file://{self.grant_dir}",
            expires_at_unix=0,
            inputs=inputs,
            outputs=[
                pb.OutputAccess(output_id=output_id, url=f"file://{self.destination(output_id)}")
                for output_id in sorted(self.request.outputs)
            ],
        )

    def _filename(self, output_id: str) -> str:
        key = opaque_key("local-output", self.request.request_id, self.request.attempt, output_id)
        return key + ".bin"


def _input_facts(path: Path) -> tuple[str, int, str]:
    """Stream one local input into its exact digest, length, and media claim."""
    digest = hashlib.sha256()
    length = 0
    head = bytearray()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            length += len(chunk)
            if len(head) < SNIFF_BYTES:
                head.extend(chunk[: SNIFF_BYTES - len(head)])
    return (
        "sha256:" + digest.hexdigest(),
        length,
        sniff(bytes(head)) or "application/octet-stream",
    )
