"""Cozy-signed artifact intent and the worker's one direct Tensorhub client."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import re
import ssl
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cozy_runtime.internal import pull
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

WORKER_TENSORHUB_HOST = "worker.tensorhub.com"
WORKER_TENSORHUB_PATH = "/v1/worker/artifacts/resolve"
MAX_REQUEST_BYTES = 128 << 10
MAX_RESPONSE_BYTES = 8 << 20
MAX_ARTIFACTS = 4096
MAX_PACKAGE_IDS = 64
MAX_MODEL_IDS = 256
RESOLVE_TIMEOUT_SECONDS = 30

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ArtifactAccessRefusal(Exception):
    """A typed intent, authorization, or direct-resolve refusal."""

    def __init__(self, kind: pb.FaultKind, code: str, detail: str, *, server_time: int = 0) -> None:
        super().__init__(f"{code}: {detail}")
        self.kind = kind
        self.code = code
        self.detail = detail
        self.server_time = server_time


@dataclass(frozen=True, slots=True)
class ArtifactIntent:
    canonical_bytes: bytes
    signature: bytes
    rental_id: str
    worker_id: str
    worker_boot_id: str
    worker_tls_certificate_digest: str
    revision: int
    package_release_ids: tuple[str, ...]
    model_checkpoint_ids: tuple[str, ...]
    delegation_id: str
    expires_at_unix: int

    def logical_identity(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.package_release_ids, self.model_checkpoint_ids


@dataclass(frozen=True, slots=True)
class PlanArtifact:
    digest: str
    length: int
    kind: str
    url: str
    expires_at_unix: int


@dataclass(frozen=True, slots=True)
class ArtifactPlan:
    server_time_unix: int = 0
    expires_at_unix: int = 0
    artifacts: Mapping[str, PlanArtifact] = field(default_factory=dict)
    allow_local: bool = False

    @property
    def subjects(self) -> dict[str, int]:
        return {digest: row.length for digest, row in self.artifacts.items()}

    @property
    def locations(self) -> dict[str, str]:
        return {digest: row.url for digest, row in self.artifacts.items()}

    def authorized(self) -> bool:
        return bool(self.artifacts)

    def merged(self, newer: ArtifactPlan) -> ArtifactPlan:
        rows = dict(self.artifacts)
        rows.update(newer.artifacts)
        return ArtifactPlan(
            server_time_unix=max(self.server_time_unix, newer.server_time_unix),
            expires_at_unix=max(self.expires_at_unix, newer.expires_at_unix),
            artifacts=dict(sorted(rows.items())),
            allow_local=self.allow_local or newer.allow_local,
        )

    def deadline(self, digest: str) -> int:
        row = self.artifacts.get(digest)
        return row.expires_at_unix if row is not None else 0


def verify_claim_proof(
    claim: pb.Claim,
    public_key: bytes,
    *,
    worker_id: str,
    worker_boot_id: str,
    worker_tls_certificate_digest: str,
) -> None:
    """Verify the canonical private-rental Claim transcript."""

    if claim.worker_id != worker_id or claim.worker_boot_id != worker_boot_id:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "claim_proof_binding_mismatch",
            "Claim does not name this worker and boot",
        )
    try:
        tls_digest = documents.raw(worker_tls_certificate_digest)
    except documents.DocumentError as exc:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "claim_proof_tls_digest_invalid",
            exc.detail,
        ) from exc
    transcript = documents.canonical_bytes(
        pb.ClaimProof(
            record_owner_epoch=claim.record_owner_epoch,
            worker_boot_id=worker_boot_id,
            worker_id=worker_id,
            worker_tls_certificate_digest=tls_digest,
        )
    )
    _verify_signature(public_key, claim.proof, transcript, "claim_proof")


def read_intent(
    update: pb.EnsureArtifacts,
    public_key: bytes,
    *,
    worker_id: str,
    worker_boot_id: str,
    worker_tls_certificate_digest: str,
) -> ArtifactIntent:
    """Verify one exact delegation before trusting or forwarding any field."""

    _verify_signature(
        public_key,
        update.creator_signature,
        update.delegation_canonical_bytes,
        "artifact_delegation",
    )
    try:
        value = documents.read(update.delegation_canonical_bytes, pb.ArtifactDelegation)
    except documents.DocumentError as exc:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            f"artifact_delegation_{exc.code}",
            exc.detail,
        ) from exc
    required = (
        "rental_id",
        "worker_id",
        "worker_boot_id",
        "worker_tls_certificate_digest",
        "revision",
        "delegation_id",
        "expires_at_unix",
    )
    if any(not value.get(name) for name in required):
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "artifact_delegation_incomplete",
            "delegation identity, revision, and expiry are required",
        )
    packages = _logical_ids(
        value.get("package_release_ids"), "package_release_ids", MAX_PACKAGE_IDS
    )
    models = _logical_ids(value.get("model_checkpoint_ids"), "model_checkpoint_ids", MAX_MODEL_IDS)
    if not packages and not models:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "artifact_delegation_empty",
            "a delegation names at least one package or model",
        )
    if any(_DIGEST.fullmatch(model) is None for model in models):
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "artifact_model_id_invalid",
            "model checkpoint ids are exact sha256 identities",
        )
    tls_digest = str(value["worker_tls_certificate_digest"])
    bindings = {
        "worker_id": (str(value["worker_id"]), worker_id),
        "worker_boot_id": (str(value["worker_boot_id"]), worker_boot_id),
        "worker_tls_certificate_digest": (tls_digest, worker_tls_certificate_digest),
    }
    mismatch = next((name for name, (got, wanted) in bindings.items() if got != wanted), "")
    if mismatch:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "artifact_delegation_binding_mismatch",
            mismatch,
        )
    return ArtifactIntent(
        canonical_bytes=bytes(update.delegation_canonical_bytes),
        signature=bytes(update.creator_signature),
        rental_id=str(value["rental_id"]),
        worker_id=worker_id,
        worker_boot_id=worker_boot_id,
        worker_tls_certificate_digest=tls_digest,
        revision=int(value["revision"]),
        package_release_ids=packages,
        model_checkpoint_ids=models,
        delegation_id=str(value["delegation_id"]),
        expires_at_unix=int(value["expires_at_unix"]),
    )


def _logical_ids(value: object, name: str, maximum: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or not all(isinstance(item, str) and 0 < len(item) <= 256 for item in value)
        or value != sorted(set(value))
    ):
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "artifact_delegation_ids_invalid",
            f"{name} is a bounded sorted unique string set",
        )
    return tuple(value)


def _verify_signature(public_key: bytes, signature: bytes, payload: bytes, subject: str) -> None:
    if len(public_key) != 32 or len(signature) != 64:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            f"{subject}_signature_invalid",
            "the configured Ed25519 key or carried signature has the wrong length",
        )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError) as exc:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            f"{subject}_signature_invalid",
            f"{subject} was not signed by this rental's Cozy key",
        ) from exc


class _Response(Protocol):
    status: int

    def read(self, amount: int | None = None) -> bytes: ...


class _Connection(Protocol):
    def request(self, method: str, url: str, body: bytes, headers: Mapping[str, str]) -> None: ...

    def getresponse(self) -> _Response: ...

    def close(self) -> None: ...


class ArtifactClient:
    """Redeem signed intent at Tensorhub's fixed mTLS-only worker authority."""

    def __init__(
        self,
        certificate: Path,
        private_key: Path,
        *,
        connection: Callable[[float], _Connection] | None = None,
    ) -> None:
        self.certificate = certificate
        self.private_key = private_key
        self.connection = connection

    def resolve(
        self,
        intent: ArtifactIntent,
        *,
        package_release_ids: Sequence[str],
        model_checkpoint_ids: Sequence[str],
    ) -> ArtifactPlan:
        packages = tuple(package_release_ids)
        models = tuple(model_checkpoint_ids)
        if packages != tuple(sorted(set(packages))) or models != tuple(sorted(set(models))):
            raise ArtifactAccessRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "artifact_resolve_subset_invalid",
                "requested logical ids are sorted unique sets",
            )
        if not set(packages).issubset(intent.package_release_ids) or not set(models).issubset(
            intent.model_checkpoint_ids
        ):
            raise ArtifactAccessRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "artifact_resolve_subset_invalid",
                "requested logical ids exceed the signed delegation",
            )
        payload = json.dumps(
            {
                "creator_signature": base64.urlsafe_b64encode(intent.signature)
                .rstrip(b"=")
                .decode("ascii"),
                "delegation_canonical_bytes": base64.b64encode(intent.canonical_bytes).decode(
                    "ascii"
                ),
                "model_checkpoint_ids": list(models),
                "package_release_ids": list(packages),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(payload) > MAX_REQUEST_BYTES:
            raise ArtifactAccessRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "artifact_resolve_request_too_large",
                f"{len(payload)} bytes",
            )
        now = int(time.time())
        if now >= intent.expires_at_unix:
            raise ArtifactAccessRefusal(
                pb.FaultKind.FAULT_KIND_AUTHORIZATION_EXPIRED,
                "worker_artifacts.delegation_expired",
                f"delegation expired at {intent.expires_at_unix}",
            )
        timeout = float(max(1, min(RESOLVE_TIMEOUT_SECONDS, intent.expires_at_unix - now)))
        connection = self._connection(timeout)
        try:
            connection.request(
                "POST",
                WORKER_TENSORHUB_PATH,
                payload,
                {"Accept": "application/json", "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise ArtifactAccessRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "worker_artifacts.unreachable",
                type(exc).__name__,
            ) from exc
        finally:
            connection.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ArtifactAccessRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "worker_artifacts.response_too_large",
                f"more than {MAX_RESPONSE_BYTES} bytes",
            )
        value = _json_object(raw)
        if response.status != 200:
            raise _error_response(value, response.status)
        return _plan(
            value,
            intent,
            allow_local=pull.local_hub(f"https://{WORKER_TENSORHUB_HOST}"),
        )

    def _connection(self, timeout: float) -> _Connection:
        if self.connection is not None:
            return self.connection(timeout)
        context = ssl.create_default_context()
        context.load_cert_chain(self.certificate, self.private_key)
        return http.client.HTTPSConnection(
            WORKER_TENSORHUB_HOST,
            443,
            timeout=timeout,
            context=context,
        )


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "worker_artifacts.response_invalid",
            "response is not JSON",
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "worker_artifacts.response_invalid",
            "response is not an object",
        )
    return value


def _error_response(value: Mapping[str, Any], status: int) -> ArtifactAccessRefusal:
    error = value.get("error")
    server_time = value.get("server_time_unix")
    if (
        set(value) != {"error", "server_time_unix"}
        or not isinstance(error, dict)
        or not {"code", "message"}.issubset(error)
        or not set(error).issubset({"code", "message", "remedy"})
        or not isinstance(error.get("code"), str)
        or not isinstance(error.get("message"), str)
        or not isinstance(server_time, int)
        or isinstance(server_time, bool)
        or server_time <= 0
    ):
        return ArtifactAccessRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "worker_artifacts.error_invalid",
            f"HTTP {status}",
        )
    code = str(error["code"])
    return ArtifactAccessRefusal(
        (
            pb.FaultKind.FAULT_KIND_AUTHORIZATION_EXPIRED
            if code == "worker_artifacts.delegation_expired"
            else pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED
        ),
        code,
        str(error["message"]),
        server_time=server_time,
    )


def _plan(
    value: Mapping[str, Any], intent: ArtifactIntent, *, allow_local: bool = False
) -> ArtifactPlan:
    if set(value) != {"artifacts", "expires_at_unix", "server_time_unix"}:
        raise _plan_invalid("fields")
    server_time = value.get("server_time_unix")
    expires = value.get("expires_at_unix")
    rows = value.get("artifacts")
    if (
        not isinstance(server_time, int)
        or isinstance(server_time, bool)
        or server_time <= 0
        or not isinstance(expires, int)
        or isinstance(expires, bool)
        or not server_time < expires <= intent.expires_at_unix
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_ARTIFACTS
    ):
        raise _plan_invalid("header")
    artifacts: dict[str, PlanArtifact] = {}
    prior = ""
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"digest", "kind", "length", "url"}:
            raise _plan_invalid("artifact fields")
        digest = row.get("digest")
        length = row.get("length")
        kind = row.get("kind")
        url = row.get("url")
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or digest <= prior
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            or not isinstance(kind, str)
            or not 0 < len(kind) <= 64
            or not isinstance(url, str)
            or not url.startswith("https://")
        ):
            raise _plan_invalid("artifact value")
        artifacts[digest] = PlanArtifact(digest, length, kind, url, expires)
        prior = digest
    return ArtifactPlan(server_time, expires, artifacts, allow_local)


def _plan_invalid(detail: str) -> ArtifactAccessRefusal:
    return ArtifactAccessRefusal(
        pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
        "worker_artifacts.plan_invalid",
        detail,
    )


def tls_certificate_digest(der: bytes) -> str:
    return "sha256:" + hashlib.sha256(der).hexdigest()
