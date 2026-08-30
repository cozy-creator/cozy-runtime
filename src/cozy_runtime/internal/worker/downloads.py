"""The worker's one direct, mTLS-authenticated Tensorhub download client."""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cozy_runtime.internal import pull
from cozy_runtime.internal.config import TensorhubOrigin
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

WORKER_DOWNLOADS_PATH = "/v1/worker/downloads/resolve"
MAX_REQUEST_BYTES = 128 << 10
MAX_RESPONSE_BYTES = 8 << 20
MAX_FILES = 4096
RESOLVE_TIMEOUT_SECONDS = 30

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class DownloadRefusal(Exception):
    """A typed Claim or package/model download refusal."""

    def __init__(self, kind: pb.FaultKind, code: str, detail: str, *, server_time: int = 0) -> None:
        super().__init__(f"{code}: {detail}")
        self.kind = kind
        self.code = code
        self.detail = detail
        self.server_time = server_time


@dataclass(frozen=True, slots=True)
class DownloadFile:
    digest: str
    length: int
    kind: str
    url: str
    expires_at_unix: int


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    server_time_unix: int = 0
    expires_at_unix: int = 0
    files: Mapping[str, DownloadFile] = field(default_factory=dict)
    allow_local: bool = False

    @property
    def subjects(self) -> dict[str, int]:
        return {digest: row.length for digest, row in self.files.items()}

    @property
    def locations(self) -> dict[str, str]:
        return {digest: row.url for digest, row in self.files.items()}

    def authorized(self) -> bool:
        return bool(self.files)

    def merged(self, newer: DownloadPlan) -> DownloadPlan:
        rows = dict(self.files)
        rows.update(newer.files)
        return DownloadPlan(
            server_time_unix=max(self.server_time_unix, newer.server_time_unix),
            expires_at_unix=max(self.expires_at_unix, newer.expires_at_unix),
            files=dict(sorted(rows.items())),
            allow_local=self.allow_local or newer.allow_local,
        )

    def deadline(self, digest: str) -> int:
        row = self.files.get(digest)
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
        raise DownloadRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            "claim_proof_binding_mismatch",
            "Claim does not name this worker and boot",
        )
    try:
        tls_digest = documents.raw(worker_tls_certificate_digest)
    except documents.DocumentError as exc:
        raise DownloadRefusal(
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


def _verify_signature(public_key: bytes, signature: bytes, payload: bytes, subject: str) -> None:
    if len(public_key) != 32 or len(signature) != 64:
        raise DownloadRefusal(
            pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
            f"{subject}_signature_invalid",
            "the configured Ed25519 key or carried signature has the wrong length",
        )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, payload)
    except (InvalidSignature, ValueError) as exc:
        raise DownloadRefusal(
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


class DownloadClient:
    """Resolve one active PlacementSet at Tensorhub's mTLS-only worker authority.

    The pinned client certificate identifies the rental, worker and boot. Tensorhub owns the
    accepted PlacementSet and derives its complete package/model file closure. Restating either
    identity or inventory in this request would create a second download request.
    """

    def __init__(
        self,
        certificate: Path,
        private_key: Path,
        origin: TensorhubOrigin,
        *,
        connection: Callable[[float], _Connection] | None = None,
    ) -> None:
        self.certificate = certificate
        self.private_key = private_key
        self.origin = origin
        self.connection = connection

    def resolve(self, placement_set_digest: bytes) -> DownloadPlan:
        try:
            spelled = documents.spell(placement_set_digest)
        except documents.DocumentError as exc:
            raise DownloadRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "worker_downloads.placement_set_digest_invalid",
                exc.detail,
            ) from exc
        payload = json.dumps(
            {"placement_set_digest": spelled},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(payload) > MAX_REQUEST_BYTES:
            raise DownloadRefusal(
                pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                "worker_downloads.request_too_large",
                f"{len(payload)} bytes",
            )
        connection = self._connection(float(RESOLVE_TIMEOUT_SECONDS))
        try:
            connection.request(
                "POST",
                WORKER_DOWNLOADS_PATH,
                payload,
                {"Accept": "application/json", "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise DownloadRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "worker_downloads.unreachable",
                type(exc).__name__,
            ) from exc
        finally:
            connection.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise DownloadRefusal(
                pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                "worker_downloads.response_too_large",
                f"more than {MAX_RESPONSE_BYTES} bytes",
            )
        value = _json_object(raw)
        if response.status != 200:
            raise _error_response(value, response.status)
        return _plan(value, allow_local=pull.local_hub(self.origin.value))

    def _connection(self, timeout: float) -> _Connection:
        if self.connection is not None:
            return self.connection(timeout)
        context = ssl.create_default_context()
        context.load_cert_chain(self.certificate, self.private_key)
        return http.client.HTTPSConnection(
            self.origin.host,
            self.origin.port,
            timeout=timeout,
            context=context,
        )


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownloadRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "worker_downloads.response_invalid",
            "response is not JSON",
        ) from exc
    if not isinstance(value, dict):
        raise DownloadRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "worker_downloads.response_invalid",
            "response is not an object",
        )
    return value


def _error_response(value: Mapping[str, Any], status: int) -> DownloadRefusal:
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
        return DownloadRefusal(
            pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
            "worker_downloads.error_invalid",
            f"HTTP {status}",
        )
    code = str(error["code"])
    return DownloadRefusal(
        pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
        code,
        str(error["message"]),
        server_time=server_time,
    )


def _plan(value: Mapping[str, Any], *, allow_local: bool = False) -> DownloadPlan:
    if set(value) != {"files", "expires_at_unix", "server_time_unix"}:
        raise _plan_invalid("fields")
    server_time = value.get("server_time_unix")
    expires = value.get("expires_at_unix")
    rows = value.get("files")
    if (
        not isinstance(server_time, int)
        or isinstance(server_time, bool)
        or server_time <= 0
        or not isinstance(expires, int)
        or isinstance(expires, bool)
        or not server_time < expires
        or not isinstance(rows, list)
        or not 1 <= len(rows) <= MAX_FILES
    ):
        raise _plan_invalid("header")
    files: dict[str, DownloadFile] = {}
    prior = ""
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"digest", "kind", "length", "url"}:
            raise _plan_invalid("file fields")
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
            raise _plan_invalid("file value")
        files[digest] = DownloadFile(digest, length, kind, url, expires)
        prior = digest
    return DownloadPlan(server_time, expires, files, allow_local)


def _plan_invalid(detail: str) -> DownloadRefusal:
    return DownloadRefusal(
        pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
        "worker_downloads.plan_invalid",
        detail,
    )


def tls_certificate_digest(der: bytes) -> str:
    return "sha256:" + hashlib.sha256(der).hexdigest()
