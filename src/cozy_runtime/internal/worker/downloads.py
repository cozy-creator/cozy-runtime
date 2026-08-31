"""Worker identity verification and transitional local download facts.

Tensorhub/R2 networking is owned by pod-supervisor. Runtime receives only
verified local paths; there is no HTTP client, origin, TLS key use, or URL
redemption in this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class DownloadRefusal(Exception):
    """A typed Claim or supervisor-supplied local-file refusal."""

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
    """Transitional in-memory input for the existing placement acquirer.

    Pod workers no longer construct this from URLs. It remains while local
    Creator and the PreparedPackageSet handoff converge on verified paths.
    """

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


def tls_certificate_digest(der: bytes) -> str:
    return "sha256:" + hashlib.sha256(der).hexdigest()
