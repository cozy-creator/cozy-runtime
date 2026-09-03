"""Worker identity verification: the canonical private-rental Claim transcript.

Runtime downloads nothing (cr-090): model bytes reach the local TensorFS Store
through TensorFS's one transport, and operation wheels through pod-supervisor.
There is no HTTP client, origin, TLS key use, URL, or download plan here.
"""

from __future__ import annotations

import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class DownloadRefusal(Exception):
    """A typed Claim refusal."""

    def __init__(self, kind: pb.FaultKind, code: str, detail: str, *, server_time: int = 0) -> None:
        super().__init__(f"{code}: {detail}")
        self.kind = kind
        self.code = code
        self.detail = detail
        self.server_time = server_time


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
