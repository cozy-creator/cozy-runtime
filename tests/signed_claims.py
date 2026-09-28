"""The RecordOwner's signed Claim: the one lane a TCP listener admits."""

from __future__ import annotations

from typing import Any, TypedDict

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

#: Fixed, so a test and the worker process it spawns agree on it.
KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32, 64)))
PUBLIC_KEY = KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
WORKER_ID = "test-worker"
BOOT_ID = "test-boot"
TLS_DIGEST = "sha256:" + "a" * 64


class Identity(TypedDict):
    """The WorkerOptions a signed Claim is bound to, as a Host passes them."""

    worker_id: str
    worker_boot_id: str
    worker_tls_certificate_digest: str


IDENTITY = Identity(
    worker_id=WORKER_ID, worker_boot_id=BOOT_ID, worker_tls_certificate_digest=TLS_DIGEST
)


def claim(owner: str = "owner", epoch: int = 1, boot: str = BOOT_ID, **fields: Any) -> pb.Claim:
    """A Claim the worker booted with IDENTITY accepts from `owner` at `epoch`."""
    signed = pb.Claim(
        record_owner_id=owner,
        record_owner_epoch=epoch,
        worker_id=WORKER_ID,
        worker_boot_id=boot,
        wire_minor=WIRE_MINOR,
        **fields,
    )
    signed.proof = KEY.sign(
        documents.canonical_bytes(
            pb.ClaimProof(
                record_owner_epoch=epoch,
                worker_boot_id=boot,
                worker_id=WORKER_ID,
                worker_tls_certificate_digest=documents.raw(TLS_DIGEST),
            )
        )
    )
    return signed
