#!/usr/bin/env python3
"""Live proto-013 intent/signature/direct-resolve acceptance and red arms."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from cozy_runtime.internal.config import parse_worker_artifact_origin
from cozy_runtime.internal.worker import acquire, artifact_access
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class Response:
    def __init__(self, status: int, value: object) -> None:
        self.status = status
        self.raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def read(self, amount: int | None = None) -> bytes:
        return self.raw if amount is None else self.raw[:amount]


class Connection:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.request_value: tuple[str, str, bytes, dict[str, str]] | None = None
        self.closed = False

    def request(self, method: str, url: str, body: bytes, headers: Any) -> None:
        self.request_value = method, url, body, dict(headers)

    def getresponse(self) -> Response:
        return self.response

    def close(self) -> None:
        self.closed = True


def main() -> int:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    worker_id = "worker-proto013"
    boot_id = "boot-proto013"
    tls_digest = "sha256:" + hashlib.sha256(b"worker leaf DER").hexdigest()

    claim = pb.Claim(
        record_owner_epoch=7,
        worker_boot_id=boot_id,
        worker_id=worker_id,
    )
    claim_bytes = documents.canonical_bytes(
        pb.ClaimProof(
            record_owner_epoch=claim.record_owner_epoch,
            worker_boot_id=boot_id,
            worker_id=worker_id,
            worker_tls_certificate_digest=documents.raw(tls_digest),
        )
    )
    claim.proof = private.sign(claim_bytes)
    artifact_access.verify_claim_proof(
        claim,
        public,
        worker_id=worker_id,
        worker_boot_id=boot_id,
        worker_tls_certificate_digest=tls_digest,
    )
    planted_claim = pb.Claim()
    planted_claim.CopyFrom(claim)
    planted_claim.record_owner_epoch += 1
    try:
        artifact_access.verify_claim_proof(
            planted_claim,
            public,
            worker_id=worker_id,
            worker_boot_id=boot_id,
            worker_tls_certificate_digest=tls_digest,
        )
    except artifact_access.ArtifactAccessRefusal as exc:
        assert exc.code == "claim_proof_signature_invalid"
    else:
        raise AssertionError("changed Claim epoch retained authority")

    now = 4_000_000_000
    delegation = pb.ArtifactDelegation(
        rental_id="rental-proto013",
        worker_id=worker_id,
        worker_boot_id=boot_id,
        worker_tls_certificate_digest=documents.raw(tls_digest),
        revision=9,
        package_release_ids=["cozy/marco-polo@v1"],
        model_checkpoint_ids=["sha256:" + "4" * 64],
        delegation_id="delegation-proto013",
        expires_at_unix=now,
    )
    delegation_bytes = documents.canonical_bytes(delegation)
    update = pb.EnsureArtifacts(
        delegation_canonical_bytes=delegation_bytes,
        creator_signature=private.sign(delegation_bytes),
    )
    intent = artifact_access.read_intent(
        update,
        public,
        worker_id=worker_id,
        worker_boot_id=boot_id,
        worker_tls_certificate_digest=tls_digest,
    )
    bad_update = pb.EnsureArtifacts()
    bad_update.CopyFrom(update)
    bad_update.creator_signature = bytes([bad_update.creator_signature[0] ^ 1]) + bytes(
        bad_update.creator_signature[1:]
    )
    try:
        artifact_access.read_intent(
            bad_update,
            public,
            worker_id=worker_id,
            worker_boot_id=boot_id,
            worker_tls_certificate_digest=tls_digest,
        )
    except artifact_access.ArtifactAccessRefusal as exc:
        assert exc.code == "artifact_delegation_signature_invalid"
    else:
        raise AssertionError("changed artifact signature was accepted")
    digest = "sha256:" + "1" * 64
    response = Response(
        200,
        {
            "artifacts": [
                {
                    "digest": digest,
                    "kind": "wheel",
                    "length": 12,
                    "url": "https://r2.example.invalid/object",
                }
            ],
            "expires_at_unix": now - 1,
            "server_time_unix": now - 60,
        },
    )
    connection = Connection(response)
    with tempfile.TemporaryDirectory(prefix="cozy-artifact-access-") as raw:
        root = Path(raw)
        certificate = root / "tls.crt"
        private_key = root / "tls.key"
        certificate.write_text("unused by injected connection")
        private_key.write_text("unused by injected connection")
        client = artifact_access.ArtifactClient(
            certificate,
            private_key,
            parse_worker_artifact_origin("https://worker.tensorhub.com:18443"),
            connection=lambda _timeout: connection,
        )
        plan = client.resolve(
            intent,
            package_release_ids=intent.package_release_ids,
            model_checkpoint_ids=intent.model_checkpoint_ids,
        )
        try:
            client.resolve(
                intent,
                package_release_ids=("foreign/package@v1",),
                model_checkpoint_ids=intent.model_checkpoint_ids,
            )
        except artifact_access.ArtifactAccessRefusal as exc:
            assert exc.code == "artifact_resolve_subset_invalid"
        else:
            raise AssertionError("worker expanded a signed logical set")
    assert list(plan.artifacts) == [digest]
    assert connection.closed and connection.request_value is not None
    method, path, request_raw, headers = connection.request_value
    assert (method, path) == ("POST", "/v1/worker/artifacts/resolve")
    assert headers == {"Accept": "application/json", "Content-Type": "application/json"}
    request = json.loads(request_raw)
    assert (
        base64.b64decode(request["delegation_canonical_bytes"], validate=True) == delegation_bytes
    )
    assert base64.urlsafe_b64decode(request["creator_signature"] + "==") == update.creator_signature

    expired_connection = Connection(
        Response(
            403,
            {
                "error": {
                    "code": "worker_artifacts.delegation_expired",
                    "message": "renew delegation",
                },
                "server_time_unix": now - 30,
            },
        )
    )
    expired_client = artifact_access.ArtifactClient(
        Path("unused.crt"),
        Path("unused.key"),
        parse_worker_artifact_origin("https://worker.tensorhub.com"),
        connection=lambda _timeout: expired_connection,
    )
    try:
        expired_client.resolve(
            intent,
            package_release_ids=intent.package_release_ids,
            model_checkpoint_ids=intent.model_checkpoint_ids,
        )
    except artifact_access.ArtifactAccessRefusal as exc:
        assert exc.code == "worker_artifacts.delegation_expired"
        assert exc.kind == pb.FaultKind.FAULT_KIND_AUTHORIZATION_EXPIRED
        assert exc.server_time == now - 30
    else:
        raise AssertionError("Tensorhub expiry refusal was admitted")

    # Per-object resume: one completed object survives a failed sibling and is never fetched
    # again after same-revision renewal.
    first_body, second_body = b"first exact object", b"second exact object"
    expiry = int(time.time()) + 60

    def row(name: str, body: bytes) -> artifact_access.PlanArtifact:
        digest_value = "sha256:" + hashlib.sha256(body).hexdigest()
        return artifact_access.PlanArtifact(
            digest_value,
            len(body),
            "evidence-only",
            f"https://127.0.0.1:1/{name}",
            expiry,
        )

    first_row, second_row = row("a", first_body), row("b", second_body)
    resume_plan = artifact_access.ArtifactPlan(
        int(time.time()),
        expiry,
        {first_row.digest: first_row, second_row.digest: second_row},
        True,
    )
    with tempfile.TemporaryDirectory(prefix="cozy-artifact-resume-") as raw:
        root = Path(raw)
        verified = acquire.VerifiedArtifactCache()
        cache = root / "cache"
        cache.mkdir()
        first_path = cache / first_row.digest.removeprefix("sha256:")
        first_path.write_bytes(first_body)
        first_path.chmod(0o400)
        verified.record(first_row.digest, first_row.length, first_path)

        def acquirer() -> acquire.Acquirer:
            return acquire.Acquirer(
                resume_plan,
                cache_root=root / "cache",
                environment_root=root / "environment",
                plans_root=root / "plans",
                documents_root=root / "documents",
                object_sets_root=root / "sets",
                artifact_store=root / "store",
                python=Path("/usr/bin/python3"),
                verified_cache=verified,
            )

        cold = acquirer()
        cold.fetch(first_row.digest, first_row.length)
        try:
            cold.fetch(second_row.digest, second_row.length)
        except acquire.AcquisitionRefusal:
            pass
        else:
            raise AssertionError("planted partial transfer did not fail")
        second_path = cache / second_row.digest.removeprefix("sha256:")
        second_path.write_bytes(second_body)
        second_path.chmod(0o400)
        resumed = acquirer()
        resumed.fetch(first_row.digest, first_row.length)
        resumed.fetch(second_row.digest, second_row.length)
        assert resumed.trusted_reuse_bytes == len(first_body)

    print(
        "artifact access: canonical Claim/Ensure signatures, exact fixed POST bytes, "
        "all-or-nothing plan, expiry/server-time refusal, and missing-only resume passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
