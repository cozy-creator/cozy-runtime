#!/usr/bin/env python3
"""Live ClaimProof and PlacementSet-digest artifact-resolution proof."""

from __future__ import annotations

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
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    tls_digest = "sha256:" + hashlib.sha256(b"worker leaf DER").hexdigest()
    claim = pb.Claim(record_owner_epoch=7, worker_boot_id="boot-7", worker_id="worker-7")
    claim.proof = private.sign(
        documents.canonical_bytes(
            pb.ClaimProof(
                record_owner_epoch=7,
                worker_boot_id="boot-7",
                worker_id="worker-7",
                worker_tls_certificate_digest=documents.raw(tls_digest),
            )
        )
    )
    artifact_access.verify_claim_proof(
        claim,
        public,
        worker_id="worker-7",
        worker_boot_id="boot-7",
        worker_tls_certificate_digest=tls_digest,
    )
    planted = pb.Claim()
    planted.CopyFrom(claim)
    planted.record_owner_epoch += 1
    try:
        artifact_access.verify_claim_proof(
            planted,
            public,
            worker_id="worker-7",
            worker_boot_id="boot-7",
            worker_tls_certificate_digest=tls_digest,
        )
    except artifact_access.ArtifactAccessRefusal as exc:
        assert exc.code == "claim_proof_signature_invalid"
    else:
        raise AssertionError("changed ClaimProof retained authority")

    now = int(time.time())
    artifact_digest = "sha256:" + "1" * 64
    connection = Connection(
        Response(
            200,
            {
                "artifacts": [
                    {
                        "digest": artifact_digest,
                        "kind": "access-only",
                        "length": 12,
                        "url": "https://r2.example.invalid/object",
                    }
                ],
                "expires_at_unix": now + 60,
                "server_time_unix": now,
            },
        )
    )
    set_digest = hashlib.sha256(b"exact PlacementSet/1").digest()
    with tempfile.TemporaryDirectory(prefix="cozy-artifact-access-") as raw:
        root = Path(raw)
        client = artifact_access.ArtifactClient(
            root / "tls.crt",
            root / "tls.key",
            parse_worker_artifact_origin("https://worker.tensorhub.com:18443"),
            connection=lambda _timeout: connection,
        )
        plan = client.resolve(set_digest)
    assert list(plan.artifacts) == [artifact_digest]
    assert connection.closed and connection.request_value is not None
    method, request_path, request_raw, headers = connection.request_value
    assert (method, request_path) == ("POST", "/v1/worker/artifacts/resolve")
    assert headers == {"Accept": "application/json", "Content-Type": "application/json"}
    assert json.loads(request_raw) == {"placement_set_digest": documents.spell(set_digest)}
    assert set(json.loads(request_raw)) == {"placement_set_digest"}

    refused = Connection(
        Response(
            409,
            {
                "error": {
                    "code": "worker_artifacts.placement_set_mismatch",
                    "message": "digest is not the accepted set",
                },
                "server_time_unix": now,
            },
        )
    )
    client = artifact_access.ArtifactClient(
        Path("unused.crt"),
        Path("unused.key"),
        parse_worker_artifact_origin("https://worker.tensorhub.com"),
        connection=lambda _timeout: refused,
    )
    try:
        client.resolve(set_digest)
    except artifact_access.ArtifactAccessRefusal as exc:
        assert exc.code == "worker_artifacts.placement_set_mismatch"
        assert exc.kind == pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED
    else:
        raise AssertionError("foreign PlacementSet was authorized")

    body = b"already verified exact bytes"
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    with tempfile.TemporaryDirectory(prefix="cozy-artifact-cache-") as raw:
        cache = Path(raw) / "cache"
        cache.mkdir()
        path = cache / digest[7:]
        path.write_bytes(body)
        acquirer = acquire.Acquirer(
            artifact_access.ArtifactPlan(),
            cache_root=cache,
            environment_root=Path(raw) / "environment",
            selection_root=Path(raw) / "selection",
            artifact_store=Path(raw) / "store",
            python=Path("/usr/bin/python3"),
        )
        assert acquirer.fetch(digest, len(body)) == path
        assert acquirer.fetch(digest, len(body)) == path
        assert acquirer.verified_bytes == len(body) and acquirer.fetched_bytes == 0

    print(
        "artifact access: ClaimProof, one-field PlacementSet resolve, foreign-set refusal, "
        "and verified local reuse passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
