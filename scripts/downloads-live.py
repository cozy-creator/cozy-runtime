#!/usr/bin/env python3
"""Live ClaimProof and package/model download-resolution proof."""

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

from cozy_runtime.internal.config import (
    ConfigError,
    parse_tensorhub_origin,
    read_worker_host_config,
)
from cozy_runtime.internal.worker import acquire, downloads
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
    for unknown in ("COZY_UNKNOWN", "TENSORHUB_CA_DER_B64URL"):
        try:
            read_worker_host_config({unknown: "unknown"})
        except ConfigError as exc:
            assert f"unknown worker environment variable {unknown}" == exc.message
        else:
            raise AssertionError(f"worker admitted {unknown}")
    origin = "https://cozy-dev.ngrok.example"
    host_config = read_worker_host_config(
        {
            "COZY_WORKER_ID": "worker-7",
            "COZY_WORKER_INTERNAL_PORT": "18443",
            "COZY_MEDIA_INTERNAL_PORT": "18444",
            "COZY_WORKER_LEASE_EXPIRY_UNIX": str(int(time.time()) + 600),
            "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
                {
                    "control_public_key_ed25519_b64url": base64.urlsafe_b64encode(public)
                    .rstrip(b"=")
                    .decode(),
                    "media_token_sha256": ["1" * 64],
                }
            ),
            "TENSORHUB_ORIGIN": origin,
            "SSL_CERT_FILE": "/run/cozy/tensorhub-ca.pem",
        }
    )
    assert host_config.tensorhub_origin.value == origin
    assert dict(host_config.child_base_env)["SSL_CERT_FILE"] == "/run/cozy/tensorhub-ca.pem"
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
    downloads.verify_claim_proof(
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
        downloads.verify_claim_proof(
            planted,
            public,
            worker_id="worker-7",
            worker_boot_id="boot-7",
            worker_tls_certificate_digest=tls_digest,
        )
    except downloads.DownloadRefusal as exc:
        assert exc.code == "claim_proof_signature_invalid"
    else:
        raise AssertionError("changed ClaimProof retained authority")

    now = int(time.time())
    file_digest = "sha256:" + "1" * 64
    connection = Connection(
        Response(
            200,
            {
                "files": [
                    {
                        "digest": file_digest,
                        "kind": "package",
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
    with tempfile.TemporaryDirectory(prefix="cozy-worker-downloads-") as raw:
        root = Path(raw)
        client = downloads.DownloadClient(
            root / "tls.crt",
            root / "tls.key",
            parse_tensorhub_origin("https://tensorhub.example:18443"),
            connection=lambda _timeout: connection,
        )
        plan = client.resolve(set_digest)
    assert list(plan.files) == [file_digest]
    assert connection.closed and connection.request_value is not None
    method, request_path, request_raw, headers = connection.request_value
    assert (method, request_path) == ("POST", "/v1/worker/downloads/resolve")
    assert headers == {"Accept": "application/json", "Content-Type": "application/json"}
    assert json.loads(request_raw) == {"placement_set_digest": documents.spell(set_digest)}
    assert set(json.loads(request_raw)) == {"placement_set_digest"}

    refused = Connection(
        Response(
            409,
            {
                "error": {
                    "code": "worker_downloads.placement_set_mismatch",
                    "message": "digest is not the accepted set",
                },
                "server_time_unix": now,
            },
        )
    )
    client = downloads.DownloadClient(
        Path("unused.crt"),
        Path("unused.key"),
        parse_tensorhub_origin("https://tensorhub.example"),
        connection=lambda _timeout: refused,
    )
    try:
        client.resolve(set_digest)
    except downloads.DownloadRefusal as exc:
        assert exc.code == "worker_downloads.placement_set_mismatch"
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
            downloads.DownloadPlan(),
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
        "worker downloads: ClaimProof, one-field PlacementSet resolve, foreign-set refusal, "
        "and verified local reuse passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
