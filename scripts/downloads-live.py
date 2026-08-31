#!/usr/bin/env python3
"""Live ClaimProof and package/model download-resolution proof."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from cozy_runtime.internal.config import (
    ConfigError,
    read_worker_host_config,
)
from cozy_runtime.internal.worker import acquire, downloads
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def main() -> int:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    for unknown in ("COZY_UNKNOWN", "TENSORHUB_CA_DER_B64URL", "TENSORHUB_ORIGIN"):
        try:
            read_worker_host_config({unknown: "unknown"})
        except ConfigError as exc:
            assert f"unknown worker environment variable {unknown}" == exc.message
        else:
            raise AssertionError(f"worker admitted {unknown}")
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
            "SSL_CERT_FILE": "/run/cozy/tensorhub-ca.pem",
        }
    )
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
        "worker downloads: ClaimProof, credential-less Runtime host config, and verified "
        "supervisor-supplied local reuse passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
