"""Delegated claims preserve their role through real Runtime process replacement."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

WORKER = r"""
import hashlib, json, sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cozy_runtime.internal.config import RuntimeConfig, Credentials
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents, worker_pb2 as pb
root=Path(sys.argv[1]); mode=sys.argv[2]
key=Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
pod=Ed25519PrivateKey.from_private_bytes(bytes(range(1,33)))
boot='same-pod-boot'; leaf=b'\xaa'*32
worker=Worker(RuntimeConfig(cozy_home=root/'home',credentials=Credentials(),
 record_owner_public_key=key.public_key().public_bytes_raw()),
 WorkerOptions(root=root/'worker',tensorfs_root=root/'store',worker_id='execution-proof',
 worker_boot_id=boot,ownership_required=mode!='first',
 worker_tls_certificate_digest='sha256:'+leaf.hex()),InMemoryControlHost())
try:
 grant=pb.ExecutionOwnerGrant(record_owner_epoch=2,record_owner_id='pod-coordinator',
 worker_id='execution-proof',worker_boot_id=boot,worker_tls_certificate_digest=leaf,
 execution_public_key=pod.public_key().public_bytes_raw(),initial_capsule_digest=b'\xbb'*32)
 if mode=='changed':grant.initial_capsule_digest=b'\xcc'*32
 if mode=='higher':grant.record_owner_epoch=3
 raw=documents.canonical_bytes(grant); digest=hashlib.sha256(raw).digest()
 claim=pb.Claim(record_owner_epoch=grant.record_owner_epoch,record_owner_id=grant.record_owner_id,
 worker_id=grant.worker_id,worker_boot_id=boot,wire_minor=WIRE_MINOR,
 execution_owner=pb.SignedExecutionOwnerGrant(grant=grant,signature=key.sign(raw)))
 claim.proof=pod.sign(documents.canonical_bytes(pb.ExecutionClaimProof(
 execution_grant_digest=digest,record_owner_epoch=claim.record_owner_epoch,
 record_owner_id=claim.record_owner_id,worker_id=claim.worker_id,worker_boot_id=boot,
 worker_tls_certificate_digest=leaf)))
 if mode=='owner-tamper':claim.record_owner_id='foreign-coordinator'
 if mode=='grant-forged':claim.execution_owner.signature=bytes(64)
 if mode=='proof-forged':claim.proof=bytes(64)
 if mode=='old-wire':claim.wire_minor=49
 if mode=='legacy':
  claim.ClearField('execution_owner');claim.record_owner_epoch=99
  claim.proof=key.sign(documents.canonical_bytes(pb.ClaimProof(record_owner_epoch=99,
  worker_id=claim.worker_id,worker_boot_id=boot,worker_tls_certificate_digest=leaf)))
 frames=[];worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]),frames.append)
 ack=next(x.claim_ack for x in frames if x.HasField('claim_ack'))
 authorized=False
 if ack.accepted:authorized=worker.authorize_workspace(claim)=='pod-coordinator'
 print('PROOF '+json.dumps({'accepted':ack.accepted,'stream':ack.control_stream_epoch,
 'rejection':pb.ClaimRejection.Name(ack.rejection),'workspace_authorized':authorized}),flush=True)
finally:worker.shutdown()
"""


def run(root: Path, mode: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(root), mode],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(
        next(line[6:] for line in result.stdout.splitlines() if line.startswith("PROOF "))
    )


def test_execution_role_is_durable_and_cannot_be_claimed_by_observer(tmp_path: Path) -> None:
    first = run(tmp_path, "first")
    assert first["accepted"] and first["workspace_authorized"] and first["stream"] == 1
    assert run(tmp_path, "same")["stream"] == 2
    path = tmp_path / "worker/ownership.json"
    held = path.read_bytes()
    for mode in (
        "legacy",
        "changed",
        "higher",
        "owner-tamper",
        "grant-forged",
        "proof-forged",
        "old-wire",
    ):
        result = run(tmp_path, mode)
        assert not result["accepted"], (mode, result)
        assert path.read_bytes() == held
    resumed = run(tmp_path, "same")
    assert resumed["accepted"] and resumed["stream"] == 3 and resumed["workspace_authorized"]
