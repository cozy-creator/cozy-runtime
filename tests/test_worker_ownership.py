"""Signed Claims cross real Worker processes and the unchanged three-field fence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

WORKER = r"""
import json, sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cozy_runtime.internal.config import RuntimeConfig, Credentials
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents, worker_pb2 as pb
root=Path(sys.argv[1]); mode=sys.argv[2]; epoch=int(sys.argv[3]); owner=sys.argv[4]
key=Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
sole=mode.startswith('supervisor')
boot=mode if sole else 'different-boot' if mode=='wrong-boot' else 'same-pod-boot'
try:
    worker=Worker(RuntimeConfig(cozy_home=root/'home',credentials=Credentials(),
        record_owner_public_key=key.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)),
        WorkerOptions(root=root/'worker',tensorfs_root=root/'store',
            worker_id='fence-proof',worker_boot_id=boot,ownership_required=mode!='first',
            sole_supervisor=sole,process_incarnation=mode if sole else '',
            worker_tls_certificate_digest='sha256:'+'a'*64),InMemoryControlHost())
except LaunchRefusal as exc:
    print('PROOF '+json.dumps({'startup_refusal':exc.code}),flush=True)
    sys.exit(0)
try:
    claim=pb.Claim(record_owner_epoch=epoch,record_owner_id=owner,worker_id='fence-proof',
        worker_boot_id=boot,
        wire_minor=37 if mode=='legacy' else 60 if mode=='previous' else WIRE_MINOR)
    transcript=documents.canonical_bytes(pb.ClaimProof(record_owner_epoch=epoch,
        worker_boot_id=boot,worker_id='fence-proof',worker_tls_certificate_digest=b'\xaa'*32))
    claim.proof=key.sign(transcript) if not mode.endswith('forged') else bytes(64)
    authorized=''
    if sole:
        try: authorized=worker.authorize_workspace(claim)
        except ValueError: authorized='refused'
    frames=[]
    worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]),frames.append)
    ack=next(frame.claim_ack for frame in frames if frame.WhichOneof('msg')=='claim_ack')
    old=pb.CancelAttempt(record_owner_epoch=8,control_stream_epoch=1,worker_boot_id=boot)
    print('PROOF '+json.dumps({'accepted':ack.accepted,'stream':ack.control_stream_epoch,
        'rejection':pb.ClaimRejection.Name(ack.rejection),'old_fenced':worker.fenced(old),
        'authorized':authorized}),flush=True)
    if mode=='hold':sys.stdin.readline()
finally:
    worker.shutdown()
"""


def _run(root: Path, mode: str, epoch: int = 8, owner: str = "owner-A") -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(root), mode, str(epoch), owner],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    lines = [
        line.removeprefix("PROOF ")
        for line in result.stdout.splitlines()
        if line.startswith("PROOF ")
    ]
    assert len(lines) == 1, result.stdout + result.stderr
    value: dict[str, Any] = json.loads(lines[0])
    return value


def test_claim_authority_survives_worker_process_replacement(tmp_path: Path) -> None:
    first = _run(tmp_path, "first")
    assert first["accepted"] and first["stream"] == 1
    same = _run(tmp_path, "same")
    assert same["accepted"] and same["stream"] == 2 and same["old_fenced"]
    held = (tmp_path / "worker/ownership.json").read_bytes()
    for mode, epoch, identity, refusal in [
        ("stale", 7, "owner-A", "STALE_RECORD_OWNER_EPOCH"),
        ("other", 8, "owner-B", "EPOCH_HELD"),
        ("forged", 99, "owner-C", "UNAUTHENTICATED"),
    ]:
        result = _run(tmp_path, mode, epoch, identity)
        assert not result["accepted"] and result["rejection"] == "CLAIM_REJECTION_" + refusal
        assert (tmp_path / "worker/ownership.json").read_bytes() == held
    newer = _run(tmp_path, "newer", 9, "owner-B")
    assert newer["accepted"] and newer["stream"] == 3
    assert not _run(tmp_path, "old", 8)["accepted"]
    # An owner on an old wire minor still claims; operations it cannot use fail alone.
    legacy = _run(tmp_path, "legacy", 10, "owner-B")
    assert legacy["accepted"] and legacy["stream"] == 4


def test_claim_ack_is_durable_before_process_is_killed(tmp_path: Path) -> None:
    _run(tmp_path, "first")
    process = subprocess.Popen(
        [sys.executable, "-c", WORKER, str(tmp_path), "hold", "8", "owner-A"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if line.startswith("PROOF "):
                assert json.loads(line.removeprefix("PROOF "))["stream"] == 2
                break
        else:
            raise AssertionError("worker died before its ClaimAck")
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for channel in (process.stdin, process.stdout, process.stderr):
            if channel is not None:
                channel.close()
    restored = _run(tmp_path, "same")
    assert restored["accepted"] and restored["stream"] == 3 and restored["old_fenced"]


def test_missing_corrupt_and_wrong_boot_history_refuse(tmp_path: Path) -> None:
    assert _run(tmp_path, "same")["startup_refusal"] == "worker_ownership_absent"
    assert _run(tmp_path, "first")["accepted"]
    path = tmp_path / "worker/ownership.json"
    held = path.read_bytes()
    assert _run(tmp_path, "wrong-boot")["startup_refusal"] == "worker_ownership_invalid"
    assert path.read_bytes() == held
    repeated = held[:-1] + b',"control_stream_epoch":0}'
    for corrupt in (b"", b"{}", repeated):
        path.write_bytes(corrupt)
        assert _run(tmp_path, "same")["startup_refusal"] == "worker_ownership_invalid"
        assert path.read_bytes() == corrupt
    # Formatting and additive fields from another Runtime version are not corruption.
    path.write_bytes(held[:-1] + b',"written_by":"newer"}\n')
    assert _run(tmp_path, "same")["accepted"]


def test_failed_counter_write_never_accepts_claim(tmp_path: Path) -> None:
    assert _run(tmp_path, "first")["accepted"]
    path = tmp_path / "worker/ownership.json"
    held = path.read_bytes()
    pending = path.with_suffix(".pending")
    pending.mkdir()  # an actual filesystem refusal, not a mocked durability answer
    refused = _run(tmp_path, "same")
    assert not refused["accepted"] and refused["rejection"] == "CLAIM_REJECTION_UNDURABLE"
    assert path.read_bytes() == held
    pending.rmdir()
    assert _run(tmp_path, "same")["stream"] == 2


@pytest.mark.skipif(os.name != "posix", reason="fixed pods use the POSIX worker-root lease")
def test_unfinished_counter_stage_cannot_reset_or_strand_resume(tmp_path: Path) -> None:
    assert _run(tmp_path, "first")["accepted"]
    (tmp_path / "worker/ownership.pending").write_bytes(b"an interrupted pre-rename write")
    assert _run(tmp_path, "same")["stream"] == 2


def test_previous_minor_owner_remains_compatible_with_current_worker(tmp_path: Path) -> None:
    assert _run(tmp_path, "first")["accepted"]
    older = _run(tmp_path, "previous")
    assert older["accepted"] and older["stream"] == 2
    assert _run(tmp_path, "same")["stream"] == 3


def test_supervised_machine_relaunch_ignores_old_control_history(tmp_path: Path) -> None:
    assert _run(tmp_path, "first")["accepted"]
    history = tmp_path / "worker/ownership.json"
    # Adoption can encounter another boot, absent history, or a torn obsolete file.
    for index, old in enumerate((history.read_bytes(), b"invalid old history", None)):
        if old is None:
            history.unlink()
        else:
            history.write_bytes(old)
        result = _run(tmp_path, f"supervisor-{index}", 1, "cozy-local-client")
        assert result["accepted"] and result["stream"] == 1
        # Machine calls authenticate before a legacy Control conversation exists.
        assert result["authorized"] == "cozy-local-client"
        assert history.read_bytes() == old if old is not None else not history.exists()


def test_supervised_machine_still_authenticates_its_private_caller(tmp_path: Path) -> None:
    for mode, epoch, owner in (
        ("supervisor-forged", 1, "cozy-local-client"),
        ("supervisor-other", 1, "other-client"),
        ("supervisor-epoch", 9, "cozy-local-client"),
    ):
        result = _run(tmp_path, mode, epoch, owner)
        assert result["authorized"] == "refused"
        assert not result["accepted"]
        assert result["rejection"] == "CLAIM_REJECTION_UNAUTHENTICATED"
    assert not (tmp_path / "worker/ownership.json").exists()
