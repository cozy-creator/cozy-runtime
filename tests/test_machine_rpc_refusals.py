"""Known preaccept validation is terminal; unexpected exceptions remain opaque."""

from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import signed_claims
from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import GrpcControlHost, InMemoryControlHost
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import offer
from test_same_release_placements import Context, Refused

CLAIM = signed_claims.claim()


@pytest.fixture
def owned(tmp_path: Path) -> Iterator[Worker]:
    """A real in-process Worker claimed by `owner`; only the gRPC context is a stand-in."""
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        WorkerOptions(
            **signed_claims.IDENTITY, root=tmp_path / "worker", tensorfs_root=tmp_path / "store"
        ),
        InMemoryControlHost(),
    )
    try:
        worker.serve_stream(iter([pb.RecordOwnerFrame(claim=CLAIM)]), lambda _: None)
        yield worker
    finally:
        worker.shutdown()


def test_noncanonical_execution_payload_is_a_known_refusal(owned: Worker) -> None:
    rpc = MachineExecutionRPC(owned)
    with pytest.raises(Refused) as refused:
        rpc._call(CLAIM, Context(), lambda: canonical.parse_canonical(b'{ "value": 7 }'))
    assert refused.value.code == grpc.StatusCode.FAILED_PRECONDITION

    def unexpected() -> None:
        raise RuntimeError("private error contents")

    with pytest.raises(Refused) as refused:
        rpc._call(CLAIM, Context(), unexpected)
    assert (refused.value.code, str(refused.value)) == (
        grpc.StatusCode.UNAVAILABLE,
        "machine execution operation failed (RuntimeError)",
    )
    notes = [event.step for event in owned.activity if event.kind == "machine_execution"]
    assert notes[0] == "operation failed unexpectedly: RuntimeError"
    assert any("in unexpected" in note for note in notes[1:]), notes
    assert not any("private error contents" in note for note in notes), notes


def test_lookup_distinguishes_absence_from_authority_workspace_and_storage_failure(
    owned: Worker,
) -> None:
    executions = owned.executions
    assert executions is not None
    rpc = MachineExecutionRPC(owned)
    context = Context()
    query = pb.MachineExecutionQuery(
        claim=CLAIM, request_id="request", expected_execution_workspace_id=executions.workspace_id
    )
    with pytest.raises(Refused) as refused:
        rpc.GetMachineExecution(query, context)
    assert refused.value.code == grpc.StatusCode.NOT_FOUND
    query.expected_execution_workspace_id = "wrong-workspace"
    with pytest.raises(Refused) as refused:
        rpc.GetMachineExecution(query, context)
    assert refused.value.code == grpc.StatusCode.FAILED_PRECONDITION
    query.expected_execution_workspace_id = executions.workspace_id
    query.claim.record_owner_id = "other"
    with pytest.raises(Refused) as refused:
        rpc.GetMachineExecution(query, context)
    assert refused.value.code == grpc.StatusCode.FAILED_PRECONDITION
    query.claim.record_owner_id = "owner"
    executions.submit(
        "owner",
        "submission",
        b"c" * 32,
        offer("request"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    assert rpc.GetMachineExecution(query, context).state == "queued"
    # A failed read must never be interpreted as permission to submit fresh work.
    for sidecar in ("-wal", "-shm", ""):
        (executions.workspace.directory / f"journal.sqlite3{sidecar}").unlink()
    (executions.workspace.directory / "journal.sqlite3").write_bytes(b"corrupt database")
    with pytest.raises(Refused) as refused:
        rpc.GetMachineExecution(query, context)
    assert refused.value.code != grpc.StatusCode.NOT_FOUND


def test_a_verified_proof_stands_for_its_epoch_and_a_changed_one_is_verified(
    tmp_path: Path,
) -> None:
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
        ),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=tmp_path / "store",
            worker_id="proof",
            worker_boot_id="boot",
            worker_tls_certificate_digest="sha256:" + "a" * 64,
        ),
        InMemoryControlHost(),
    )
    try:
        claim = pb.Claim(
            record_owner_id="owner",
            record_owner_epoch=3,
            worker_id="proof",
            worker_boot_id="boot",
            wire_minor=WIRE_MINOR,
            proof=key.sign(
                documents.canonical_bytes(
                    pb.ClaimProof(
                        record_owner_epoch=3,
                        worker_boot_id="boot",
                        worker_id="proof",
                        worker_tls_certificate_digest=b"\xaa" * 32,
                    )
                )
            ),
        )
        worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
        assert worker.authorize_workspace(claim) == "owner"
        assert worker.authorize_workspace(claim) == "owner"
        forged = pb.Claim()
        forged.CopyFrom(claim)
        forged.proof = bytes(64)
        with pytest.raises(WorkspaceRefusal, match="proof was refused"):
            worker.authorize_workspace(forged)
        assert worker.authorize_workspace(claim) == "owner"
    finally:
        worker.shutdown()


def test_a_listener_without_the_record_owners_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="public key"):
        Worker(
            RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
            WorkerOptions(root=tmp_path / "worker", tensorfs_root=tmp_path / "store"),
            GrpcControlHost("127.0.0.1:0", tmp_path / "address"),
        )
