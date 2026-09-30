"""cozy-runtime-worker as a user under a Host-chosen root, launched the way the Host launches it.

The Host is the only stand-in here: it writes the bootstrap handoff, answers the media
readiness probe with 401, and makes the loopback preparation and machine calls a Host
forwards. The worker is the image's real console script (or, for a virtual inventory, the
same `main` called with `gpus=`), with fd 3 as its stop pipe and no argument.
"""

from __future__ import annotations

import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import pytest
import tensorfs
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from conftest import image_python
from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_end_to_end import NO_EXECUTOR
from test_job_preparation_isolation import package

pytestmark = pytest.mark.skipif(
    bool(NO_EXECUTOR) or os.geteuid() == 0, reason=NO_EXECUTOR or "the proof is non-root"
)

VIRTUAL_GPUS = [
    {
        "device_index": index,
        "device_name": "Virtual Accelerator",
        "device_uuid": f"GPU-virtual-{index}",
        "driver_version": "0.0",
        "memory_bytes": 8 << 30,
        "pci_bus_id": f"00000000:0{index}:00.0",
    }
    for index in range(4)
]

LAUNCH_VIRTUAL = (
    "import json, sys\n"
    "from cozy_runtime.cli import runtime_worker\n"
    "sys.exit(runtime_worker.main([], gpus=json.loads(sys.argv[1])))\n"
)

#: A test bound on a worker that must finish booting; the worker itself runs under no clock.
BOOT_BOUND = 180


class Machine:
    """One rooted machine: its bootstrap handoff, media stand-in and running worker."""

    def __init__(self, root: Path, gpus: list[dict[str, Any]] | None) -> None:
        self.root = root
        self.worker_id = "rooted-worker"
        self.key = Ed25519PrivateKey.generate()
        bootstrap = root / "run/cozy/bootstrap"
        bootstrap.mkdir(parents=True)
        (bootstrap / "pod-boot-id").write_text(
            base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
        )
        self.certificate_pem, key_pem = _certificate()
        (bootstrap / "tls.crt").write_bytes(self.certificate_pem)
        (bootstrap / "tls.key").write_bytes(key_pem)
        self.certificate_der = ssl.PEM_cert_to_DER_cert(self.certificate_pem.decode())
        tensorfs.Store.ensure(str(root / "var/lib/tensorfs")).prepare_readers()
        (root / "tmp").mkdir()
        # The image layout, linked the way a local controller installs it.
        image = image_python()
        for relative, target in (
            ("opt/cozy/bin/cozy-runtime-worker", image.parent / "cozy-runtime-worker"),
            ("usr/local/bin/uv", Path(shutil.which("uv") or "uv")),
        ):
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).symlink_to(target)
        self.media, self.media_port = _media_stand_in(self.certificate_pem, key_pem)
        self.worker_port = _free_port()
        public = self.key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        env = {
            "PATH": f"{root}/usr/local/bin:/usr/bin:/bin",
            "TMPDIR": str(root / "tmp"),
            "COZY_MACHINE_ROOT": str(root),
            "COZY_WORKER_ID": self.worker_id,
            "COZY_WORKER_INTERNAL_PORT": str(self.worker_port),
            "COZY_MEDIA_INTERNAL_PORT": str(self.media_port),
            "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
                {
                    "control_public_key_ed25519_b64url": base64.urlsafe_b64encode(public)
                    .rstrip(b"=")
                    .decode(),
                    "media_token_sha256": [hashlib.sha256(b"media").hexdigest()],
                }
            ),
            "TENSORHUB_OBJECT_STORAGE_HOSTS": "",
        }
        self.env = env
        self.image, self.gpus = image, gpus
        self.log = (root / "worker.log").open("ab")
        self._launch()
        self.claim, self.ack = self._claim()

    def _launch(self) -> None:
        read, self.stop = os.pipe()
        # ExtraFiles[0] is fd 3, as the Host passes it.
        redirect = f"3<&{read} {read}<&-"
        if self.gpus is None:
            worker = self.root / "opt/cozy/bin/cozy-runtime-worker"
            argv = ["/bin/bash", "-c", f'exec "$0" {redirect}', str(worker)]
        else:
            argv = [
                "/bin/bash",
                "-c",
                f'exec "$0" -c "$1" "$2" {redirect}',
                str(self.image),
                LAUNCH_VIRTUAL,
                json.dumps(self.gpus),
            ]
        self.process = subprocess.Popen(
            argv, env=self.env, pass_fds=(read,), stdout=self.log, stderr=subprocess.STDOUT
        )
        os.close(read)
        self.payload = self._await_readiness()
        self.channel = grpc.secure_channel(
            f"127.0.0.1:{self.worker_port}",
            grpc.ssl_channel_credentials(self.certificate_pem),
            options=(
                ("grpc.ssl_target_name_override", "cozy-worker"),
                ("grpc.max_receive_message_length", 8 << 20),
            ),
        )
        self.control = rpc.WorkerControlStub(self.channel)
        self.preparation = rpc.RuntimePreparationStub(self.channel)

    def restart(self) -> int:
        """Stop this Runtime and launch it again on the same root and boot, as its Host does
        after a Host or Runtime restart. Nothing claims the relaunched worker."""
        code = self._stop()
        (self.root / "run/cozy/bootstrap/readiness-payload").unlink()
        self._launch()
        return code

    def _await_readiness(self) -> dict[str, Any]:
        payload = self.root / "run/cozy/bootstrap/readiness-payload"
        until = time.monotonic() + BOOT_BOUND
        while not payload.exists():
            assert self.process.poll() is None, self.output()
            assert time.monotonic() < until, self.output()
            time.sleep(0.05)
        value: dict[str, Any] = json.loads(payload.read_bytes())
        return value

    def _claim(self) -> tuple[pb.Claim, pb.ClaimAck]:
        boot = self.payload["pod_boot_id"]
        claim = pb.Claim(
            record_owner_epoch=1,
            record_owner_id="owner",
            worker_id=self.worker_id,
            worker_boot_id=boot,
            wire_minor=WIRE_MINOR,
        )
        claim.proof = self.key.sign(
            documents.canonical_bytes(
                pb.ClaimProof(
                    record_owner_epoch=1,
                    worker_boot_id=boot,
                    worker_id=self.worker_id,
                    worker_tls_certificate_digest=hashlib.sha256(self.certificate_der).digest(),
                )
            )
        )

        def frames() -> Iterator[pb.RecordOwnerFrame]:
            yield pb.RecordOwnerFrame(claim=claim)

        stream = self.control.Control(frames(), timeout=BOOT_BOUND)
        ack = next(stream).claim_ack
        assert ack.accepted, (ack, self.output())
        stream.cancel()
        # Observe RPC closure itself. EOF may win the race with local cancellation;
        # the worker's corresponding log is asynchronous and may use either wording.
        assert stream.code() in (grpc.StatusCode.OK, grpc.StatusCode.CANCELLED), self.output()
        assert not stream.is_active()
        return claim, ack

    def output(self) -> str:
        self.log.flush()
        return (self.root / "worker.log").read_text(errors="replace")[-6000:]

    def _stop(self) -> int:
        self.channel.close()
        os.close(self.stop)  # EOF on fd 3: the Host's one stop signal
        try:
            return self.process.wait(timeout=BOOT_BOUND)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()

    def close(self) -> int:
        try:
            return self._stop()
        finally:
            self.media.shutdown()
            self.log.close()


def _certificate() -> tuple[bytes, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cozy-worker")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("cozy-worker")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    return (
        certificate.public_bytes(serialization.Encoding.PEM),
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _Media:
    def __init__(self, listener: socket.socket) -> None:
        self.listener = listener

    def shutdown(self) -> None:
        self.listener.close()


def _media_stand_in(certificate: bytes, key: bytes) -> tuple[_Media, int]:
    """The Host's media listener as far as readiness sees it: TLS, then 401 to a stranger."""
    with tempfile.TemporaryDirectory() as raw:
        (Path(raw) / "c").write_bytes(certificate)
        (Path(raw) / "k").write_bytes(key)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(Path(raw) / "c", Path(raw) / "k")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()

    def serve() -> None:
        while True:
            try:
                raw_connection, _ = listener.accept()
            except OSError:
                return
            try:
                with context.wrap_socket(raw_connection, server_side=True) as connection:
                    connection.recv(4096)
                    connection.sendall(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
            except (OSError, ssl.SSLError):
                pass

    threading.Thread(target=serve, daemon=True).start()
    return _Media(listener), int(listener.getsockname()[1])


def _run_job(
    machine: Machine,
    scratch: Path,
    staged: tuple[Path, pb.PrepareLocalPackageRequest] | None = None,
    request_id: str = "rooted-1",
    orchestration: bool = False,
) -> tuple[str, pb.AttemptOutcome]:
    """Prepare a CPU fixture package and run one job through the machine calls; an
    orchestration job takes the CPU slot, as Creator submits a job granted no device."""
    if staged is None:
        scratch.mkdir()
        staged = package(scratch, "rooted")
    binding, prepared = _prepare(machine, staged)
    return _submit(machine, scratch, binding, prepared, request_id, orchestration)


def _prepare(
    machine: Machine, staged: tuple[Path, pb.PrepareLocalPackageRequest]
) -> tuple[JobBinding, pb.PreparePackageSetResult]:
    installs = machine.root / "var/lib/cozy/installs"
    staged_environment, request = staged
    stage = installs / ".stage/rooted/wheels"
    stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged_environment / ".stage/rooted/wheels", stage, dirs_exist_ok=True)
    for row in request.files:
        row.path = str(stage / Path(row.path).name)
    request.install_root = str(installs)
    prepared = machine.preparation.PrepareLocalPackage(request, timeout=900)
    plans = [
        json.loads(path.read_bytes())
        for path in (machine.root / "run/cozy/job-plans").glob("*/*.json")
    ]
    return JobBinding.read(next(row for row in plans if row.get("job") == "rooted")), prepared


def _submit(
    machine: Machine,
    scratch: Path,
    binding: JobBinding,
    prepared: pb.PreparePackageSetResult,
    request_id: str,
    orchestration: bool,
) -> tuple[str, pb.AttemptOutcome]:
    capture, capture_digest = documents.identity(
        pb.MachineExecutionCapture(
            root_installation_id=binding.installation_id,
            installed_packages=[prepared.installed_package],
        )
    )
    payload = canonical_json.encode({"temporary": str(scratch / "temporary")})
    payload_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    spec, spec_digest = documents.identity(
        pb.InvocationSpec(
            installation_id=binding.installation_id,
            payload_digest=payload_digest,
            inputs=[
                pb.InputBinding(
                    input_id="payload",
                    digest=payload_digest,
                    length=len(payload),
                    kind_mime="application/json",
                )
            ],
            deadline_unix_ms=int(time.time() * 1000) + 600_000,
            job=pb.JobInvocationSpec(
                installation_id=binding.installation_id,
                job_descriptor_id=binding.job_descriptor_id,
                publication_contract=pb.PublicationContract(grant_id="local/_job-rooted"),
            ),
        )
    )
    workspace = machine.control.GetMachineExecutionWorkspace(
        pb.MachineExecutionWorkspaceQuery(claim=machine.claim)
    ).execution_workspace_id
    machine.control.SubmitMachineExecution(
        pb.MachineExecutionSubmit(
            claim=machine.claim,
            submission_id=request_id,
            capture_digest=capture_digest,
            capture_canonical_bytes=capture,
            offer=pb.AttemptOffer(
                request_id=request_id,
                attempt_ordinal=1,
                invocation_spec_digest=spec_digest,
                invocation_spec_canonical_bytes=spec,
                grant=pb.DeliveryGrant(
                    invocation_spec_digest=spec_digest,
                    inputs=[
                        pb.InputAccess(
                            input_id="payload",
                            url="data:application/json;base64,"
                            + base64.b64encode(payload).decode(),
                        )
                    ],
                ),
            ),
            prepared_state=pb.DesiredWorkerState(
                revision=1,
                posture=pb.Posture.POSTURE_ACCEPTING,
                wire_minor=WIRE_MINOR,
                job=pb.JobDirective(
                    installation_id=binding.installation_id,
                    job_descriptor_id=binding.job_descriptor_id,
                    orchestration=orchestration,
                    resource_caps=pb.ResourceCaps(device_required=False),
                    publication_contract=pb.PublicationContract(grant_id="local/_job-rooted"),
                ),
            ),
            payload_canonical_bytes=payload,
            expected_execution_workspace_id=workspace,
        ),
        timeout=120,
    )
    query = pb.MachineExecutionQuery(
        claim=machine.claim, request_id=request_id, expected_execution_workspace_id=workspace
    )
    until = time.monotonic() + 900
    while (state := machine.control.GetMachineExecution(query)).state not in {
        "succeeded",
        "failed",
        "canceled",
    }:
        assert machine.process.poll() is None and time.monotonic() < until, machine.output()
        time.sleep(0.2)
    outcome = machine.control.CollectMachineExecution(pb.MachineExecutionCollect(execution=query))
    assert state.state == "succeeded", outcome.outcome_canonical_bytes[:2000]
    return workspace, outcome


def _files_outside(root: Path) -> list[str]:
    """Pod paths this worker would have written if it ignored its root."""
    return [
        str(path)
        for path in (Path("/run/cozy"), Path("/var/lib/cozy"), Path("/var/lib/tensorfs"))
        if path.exists() and path.stat().st_uid == os.geteuid()
    ]


def test_user_worker_runs_a_job_under_its_root_and_reports_a_virtual_inventory() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-rooted.") as raw:
        root = Path(raw) / "machine"
        root.mkdir()
        machine = Machine(root, VIRTUAL_GPUS)
        try:
            assert [row["device_uuid"] for row in machine.payload["runtime_gpus"]] == [
                gpu["device_uuid"] for gpu in VIRTUAL_GPUS
            ]
            observed = machine.control.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=machine.claim)
            )
            assert [(d.ordinal, d.uuid, d.memory_bytes) for d in observed.devices] == [
                (index, f"GPU-virtual-{index}", 8 << 30) for index in range(4)
            ]
            assert observed.accelerator_backend == "cuda"
            assert not observed.executor_uid_isolation

            workspace, outcome = _run_job(machine, Path(raw) / "scratch")
            body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            triage = machine.control.ReadMachineExecutionTriage(
                pb.MachineExecutionTriageQuery(
                    execution=pb.MachineExecutionQuery(
                        claim=machine.claim,
                        request_id="rooted-1",
                        expected_execution_workspace_id=workspace,
                    )
                )
            )
            ref = body["triage_bundle"]
            assert triage.bundle.subject_id == ref["subject_id"]
            assert (
                documents.spell(triage.bundle.write_receipt_digest) == ref["write_receipt_digest"]
            )
            assert (
                documents.spell(hashlib.sha256(triage.bundle_canonical_bytes).digest())
                == ref["write_receipt_digest"]
            )
            assert json.loads(triage.bundle_canonical_bytes)["subject_id"] == ref["subject_id"]

            with pytest.raises(grpc.RpcError) as absent:
                machine.control.ReadMachineExecutionTriage(
                    pb.MachineExecutionTriageQuery(
                        execution=pb.MachineExecutionQuery(
                            claim=machine.claim,
                            request_id="never-submitted",
                            expected_execution_workspace_id=workspace,
                        )
                    )
                )
            assert absent.value.code() == grpc.StatusCode.NOT_FOUND

            for relative in (
                "run/cozy/worker/control.addr",
                "run/cozy/bootstrap/worker-activity",
                "var/lib/cozy/installs",
                f"var/lib/cozy/media/triage/{ref['subject_id']}.json",
            ):
                assert (root / relative).exists(), relative
            assert _files_outside(root) == []
            # Spills go to the executor's own scope under Runtime's root, not the Host's TMPDIR.
            temporary = Path((Path(raw) / "scratch/temporary").read_text())
            assert temporary.is_relative_to(root / "run/cozy/jit-cache"), temporary
        finally:
            code = machine.close()
        assert code == 0, machine.output()


def test_user_worker_measures_this_machines_driver_inventory() -> None:
    with tempfile.TemporaryDirectory(prefix="cz-measured.") as raw:
        root = Path(raw) / "machine"
        root.mkdir()
        machine = Machine(root, None)
        try:
            uuids: list[str] = []
            if shutil.which("nvidia-smi"):
                measured = subprocess.run(
                    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                )
                if measured.returncode == 0:
                    uuids = [line.strip() for line in measured.stdout.splitlines() if line.strip()]
            observed = machine.control.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=machine.claim)
            )
            assert [device.uuid for device in observed.devices] == uuids
            assert observed.accelerator_backend == ("cuda" if uuids else "none")
            assert not observed.executor_uid_isolation
        finally:
            code = machine.close()
        assert code == 0, machine.output()


def test_restarted_worker_runs_its_owners_next_submission_without_another_claim() -> None:
    """A Host or Runtime restart forgets every control stream. The owner's next submission
    carries its Claim, and runs at the minor that Claim negotiated: no Control Claim first.
    The one Claim the owner did make reads back the inventory this worker schedules on."""
    with tempfile.TemporaryDirectory(prefix="cz-restart.") as raw:
        root = Path(raw) / "machine"
        root.mkdir()
        machine = Machine(root, VIRTUAL_GPUS)
        try:
            resources = machine.ack.resources
            assert (
                resources.backend,
                resources.device_count,
                resources.device_name,
                resources.device_memory_total_bytes,
                resources.driver_version,
            ) == ("cuda", 4, "Virtual Accelerator", 8 << 30, "0.0")
            scratch = Path(raw) / "scratch"
            scratch.mkdir()
            staged = package(scratch, "rooted")
            _run_job(machine, scratch, staged, "before-restart", orchestration=True)
            assert machine.restart() == 0, machine.output()
            _, outcome = _run_job(machine, scratch, staged, "after-restart", orchestration=True)
            assert outcome.request_id == "after-restart"
            machine.log.flush()
            log = (root / "worker.log").read_text(errors="replace")
            # The boot's named negative readiness proof. _claim already observed its
            # accepted RPC close before either submission or the restart.
            assert "REFUSED (CLAIM_REJECTION_UNAUTHENTICATED) for readiness-negative-arm" in log
        finally:
            code = machine.close()
        assert code == 0, machine.output()


def test_concurrent_first_submissions_share_the_preparation_slot() -> None:
    """A fresh worker's first submissions arrive together. Each captures the Runtime's
    built-in operations in the one preparation executor slot; they share one capture and
    none is refused for another's use of the slot (hakufu run 1416)."""
    with tempfile.TemporaryDirectory(prefix="cz-concurrent.") as raw:
        root = Path(raw) / "machine"
        root.mkdir()
        machine = Machine(root, VIRTUAL_GPUS)
        try:
            scratch = Path(raw) / "scratch"
            scratch.mkdir()
            binding, prepared = _prepare(machine, package(scratch, "rooted"))
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                runs = [
                    pool.submit(_submit, machine, scratch, binding, prepared, f"first-{i}", True)
                    for i in range(3)
                ]
                outcomes = [run.result()[1] for run in runs]
            assert sorted(outcome.request_id for outcome in outcomes) == [
                "first-0",
                "first-1",
                "first-2",
            ]
        finally:
            code = machine.close()
        assert code == 0, machine.output()
