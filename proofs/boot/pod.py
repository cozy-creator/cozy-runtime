#!/usr/bin/env python3
"""cr-048's boot proof: a pod that boots with NO provision document, ready and empty.

This process is the Runtime-side contract stand-in for cozy-creator's single `pod-supervisor`
binary. It makes the same grants (TLS keypair, boot id, renter-token hashes, its in-process
media listener, and the closed COZY_ADAPTER_* environment), execs the real adapter, and then
observes what a rented pod's two consumers observe: tensorhub's receipt, and a
RecordOwner's Claim.

What it proves, and every line of it is measured rather than asserted by construction:

  * the adapter boots on nine environment names and reads no document at all;
  * the attempt id is spent as worker identity, not echoed through the receipt;
  * the pod does not re-measure image-resident Runtime/base bytes;
  * the old closure and identity-echo fields are GONE from the receipt;
  * both listeners bound and both refused a foreign credential;
  * the worker admits the renter's real token and reports ZERO placements, an empty
    accepted set, and no endpoint release — ready, and empty.

Then cr-049's half, on the SAME pod and through the SAME connection, because a first
placement that took a private door would prove nothing about the tenth:

  * a standing `ArtifactGrantUpdate` lands and is echoed back, changing no revision;
  * one `DesiredPlacementSet` names an environment, a receipt and a binding plan by digest —
    and NOTHING else crosses;
  * the worker fetches source/custom wheels, publishes a small installed-base overlay, and
    derives the same receipt the hub derived independently;
  * it reports `STAGED` x `DISPATCHABLE` with `converged_revision` = the accepted revision;
  * it SERVES a real attempt against the endpoint installed in that generation;
  * and a following revision naming an unauthorized digest is refused typed while the
    converged placement KEEPS SERVING (fallback-retention, #474).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import os
import platform
import queue
import secrets
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import grpc
import hub

from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

STATE = Path("/run/cozy/pod")
# The adapter gives the worker this exact co-resident media root. Keeping a second proof-only
# spelling made the real attempt fail at the grant-root fence before the convergence arms ran.
MEDIA_ROOT = Path("/var/lib/cozy/media")
WHEELS = Path("/wheels")
ENVIRONMENT_ROOT = Path("/var/lib/cozy/environments")
ARTIFACT_CACHE = Path("/var/lib/cozy/artifacts")
APPLICATION = "proof_cli_endpoint:app"
WORKER_PORT = 43100
MEDIA_PORT = 43101
ARTIFACT_PORT = 43102
DEADLINE = int(time.time()) + 1800


def access_as(path: Path, uid: int, gid: int, flags: int) -> bool:
    """One DAC probe in a short child; no Python runs after the identity transition."""

    pid = os.fork()
    if pid == 0:
        try:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            descriptor = os.open(path, flags)
            os.close(descriptor)
        except OSError:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) == 0


def prove_adapter_parent_death() -> None:
    """The exact adapter arming code kills its process when its direct parent exits."""

    source = """
import os
import time
from cozy_runtime.cli.materialize_launch import _arm_supervisor_lifetime

read_fd, write_fd = os.pipe()
if read_fd != 3:
    os.dup2(read_fd, 3)
    os.close(read_fd)
pid = os.fork()
if pid == 0:
    os.close(write_fd)
    _arm_supervisor_lifetime()
    time.sleep(60)
    os._exit(99)
print(pid, flush=True)
time.sleep(0.5)
os._exit(0)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert parent.stdout is not None
    child_pid = int(parent.stdout.readline().strip())
    assert parent.wait(timeout=5) == 0
    deadline = time.monotonic() + 5
    state = ""
    while time.monotonic() < deadline:
        try:
            status = Path(f"/proc/{child_pid}/status").read_text()
        except OSError:
            state = "gone"
            break
        state = next(
            (line.split()[1] for line in status.splitlines() if line.startswith("State:")),
            "",
        )
        if state == "Z":
            break
        time.sleep(0.05)
    assert state in {"gone", "Z"}, f"adapter child survived parent exit in state {state!r}"
    with contextlib.suppress(ChildProcessError):
        os.waitpid(child_pid, os.WNOHANG)


RETIRED_RECEIPT_FIELDS = (
    "acquisition_attempt_id",
    "acquisition_attempt_ordinal",
    "artifact_sources",
    "measured_control_runtime_wheel_digest",
    "measured_control_runtime_wheel_length",
    "measured_installed_environment_receipt_digest",
    "network_cache_degradations",
    "provision_bundle_digest",
    "provision_spec_digest",
    "rental_id",
    "staged_placement_set_digest",
)
RETIRED_ADAPTER_ENV = (
    "COZY_ADAPTER_ACQUISITION_ATTEMPT_ORDINAL",
    "COZY_ADAPTER_PROVISION_SPEC_PATH",
    "COZY_ADAPTER_PROVISION_SPEC_DIGEST",
    "COZY_ADAPTER_PROVISION_BUNDLE_PATH",
    "COZY_ADAPTER_PROVISION_BUNDLE_DIGEST",
    "COZY_ADAPTER_RENTAL_ID",
)


def note(line: str) -> None:
    print(f"[boot-proof] {line}", flush=True)


def mint_origin_tls() -> tuple[Path, Path]:
    """The ARTIFACT ORIGIN's own keypair, which is the hub's and not the pod's.

    It is separate on purpose: the pod's certificate is bound to `DNS:cozy-worker` and
    `readiness.certificate` refuses any other SAN on it, because that certificate is the one
    a renter pins. The origin is a different peer with a different name, so it gets a
    different key — which is also what a real deployment looks like.
    """

    certificate, key = STATE / "origin.crt", STATE / "origin.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=cozy-artifact-origin",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return certificate, key


def mint_tls() -> tuple[Path, Path]:
    certificate, key = STATE / "tls.crt", STATE / "tls.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-days",
            "1",
            "-subj",
            "/CN=cozy-worker",
            "-addext",
            "subjectAltName=DNS:cozy-worker",
        ],
        check=True,
        capture_output=True,
    )
    return certificate, key


class MediaStandIn(http.server.BaseHTTPRequestHandler):
    """pod-supervisor's in-process authenticated health route, and nothing else.

    The adapter proves the media listener refuses a FOREIGN bearer before it will publish a
    receipt; that refusal is the property, so the stand-in implements exactly it.
    """

    token = ""
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        offered = self.headers.get("Authorization", "")
        ok = self.path == "/v1/health" and offered == f"Bearer {MediaStandIn.token}"
        body = b'{"status":"ok"}' if ok else b'{"error":"unauthorized"}'
        self.send_response(200 if ok else 401)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_media(certificate: Path, key: Path, token: str) -> None:
    MediaStandIn.token = token
    server = http.server.ThreadingHTTPServer(("0.0.0.0", MEDIA_PORT), MediaStandIn)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(certificate), str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()


class RecordOwner:
    """One long-lived RecordOwner over one control stream, driven by this proof.

    It exists because the cr-048 arm and the cr-049 arm must run on the SAME connection: a
    first placement delivered over a second, private door would prove nothing about the
    tenth, and "placement 1 is treated identically to placements 2..N" is the property.
    """

    def __init__(self, certificate: Path, token: bytes) -> None:
        credentials = grpc.ssl_channel_credentials(certificate.read_bytes())
        self.channel = grpc.secure_channel(
            f"127.0.0.1:{WORKER_PORT}",
            credentials,
            options=(
                ("grpc.ssl_target_name_override", "cozy-worker"),
                ("grpc.default_authority", "cozy-worker"),
            ),
        )
        self.outbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
        self.inbound: queue.Queue[pb.WorkerFrame] = queue.Queue()
        self.epoch = 1
        self.generation = 0
        self.worker_boot_id = ""
        self.admission_generation = 0
        self.claim_ack: pb.ClaimAck | None = None
        self.snapshot: dict[str, Any] = {}
        self.states: list[pb.ObservedWorkerState] = []

        def frames() -> Any:
            while (frame := self.outbound.get()) is not None:
                yield frame

        self.call = pb_grpc.WorkerControlStub(self.channel).Control(frames())
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.send(
            pb.RecordOwnerFrame(
                claim=pb.Claim(
                    record_owner_epoch=self.epoch,
                    record_owner_id="cr-049-boot-proof",
                    wire_minor=WIRE_MINOR,
                    proof=token,
                )
            )
        )
        self.reader.start()

    def _read(self) -> None:
        try:
            for frame in self.call:
                self.inbound.put(frame)
        except grpc.RpcError:
            return

    def send(self, frame: pb.RecordOwnerFrame) -> None:
        self.outbound.put(frame)

    def envelope(self, message: Any) -> Any:
        message.record_owner_epoch = self.epoch
        message.control_stream_generation = self.generation
        message.worker_boot_id = self.worker_boot_id
        return message

    def close(self) -> None:
        self.outbound.put(None)
        self.channel.close()

    def pump(self, deadline: float) -> pb.WorkerFrame | None:
        """One inbound frame, with the barrier and the ack lane already handled."""

        try:
            frame = self.inbound.get(timeout=max(0.05, deadline - time.monotonic()))
        except queue.Empty:
            return None
        kind = frame.WhichOneof("msg")
        if kind == "claim_ack":
            ack = frame.claim_ack
            assert ack.accepted, f"Claim REFUSED: {pb.ClaimRejection.Name(ack.rejection)}"
            self.generation = ack.control_stream_generation
            self.worker_boot_id = ack.worker_boot_id
            self.claim_ack = ack
        elif kind == "snapshot":
            # RECOMPUTE before parsing, then ack by id AND digest: the worker's dispatch
            # barrier stays closed until the RecordOwner has reconciled.
            body = frame.snapshot.snapshot_canonical_bytes
            assert documents.digest_of(body) == frame.snapshot.snapshot_digest
            self.snapshot = documents.read(body, pb.WorkerSnapshotBody)
            self.send(
                pb.RecordOwnerFrame(
                    snapshot_ack=self.envelope(
                        pb.SnapshotAck(
                            snapshot_id=frame.snapshot.snapshot_id,
                            snapshot_digest=frame.snapshot.snapshot_digest,
                        )
                    )
                )
            )
        elif kind == "observed_state":
            self.admission_generation = frame.observed_state.admission_generation
            retained = pb.ObservedWorkerState()
            retained.CopyFrom(frame.observed_state)
            self.states.append(retained)
        return frame

    def await_state(self, predicate: Any, why: str, *, seconds: float) -> pb.ObservedWorkerState:
        """Watch the OBSERVED state until it satisfies `predicate`, or say what it was."""

        deadline = time.monotonic() + seconds
        last: pb.ObservedWorkerState | None = None
        while time.monotonic() < deadline:
            frame = self.pump(deadline)
            if frame is None or frame.WhichOneof("msg") != "observed_state":
                continue
            last = frame.observed_state
            if predicate(last):
                return last
        raise SystemExit(f"never observed {why}; last state was {describe(last)}")

    def await_frame(self, kind: str, *, seconds: float) -> pb.WorkerFrame:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self.pump(deadline)
            if frame is not None and frame.WhichOneof("msg") == kind:
                return frame
        raise SystemExit(f"the worker never sent a {kind}")


def describe(state: pb.ObservedWorkerState | None) -> str:
    if state is None:
        return "(no ObservedWorkerState arrived)"
    rows = [
        f"phase={pb.WorkerPhase.Name(state.worker_phase)}",
        f"accepted={state.accepted_desired_state_revision}",
        f"converged={state.converged_revision}",
        f"grant={state.applied_artifact_grant_id!r}@{state.applied_grant_revision}",
    ]
    for placement in state.placements:
        rows.append(
            f"{placement.placement_id}:"
            f"{pb.MaterializationState.Name(placement.materialization)}x"
            f"{pb.ServingState.Name(placement.serving)}"
        )
    rows.extend(f"fault[{f.reason}]" for f in state.faults)
    return " ".join(rows)


def placement_is(state: pb.ObservedWorkerState, materialization: Any, serving: Any) -> bool:
    return len(state.placements) == 1 and (
        state.placements[0].materialization == materialization
        and state.placements[0].serving == serving
    )


def grant_update(release: hub.Release, revision: int) -> pb.ArtifactGrantUpdate:
    """The standing acquisition authority: WHICH digests, WHERE, and until when."""

    base = f"https://127.0.0.1:{ARTIFACT_PORT}"
    return pb.ArtifactGrantUpdate(
        grant_revision=revision,
        grant=pb.ArtifactGrant(
            grant_id=f"cr049-grant-{revision}",
            subjects=release.subjects(),
            credential=pb.DeliveryAccessCredential(
                issuer="cr-049-boot-proof",
                key_id="artifacts",
                epoch=1,
                expires_at_unix=DEADLINE,
                token=b"proof-artifact-token",
            ),
            base_url=base,
            expires_at_unix=DEADLINE,
            locations=release.locations(base),
        ),
    )


def desired_state(placement_set: bytes, revision: int) -> pb.DesiredWorkerState:
    return pb.DesiredWorkerState(
        revision=revision,
        posture=pb.Posture.POSTURE_ACCEPTING,
        wire_minor=WIRE_MINOR,
        placement_set=pb.DesiredPlacementSet(
            placement_set_digest=documents.digest_of(placement_set),
            placement_set_canonical_bytes=placement_set,
        ),
    )


def serve_one_attempt(
    owner: RecordOwner,
    plan_id: str,
    release_id: str,
    *,
    text: str = "cr-049",
    count: int = 2,
    request_id: str = "cr049-req-1",
) -> dict[str, Any]:
    """One real attempt against the endpoint installed in the materialized generation."""

    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"text": text, "count": count}).encode()
    payload_path = MEDIA_ROOT / f"{request_id}-payload.json"
    payload_path.write_bytes(payload)
    payload_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    spec = pb.InvocationSpec(
        endpoint_release_id=release_id,
        environment_spec_digest="sha256:" + "11" * 32,
        config_digest="sha256:" + "22" * 32,
        payload_digest=payload_digest,
        inputs=[
            pb.InputBinding(
                input_id="payload",
                digest=payload_digest,
                length=len(payload),
                kind_mime="application/json",
                order=0,
            )
        ],
        # `echo` returns a typed struct, not an artifact: it names no outputs, so the
        # grant names none either. Exact set equality between produced and granted outputs
        # is enforced by the tail, so an invented destination would refuse the attempt.
    )
    spec.serving.entrypoint_binding_plan_id = plan_id
    spec.serving.attempt_binding_id = plan_id
    canonical_bytes, spec_digest = documents.identity(spec)
    owner.send(
        pb.RecordOwnerFrame(
            attempt_offer=owner.envelope(
                pb.AttemptOffer(
                    request_id=request_id,
                    attempt_ordinal=1,
                    invocation_spec_digest=spec_digest,
                    invocation_spec_canonical_bytes=canonical_bytes,
                    placement_id=hub.PLACEMENT_ID,
                    admission_generation=owner.admission_generation,
                    grant=pb.DeliveryGrant(
                        invocation_spec_digest=spec_digest,
                        credential=pb.DeliveryAccessCredential(
                            issuer="cr-049-boot-proof",
                            key_id="delivery",
                            epoch=1,
                            expires_at_unix=DEADLINE,
                            token=b"proof-delivery-token",
                        ),
                        file_base_url=f"file://{MEDIA_ROOT}",
                        expires_at_unix=DEADLINE,
                        inputs=[pb.InputAccess(input_id="payload", url=f"file://{payload_path}")],
                    ),
                )
            )
        )
    )
    frame = owner.await_frame("attempt_outcome", seconds=300)
    outcome = frame.attempt_outcome
    assert outcome.request_id == request_id
    body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
    owner.send(
        pb.RecordOwnerFrame(
            outcome_ack=owner.envelope(
                pb.AttemptOutcomeAck(
                    request_id=outcome.request_id,
                    attempt_ordinal=outcome.attempt_ordinal,
                    outcome_id=outcome.outcome_id,
                    outcome_digest=outcome.outcome_digest,
                )
            )
        )
    )
    envelope = body.get("result", {})
    inline = base64.b64decode(str(envelope.get("inline_result", "")))
    return {"body": body, "result": json.loads(inline) if inline else None}


def author_release() -> hub.Release:
    """Author one endpoint release against the stand-in's installed base family.

    The wheelhouse is the control Runtime and its three ordinary dependencies, already
    installed in image Python. The project wheel is absent until the grant lands.
    """

    project = next(WHEELS.glob("proof_cli_endpoint-*.whl"))
    wheelhouse = sorted([next(WHEELS.glob("cozy_runtime-*.whl")), *(WHEELS / "deps").glob("*.whl")])
    version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    libc = "glibc{}.{}".format(*platform.libc_ver()[1].split(".")[:2])
    return hub.author(
        application=APPLICATION,
        project_wheel=project,
        wheelhouse_wheels=wheelhouse,
        target=hub.platform_target(version, libc),
        build_root=Path("/run/cozy/hub-build"),
        python=Path(sys.executable),
        base_manifest=Path("/opt/cozy/base-runtime/WheelhouseManifest.json"),
    )


def main() -> int:
    prove_adapter_parent_death()
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chown(STATE, 65532, 65532)
    STATE.chmod(0o700)
    stop_read, stop_write = os.pipe()
    if stop_read != 3:
        os.dup2(stop_read, 3)
        os.close(stop_read)
        stop_read = 3
    os.set_inheritable(stop_read, True)
    certificate, key = mint_tls()
    origin_certificate, origin_key = mint_origin_tls()
    boot_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    (STATE / "pod-boot-id").write_text(boot_id + "\n")
    # Creator is the private pod's sole RecordOwner. It mints the renter token and hands
    # the pod only its hash; the proof holds the preimage because it plays Creator.
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    serve_media(certificate, key, token)
    # Authoring also installs the stand-in's exact complete base manifest. The adapter now
    # measures its cozy-runtime wheel fact before it starts the control worker.
    release = author_release()

    environment = {
        "LANG": "C.UTF-8",
        # The artifact origin is this proof's own HTTPS listener, so the pod trusts its
        # certificate the way a pod trusts any private CA: through the ONE platform name the
        # base-worker-image bootstrap already passes through
        # (`config.ADAPTER_CHILD_PLATFORM_ENV`). No new
        # adapter variable, and no verification is skipped anywhere.
        "SSL_CERT_FILE": str(origin_certificate),
        "COZY_ADAPTER_ACQUISITION_ATTEMPT_ID": "ra-cr048bootproof",
        "COZY_ADAPTER_POD_BOOT_ID_PATH": str(STATE / "pod-boot-id"),
        "COZY_ADAPTER_TLS_CERT_PATH": str(certificate),
        "COZY_ADAPTER_TLS_KEY_PATH": str(key),
        "COZY_ADAPTER_RECEIPT_PAYLOAD_PATH": str(STATE / "readiness-payload"),
        "COZY_ADAPTER_WORKER_INTERNAL_PORT": str(WORKER_PORT),
        "COZY_ADAPTER_MEDIA_INTERNAL_PORT": str(MEDIA_PORT),
        "COZY_ADAPTER_RENTER_TOKEN_SHA256_JSON": json.dumps([token_hash]),
        "COZY_ADAPTER_READINESS_DEADLINE_UNIX": str(DEADLINE),
    }
    for retired in RETIRED_ADAPTER_ENV:
        assert retired not in environment
    note(f"handing the adapter {len(environment) - 2} COZY_ADAPTER_* names and no document")

    adapter = subprocess.Popen(
        [str(Path(sys.executable).parent / "cozy-materialize-launch")],
        cwd="/",
        env=environment,
        stdin=subprocess.DEVNULL,
        pass_fds=(stop_read,),
    )
    os.close(stop_read)
    receipt_path = Path(environment["COZY_ADAPTER_RECEIPT_PAYLOAD_PATH"])
    try:
        while not receipt_path.exists():
            if adapter.poll() is not None:
                raise SystemExit(f"the adapter exited {adapter.returncode} before readiness")
            if time.time() >= DEADLINE:
                raise SystemExit("the adapter never published a readiness receipt")
            time.sleep(0.1)
        payload = receipt_path.read_bytes()
        receipt_info = receipt_path.stat()
        assert (receipt_info.st_uid, receipt_info.st_gid) == (65532, 65532)
        assert receipt_info.st_mode & 0o777 == 0o400
        assert access_as(receipt_path, 65532, 65532, os.O_RDONLY)
        assert not access_as(receipt_path, 65533, 65533, os.O_RDONLY)
        receipt = json.loads(payload)
        note(f"receipt published: {len(payload)} B, {len(receipt)} fields")

        assert receipt["format"] == "cozy.pod-readiness/1"
        assert receipt["pod_boot_id"] == boot_id
        assert receipt["observed_renter_token_sha256"] == [token_hash]
        assert receipt["worker_internal_port"] == WORKER_PORT
        assert receipt["media_internal_port"] == MEDIA_PORT
        assert receipt["worker_listener_bound"] and receipt["media_listener_bound"]
        assert receipt["worker_foreign_credential_refused"]
        assert receipt["media_foreign_credential_refused"]
        assert [row["device_name"] for row in receipt["runtime_gpus"]] == ["NVIDIA H200"]
        for field in RETIRED_RECEIPT_FIELDS:
            assert field not in receipt, f"{field} survived the closure retirement"

        note(f"{len(RETIRED_RECEIPT_FIELDS)} closure/image echo fields are absent")

        # READY, AND EMPTY: nothing was materialized, because nothing named anything.
        object_set = Path("/var/lib/cozy/boot/artifact-object-set.json").read_bytes()
        assert object_set == b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}'
        assert sorted(p.name for p in Path("/var/lib/cozy/store").iterdir()) == []
        assert not Path("/var/lib/cozy/environments").exists()
        assert not Path("/var/lib/cozy/provision").exists()
        note("artifact store empty, object set empty, no environment generation")

        owner = RecordOwner(certificate, token.encode())
        try:
            empty = owner.await_state(
                lambda state: state.worker_phase != pb.WorkerPhase.WORKER_PHASE_UNSPECIFIED,
                "the empty pod's first ObservedWorkerState",
                seconds=60,
            )
            ack = owner.claim_ack
            assert ack is not None
            assert len(empty.placements) == 0, describe(empty)
            assert empty.accepted_desired_state_revision == 0, describe(empty)
            assert empty.converged_revision == 0, describe(empty)
            assert list(empty.faults) == [], describe(empty)
            assert owner.snapshot.get("held_attempts", []) == []
            assert ack.worker_id == "ra-cr048bootproof", ack
            assert ack.worker_release_id == "", ack
            base_manifest = json.loads(
                Path("/opt/cozy/base-runtime/WheelhouseManifest.json").read_bytes()
            )
            runtime_digest = next(
                row["digest"]
                for row in base_manifest["wheels"]
                if row["distribution"] == "cozy-runtime"
            )
            assert ack.control_runtime_digest == runtime_digest, ack
            note(
                f"renter Claim ACCEPTED: phase={pb.WorkerPhase.Name(empty.worker_phase)} "
                f"placements=0 release_id={ack.worker_release_id!r}"
            )

            # ---- cr-049: the pod is empty; make it serve, over this same stream ----------
            hub.serve_artifacts(release, origin_certificate, origin_key, ARTIFACT_PORT)
            note(
                f"artifact origin serving {len(release.objects)} exact object(s) on "
                f"127.0.0.1:{ARTIFACT_PORT}; the hub measured receipt "
                f"{hashlib.sha256(release.receipt).hexdigest()[:16]}…"
            )

            owner.send(
                pb.RecordOwnerFrame(artifact_grant_update=owner.envelope(grant_update(release, 1)))
            )
            granted = owner.await_state(
                lambda state: state.applied_grant_revision == 1,
                "the standing grant echoed back",
                seconds=60,
            )
            # REFRESHING ACCESS NEVER CHANGES IDENTITY (§3): no revision moved.
            assert granted.accepted_desired_state_revision == 0, describe(granted)
            assert granted.converged_revision == 0, describe(granted)
            assert granted.applied_artifact_grant_id == "cr049-grant-1", describe(granted)
            note(f"grant landed: {describe(granted)}")

            owner.send(
                pb.RecordOwnerFrame(
                    desired_state=owner.envelope(desired_state(release.placement_set, 1))
                )
            )
            staged = owner.await_state(
                lambda state: placement_is(
                    state,
                    pb.MaterializationState.MATERIALIZATION_STATE_STAGED,
                    pb.ServingState.SERVING_STATE_DISPATCHABLE,
                ),
                "STAGED x DISPATCHABLE for the first placement",
                seconds=900,
            )
            assert staged.accepted_desired_state_revision == 1, describe(staged)
            assert staged.converged_revision == 1, describe(staged)
            assert staged.placements[0].placement_id == hub.PLACEMENT_ID, describe(staged)
            assert staged.placements[0].dispatchable_plan_ids, describe(staged)
            assert staged.admission_state == pb.AdmissionState.ADMISSION_STATE_OPEN
            assert hub.overlap_proven(), "endpoint wheel and model object-set fetches serialized"
            cold_acquisition = staged.placements[0].acquisition
            assert cold_acquisition.endpoint.started_monotonic_ns > 0
            assert cold_acquisition.model.started_monotonic_ns > 0
            assert cold_acquisition.endpoint.ended_monotonic_ns >= (
                cold_acquisition.endpoint.started_monotonic_ns
            )
            assert cold_acquisition.model.ended_monotonic_ns >= (
                cold_acquisition.model.started_monotonic_ns
            )
            assert max(
                cold_acquisition.endpoint.started_monotonic_ns,
                cold_acquisition.model.started_monotonic_ns,
            ) <= min(
                cold_acquisition.endpoint.ended_monotonic_ns,
                cold_acquisition.model.ended_monotonic_ns,
            ), "terminal acquisition intervals do not overlap"
            assert cold_acquisition.endpoint.downloaded_bytes > 0
            assert cold_acquisition.endpoint.reused_bytes == 0
            assert any(
                state.placements
                and (
                    (
                        state.placements[0].acquisition.endpoint.started_monotonic_ns > 0
                        and state.placements[0].acquisition.endpoint.ended_monotonic_ns == 0
                    )
                    or (
                        state.placements[0].acquisition.model.started_monotonic_ns > 0
                        and state.placements[0].acquisition.model.ended_monotonic_ns == 0
                    )
                )
                for state in owner.states
            ), "no partial acquisition observation was emitted"
            assert any(
                state.placements
                and state.placements[0].executor_generation > 0
                and state.placements[0].acquisition.endpoint.ended_monotonic_ns > 0
                and state.placements[0].acquisition.model.started_monotonic_ns > 0
                and state.placements[0].acquisition.model.ended_monotonic_ns == 0
                for state in owner.states
            ), "executor did not spawn after endpoint admission while model hydration continued"
            note(f"CONVERGED: {describe(staged)}")
            note("endpoint-source and model-object-set GETs overlapped at the origin barrier")

            # THE POD BUILT ONLY THE ENDPOINT OVERLAY and derived the hub's exact receipt.
            expected = "sha256:" + hashlib.sha256(release.receipt).hexdigest()
            receipt_path = (
                ENVIRONMENT_ROOT / "receipts" / f"{expected.removeprefix('sha256:')}.json"
            )
            receipt_value = json.loads(release.receipt)
            content_identity = json.dumps(
                {
                    "base_family_digest": receipt_value["base_family_digest"],
                    "format": "cozy.runtime.EndpointOverlayContent/1",
                    "wheel_digests": sorted(
                        row["digest"] for row in receipt_value["overlay_wheels"]
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            content_digest = hashlib.sha256(content_identity).hexdigest()
            generation = ENVIRONMENT_ROOT / "contents" / content_digest
            assert generation.is_dir(), f"no generation at {generation}"
            assert receipt_path.read_bytes() == release.receipt, "the pod receipt differs"
            assert (
                generation / "endpoint-overlay-content.json"
            ).read_bytes() == content_identity, "the installed-content identity differs"
            assert (generation / "site-packages" / "proof_cli_endpoint").is_dir()
            endpoint_file = generation / "site-packages" / "proof_cli_endpoint" / "__init__.py"
            assert access_as(endpoint_file, 65533, 65533, os.O_RDONLY)
            assert not access_as(endpoint_file, 65533, 65533, os.O_WRONLY)
            assert not access_as(receipt_path, 65533, 65533, os.O_RDONLY)
            journal = STATE.parent / "worker" / "journal.log"
            assert journal.is_file() and not access_as(journal, 65533, 65533, os.O_RDONLY)
            # THE WORKER FETCHED THE REACHABLE CLOSURE AND NOTHING ELSE. The grant
            # authorizes the evaluated config too, and no document in the walk names it, so
            # it stays INERT and unfetched — rev-2 §3's rule that a grant may authorize more
            # than a spec reaches, observed rather than asserted.
            fetched = {"sha256:" + p.name for p in ARTIFACT_CACHE.iterdir()}
            reached = {d for d, kind in release.kinds.items() if kind not in {"config", "receipt"}}
            inert = {d for d, kind in release.kinds.items() if kind in {"config", "receipt"}}
            assert fetched == reached, sorted(fetched ^ reached)
            assert inert and not (fetched & inert), sorted(inert)
            note(
                f"the pod fetched the {len(fetched)} object(s) the spec REACHES, left "
                f"{len(inert)} authorized-but-unreached subject(s) inert, and published "
                f"generation {generation.name[:16]}… whose receipt is byte-identical to "
                "the hub's"
            )

            served = serve_one_attempt(
                owner, staged.placements[0].dispatchable_plan_ids[0], hub.ENDPOINT_RELEASE
            )
            assert served["body"]["status"] == pb.OutcomeStatus.OUTCOME_STATUS_SUCCEEDED, served[
                "body"
            ]
            assert served["body"]["execution_started"] is True, served["body"]
            assert served["result"] == {"text": "cr-049cr-049"}, served
            note(f"SERVED one attempt: {served['result']} from the materialized environment")

            isolation = serve_one_attempt(
                owner,
                staged.placements[0].dispatchable_plan_ids[0],
                hub.ENDPOINT_RELEASE,
                text="__cozy_isolation_probe__",
                count=1,
                request_id="cr045-isolation-probe",
            )
            isolation_facts = json.loads(isolation["result"]["text"])
            assert isolation_facts == {
                "euid": 65533,
                "worker_proc_mem_refused": True,
                "worker_secret_read_refused": True,
            }, isolation_facts
            note(
                "actual endpoint executor ran as uid 65533 and refused worker journal plus "
                "/proc/<worker>/mem"
            )

            first_executor_generation = staged.placements[0].executor_generation
            first_scratch = STATE.parent / "scratch" / f"generation-{first_executor_generation}"
            assert first_scratch.is_dir() and first_scratch.stat().st_uid == 65533
            scratch_sentinel = first_scratch / "executor-private"
            scratch_sentinel.write_bytes(b"generation one\n")
            os.chown(scratch_sentinel, 65533, 65533)
            scratch_sentinel.chmod(0o600)
            assert access_as(scratch_sentinel, 65533, 65533, os.O_RDONLY)

            # Leave observed state, then request the exact same placement again. Every
            # endpoint byte is already verified in the artifact cache and the empty model
            # set reaches zero model bytes: both legs must download zero on the warm pass.
            empty_set, _ = documents.identity(pb.PlacementSet())
            owner.send(
                pb.RecordOwnerFrame(desired_state=owner.envelope(desired_state(empty_set, 2)))
            )
            owner.await_state(
                lambda state: state.accepted_desired_state_revision == 2 and not state.placements,
                "the explicit empty placement set",
                seconds=60,
            )
            owner.send(
                pb.RecordOwnerFrame(
                    desired_state=owner.envelope(desired_state(release.placement_set, 3))
                )
            )
            warm = owner.await_state(
                lambda state: (
                    state.accepted_desired_state_revision == 3
                    and state.converged_revision == 3
                    and placement_is(
                        state,
                        pb.MaterializationState.MATERIALIZATION_STATE_STAGED,
                        pb.ServingState.SERVING_STATE_DISPATCHABLE,
                    )
                ),
                "warm zero-download placement convergence",
                seconds=300,
            )
            warm_acquisition = warm.placements[0].acquisition
            assert warm_acquisition.endpoint.downloaded_bytes == 0
            assert warm_acquisition.endpoint.reused_bytes == (
                cold_acquisition.endpoint.downloaded_bytes
            )
            assert warm_acquisition.model.downloaded_bytes == 0
            assert warm_acquisition.model.reused_bytes == 0
            successor_generation = warm.placements[0].executor_generation
            assert successor_generation > first_executor_generation
            assert not first_scratch.exists(), "retired executor scratch survived replacement"
            assert (STATE.parent / "scratch" / f"generation-{successor_generation}").is_dir()
            note(
                "WARM acquisition downloaded 0 B and reused "
                f"{warm_acquisition.endpoint.reused_bytes} endpoint B"
            )

            # ---- the red arm: an incoming spec the grant does not authorize --------------
            broken = release.placement_set.replace(
                expected.encode(),
                (b"sha256:" + b"ee" * 32),
            )
            assert broken != release.placement_set
            owner.send(pb.RecordOwnerFrame(desired_state=owner.envelope(desired_state(broken, 4))))
            refused = owner.await_state(
                lambda state: (
                    state.accepted_desired_state_revision == 4
                    and any(f.reason == "overlay_receipt_mismatch" for f in state.faults)
                ),
                "a typed refusal of the unauthorized revision",
                seconds=180,
            )
            # FALLBACK-RETENTION: the converged placement never stopped serving, and the
            # convergence fact stayed behind the accepted one rather than lying.
            assert refused.converged_revision == 3, describe(refused)
            # The status reports what the placement ACTUALLY HOLDS: the outgoing spec, on
            # disk and serving. The incoming failure is the typed fault beside it.
            assert placement_is(
                refused,
                pb.MaterializationState.MATERIALIZATION_STATE_STAGED,
                pb.ServingState.SERVING_STATE_DISPATCHABLE,
            ), describe(refused)
            assert (
                refused.placements[0].placement_spec_digest
                == staged.placements[0].placement_spec_digest
            ), describe(refused)
            note(f"mismatched revision 4 REFUSED, outgoing keeps serving: {describe(refused)}")

            # A GRANT refresh repairs access; it cannot repair a terminal verdict about the
            # desired spec's exact receipt bytes. Before this arm existed, every newer grant
            # cleared the latch, repeated the full acquisition walk, and appended the same
            # `overlay_receipt_mismatch` fault again while claiming failures were latched.
            mismatch_faults = sum(
                fault.reason == "overlay_receipt_mismatch" for fault in refused.faults
            )
            owner.send(
                pb.RecordOwnerFrame(artifact_grant_update=owner.envelope(grant_update(release, 2)))
            )
            refreshed = owner.await_state(
                lambda state: state.applied_grant_revision == 2,
                "the refreshed grant beside a terminal materialization failure",
                seconds=60,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                owner.pump(deadline)
            after_refresh = owner.states[-1]
            assert after_refresh.accepted_desired_state_revision == 4, describe(after_refresh)
            assert after_refresh.converged_revision == 3, describe(after_refresh)
            assert placement_is(
                after_refresh,
                pb.MaterializationState.MATERIALIZATION_STATE_STAGED,
                pb.ServingState.SERVING_STATE_DISPATCHABLE,
            ), describe(after_refresh)
            assert (
                sum(fault.reason == "overlay_receipt_mismatch" for fault in after_refresh.faults)
                == mismatch_faults
            ), "a grant refresh repeated a terminal materialization fault"
            note(
                "grant revision 2 landed without re-driving terminal revision 4: "
                f"{describe(refreshed)}"
            )
        finally:
            owner.close()
    finally:
        # PID1's capability-free cooperative stop: EOF, not a signal to the root child.
        os.close(stop_write)
        try:
            code = adapter.wait(timeout=20)
        except subprocess.TimeoutExpired:
            adapter.kill()
            code = adapter.wait(timeout=20)
        assert code == 0, f"adapter stop-pipe shutdown exited {code}"
    print(
        "boot proof: PASS — ready and empty with no provision document, then a first "
        "DesiredPlacementSet fetched with observed overlapping legs and early executor spawn; "
        "the uid-65533 endpoint refused worker secret/proc access, scratch rotated, warm reuse "
        "downloaded zero bytes, a mismatched revision retained serving, and fd3 EOF stopped "
        "the root worker cooperatively",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
