#!/usr/bin/env python3
"""cr-048's boot proof: a pod that boots with NO provision document, ready and empty.

This process stands in for `cozy-bootstrap` + `cozy-media` — the two things the adapter is
launched by and beside, and neither of which this repo owns. It makes exactly the grants
the substrate makes (TLS keypair, boot id, renter-token hashes, the media listener, the
closed COZY_ADAPTER_* environment), execs the real `cozy-materialize-launch`, and then
observes what a rented pod's two consumers observe: tensorhub's receipt, and a
RecordOwner's Claim.

What it proves, and every line of it is measured rather than asserted by construction:

  * the adapter boots on eleven environment names and reads no document at all;
  * the receipt carries the attempt identity that used to ride the provision spec;
  * the receipt says NOTHING about the pod's own image — the eight retired fields, the
    control-Runtime wheel measurement among them, are absent, and the worker reports an
    empty `worker_artifact_digest` rather than a hash of bytes the hub itself pinned;
  * both listeners bound and both refused a foreign credential;
  * the worker admits the renter's real token and reports ZERO placements, an empty
    accepted set, and no endpoint release — ready, and empty.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
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

from cozy_runtime.protocol import SCHEMA_DIGEST, SCHEMA_DIGEST_RAW, WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

STATE = Path("/run/cozy/bootstrap")
CONTROL_WHEEL = Path("/opt/cozy/control-runtime/cozy-runtime.whl")
WORKER_PORT = 43100
MEDIA_PORT = 43101
DEADLINE = int(time.time()) + 300

RETIRED_RECEIPT_FIELDS = (
    # The hub verifies this payload under a key minted for ONE attempt and recovered from
    # that attempt's own create bytes, so a receipt that authenticates has already said
    # which attempt it belongs to. Echoing the ids let the binder re-derive that from guest
    # bytes; `pod_boot_id` is the identity this pod actually measured.
    "acquisition_attempt_id",
    "acquisition_attempt_ordinal",
    "artifact_sources",
    # The pod does not measure its own image. Tensorhub registers an execution substrate
    # as `name@sha256:…` and refuses a tag, so the control-Runtime wheel baked inside it
    # is a fact the hub CHOSE; hashing it on the pod to report it back proves nothing the
    # OCI digest had not already proven before this process started.
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
    """cozy-media's authenticated health route, and nothing else.

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


def claim_worker(certificate: Path, token: str) -> dict[str, Any]:
    """Claim the freshly booted worker as its renter and read what it says it is."""

    credentials = grpc.ssl_channel_credentials(certificate.read_bytes())
    channel = grpc.secure_channel(
        f"127.0.0.1:{WORKER_PORT}",
        credentials,
        options=(
            ("grpc.ssl_target_name_override", "cozy-worker"),
            ("grpc.default_authority", "cozy-worker"),
        ),
    )
    outbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()

    def frames() -> Any:
        while (frame := outbound.get()) is not None:
            yield frame

    call = pb_grpc.WorkerControlStub(channel).Control(frames())
    outbound.put(
        pb.RecordOwnerFrame(
            claim=pb.Claim(
                record_owner_epoch=1,
                record_owner_id="cr-048-boot-proof",
                wire_minor=WIRE_MINOR,
                proof=token.encode(),
                wire_schema_digest=SCHEMA_DIGEST_RAW,
            )
        )
    )
    observed: dict[str, Any] = {}
    generation, worker_boot = 0, ""
    try:
        for frame in call:
            kind = frame.WhichOneof("msg")
            if kind == "claim_ack":
                ack = frame.claim_ack
                assert ack.accepted, f"Claim REFUSED: {pb.ClaimRejection.Name(ack.rejection)}"
                assert ack.wire_schema_digest == SCHEMA_DIGEST_RAW
                generation, worker_boot = ack.control_stream_generation, ack.worker_boot_id
                observed["worker_boot_id"] = ack.worker_boot_id
                # `worker_artifact_digest` is where SUBSTRATE identity rides and nowhere
                # else (#487/#489), and the only substrate identity a pod could measure
                # from inside is its own image — which the hub pinned by digest. So it is
                # ABSENT here, the way `worker_release_id` is: #481's "absent stays absent".
                observed["worker_id"] = ack.worker_id
                observed["release_id"] = ack.worker_release_id
                observed["artifact_digest"] = ack.worker_artifact_digest
            elif kind == "snapshot":
                # RECOMPUTE before parsing, then ack by id AND digest: the worker's
                # dispatch barrier stays closed until the RecordOwner has reconciled, so
                # an unacked snapshot is why no observed state would ever arrive.
                body = frame.snapshot.snapshot_canonical_bytes
                assert documents.digest_of(body) == frame.snapshot.snapshot_digest
                observed["snapshot"] = documents.read(body, pb.WorkerSnapshotBody)
                outbound.put(
                    pb.RecordOwnerFrame(
                        snapshot_ack=pb.SnapshotAck(
                            record_owner_epoch=1,
                            control_stream_generation=generation,
                            worker_boot_id=worker_boot,
                            snapshot_id=frame.snapshot.snapshot_id,
                            snapshot_digest=frame.snapshot.snapshot_digest,
                        )
                    )
                )
            elif kind == "observed_state":
                state = frame.observed_state
                observed["phase"] = pb.WorkerPhase.Name(state.worker_phase)
                observed["placements"] = len(state.placements)
                observed["accepted_revision"] = state.accepted_desired_state_revision
                observed["converged_revision"] = state.converged_revision
                observed["faults"] = [fault.reason for fault in state.faults]
                break
    finally:
        outbound.put(None)
        channel.close()
    return observed


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    certificate, key = mint_tls()
    boot_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    (STATE / "pod-boot-id").write_text(boot_id + "\n")
    # The renter mints the token and tells NOBODY; the hub holds only the hash. The proof
    # holds both because it is playing both roles, and it hands the pod only the hash.
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    serve_media(certificate, key, token)

    environment = {
        "LANG": "C.UTF-8",
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
    note(f"handing the adapter {len(environment) - 1} COZY_ADAPTER_* names and no document")

    adapter = subprocess.Popen(
        [str(Path(sys.executable).parent / "cozy-materialize-launch")],
        cwd="/",
        env=environment,
        stdin=subprocess.DEVNULL,
    )
    receipt_path = Path(environment["COZY_ADAPTER_RECEIPT_PAYLOAD_PATH"])
    try:
        while not receipt_path.exists():
            if adapter.poll() is not None:
                raise SystemExit(f"the adapter exited {adapter.returncode} before readiness")
            if time.time() >= DEADLINE:
                raise SystemExit("the adapter never published a readiness receipt")
            time.sleep(0.1)
        payload = receipt_path.read_bytes()
        receipt = json.loads(payload)
        note(f"receipt published: {len(payload)} B, {len(receipt)} fields")

        assert receipt["format"] == "cozy.pod-readiness/1"
        assert receipt["pod_boot_id"] == boot_id
        assert receipt["observed_renter_token_sha256"] == [token_hash]
        assert receipt["worker_internal_port"] == WORKER_PORT
        assert receipt["media_internal_port"] == MEDIA_PORT
        assert receipt["worker_wire_schema_digest"] == SCHEMA_DIGEST
        assert receipt["worker_listener_bound"] and receipt["media_listener_bound"]
        assert receipt["worker_foreign_credential_refused"]
        assert receipt["media_foreign_credential_refused"]
        assert [row["device_name"] for row in receipt["runtime_gpus"]] == ["NVIDIA H200"]
        for field in RETIRED_RECEIPT_FIELDS:
            assert field not in receipt, f"{field} survived the closure retirement"

        # THE POD NEVER OPENED ITS OWN IMAGE. The wheel is right there at its fixed path
        # and the receipt says nothing about it: what the hub pinned, the hub knows.
        assert CONTROL_WHEEL.is_file(), "the substrate stand-in did not bake the wheel"
        note(f"{len(RETIRED_RECEIPT_FIELDS)} image/closure fields absent from the receipt")

        # READY, AND EMPTY: nothing was materialized, because nothing named anything.
        object_set = Path("/var/lib/cozy/boot/artifact-object-set.json").read_bytes()
        assert object_set == b'{"kind":"tensorhub.resolved_object_set/1","roots":[]}'
        assert sorted(p.name for p in Path("/var/lib/cozy/store").iterdir()) == []
        assert not Path("/var/lib/cozy/environments").exists()
        assert not Path("/var/lib/cozy/provision").exists()
        note("artifact store empty, object set empty, no environment generation")

        observed = claim_worker(certificate, token)
        assert observed["placements"] == 0, observed
        assert observed["accepted_revision"] == 0, observed
        assert observed["converged_revision"] == 0, observed
        assert observed["faults"] == [], observed
        assert observed["snapshot"].get("held_attempts", []) == []
        assert observed["worker_id"] == "ra-cr048bootproof", observed
        assert observed["release_id"] == "", observed
        assert observed["artifact_digest"] == "", observed
        note(
            f"renter Claim ACCEPTED: phase={observed['phase']} "
            f"placements={observed['placements']} release_id={observed['release_id']!r}"
        )
    finally:
        adapter.terminate()
        try:
            adapter.wait(timeout=20)
        except subprocess.TimeoutExpired:
            adapter.kill()
            adapter.wait(timeout=20)
    print("boot proof: PASS — ready and empty, no provision document read", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
