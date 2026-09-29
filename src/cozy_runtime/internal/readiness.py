"""Measured bootstrap-readiness facts for the fixed Runtime worker.

No proof here runs under a clock (xs-007 row 18). The pod's lease expiry used to be every
proof's deadline — a GPU inventory, a loopback TLS handshake and two negative-arm dials cut
at the rental cap, and a boot near the cap's end refused `readiness_deadline_elapsed` for
being late rather than wrong. The cap is gone from tensorhub; what ends a boot is the
supervisor cancelling it (fd 3, SIGTERM) or the leg itself exiting, and the supervisor's own
liveness rule is what notices a boot that has stopped.
"""

from __future__ import annotations

import base64
import csv
import os
import socket
import ssl
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import grpc

from cozy_runtime.internal import canonical
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

MAX_HTTP_PROOF_BYTES = 64 << 10
WORKER_TLS_SERVER_NAME = "cozy-worker"


class RuntimeGPU(TypedDict):
    device_index: int
    device_name: str
    device_uuid: str
    driver_version: str
    memory_bytes: int
    pci_bus_id: str


def pod_boot_id(path: Path) -> str:
    try:
        value = path.read_text(encoding="ascii").strip()
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (OSError, UnicodeError, ValueError) as exc:
        raise LaunchRefusal("pod_boot_id_invalid", "boot-id handoff is unreadable") from exc
    encoded = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode()
    if "=" in value or len(decoded) != 32 or encoded != value:
        raise LaunchRefusal(
            "pod_boot_id_invalid", "boot id is not unpadded base64url for exactly 32 bytes"
        )
    return value


def certificate(path: Path) -> bytes:
    try:
        pem = path.read_text(encoding="ascii")
        der = ssl.PEM_cert_to_DER_cert(pem)
        # CPython exposes its certificate decoder through ssl; it is read-only and avoids
        # growing an X.509 parser merely to supply gRPC's target-name override.
        decoded = ssl._ssl._test_decode_cert(str(path))  # type: ignore[attr-defined]
        sans = decoded.get("subjectAltName", ())
    except (OSError, UnicodeError, ValueError, ssl.SSLError) as exc:
        raise LaunchRefusal("tls_certificate_invalid", "certificate handoff is unreadable") from exc
    if not der or tuple(sans) != (("DNS", WORKER_TLS_SERVER_NAME),):
        raise LaunchRefusal(
            "tls_certificate_invalid",
            f"certificate requires only DNS SAN {WORKER_TLS_SERVER_NAME}",
        )
    return der


def runtime_gpus(child_env: tuple[tuple[str, str], ...], *, required: bool) -> list[RuntimeGPU]:
    """Measure the driver's GPU inventory; the Runtime's own Python needs no torch for it.

    `required` is a CUDA base: an absent or unreadable driver refuses the boot. Otherwise a
    machine without NVIDIA tooling has no GPUs, and an unreadable driver serves CPU only
    with the reason on the worker's log.
    """

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        run = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=dict(child_env),
        )
    except FileNotFoundError as exc:
        if required:
            raise LaunchRefusal("runtime_gpus_unreadable", type(exc).__name__) from exc
        return []
    except (OSError, subprocess.SubprocessError) as exc:
        if required:
            raise LaunchRefusal("runtime_gpus_unreadable", type(exc).__name__) from exc
        print(
            f"[worker] GPU inventory unreadable ({type(exc).__name__}); serving CPU only",
            file=sys.stderr,
            flush=True,
        )
        return []
    rows: list[RuntimeGPU] = []
    try:
        for fields in csv.reader(run.stdout.splitlines(), skipinitialspace=True):
            if len(fields) != 6:
                raise ValueError("field count")
            index = int(fields[0])
            memory_bytes = int(fields[4]) * (1 << 20)
            if index < 0 or memory_bytes <= 0 or not all(field.strip() for field in fields[1:]):
                raise ValueError("incomplete row")
            rows.append(
                {
                    "device_index": index,
                    "device_name": fields[1].strip(),
                    "device_uuid": fields[2].strip(),
                    "driver_version": fields[5].strip(),
                    "memory_bytes": memory_bytes,
                    "pci_bus_id": fields[3].strip(),
                }
            )
    except ValueError as exc:
        raise LaunchRefusal("runtime_gpus_invalid", "nvidia-smi returned an invalid row") from exc
    return validated_gpus(rows, required=required)


def validated_gpus(rows: list[RuntimeGPU], *, required: bool) -> list[RuntimeGPU]:
    """One inventory shape for a measured or a virtual device list."""
    rows = sorted(rows, key=lambda row: row["device_index"])
    indexes = [row["device_index"] for row in rows]
    if len(rows) > 16 or indexes != sorted(set(indexes)) or (required and not rows):
        raise LaunchRefusal(
            "runtime_gpus_invalid", "GPU inventory is empty, duplicate, or oversized"
        )
    return rows


def prove_tls_leaf(port: int, expected_der: bytes, *, owner: str) -> None:
    """Compare one listener's exact TLS leaf DER, never only its trust chain or SAN."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with (
            socket.create_connection(("127.0.0.1", port)) as raw,
            context.wrap_socket(raw, server_hostname=WORKER_TLS_SERVER_NAME) as connection,
        ):
            if connection.getpeercert(binary_form=True) != expected_der:
                raise LaunchRefusal(
                    f"{owner}_tls_peer_mismatch",
                    f"{owner} listener did not present the exact bootstrap certificate",
                )
    except LaunchRefusal:
        raise
    except (OSError, ssl.SSLError) as exc:
        raise LaunchRefusal(f"{owner}_listener_unproven", type(exc).__name__) from exc


def prove_worker_foreign_refusal(port: int, certificate_pem: bytes) -> None:
    """Observe a current-schema Claim rejected specifically for a foreign credential."""

    address = f"127.0.0.1:{port}"
    credentials = grpc.ssl_channel_credentials(certificate_pem)
    channel = grpc.secure_channel(
        address,
        credentials,
        options=(
            ("grpc.ssl_target_name_override", WORKER_TLS_SERVER_NAME),
            ("grpc.default_authority", WORKER_TLS_SERVER_NAME),
        ),
    )
    try:
        claim = pb.Claim(
            record_owner_epoch=1,
            record_owner_id="readiness-negative-arm",
            wire_minor=WIRE_MINOR,
            proof=("foreign-" + uuid.uuid4().hex).encode(),
        )
        call = pb_grpc.WorkerControlStub(channel).Control(iter((pb.RecordOwnerFrame(claim=claim),)))
        frame = next(call)
    except (grpc.RpcError, StopIteration) as exc:
        raise LaunchRefusal("worker_listener_unproven", type(exc).__name__) from exc
    finally:
        channel.close()
    if (
        frame.WhichOneof("msg") != "claim_ack"
        or frame.claim_ack.accepted
        or frame.claim_ack.rejection != pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED
    ):
        raise LaunchRefusal("worker_foreign_credential_unproven", "negative Claim was not refused")


def prove_media_foreign_refusal(port: int, expected_der: bytes) -> None:
    """Observe the media listener's authenticated health route refuse a foreign bearer."""

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with (
            socket.create_connection(("127.0.0.1", port)) as raw,
            context.wrap_socket(raw, server_hostname=WORKER_TLS_SERVER_NAME) as connection,
        ):
            if connection.getpeercert(binary_form=True) != expected_der:
                raise LaunchRefusal(
                    "media_tls_peer_mismatch", "media listener did not present bootstrap cert"
                )
            request = (
                "GET /v1/health HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                f"Authorization: Bearer foreign-{uuid.uuid4().hex}\r\n"
                "Connection: close\r\n\r\n"
            )
            connection.sendall(request.encode("ascii"))
            response = bytearray()
            while b"\r\n" not in response and len(response) <= MAX_HTTP_PROOF_BYTES:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response.extend(chunk)
    except LaunchRefusal:
        raise
    except (OSError, ssl.SSLError) as exc:
        raise LaunchRefusal("media_listener_unproven", type(exc).__name__) from exc
    status = bytes(response).split(b"\r\n", 1)[0]
    if status not in {b"HTTP/1.1 401 Unauthorized", b"HTTP/1.0 401 Unauthorized"}:
        raise LaunchRefusal("media_foreign_credential_unproven", "health did not return 401")


def receipt_payload(
    *,
    control_public_key_ed25519_b64url: str,
    media_token_sha256: Sequence[str],
    pod_boot_id_value: str,
    certificate_der: bytes,
    worker_internal_port: int,
    media_internal_port: int,
    gpus: list[RuntimeGPU],
    process_incarnation: str = "",
) -> bytes:
    """The bootstrap payload: everything the pod MEASURED about itself.

    Six fields went with cr-048 —
    `provision_spec_digest`, `provision_bundle_digest`,
    `measured_installed_environment_receipt_digest`, `staged_placement_set_digest`,
    `artifact_sources` and cache diagnostics — because a pod that boots empty has
    staged nothing, and a boot-frozen "staged" claim is exactly the acceptance-as-reality
    shape #473 retired. That state is per-placement observed state on the rev-2 wire
    (`converged_revision`, MaterializationState x ServingState), reported continuously
    instead of asserted once.

    `measured_control_runtime_wheel_digest`/`_length` went next, and for a different
    reason: they were image-resident bytes measured on the pod and reported back to the
    hub that had already pinned the image BY DIGEST. A value the reader chose is not a
    value the reader learns.

    `acquisition_attempt_id`, `acquisition_attempt_ordinal` and `rental_id` went last, for
    a third reason: the hub verifies this payload under a key it minted for ONE attempt and
    recovered from that attempt's own persisted create bytes. A receipt that authenticates
    has already said which attempt it belongs to; echoing the ids back let the binder
    re-derive, from guest bytes, a fact the signature had established first. `pod_boot_id`
    remains — it is measured here, not told to us, and it is what makes two boots of the
    same pod distinguishable.

    Protocol identity is the package major plus the additive minor exchanged on the control
    stream. Exact generated-binding provenance belongs to the build and CI, not a pod receipt.

    The one-shot HMAC envelope and the fixed bootstrap route already identify this payload;
    embedding a second version tag in the authenticated bytes created another schema authority.
    """

    return canonical.write(
        {
            "media_foreign_credential_refused": True,
            "media_internal_port": media_internal_port,
            "media_listener_bound": True,
            "media_protocol": "https",
            "observed_record_owner_auth": {
                "control_public_key_ed25519_b64url": control_public_key_ed25519_b64url,
                "media_token_sha256": list(media_token_sha256),
            },
            "pod_boot_id": pod_boot_id_value,
            "runtime_gpus": gpus,
            "tls_certificate_der_base64": base64.b64encode(certificate_der).decode("ascii"),
            "worker_foreign_credential_refused": True,
            "worker_internal_port": worker_internal_port,
            "worker_listener_bound": True,
            "worker_protocol": "cozy.worker.v1",
            **({"process_incarnation": process_incarnation} if process_incarnation else {}),
        }
    )


#: The facts the supervisor seals and the Hub attests: which boot, which listener identity,
#: which owner credentials and which GPUs. Every other payload field (ports, proofs, fields a
#: newer Runtime adds) may differ when a worker restarts within the same boot.
ATTESTED_FACTS = (
    "pod_boot_id",
    "tls_certificate_der_base64",
    "observed_record_owner_auth.control_public_key_ed25519_b64url",
    "observed_record_owner_auth.media_token_sha256",
    "runtime_gpus[].device_uuid",
)


def attested(payload: bytes) -> dict[str, object]:
    """The attested facts of one readiness payload, by their `ATTESTED_FACTS` names."""
    value = canonical.parse(payload)
    if not isinstance(value, dict):
        raise ValueError("readiness payload is not an object")
    auth = value.get("observed_record_owner_auth") or {}
    gpus = value.get("runtime_gpus") or []
    if not isinstance(auth, dict) or not isinstance(gpus, list):
        raise ValueError("readiness payload facts are malformed")
    return {
        "pod_boot_id": value.get("pod_boot_id"),
        "tls_certificate_der_base64": value.get("tls_certificate_der_base64"),
        "observed_record_owner_auth.control_public_key_ed25519_b64url": auth.get(
            "control_public_key_ed25519_b64url"
        ),
        "observed_record_owner_auth.media_token_sha256": sorted(
            auth.get("media_token_sha256") or []
        ),
        "runtime_gpus[].device_uuid": sorted(
            str(row.get("device_uuid")) for row in gpus if isinstance(row, dict)
        ),
    }


def publish(path: Path, payload: bytes, *, owner: tuple[int, int] | None = None) -> None:
    """Publish atomically. A worker restarted within the same boot republishes: the new
    payload replaces the old one when every attested fact agrees, and refuses otherwise."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".receipt-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if owner is not None:
            os.chown(temporary, owner[0], owner[1])
            owned = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(owned)
            finally:
                os.close(owned)
        try:
            existing = attested(path.read_bytes())
        except (OSError, ValueError, canonical.CanonicalError):
            existing = None
        if existing is not None:
            changed = [name for name, fact in attested(payload).items() if existing[name] != fact]
            if changed:
                raise LaunchRefusal(
                    "readiness_receipt_conflict",
                    f"this boot already attested different {', '.join(changed)}; "
                    "a restarted worker cannot change what the pod attested",
                )
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
