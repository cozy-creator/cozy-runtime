"""Measured PodReadinessReceipt/1 facts for the fixed materialize/launch adapter."""

from __future__ import annotations

import base64
import csv
import os
import socket
import ssl
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import grpc

from cozy.worker.v1 import worker_pb2 as pb
from cozy.worker.v1 import worker_pb2_grpc as pb_grpc
from cozy_runtime.internal import canonical
from cozy_runtime.internal.materialize_launch import LaunchRefusal
from cozy_runtime.protocol import SCHEMA_DIGEST_RAW, WIRE_MINOR

MAX_HTTP_PROOF_BYTES = 64 << 10


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


def certificate(path: Path) -> tuple[bytes, str]:
    try:
        pem = path.read_text(encoding="ascii")
        der = ssl.PEM_cert_to_DER_cert(pem)
        # CPython exposes its certificate decoder through ssl; it is read-only and avoids
        # growing an X.509 parser merely to supply gRPC's target-name override.
        decoded = ssl._ssl._test_decode_cert(str(path))  # type: ignore[attr-defined]
        sans = decoded.get("subjectAltName", ())
        ips = [value for kind, value in sans if kind == "IP Address"]
    except (OSError, UnicodeError, ValueError, ssl.SSLError) as exc:
        raise LaunchRefusal("tls_certificate_invalid", "certificate handoff is unreadable") from exc
    if not der or len(ips) != 1:
        raise LaunchRefusal("tls_certificate_invalid", "certificate requires exactly one IP SAN")
    return der, ips[0]


def runtime_gpus(child_env: tuple[tuple[str, str], ...], *, deadline: int) -> list[RuntimeGPU]:
    """Measure the exact executor-visible NVIDIA inventory, sorted by device index."""

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
            timeout=_remaining(deadline),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchRefusal("runtime_gpus_unreadable", type(exc).__name__) from exc
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
    rows.sort(key=lambda row: row["device_index"])
    indexes = [row["device_index"] for row in rows]
    if not 1 <= len(rows) <= 16 or indexes != sorted(set(indexes)):
        raise LaunchRefusal(
            "runtime_gpus_invalid", "GPU inventory is empty, duplicate, or oversized"
        )
    return rows


def wait_worker_listener(root: Path, process: subprocess.Popen[bytes], *, deadline: int) -> str:
    address_file = root / "control.addr"
    while time.time() < deadline:
        if process.poll() is not None:
            raise LaunchRefusal(
                "worker_exited_before_readiness", f"worker exited {process.returncode}"
            )
        try:
            address = address_file.read_text(encoding="ascii").strip()
        except OSError:
            address = ""
        if address:
            return address
        time.sleep(0.02)
    raise LaunchRefusal("readiness_deadline_elapsed", str(deadline))


def prove_worker_foreign_refusal(
    port: int, certificate_pem: bytes, certificate_ip: str, *, deadline: int
) -> None:
    """Observe a current-schema Claim rejected specifically for a foreign credential."""

    address = f"127.0.0.1:{port}"
    credentials = grpc.ssl_channel_credentials(certificate_pem)
    channel = grpc.secure_channel(
        address,
        credentials,
        options=(
            ("grpc.ssl_target_name_override", certificate_ip),
            ("grpc.default_authority", certificate_ip),
        ),
    )
    try:
        claim = pb.Claim(
            record_owner_epoch=1,
            record_owner_id="readiness-negative-arm",
            wire_minor=WIRE_MINOR,
            proof=("foreign-" + uuid.uuid4().hex).encode(),
            wire_schema_digest=SCHEMA_DIGEST_RAW,
        )
        call = pb_grpc.WorkerControlStub(channel).Control(
            iter((pb.RecordOwnerFrame(claim=claim),)), timeout=_remaining(deadline)
        )
        frame = next(call)
    except (grpc.RpcError, StopIteration) as exc:
        raise LaunchRefusal("worker_listener_unproven", type(exc).__name__) from exc
    finally:
        channel.close()
    if (
        frame.WhichOneof("msg") != "claim_ack"
        or frame.claim_ack.accepted
        or frame.claim_ack.rejection != pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED
        or frame.claim_ack.wire_schema_digest != SCHEMA_DIGEST_RAW
    ):
        raise LaunchRefusal("worker_foreign_credential_unproven", "negative Claim was not refused")


def prove_media_foreign_refusal(
    port: int, expected_der: bytes, certificate_ip: str, *, deadline: int
) -> None:
    """Observe the media listener's authenticated health route refuse a foreign bearer."""

    timeout = _remaining(deadline)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with (
            socket.create_connection(("127.0.0.1", port), timeout=timeout) as raw,
            context.wrap_socket(raw, server_hostname=certificate_ip) as connection,
        ):
            connection.settimeout(_remaining(deadline))
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
    spec: Mapping[str, Any],
    provision_spec_digest: str,
    provision_bundle_digest: str,
    provider_pod_id: str,
    pod_boot_id_value: str,
    certificate_der: bytes,
    installed_receipt_digest: str,
    placement_set_digest: str,
    gpus: list[RuntimeGPU],
    artifact_sources: Mapping[str, Mapping[str, int]],
    network_cache_degradations: Sequence[str],
) -> bytes:
    expected_sources = {"cloud", "local", "network_cache"}
    if set(artifact_sources) != expected_sources or any(
        set(row) != {"bytes", "objects"}
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in row.values()
        )
        for row in artifact_sources.values()
    ):
        raise LaunchRefusal("artifact_source_evidence_invalid", "source rows are not closed counts")
    degradations = list(network_cache_degradations)
    if degradations != sorted(set(degradations)) or any(
        value
        not in {
            "backfill_unavailable",
            "corrupt",
            "read_unavailable",
            "unavailable",
        }
        for value in degradations
    ):
        raise LaunchRefusal(
            "network_cache_evidence_invalid", "degradations are not a canonical closed set"
        )
    network = spec["network"]
    return canonical.write(
        {
            "acquisition_attempt_id": spec["acquisition_attempt_id"],
            "acquisition_attempt_ordinal": spec["acquisition_attempt_ordinal"],
            "artifact_sources": {
                source: {"bytes": row["bytes"], "objects": row["objects"]}
                for source, row in artifact_sources.items()
            },
            "format": "cozy.pod-readiness/1",
            "media_foreign_credential_refused": True,
            "media_internal_port": network["media_port"],
            "media_listener_bound": True,
            "media_protocol": network["media_protocol"],
            "measured_installed_environment_receipt_digest": installed_receipt_digest,
            "network_cache_degradations": degradations,
            "observed_renter_token_sha256": spec["renter_token_sha256"],
            "pod_boot_id": pod_boot_id_value,
            "provider_pod_id": provider_pod_id,
            "provision_bundle_digest": provision_bundle_digest,
            "provision_spec_digest": provision_spec_digest,
            "rental_id": spec["rental_id"],
            "runtime_gpus": gpus,
            "staged_placement_set_digest": placement_set_digest,
            "tls_certificate_der_base64": base64.b64encode(certificate_der).decode("ascii"),
            "worker_foreign_credential_refused": True,
            "worker_internal_port": network["worker_port"],
            "worker_listener_bound": True,
            "worker_protocol": network["worker_protocol"],
            "worker_wire_schema_digest": network["worker_wire_schema_digest"],
        }
    )


def publish(path: Path, payload: bytes) -> None:
    """Publish atomically once; an exact replay converges and different bytes refuse."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".receipt-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise LaunchRefusal(
                    "readiness_receipt_conflict", "existing receipt is unreadable"
                ) from exc
            if existing != payload:
                raise LaunchRefusal("readiness_receipt_conflict", "receipt bytes differ") from None
            return
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _remaining(deadline: int) -> float:
    remaining = float(deadline) - time.time()
    if remaining <= 0:
        raise LaunchRefusal("readiness_deadline_elapsed", str(deadline))
    return remaining
