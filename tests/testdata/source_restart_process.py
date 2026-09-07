"""Operator-only proof inside an isolated CPU worker container; never a live pod.

Arguments: candidate pod-supervisor binary, native restart fixture JSON, H3 geometry.
Uses real fixed Runtime startup, native conversion/checkpoint pages and SIGHUP/SIGTERM.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import http.client
import importlib.metadata
import json
import math
import os
import shutil
import signal
import ssl
import struct
import subprocess
import sys
import time
from pathlib import Path

import grpc
import tensorfs

from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc

binary, fixture_path, geometry_path = sys.argv[1:]
fixture = json.loads(Path(fixture_path).read_text())
store_root = Path("/var/lib/tensorfs")
store_root.mkdir(parents=True, exist_ok=True)
os.chown(store_root, 65532, 65532)
subprocess.run(
    ["cp", "-R", str(Path(fixture_path).parent / "store") + "/.", str(store_root)],
    user=65532,
    group=65532,
    check=True,
)
check = subprocess.run(
    ["tfs", "store", "info", str(store_root)],
    user=65532,
    group=65532,
    capture_output=True,
    text=True,
)
assert check.returncode == 0, check.stderr + check.stdout
root = Path("/run/cozy/bootstrap")
key_name = "COZY_BOOTSTRAP_RECEIPT_HMAC_KEY_B64URL"
key = os.urandom(32)
base = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/root",
    "COZY_WORKER_ID": "source-restart-proof",
    "COZY_WORKER_INTERNAL_PORT": "18441",
    "COZY_MEDIA_INTERNAL_PORT": "18442",
    "TENSORHUB_ORIGIN": "https://hub.invalid",
    "COZY_RECORD_OWNER_AUTH_JSON": json.dumps(
        {
            "control_public_key_ed25519_b64url": base64.urlsafe_b64encode(os.urandom(32))
            .rstrip(b"=")
            .decode(),
            "media_token_sha256": [hashlib.sha256(b"source-restart-proof").hexdigest()],
        },
        separators=(",", ":"),
    ),
}
original = None
proof = []


def source_request() -> pb.PrepareModelSourceRequest:
    raw_root = Path("/tmp/hup-source-carriers")
    raw_root.mkdir()
    geometry = json.loads(gzip.decompress(Path(geometry_path).read_bytes()))
    request = pb.PrepareModelSourceRequest(
        operation_id="hup-native-h3",
        source_selection_digest=hashlib.sha256(b"hup-native-h3").digest(),
        profiles=[pb.ModelSourceProfile(slot="dits", profile="hf/minimax-h3/native-dual-bf16/1")],
    )
    for task, tensors in geometry.items():
        sizes = {
            k: math.prod(v["shape"]) * {"BF16": 2, "F32": 4}[v["dtype"]] for k, v in tensors.items()
        }
        chosen = min((k for k, n in sizes.items() if n > 256), key=lambda k: sizes[k])
        assert sizes[chosen] == 384
        index = json.dumps(
            {
                "weight_map": {
                    k: "small.safetensors" if k == chosen else "rest.safetensors" for k in tensors
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        for name, selected in [
            ("model.safetensors.index.json", None),
            ("small.safetensors", [chosen]),
            ("rest.safetensors", [k for k in tensors if k != chosen]),
        ]:
            member = f"{task}/transformer/{name}"
            data: bytes | None
            if selected is None:
                header = index
                length = len(index)
                data = index
            else:
                offset = 0
                table = {}
                for k in sorted(selected):
                    table[k] = {**tensors[k], "data_offsets": [offset, offset + sizes[k]]}
                    offset += sizes[k]
                encoded = json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
                header = struct.pack("<Q", len(encoded)) + encoded
                length = len(header) + offset
                data = header + bytes(offset) if name == "small.safetensors" else None
            path = raw_root / (task + "-" + name)
            header_path = path.with_suffix(path.suffix + ".header")
            header_path.write_bytes(header)
            if data is not None:
                path.write_bytes(data)
            digest = hashlib.sha256(data if data is not None else member.encode()).hexdigest()
            request.files.append(
                pb.LocalModelSourceFile(
                    member=member,
                    object_id="sha256:" + digest,
                    length=length,
                    path=str(path),
                    header_path=str(header_path),
                    verified=data is not None,
                )
            )
    request.files.sort(key=lambda f: f.member)
    return request


def start(first: bool) -> tuple[subprocess.Popen[bytes], Path, bytes, int]:
    env = dict(base)
    if first:
        env[key_name] = base64.urlsafe_b64encode(key).rstrip(b"=").decode()
    log_path = Path("/tmp/source-restart-first.log" if first else "/tmp/source-restart-second.log")
    log = log_path.open("wb")
    process = subprocess.Popen([binary], env=env, stdout=log, stderr=log)
    log.close()
    started = time.monotonic()
    pending = 0
    try:
        while process.poll() is None:
            assert time.monotonic() - started < 60, "readiness observation bound exceeded"
            connection = http.client.HTTPSConnection(
                "127.0.0.1", 18442, timeout=0.5, context=ssl._create_unverified_context()
            )
            try:
                connection.connect()
                assert isinstance(connection.sock, ssl.SSLSocket)
                peer = connection.sock.getpeercert(binary_form=True)
                connection.request("GET", "/v1/bootstrap/receipt")
                response = connection.getresponse()
                body = response.read()
            except (OSError, http.client.HTTPException):
                time.sleep(0.05)
                continue
            finally:
                connection.close()
            if response.status == 425:
                pending += 1
                time.sleep(0.05)
                continue
            assert response.status == 200
            envelope = json.loads(body)
            payload = base64.b64decode(envelope["payload"])
            assert hmac.compare_digest(
                envelope["hmac_sha256"],
                hmac.new(key, b"cozy.pod-readiness/1\0" + payload, hashlib.sha256).hexdigest(),
            )
            assert base64.b64decode(json.loads(payload)["tls_certificate_der_base64"]) == peer
            return process, log_path, body, pending
        raise AssertionError(log_path.read_text())
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        raise


def runtime_child(host_pid: int) -> int:
    children = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            rows = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
        except (OSError, ValueError):
            continue
        if int(rows.get("PPid", "0").strip()) == host_pid and rows.get("Uid", "").split()[0] == "0":
            children.append(int(path.parent.name))
    assert len(children) == 1, children
    return children[0]


first, log, original, pending = start(True)
try:
    request = source_request()
    address = Path("/run/cozy/worker/control.addr").read_text().strip()
    certificate = Path("/run/cozy/bootstrap/tls.crt").read_bytes()
    with grpc.secure_channel(
        address,
        grpc.ssl_channel_credentials(certificate),
        options=[("grpc.ssl_target_name_override", "cozy-worker")],
    ) as channel:
        result = rpc.RuntimePreparationStub(channel).PrepareModelSource(request, timeout=60)
    assert (
        result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE
        and len(result.checkpoints) == 1
    ), result
    checkpoint = result.checkpoints[0]
    fixture["heads"].append(
        {
            "head_id": "sha256:" + checkpoint.head.digest.hex(),
            "head_length": checkpoint.head.length,
            "slot": checkpoint.slot,
            "plan_digest": "sha256:" + checkpoint.plan_digest.hex(),
            "operation": request.operation_id,
        }
    )
    before = tensorfs.Store.open(str(store_root)).model_source_operations()
    assert before == sorted(
        [fixture["operation"], fixture["open_operation"], request.operation_id]
    ), before
    stored_pages = []
    for row in fixture["heads"]:
        stored_pages.append(
            tensorfs.Store.open(str(store_root)).checkpoint_page(
                row["head_id"],
                row["head_length"],
                operation_id=row.get("operation", fixture["operation"]),
                slot=row["slot"],
                plan_digest=row["plan_digest"],
            )
        )
    shutil.rmtree("/tmp/hup-source-carriers")
    started = time.monotonic()
    os.kill(runtime_child(first.pid), signal.SIGHUP)
    first.wait(timeout=15)
    assert "model source cleanup" not in log.read_text(), log.read_text()
    summaries = [
        json.loads(line.removeprefix("[worker] summary "))
        for line in log.read_text().splitlines()
        if line.startswith("[worker] summary ")
    ]
    assert len(summaries) == 1 and summaries[0]["restart_requested"], log.read_text()
    assert tensorfs.Store.open(str(store_root)).model_source_operations() == before
    proof.append(
        {
            "phase": "SIGHUP",
            "seconds": round(time.monotonic() - started, 3),
            "operations": before,
            "heads": fixture["heads"],
            "media_pending_observed": pending,
        }
    )
finally:
    if first.poll() is None:
        first.terminate()
        first.wait(timeout=15)

second, log, envelope, pending = start(False)
try:
    assert envelope == original
    assert tensorfs.Store.open(str(store_root)).model_source_operations() == before
    for row, expected in zip(fixture["heads"], stored_pages, strict=True):
        assert (
            tensorfs.Store.open(str(store_root)).checkpoint_page(
                row["head_id"],
                row["head_length"],
                operation_id=row.get("operation", fixture["operation"]),
                slot=row["slot"],
                plan_digest=row["plan_digest"],
            )
            == expected
        )
    # A normal process stop is not the owner's source disposition. A blocked
    # transaction's source survives even when no prepare was replayed here.
    started = time.monotonic()
    second.send_signal(signal.SIGTERM)
    second.wait(timeout=15)
    assert second.returncode == 0, log.read_text()
    assert tensorfs.Store.open(str(store_root)).model_source_operations() == before
    proof.append(
        {
            "phase": "SIGTERM",
            "seconds": round(time.monotonic() - started, 3),
            "operations": before,
            "same_envelope": True,
            "media_pending_observed": pending,
        }
    )
finally:
    if second.poll() is None:
        second.terminate()
        second.wait(timeout=15)

history = Path("/run/cozy/worker/ownership.json")
held = history.read_bytes()
for state in ["absent", "corrupt"]:
    if state == "absent":
        history.unlink()
    else:
        history.write_bytes(b"{}")
    refused = subprocess.run([binary], env=dict(base), capture_output=True, timeout=15)
    assert refused.returncode != 0 and b"worker_ownership_" in refused.stderr, refused.stderr
    assert tensorfs.Store.open(str(store_root)).model_source_operations() == before
    assert root.joinpath("readiness-envelope.json").read_bytes() == original
    proof.append({"history": state, "refused": True})
history.write_bytes(held)
for operation in before:
    tensorfs.Store.open(str(store_root)).release_model_source(operation)
assert tensorfs.Store.open(str(store_root)).model_source_operations() == []
proof.append({"phase": "explicit owner release", "operations": []})
print(
    json.dumps(
        {
            "proof": proof,
            "software": {
                "runtime": importlib.metadata.version("cozy-runtime"),
                "tensorfs": importlib.metadata.version("tensorfs"),
            },
        },
        indent=2,
    )
)
