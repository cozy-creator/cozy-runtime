"""Checkpoint upload stages bytes idempotently; only finalize commits.

A real retained TensorFS model is pushed by the real effect worker through the real
publication client (verified TLS) to a stand-in Hub whose grants address a stand-in
bucket. The bucket fails pushes the way R2 does (HTTP 503 bursts).
"""

from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import json
import ssl
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from cozy_runtime import canonical_json
from cozy_runtime.author import ModelArtifact
from cozy_runtime.internal import effect_interfaces
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.worker.machine_effects import Effects
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Call, Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_calls import running
from test_machine_execution import offer
from test_workspace_custody import produced

AUTHORIZATION = "019aaaab-0000-7000-8000-000000000003"
WORKER_TOKEN = "A" * 43
MODELS = "/v1/models/alice/model"


class Bucket:
    """Immutable content-addressed keys; ``failing`` answers 503 until the Hub reopens."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.objects: dict[str, bytes] = {}
        self.puts: Counter[str] = Counter()
        self.failing = ""
        self.reopened = threading.Event()


class Hub:
    """Tensorhub's publication contract for a private model, as a machine grant sees it.

    Finalize enqueues a durable job (HTTP 202) that commits later, like tensorhub's River
    worker; a private checkpoint stays unreadable (404) to a checkpoint-only grant, and
    reopening a publication whose finalization is in flight conflicts.
    """

    def __init__(self, bucket: Bucket) -> None:
        self.bucket = bucket
        self.lock = threading.Lock()
        self.bucket_url = ""
        self.objects: dict[str, dict[str, int]] = {}
        self.accepted: dict[str, set[str]] = {}
        self.opens: Counter[str] = Counter()
        self.finalizations: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, str] = {}
        self.refuse_grants = 0
        self.finalize_seconds = 0.3
        self.fail_finalization = ""
        self.lose_finalize = False

    def held(self, object_id: str) -> bool:
        with self.bucket.lock:
            body = self.bucket.objects.get(object_id)
        return body is not None and "sha256:" + hashlib.sha256(body).hexdigest() == object_id

    def finish(self, operation: str) -> None:
        with self.lock:
            view = self.finalizations[operation]
            declared = self.objects[operation]
            if self.fail_finalization:
                view.update(
                    state="failed",
                    error={"code": self.fail_finalization, "message": "finalization failed"},
                )
                return
            manifest, length = view.pop("manifest")
            assert all(map(self.held, declared))
            self.checkpoints[manifest] = operation
            view.update(
                state="completed",
                result={
                    "publish_id": operation,
                    "checkpoint_id": manifest,
                    "manifest": {"sha256": manifest[7:], "length": length},
                    "objects": len(declared) - 1,
                    "bytes": sum(declared.values()) - length,
                    "state": "checkpointed",
                    "duplicate": False,
                },
            )

    def route(self, method: str, path: str, body: bytes) -> tuple[int, Any] | None:
        """``None``: the connection drops before Hub reads the request."""
        if method == "POST" and path == f"/v1/worker/machine-authorizations/{AUTHORIZATION}/token":
            return 200, {"token": "hub-token", "expires_at": "2099-01-01T00:00:00Z"}
        if path.startswith(MODELS + "/checkpoints/") and method == "GET":
            return 404, {"code": "checkpoint.absent"}
        prefix = MODELS + "/publications/"
        if not path.startswith(prefix):
            return 404, {"code": "route.absent"}
        operation, _, action = path[len(prefix) :].partition("/")
        status_url = prefix + operation + "/finalization"
        request = json.loads(body) if body else {}
        with self.lock:
            finalization = self.finalizations.get(operation)
            if method == "GET" and action == "finalization":
                if finalization is None:
                    return 404, {"code": "publication.finalization_absent"}
                return 200, {k: v for k, v in finalization.items() if k != "manifest"}
            if method == "PUT" and not action:
                declared = {row["object_id"]: row["length"] for row in request["objects"]}
                if self.objects.setdefault(operation, declared) != declared:
                    return 409, {"code": "publication.object_set_changed"}
                if finalization is not None and finalization["state"] in {"queued", "running"}:
                    return 409, {"code": "publication.finalizing"}
                self.opens[operation] += 1
                if self.opens[operation] > 1:
                    self.bucket.reopened.set()
                accepted = self.accepted.setdefault(operation, set())
                return 200, {
                    "created": self.opens[operation] == 1,
                    "publication": {
                        "operation": operation,
                        "state": "checkpointed"
                        if operation in self.checkpoints.values()
                        else "open",
                        "objects": [
                            {
                                "object_id": key,
                                "length": length,
                                "state": "accepted" if key in accepted else "claimed",
                            }
                            for key, length in sorted(declared.items())
                        ],
                    },
                }
            declared = self.objects[operation]
            accepted = self.accepted[operation]
            if method == "POST" and action == "grants":
                if self.refuse_grants:
                    return self.refuse_grants, {"code": "publication.hold_key_conflict"}
                selected = request["object_ids"]
                assert 0 < len(selected) <= 128 and len(set(selected)) == len(selected)
                grants, held = [], []
                for key in selected:
                    if key in accepted or self.held(key):
                        accepted.add(key)
                        held.append(
                            {"object_id": key, "length": declared[key], "state": "accepted"}
                        )
                        continue
                    grants.append(
                        {
                            "object_id": key,
                            "length": declared[key],
                            "url": f"{self.bucket_url}/objects/{key}?X-Amz-Signature=test",
                            "required_headers": {
                                "if-none-match": "*",
                                "x-amz-checksum-sha256": base64.b64encode(
                                    bytes.fromhex(key[7:])
                                ).decode(),
                            },
                            "expires_at_unix": int(time.time()) + 3600,
                        }
                    )
                return 200, {
                    "grants": grants,
                    "held": held,
                    "server_time_unix": int(time.time()),
                }
            if method == "POST" and action == "finalize":
                if self.lose_finalize:
                    return None
                manifest, length = request["manifest_id"], request["manifest_length"]
                if declared.get(manifest) != length or not all(map(self.held, declared)):
                    return 409, {"code": "publication.objects_unverified"}
                if finalization is None or finalization["state"] == "failed":
                    finalization = self.finalizations[operation] = {
                        "operation": operation,
                        "state": "queued",
                        "status_url": status_url,
                        "manifest": (manifest, length),
                    }
                    job = threading.Timer(self.finalize_seconds, self.finish, (operation,))
                    job.daemon = True
                    job.start()
                return 202, {k: v for k, v in finalization.items() if k != "manifest"}
        return 404, {"code": "route.absent"}


@contextlib.contextmanager
def serving(
    handle: Callable[[BaseHTTPRequestHandler, str], None], context: ssl.SSLContext | None = None
) -> Iterator[int]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            handle(self, "GET")

        def do_PUT(self) -> None:
            handle(self, "PUT")

        def do_POST(self) -> None:
            handle(self, "POST")

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if context is not None:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def reply(handler: BaseHTTPRequestHandler, status: int, body: Any) -> None:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def hub_handler(hub: Hub) -> Callable[[BaseHTTPRequestHandler, str], None]:
    def handle(handler: BaseHTTPRequestHandler, method: str) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        assert handler.headers["X-Cozy-Worker-Token"] == WORKER_TOKEN
        if handler.path.startswith("/v1/models/"):
            assert handler.headers["Authorization"] == "Bearer hub-token"
        answer = hub.route(method, handler.path, body)
        if answer is None:
            handler.close_connection = True
            return
        reply(handler, *answer)

    return handle


def bucket_handler(bucket: Bucket) -> Callable[[BaseHTTPRequestHandler, str], None]:
    def handle(handler: BaseHTTPRequestHandler, method: str) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        key = handler.path.split("?", 1)[0].rsplit("/", 1)[1]
        assert method == "PUT" and handler.headers["If-None-Match"] == "*"
        digest = hashlib.sha256(body).digest()
        assert handler.headers["x-amz-checksum-sha256"] == base64.b64encode(digest).decode()
        with bucket.lock:
            bucket.puts[key] += 1
            if key == bucket.failing and not bucket.reopened.is_set():
                status = 503
            elif key in bucket.objects:
                status = 412
            else:
                bucket.objects[key] = body
                status = 200
        reply(handler, status, b"")

    return handle


def certificate(root: Path) -> tuple[Path, ssl.SSLContext]:
    """A throwaway CA and a localhost leaf; the client trusts only that CA."""
    now = datetime.datetime.now(datetime.UTC)
    authority_key, leaf_key = (
        ec.generate_private_key(ec.SECP256R1()),
        ec.generate_private_key(ec.SECP256R1()),
    )
    authority_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "stand-in hub CA")])
    authority = (
        x509.CertificateBuilder()
        .subject_name(authority_name)
        .issuer_name(authority_name)
        .public_key(authority_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(False, False, False, False, False, True, True, False, False),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(authority_key.public_key()), critical=False
        )
        .sign(authority_key, hashes.SHA256())
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(authority_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(authority_key.public_key()),
            critical=False,
        )
        .sign(authority_key, hashes.SHA256())
    )
    trusted, chain, key = root / "ca.pem", root / "leaf.pem", root / "leaf.key"
    trusted.write_bytes(authority.public_bytes(serialization.Encoding.PEM))
    chain.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(chain, key)
    return trusted, server


class Stage:
    def __init__(self, workspace: Workspace, executions: Executions, artifact: ModelArtifact):
        self.workspace, self.executions, self.artifact = workspace, executions, artifact
        self.calls = Calls(workspace)

    def call(self) -> Call:
        return self.calls.get("owner", "parent", 0)

    def hold_states(self) -> list[str]:
        recipient = self.call().child_request
        with self.workspace.locked() as db:
            return [
                db.execute(
                    "SELECT state FROM holds WHERE owner='owner' AND id=?",
                    (pb.DerivedRetentionRequest.FromString(row[0]).retention_id,),
                ).fetchone()[0]
                for row in db.execute(
                    "SELECT retention FROM execution_model_holds "
                    "WHERE owner='owner' AND recipient=?",
                    (recipient,),
                ).fetchall()
            ]

    def progress(self) -> list[dict[str, Any]]:
        page = self.executions.events("owner", "parent")
        return [
            json.loads(event.body)["payload"] for event in page.events if event.kind == "progress"
        ]


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Stage, Hub, Effects]]:
    _, workspace, receipt = produced(tmp_path, parts=6)
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer("parent"),
        publication_authorization_id=AUTHORIZATION,
        expected_execution_workspace_id=executions.workspace_id,
    )
    parent = running(executions)
    request = pb.ChildCallRequest(
        parent_request_id=parent.request_id,
        parent_attempt_ordinal=parent.attempt_ordinal,
        parent_invocation_spec_digest=parent.invocation_spec_digest,
        call_index=0,
        module=effect_interfaces.MODULE,
        export="upload_checkpoint",
        request_canonical_bytes=canonical_json.encode(
            msgspec.to_builtins(effect_interfaces.Upload(receipt.artifact, "alice/model"))
        ),
    )
    request.intent_digest = hashlib.sha256(canonical_intent(request)).digest()
    staged = Stage(workspace, executions, receipt.artifact)
    staged.calls.accept("owner", request)
    trusted, context = certificate(tmp_path)
    # The real client builds its default context; the throwaway CA is its only trust root.
    monkeypatch.setenv("SSL_CERT_FILE", str(trusted))
    bucket = Bucket()
    hub = Hub(bucket)
    with (
        serving(bucket_handler(bucket)) as bucket_port,
        serving(hub_handler(hub), context) as hub_port,
    ):
        hub.bucket_url = f"http://127.0.0.1:{bucket_port}"
        authority = PublicationAuthority(f"https://localhost:{hub_port}", "worker", WORKER_TOKEN)
        effects = Effects(workspace, executions, authority, allow_local=True)
        try:
            yield staged, hub, effects
        finally:
            effects.close()


def drive(effects: Effects, done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 60
    while not done():
        effects.drive("owner")
        assert time.monotonic() < deadline, "effect did not settle"
        time.sleep(0.02)


def settle(staged: Stage, effects: Effects) -> Call:
    """Drive until the call has a result or a refusal, the parent's view of it."""
    drive(effects, lambda: bool(staged.call().result or staged.call().safe_code))
    return staged.call()


def test_push_failure_retries_the_frozen_operation_without_repushing(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    closure = {staged.artifact.manifest.digest} | {
        row["id"]
        for row in tensorfs.Store.ensure(staged.workspace.store_root).walk_cozytensors(
            staged.artifact.manifest.digest
        )
    }
    assert len(closure) == 8
    failing = sorted(closure - {staged.artifact.manifest.digest})[3]
    hub.bucket.failing = failing
    call = settle(staged, effects)
    assert not call.safe_code and call.sent
    result = canonical_json.decode(call.result)
    assert result["checkpoint"] == staged.artifact.manifest.digest
    assert result["observation"] == "acknowledged"
    assert hub.opens[call.child_request] == 2
    assert set(hub.bucket.objects) == closure
    # The burst outlived TensorFS's own PUT retries; the effect re-granted only that object.
    assert hub.bucket.puts[failing] > 1
    assert all(hub.bucket.puts[key] == 1 for key in closure - {failing})
    assert any(
        event.get("phase") == "retrying"
        and event.get("code") == "publication.adapter_failed.TransferFailed.TRANSFER_FAILED"
        for event in staged.progress()
    )
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_asynchronous_finalization_is_pending_until_the_hub_commits(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    """Production run 1213: refused 0.4 s after Hub committed its checkpoint."""
    staged, hub, effects = stage
    hub.finalize_seconds = 2.0
    call = settle(staged, effects)
    assert not call.safe_code and call.sent
    result = canonical_json.decode(call.result)
    assert result["checkpoint"] == staged.artifact.manifest.digest
    assert result["observation"] == "acknowledged"
    assert {"phase": "reconciling", "code": "publication.finalization_pending"} in [
        {"phase": event["phase"], "code": event["code"]} for event in staged.progress()
    ]
    # Reconciliation reads the durable job; it never reopens or re-finalizes.
    assert hub.opens[call.child_request] == 1
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_failed_finalization_settles_uncommitted_and_releases_its_hold(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.fail_finalization = "checkpoint.custody_changed"
    call = settle(staged, effects)
    assert call.safe_code == "checkpoint.custody_changed"
    assert call.safe_detail == "publication commit did not land"
    assert not call.sent and not call.result and not hub.checkpoints
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_canceled_parent_settles_a_finalize_that_never_reached_the_hub(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.lose_finalize = True
    drive(effects, lambda: staged.call().sent and bool(staged.progress()))
    assert staged.progress()[-1] == {
        "operation": "publication",
        "call_index": 0,
        "phase": "reconciling",
        "code": "publication.control_transport_unavailable",
    }
    assert not staged.call().safe_code
    staged.executions.control("owner", "parent", "cancel", 1, "cancel")
    drive(effects, lambda: staged.hold_states() == ["released"])
    call = staged.call()
    assert call.safe_code == "publication.finalize_absent"
    assert not call.sent and not call.result and not hub.finalizations
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_refused_publication_settles_before_commit_and_releases_its_hold(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.refuse_grants = 409
    call = settle(staged, effects)
    assert call.safe_code == "publication.http_refused"
    assert call.safe_detail == "publication was refused before its commit"
    assert not call.sent and not call.result
    assert not hub.bucket.puts and not hub.checkpoints
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_canceled_parent_releases_an_uncommitted_upload(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.bucket.failing = staged.artifact.manifest.digest
    drive(effects, lambda: any(e.get("phase") == "retrying" for e in staged.progress()))
    staged.executions.control("owner", "parent", "cancel", 1, "cancel")
    drive(effects, lambda: staged.hold_states() == ["released"])
    call = staged.call()
    assert not call.sent and not call.result and not hub.checkpoints
    assert not staged.calls.has_unsettled_effects("owner", "parent")
