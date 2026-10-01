"""Checkpoint upload stages bytes idempotently; only finalize commits.

A real retained TensorFS model is pushed by the real effect worker through the real
publication client (verified TLS) to a stand-in Hub whose grants address a stand-in
bucket. The bucket refuses pushes the way R2 does (HTTP 403), for one key or past a grant's
expiry. TensorFS reports a refused PUT at once; it re-sends a 503 only after its measured
patience, so a 503 burst would exercise the same effect retry, far slower.
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
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal import effect_interfaces
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.worker import machine_publication
from cozy_runtime.internal.worker.machine_effects import _STALLED_ATTEMPTS, Effects
from cozy_runtime.internal.worker.machine_publication import (
    PublicationAuthority,
    PublicationRefusal,
    reconcile_checkpoint,
    upload_checkpoint,
)
from cozy_runtime.internal.worker.session import _progress_document
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
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
    """Immutable content-addressed keys; ``failing`` answers 403 until the Hub reopens.

    Each PUT takes ``seconds`` (``slow`` for one key), standing in for transfer time; past
    ``capacity`` concurrent PUTs (0: unbounded) a link is congested and each takes longer
    with the square of its overload. ``starts`` records the PUTs in flight as each began.
    A PUT arriving after its grant's expiry is refused (403) and counted in ``expired``.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.objects: dict[str, bytes] = {}
        self.puts: Counter[str] = Counter()
        self.failing = ""
        self.reopened = threading.Event()
        self.seconds = 0.0
        self.slow: tuple[str, float] = ("", 0.0)
        self.capacity = 0
        self.starts: list[int] = []
        self.active = self.peak = self.expired = 0


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
        self.grant_batches: list[int] = []
        self.verify_batches: list[int] = []
        # Objects the commit itself had to verify: none when staging verified them.
        self.unverified_at_finalize: list[int] = []
        self.finalize_seconds = 0.3
        self.fail_finalization = ""
        self.lose_finalize = False
        self.grant_seconds = 3600
        # Abandoned publications: Hub released their holds and refuses to stage them again.
        self.abandoned: set[str] = set()
        # The machine authorization was revoked: its token and every token it holds refuse.
        self.revoked = False

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
        if self.revoked:
            return (403 if path.endswith("/token") else 401), {"code": "auth.revoked"}
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
            if operation in self.abandoned:
                return 409, {"code": "publication.not_open"}
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
                self.grant_batches.append(len(selected))
                grants, held = [], []
                for key in selected:
                    if key in accepted or self.held(key):
                        accepted.add(key)
                        held.append(
                            {"object_id": key, "length": declared[key], "state": "accepted"}
                        )
                        continue
                    expires = int(time.time()) + self.grant_seconds
                    grants.append(
                        {
                            "object_id": key,
                            "length": declared[key],
                            "url": f"{self.bucket_url}/objects/{key}?expires={expires}",
                            "required_headers": {
                                "if-none-match": "*",
                                "x-amz-checksum-sha256": base64.b64encode(
                                    bytes.fromhex(key[7:])
                                ).decode(),
                            },
                            "expires_at_unix": expires,
                        }
                    )
                return 200, {
                    "grants": grants,
                    "held": held,
                    "server_time_unix": int(time.time()),
                }
            if method == "POST" and action == "verify":
                selected = request["object_ids"]
                assert 0 < len(selected) <= 128 and len(set(selected)) == len(selected)
                self.verify_batches.append(len(selected))
                accepted.update(key for key in selected if self.held(key))
                rows = [
                    {
                        "object_id": key,
                        "length": declared[key],
                        "state": "accepted" if key in accepted else "claimed",
                    }
                    for key in selected
                ]
                return 200, {"publication": {"operation": operation, "objects": rows}}
            if method == "POST" and action == "finalize":
                if self.lose_finalize:
                    return None
                self.unverified_at_finalize.append(len(set(declared) - accepted))
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
        status, value = answer
        # Tensorhub's error envelope: every refusal names its code under "error".
        reply(handler, status, {"error": value} if status >= 400 else value)

    return handle


def bucket_handler(bucket: Bucket) -> Callable[[BaseHTTPRequestHandler, str], None]:
    def handle(handler: BaseHTTPRequestHandler, method: str) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        target, _, expires = handler.path.partition("?expires=")
        key = target.rsplit("/", 1)[1]
        assert method == "PUT" and handler.headers["If-None-Match"] == "*"
        if time.time() >= int(expires):
            with bucket.lock:
                bucket.expired += 1
            reply(handler, 403, b"")
            return
        digest = hashlib.sha256(body).digest()
        assert handler.headers["x-amz-checksum-sha256"] == base64.b64encode(digest).decode()
        with bucket.lock:
            bucket.active += 1
            bucket.peak = max(bucket.peak, bucket.active)
            bucket.starts.append(bucket.active)
            overload = bucket.active / bucket.capacity if bucket.capacity else 1.0
        seconds = bucket.slow[1] if key == bucket.slow[0] else bucket.seconds
        time.sleep(seconds * max(1.0, overload) ** 2)
        with bucket.lock:
            bucket.active -= 1
            bucket.puts[key] += 1
            if key == bucket.failing and not bucket.reopened.is_set():
                status = 403
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

    def progress(self, kind: str = "progress") -> list[dict[str, Any]]:
        page = self.executions.events("owner", "parent")
        return [json.loads(event.body)["payload"] for event in page.events if event.kind == kind]

    def notices(self, kind: str) -> list[dict[str, Any]]:
        page = self.executions.events("owner", "parent")
        return [json.loads(event.body) for event in page.events if event.kind == kind]

    def report(self, request: str, attempt: int, _index: int, frame: dict[str, Any]) -> None:
        """The worker's lane: a frame journals on its parent execution, as Creator reads it."""
        self.executions.progress("owner", request, attempt, _progress_document(frame))


@contextlib.contextmanager
def staging(
    root: Path, setenv: Callable[[str, str], None], *, parts: int = 6
) -> Iterator[tuple[Stage, Hub, Effects]]:
    """One admitted upload_checkpoint call of a ``parts``-tensor model, Hub and bucket up."""
    _, workspace, receipt = produced(root, parts=parts)
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
    trusted, context = certificate(root)
    # The real client builds its default context; the throwaway CA is its only trust root.
    setenv("SSL_CERT_FILE", str(trusted))
    bucket = Bucket()
    hub = Hub(bucket)
    with (
        serving(bucket_handler(bucket)) as bucket_port,
        serving(hub_handler(hub), context) as hub_port,
    ):
        hub.bucket_url = f"http://127.0.0.1:{bucket_port}"
        authority = PublicationAuthority(f"https://localhost:{hub_port}", "worker", WORKER_TOKEN)
        effects = Effects(
            workspace, executions, authority, allow_local=True, progress=staged.report
        )
        try:
            yield staged, hub, effects
        finally:
            effects.close()


@pytest.fixture
def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Stage, Hub, Effects]]:
    with staging(tmp_path, monkeypatch.setenv) as staged:
        yield staged


def drive(effects: Effects, done: Callable[[], bool]) -> None:
    """Try every owed effect once per pass, as its unit does between its backoff waits,
    and release what settled effects held, until `done`: the rule counts tries, not time."""
    deadline = time.monotonic() + 60
    while not done():
        with effects.workspace.locked() as db:
            rows = db.execute(
                "SELECT parent_request,call_index FROM execution_calls WHERE owner='owner' "
                "AND json_extract(CAST(intent AS TEXT),'$.module')=?",
                (effect_interfaces.MODULE,),
            ).fetchall()
        for parent, index in rows:
            call = effects.journal.get("owner", parent, index)
            if not call.result and (not call.safe_code or call.sent):
                with contextlib.suppress(Exception):
                    effects._run("owner", call)
            effects._release("owner", effects.journal.get("owner", parent, index))
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
    # TensorFS reported the refused PUT; the effect re-granted only that object.
    assert hub.bucket.puts[failing] > 1
    assert all(hub.bucket.puts[key] == 1 for key in closure - {failing})
    assert any(
        event.get("stage") == "Uploading checkpoint / retrying"
        and event.get("code") == "publication.adapter_failed.TransferFailed.TRANSFER_FAILED"
        for event in staged.progress()
    )
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_pushes_in_flight_grow_while_they_land_faster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncongested link: more pushes land more bytes, so the window keeps doubling."""
    with staging(tmp_path, monkeypatch.setenv, parts=100) as (staged, hub, effects):
        hub.bucket.seconds = 0.2
        call = settle(staged, effects)
        assert not call.safe_code and canonical_json.decode(call.result)["observation"] == (
            "acknowledged"
        )
        assert len(hub.bucket.objects) == 102 and set(hub.bucket.puts.values()) == {1}
        # Sixteen first, then more once sixteen had landed faster than none.
        assert max(hub.bucket.starts[:17]) <= 16 and hub.bucket.peak >= 32
        # Grants were fetched two windows ahead, within the Hub's bound per request.
        assert hub.grant_batches[0] == 32 and max(hub.grant_batches) <= 128
        assert sum(hub.grant_batches) == 102
        # Staging verified every pushed object; the commit verified none itself.
        assert sum(hub.verify_batches) == 102 and hub.unverified_at_finalize == [0]


def test_congested_upload_verifies_and_releases_custody_after_pushes_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real HTTP pushes and TLS grants settle every object under server-side congestion."""
    with staging(tmp_path, monkeypatch.setenv, parts=200) as (staged, hub, effects):
        hub.bucket.seconds, hub.bucket.capacity = 0.5, 16
        call = settle(staged, effects)
        assert not call.safe_code and call.result
        assert len(hub.bucket.objects) == 202 and set(hub.bucket.puts.values()) == {1}
        # The bucket counts only requests whose bodies arrived. Client measurements
        # also include connection/setup, grant and scheduling delays: server capacity
        # cannot prescribe one exact chosen client window on a loaded CI host.
        assert hub.bucket.active == 0
        assert sum(hub.verify_batches) == 202 and hub.unverified_at_finalize == [0]
        drive(effects, lambda: staged.hold_states() == ["released"])
        assert not staged.calls.has_unsettled_effects("owner", "parent")


@pytest.mark.parametrize(
    "seconds,levels",
    [((1.0, 4.0, 0.25), (32, 16, 16)), ((1.0, 1.0, 1.0), (32, 64, 64))],
)
def test_push_window_follows_observed_throughput_and_keeps_the_best_level(
    monkeypatch: pytest.MonkeyPatch,
    seconds: tuple[float, ...],
    levels: tuple[int, ...],
) -> None:
    """Exercise the real policy with known byte/rate observations, apart from TLS noise."""
    now = 0.0

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return now

    # Control only this policy's clock; the integration above uses real transport
    # and clocks. No production callback or replacement pacing policy is involved.
    monkeypatch.setattr(machine_publication, "time", Clock)
    pace = machine_publication._Pace(16, 64)
    pace.started()
    now += 0.25
    pace.ended(16, None)
    assert (pace.level, pace.count, pace.settled) == (16, 0, False)

    for duration, expected in zip(seconds, levels, strict=True):
        level = pace.level
        for _ in range(level):
            pace.started()
        now += duration
        for _ in range(level):
            pace.ended(level, (1 << 20, duration))
        assert pace.level == expected and pace.running == 0
    assert pace.settled and pace.best[1] == levels[-1]


def test_upload_reports_held_bytes_then_publication_with_stage_timings(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    digest = staged.artifact.manifest.digest
    total = staged.artifact.manifest.length + sum(
        int(row["length"])
        for row in tensorfs.Store.ensure(staged.workspace.store_root).walk_cozytensors(digest)
    )
    hub.finalize_seconds = 1.5
    call = settle(staged, effects)
    assert not call.safe_code and call.result
    # The parent's poll reads the call's plan; an effect's frozen plan names no kind or
    # result schema (production run 1511 refused `child_call_refused` here).
    assert call.plan().result_schema is None
    progress = staged.progress()
    upload = [event for event in progress if event.get("stage") == "Uploading checkpoint"]
    assert upload[0] == {"stage": "Uploading checkpoint", "step_ms": 0.0}
    assert upload[-1]["position"] == upload[-1]["total"] == total
    assert upload[-1]["unit"] == "bytes"
    # Every byte is held before Hub's finalization publishes the checkpoint.
    stages = [event["stage"] for event in progress if "stage" in event]
    assert stages.index("Publishing checkpoint") > stages.index("Uploading checkpoint")
    assert "Publishing checkpoint / reconciling" in stages
    phases = {
        log["name"]: log["fields"]
        for log in staged.progress("log")
        if "phase" in log.get("fields", {})
    }
    assert list(phases) == ["Uploading checkpoint", "Publishing checkpoint"]
    uploaded, published = phases["Uploading checkpoint"], phases["Publishing checkpoint"]
    assert uploaded["completed"] and uploaded["bytes"] == uploaded["moved_bytes"] == total
    assert uploaded["rate_bytes_per_second"] > 0
    # The publication stage spans the finalization retries, not only the last attempt.
    assert published["completed"] and published["elapsed_ms"] >= 1500
    # The settled upload is a call of its run, recorded on the root like a child call: its
    # label, status, and each stage's time and bytes (`cozy run show` lists it).
    (record,) = staged.notices("call")
    assert record["label"].startswith("Upload checkpoint to ") and record["status"] == "succeeded"
    assert record["module"] == effect_interfaces.MODULE and record["export"] == "upload_checkpoint"
    assert record["index"] == call.call_index and record["called_unix_ms"] == call.created_ms
    spans = record["stages"]
    # A record is canonical JSON (sorted keys); its spans order by when they started.
    assert sorted(spans, key=lambda name: spans[name]["started_unix_ms"]) == [
        "Uploading checkpoint",
        "Publishing checkpoint",
    ]
    assert spans["Uploading checkpoint"]["bytes"] == total
    assert spans["Publishing checkpoint"]["total_ms"] >= 1500
    assert (
        spans["Uploading checkpoint"]["ended_unix_ms"]
        <= spans["Publishing checkpoint"]["started_unix_ms"] + 1
    )


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
        {"phase": event["phase"], "code": event["code"]}
        for event in staged.progress()
        if "phase" in event
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
    # Earlier phases' rows land first; the lost finalize is settled once it reconciles.
    drive(
        effects,
        lambda: (
            staged.call().sent
            and any(row.get("phase") == "reconciling" for row in staged.progress())
        ),
    )
    assert staged.progress()[-1] == {
        "stage": "Publishing checkpoint / reconciling",
        "operation": "publication",
        "call_index": 0,
        "publication": staged.call().child_request,
        "destination": "alice/model",
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
    (record,) = staged.notices("call")
    assert record["status"] == "failed" and record["error"] == call.safe_detail
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


def upload(staged: Stage, effects: Effects, window: int) -> CheckpointRef:
    """The real upload of the stage's retained closure, at the window a pod's memory allows."""
    assert effects.authority is not None
    client = effects.authority.client(AUTHORIZATION)
    store = tensorfs.Store.ensure(staged.workspace.store_root)
    manifest = staged.artifact.manifest
    objects = {manifest.digest: manifest.length} | {
        str(row["id"]): int(row["length"]) for row in store.walk_cozytensors(manifest.digest)
    }
    checkpoint = CheckpointRef("alice/model", manifest.digest, manifest, "direct", "")

    def push(key: str, length: int, grant: str) -> None:
        store.checkpoint_push(key, length, key == manifest.digest, grant, allow_local=True)

    try:
        return upload_checkpoint(
            client,
            operation="direct",
            checkpoint=checkpoint,
            objects=objects,
            before_stage=lambda: None,
            before_write=lambda: None,
            push=push,
            landed=lambda count, held: None,
            streams=window,
        )
    except PublicationRefusal as exc:
        assert exc.code == "publication.finalization_pending"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        with contextlib.suppress(PublicationRefusal):
            settled = reconcile_checkpoint(
                client, operation="direct", checkpoint=checkpoint, objects=objects
            )
            assert settled is not None
            return settled
        time.sleep(0.05)
    raise AssertionError("finalization did not complete")


@pytest.mark.parametrize("window", [1, 3])
def test_a_narrow_window_still_uploads_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: int
) -> None:
    """No RAM floor: a small pod pushes fewer at once, down to one, and still commits."""
    with staging(tmp_path, monkeypatch.setenv, parts=20) as (staged, hub, effects):
        hub.bucket.seconds = 0.2
        assert upload(staged, effects, window).observation == "acknowledged"
        assert len(hub.bucket.objects) == 22 and set(hub.bucket.puts.values()) == {1}
        assert hub.bucket.peak == window
        # Grants are fetched two windows ahead.
        batch = 2 * window
        assert hub.grant_batches == [batch] * (22 // batch) + ([22 % batch] if 22 % batch else [])
        assert sum(hub.verify_batches) == 22 and hub.unverified_at_finalize == [0]


def test_a_grant_gone_cold_in_the_queue_is_asked_for_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-push window outlasts a queued grant: its object is re-granted, never sent cold."""
    with staging(tmp_path, monkeypatch.setenv, parts=2) as (staged, hub, effects):
        hub.grant_seconds, hub.bucket.seconds = 2, 1.2
        assert upload(staged, effects, 1).observation == "acknowledged"
        assert set(hub.bucket.puts.values()) == {1} and len(hub.bucket.objects) == 4
        assert hub.bucket.expired == 0
        # The second grant of each batch went cold behind the push ahead of it.
        assert hub.grant_batches == [2, 2, 2, 1]
        assert hub.opens["direct"] == 1


def test_sent_upload_settles_when_the_hub_refuses_to_stage_it_again(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    """The finalize never arrived and Hub abandoned the publication: settle, parent live."""
    staged, hub, effects = stage
    hub.lose_finalize = True
    drive(effects, lambda: any(e.get("phase") == "reconciling" for e in staged.progress()))
    operation = staged.call().child_request
    hub.lose_finalize = False
    hub.abandoned.add(operation)
    call = settle(staged, effects)
    assert call.safe_code == "publication.http_refused"
    assert call.safe_detail == "publication commit did not land"
    assert not call.sent and not call.result and not hub.finalizations
    assert hub.opens[operation] == 1
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")
    assert not staged.calls.canceled("owner", call)


def test_sent_upload_settles_once_restaging_stops_landing_objects(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    """Every finalize is lost: after the stall rule, a fresh read still finds none, so settle."""
    staged, hub, effects = stage
    hub.lose_finalize = True
    operation = staged.call().child_request
    drive(effects, lambda: bool(staged.call().safe_code))
    call = staged.call()
    assert call.safe_code == "publication.control_transport_unavailable"
    assert call.safe_detail == "publication commit did not land"
    assert not call.sent and not call.result and not hub.finalizations
    # One attempt landed the closure; each later one staged it again and landed nothing.
    assert hub.opens[operation] == 1 + _STALLED_ATTEMPTS
    assert set(hub.bucket.puts.values()) == {1}
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_a_retry_opens_only_the_objects_it_sends(stage: tuple[Stage, Hub, Effects]) -> None:
    """The retention guards the closure; a retry leases just what Hub still lacks.

    A tensor part Hub already holds is removed locally between attempts. Retaining the
    closure again or leasing all of it would refuse on it; the retry only checks the hold,
    sends the failed part and commits.
    """
    staged, hub, effects = stage
    store = tensorfs.Store.ensure(staged.workspace.store_root)
    manifest = staged.artifact.manifest.digest
    parts = sorted(
        str(row["id"]) for row in store.walk_cozytensors(manifest) if row["kind"] == "part"
    )
    failing, landed = parts[3], parts[0]
    hub.bucket.failing = failing
    drive(effects, lambda: any(e.get("phase") == "retrying" for e in staged.progress()))
    assert hub.held(landed)
    digest = landed.removeprefix("sha256:")
    (staged.workspace.store_root / "blobs" / digest[:2] / digest[2:4] / digest).unlink()
    call = settle(staged, effects)
    assert not call.safe_code and canonical_json.decode(call.result)["observation"] == (
        "acknowledged"
    )
    assert hub.bucket.puts[failing] > 1 and hub.bucket.puts[landed] == 1
    drive(effects, lambda: staged.hold_states() == ["released"])


def owner_read(hub: Hub, operation: str) -> tuple[int, bytes]:
    """The owner's own GET of the finalization status, as Hub answers it."""
    with hub.lock:
        view = hub.finalizations.get(operation)
        if view is None:
            body: dict[str, Any] = {
                "error": {"code": "publication.finalization_absent", "message": "absent"}
            }
            return 404, json.dumps(body).encode()
        return 200, json.dumps({k: v for k, v in view.items() if k != "manifest"}).encode()


def revoke_after_send(staged: Stage, hub: Hub, effects: Effects) -> Call:
    """The finalize is sent, then the machine authorization stops answering."""
    drive(effects, lambda: staged.call().sent)
    hub.revoked = True
    drive(effects, lambda: bool(staged.notices("publication_unresolved")))
    return staged.call()


def test_expired_machine_authority_is_settled_committed_from_the_owners_read(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.finalize_seconds = 3600  # the job commits when the test says so
    call = revoke_after_send(staged, hub, effects)
    publication = {"call_index": 0, "publication": call.child_request, "destination": "alice/model"}
    assert staged.notices("publication_unresolved") == [
        {**publication, "code": "publication.authority_refused"}
    ]
    drive(effects, lambda: any(e.get("publication") for e in staged.progress()))
    assert {
        **publication,
        "phase": "reconciling",
        "code": "publication.authority_refused",
    }.items() <= [e for e in staged.progress() if e.get("publication")][-1].items()
    # The owner reads the job still running: pending is not a settlement.
    with pytest.raises(WorkspaceRefusal, match="finalization_pending"):
        effects.reconcile("owner", "parent", 0, *owner_read(hub, call.child_request))
    # Another operation's document does not settle this call.
    status, body = owner_read(hub, call.child_request)
    forged = json.loads(body) | {"operation": "call-other"}
    with pytest.raises(WorkspaceRefusal):
        effects.reconcile("owner", "parent", 0, status, json.dumps(forged).encode())
    hub.finish(call.child_request)
    effects.reconcile("owner", "parent", 0, *owner_read(hub, call.child_request))
    settled = staged.call()
    assert canonical_json.decode(settled.result)["checkpoint"] == staged.artifact.manifest.digest
    assert staged.notices("publication_settled") == [{**publication, "committed": True}]
    effects.reconcile("owner", "parent", 0, *owner_read(hub, call.child_request))  # idempotent
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_expired_machine_authority_with_no_finalization_releases_its_hold(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.lose_finalize = True
    call = revoke_after_send(staged, hub, effects)
    # A 401 of the owner's own is not an answer.
    with pytest.raises(WorkspaceRefusal, match="not a settlement"):
        effects.reconcile("owner", "parent", 0, 401, b'{"error":{"code":"auth.required"}}')
    effects.reconcile("owner", "parent", 0, *owner_read(hub, call.child_request))
    settled = staged.call()
    assert settled.safe_code == "publication.finalize_absent" and not settled.sent
    assert staged.notices("publication_settled")[-1]["committed"] is False
    drive(effects, lambda: staged.hold_states() == ["released"])
    assert not staged.calls.has_unsettled_effects("owner", "parent")


def test_a_machine_whose_authority_answers_refuses_owner_reconciliation(
    stage: tuple[Stage, Hub, Effects],
) -> None:
    staged, hub, effects = stage
    hub.finalize_seconds = 5.0
    drive(effects, lambda: staged.call().sent)
    with pytest.raises(WorkspaceRefusal, match="still reads Hub"):
        effects.reconcile("owner", "parent", 0, *owner_read(hub, staged.call().child_request))
    assert not staged.call().result
