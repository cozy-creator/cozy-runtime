"""Direct weights upload: a real PUT to a real store, under one presigned grant.

Runtime holds produced weights, so Runtime streams them. These tests exercise that path the
way it actually runs — a real `WeightsExchange`, a real HTTP object store on a real socket,
the real `internal/egress.py` boundary — and assert the things that would silently corrupt
or over-authorize a transfer: that the store receives the exact bytes, that the grant's own
signed headers arrive verbatim, that HTTP 412 reads as already-present rather than failure,
that a request which does not bind the held object refuses typed, and that a grant naming an
address this runtime may not reach is refused by the one address predicate.

Both cached manifest bytes and a real native output object are covered. Native roots own
idle output retention; an active object transfer alone holds a read lease against GC.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.weights_sink import protocol_receipt
from cozy_runtime.internal.worker import weights
from cozy_runtime.protocol import worker_pb2 as pb
from test_weights_resume_native import _committed

TRANSACTION = "sha256:" + "a" * 64
WRITER_EPOCH = 9
OBJECT = b"produced-manifest-bytes" * 512
OBJECT_ID = "sha256:" + hashlib.sha256(OBJECT).hexdigest()
SOURCE_REF = "tensorfs-manifest:" + OBJECT_ID
CHECKSUM_HEADER = base64.b64encode(hashlib.sha256(OBJECT).digest()).decode()


class ObjectStore:
    """A minimal immutable object store: it records ONE PUT and answers a chosen status."""

    def __init__(self) -> None:
        self.body = b""
        self.headers: dict[str, str] = {}
        self.raw_headers: list[tuple[str, str]] = []
        self.status = 200
        self.received = threading.Event()
        self.response_gate: threading.Event | None = None


@contextlib.contextmanager
def serving(store: ObjectStore) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            store.body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            store.raw_headers = list(self.headers.items())
            store.headers = {k.lower(): v for k, v in store.raw_headers}
            store.received.set()
            if store.response_gate is not None:
                store.response_gate.wait(10)
            self.send_response(store.status)
            self.send_header("ETag", '"stored"')
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def exchange(tmp_path: Path, *, allow_private: bool) -> weights.WeightsExchange:
    """One exchange already holding a committed receipt for the manifest object."""
    made = weights.WeightsExchange(
        store_root=tmp_path,
        stop=threading.Event(),
        owner_scope=lambda: "test-owner",
        allow_private_egress=allow_private,
    )
    made._held[TRANSACTION] = weights._Held(
        transaction_id=TRANSACTION,
        writer_epoch=WRITER_EPOCH,
        receipt=pb.WeightsReceiptRef(),
        objects={
            OBJECT_ID: pb.WeightsObjectSource(
                object_id=OBJECT_ID, length=len(OBJECT), source_ref=SOURCE_REF
            )
        },
        manifest_id=OBJECT_ID,
        manifest_bytes=OBJECT,
        manifest=pb.Ref(digest=hashlib.sha256(OBJECT).digest(), length=len(OBJECT)),
    )
    return made


def request(base: str, **overrides: object) -> pb.WeightsUploadRequest:
    fields: dict[str, object] = {
        "weights_transaction_id": TRANSACTION,
        "writer_epoch": WRITER_EPOCH,
        "object_id": OBJECT_ID,
        "source_ref": SOURCE_REF,
        "length": len(OBJECT),
        "operation_id": "test-upload",
        "grant_revision": 1,
        "grant": pb.WeightsUploadGrant(
            object_id=OBJECT_ID,
            length=len(OBJECT),
            url=f"{base}/objects/{OBJECT_ID}?X-Amz-Signature=test",
            required_headers=[
                pb.WeightsUploadHeader(name="if-none-match", value="*"),
                pb.WeightsUploadHeader(name="x-amz-checksum-sha256", value=CHECKSUM_HEADER),
            ],
            expires_at_unix=2000000000,
        ),
    }
    fields.update(overrides)
    return pb.WeightsUploadRequest(**fields)  # type: ignore[arg-type]


def test_upload_streams_the_exact_object_under_the_grant(tmp_path: Path) -> None:
    store = ObjectStore()
    with serving(store) as base:
        result = exchange(tmp_path, allow_private=True).upload(request(base))

    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED, result.safe_detail
    assert store.body == OBJECT
    assert result.transferred_bytes == len(OBJECT)
    # The digest Runtime reports is the object it claims to be. The store enforced the same
    # digest independently, through the grant's signed checksum header.
    assert result.checksum_sha256 == OBJECT_ID
    # Those signed headers must reach the store unaltered: they are what bind this capability
    # to one object and to a first write.
    assert store.headers.get("x-amz-checksum-sha256") == CHECKSUM_HEADER
    assert store.headers.get("if-none-match") == "*"
    # Collapsing names into a dict concealed duplicate headers on the real wire.
    for name in ("if-none-match", "x-amz-checksum-sha256", "content-length"):
        assert sum(key.lower() == name for key, _ in store.raw_headers) == 1, store.raw_headers


def test_precondition_failure_is_already_present_not_failure(tmp_path: Path) -> None:
    store = ObjectStore()
    store.status = 412
    with serving(store) as base:
        result = exchange(tmp_path, allow_private=True).upload(request(base))

    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
    assert result.http_status == 412


@pytest.mark.parametrize(
    ("overrides", "want"),
    [
        ({"writer_epoch": 8}, pb.WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER),
        (
            {"source_ref": "tensorfs-object:sha256:" + "0" * 64},
            pb.WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH,
        ),
        (
            {"weights_transaction_id": "sha256:" + "b" * 64},
            pb.WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION,
        ),
        ({"object_id": "sha256:" + "c" * 64}, pb.WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT),
        ({"length": len(OBJECT) + 1}, pb.WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE),
    ],
)
def test_a_request_that_does_not_bind_the_held_object_refuses_typed(
    tmp_path: Path, overrides: dict[str, object], want: pb.WeightsUploadRefusal
) -> None:
    store = ObjectStore()
    with serving(store) as base:
        result = exchange(tmp_path, allow_private=True).upload(request(base, **overrides))

    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
    assert result.refusal == want
    assert store.body == b"", "a refused upload must not have spent the grant"


def test_a_grant_naming_an_unreachable_address_is_refused(tmp_path: Path) -> None:
    """On the PRODUCTION setting, which is the only one worth arming this against.

    `upload` builds its allowlist from the grant's own host, exactly as `write_output` does,
    so the allowlist can never refuse. The whole protection is `egress.blocked()` on every
    resolved address — and `allow_private=True` turns that off wholesale rather than per
    host. An arm written on a private-allowing exchange therefore proves nothing, which is
    why this one does not set it.
    """
    store = ObjectStore()
    with serving(store) as base:
        message = request(base)
        message.grant.url = "http://169.254.169.254/objects/steal"
        result = exchange(tmp_path, allow_private=False).upload(message)

    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
    assert result.refusal == pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED


def test_native_output_leases_belong_only_to_active_transfers(tmp_path: Path) -> None:
    native, _, _, receipt = _committed(tmp_path)
    _, raw, digest = protocol_receipt(
        receipt,
        owner_scope="resume-proof",
        request_id="request",
        invocation_spec_digest="sha256:" + "41" * 32,
    )
    exchange = weights.WeightsExchange(
        store_root=tmp_path / "store",
        stop=threading.Event(),
        owner_scope=lambda: "resume-proof",
        allow_private_egress=True,
    )
    held = exchange._hold(
        receipt.weights_transaction_id,
        1,
        pb.WeightsReceiptRef(weights_receipt_canonical_bytes=raw, weights_receipt_digest=digest),
    )
    row = next(value for value in held.objects.values() if value.length == 2048)
    assert row.source_ref.startswith("tensorfs-object:")
    tensorfs.gc(str(native.root))  # idle metadata is not a second retention authority
    backend = ObjectStore()
    backend.response_gate = threading.Event()
    with serving(backend) as base, ThreadPoolExecutor(max_workers=1) as pool:
        upload = pb.WeightsUploadRequest(
            weights_transaction_id=receipt.weights_transaction_id,
            writer_epoch=1,
            object_id=row.object_id,
            source_ref=row.source_ref,
            length=row.length,
            operation_id="native-transfer",
            grant_revision=1,
            grant=pb.WeightsUploadGrant(
                object_id=row.object_id, length=row.length, url=base + "/object"
            ),
        )
        pending = pool.submit(exchange.upload, upload)
        try:
            assert backend.received.wait(5)
            with pytest.raises(Exception, match="STORE_BUSY"):
                tensorfs.gc(str(native.root), dry_run=True)
        finally:
            backend.response_gate.set()
        result = pending.result(timeout=5)
    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED, result.safe_detail
    assert backend.body == b"\x31" * 2048
    tensorfs.gc(str(native.root))
    assert native.manifest(held.manifest_id)
    exchange.close()
