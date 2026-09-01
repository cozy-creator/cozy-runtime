"""Direct weights upload: a real PUT to a real store, under one presigned grant.

Runtime holds produced weights, so Runtime streams them. These tests exercise that path the
way it actually runs — a real `ArtifactExchange`, a real HTTP object store on a real socket,
the real `internal/egress.py` boundary — and assert the things that would silently corrupt
or over-authorize a transfer: that the store receives the exact bytes, that the grant's own
signed headers arrive verbatim, that HTTP 412 reads as already-present rather than failure,
that a request which does not bind the held object refuses typed, and that a grant naming an
address this runtime may not reach is refused by the one address predicate.

WHAT IS NOT COVERED HERE, AND WHY. `_LeaseReader`, the streaming view over a live TensorFS
lease, needs a real checkpoint, and TensorFS is an optional extra (`model-execution`) that
`uv sync` does not install. So these tests drive the OTHER production source — the produced
Manifest, which is itself one row in the receipt inventory and is uploaded by exactly this
code path. Nothing is stubbed to make that work: the manifest source needs no lease, so the
held row carries none. The lease adapter's own arithmetic is unproven in CI, and saying so
is better than a fake lease that would agree with whatever the adapter did.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cozy_runtime.internal.worker import artifacts
from cozy_runtime.protocol import worker_pb2 as pb

TRANSACTION = "sha256:" + "a" * 64
WRITER_GENERATION = 9
OBJECT = b"produced-manifest-bytes" * 512
OBJECT_ID = "sha256:" + hashlib.sha256(OBJECT).hexdigest()
SOURCE_REF = "tensorfs-manifest:" + OBJECT_ID
CHECKSUM_HEADER = base64.b64encode(hashlib.sha256(OBJECT).digest()).decode()


class ObjectStore:
    """A minimal immutable object store: it records ONE PUT and answers a chosen status."""

    def __init__(self) -> None:
        self.body = b""
        self.headers: dict[str, str] = {}
        self.status = 200


@contextlib.contextmanager
def serving(store: ObjectStore) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self) -> None:
            store.body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            store.headers = {k.lower(): v for k, v in self.headers.items()}
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


def exchange(tmp_path: Path, *, allow_private: bool) -> artifacts.ArtifactExchange:
    """One exchange already holding a committed receipt for the manifest object."""
    made = artifacts.ArtifactExchange(
        store_root=tmp_path,
        send=lambda _frame: None,
        stop=threading.Event(),
        owner_scope=lambda: "test-owner",
        allow_private_egress=allow_private,
    )
    made._held[TRANSACTION] = artifacts._Held(
        transaction_id=TRANSACTION,
        writer_generation=WRITER_GENERATION,
        receipt=pb.WeightsReceiptRef(),
        objects={
            OBJECT_ID: pb.WeightsObjectSource(
                object_id=OBJECT_ID, length=len(OBJECT), source_ref=SOURCE_REF
            )
        },
        manifest_id=OBJECT_ID,
        manifest_bytes=OBJECT,
        manifest=pb.Ref(digest=hashlib.sha256(OBJECT).digest(), length=len(OBJECT)),
        lease=None,  # the manifest source needs none, and this path never reaches for one
    )
    return made


def request(base: str, **overrides: object) -> pb.WeightsUploadRequest:
    fields: dict[str, object] = {
        "weights_transaction_id": TRANSACTION,
        "writer_generation": WRITER_GENERATION,
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
        result = exchange(tmp_path, allow_private=True).upload(request(base)).weights_upload_result

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


def test_precondition_failure_is_already_present_not_failure(tmp_path: Path) -> None:
    store = ObjectStore()
    store.status = 412
    with serving(store) as base:
        result = exchange(tmp_path, allow_private=True).upload(request(base)).weights_upload_result

    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
    assert result.http_status == 412


@pytest.mark.parametrize(
    ("overrides", "want"),
    [
        ({"writer_generation": 8}, pb.WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER),
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
        result = (
            exchange(tmp_path, allow_private=True)
            .upload(request(base, **overrides))
            .weights_upload_result
        )

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
        result = exchange(tmp_path, allow_private=False).upload(message).weights_upload_result

    assert result.outcome == pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
    assert result.refusal == pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED
