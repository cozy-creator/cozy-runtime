"""Actual TensorFS checkpoint pages and PUT/GET across two independent Stores."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import tensorfs

from cozy_runtime.internal.worker import checkpoint_transport as checkpoint
from cozy_runtime.protocol import worker_pb2 as pb


def _ref(raw: bytes) -> pb.Ref:
    return pb.Ref(digest=hashlib.sha256(raw).digest(), length=len(raw))


def _link(store: Any) -> tuple[pb.CheckpointPageRequest, dict[str, bytes]]:
    payloads = [b"first tensor" * 341, b"second tensor" * 257]
    progress = b'{"converter":"opaque-to-transport"}'
    refs = sorted((_ref(raw) for raw in payloads), key=lambda row: row.digest)
    document = {
        "format": "tensorfs.ingest.durability/1",
        "operation": "op-resume/dit",
        "plan": "sha256:" + "b" * 64,
        "index": 0,
        "blobs": [{"sha256": row.digest.hex(), "length": row.length} for row in refs],
        "manifests": [],
        "bytes": sum(row.length for row in refs),
        "progress": {"sha256": _ref(progress).digest.hex(), "length": len(progress)},
    }
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    objects = {_ref(body).digest.hex(): body for body in [*payloads, progress, raw]}
    for body in objects.values():
        store.put_reader(io.BytesIO(body), _ref(body).digest.hex(), len(body))
    return pb.CheckpointPageRequest(
        subject=pb.CheckpointSubject(
            source=pb.SourceCheckpointSubject(
                operation_id="op-resume",
                source_selection_digest=b"s" * 32,
                slot="dit",
            )
        ),
        plan_digest=b"\xbb" * 32,
        head=_ref(raw),
        limit=1,
    ), objects


@contextmanager
def _origin() -> Iterator[tuple[str, dict[str, bytes], list[tuple[str, str]]]]:
    objects: dict[str, bytes] = {}
    requests: list[tuple[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_PUT(self) -> None:
            key = urlsplit(self.path).path.removeprefix("/")
            requests.append(("PUT", key))
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            assert self.headers["If-None-Match"] == "*"
            assert (
                self.headers["X-Amz-Checksum-Sha256"]
                == base64.b64encode(hashlib.sha256(raw).digest()).decode()
            )
            status = 412 if key in objects else 200
            objects.setdefault(key, raw)
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            key = urlsplit(self.path).path.removeprefix("/")
            requests.append(("GET", key))
            raw = objects[key]
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", objects, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _transfer(subject: pb.CheckpointPageRequest, ref: pb.Ref) -> pb.CheckpointTransferRequest:
    return pb.CheckpointTransferRequest(
        subject=subject.subject,
        plan_digest=subject.plan_digest,
        head=subject.head,
        object=pb.CheckpointObject(ref=ref),
        transfer_id="transfer-" + ref.digest.hex(),
        grant_revision=1,
    )


def test_pages_refuse_cross_subject_and_restore_exact_objects_to_a_new_store(
    tmp_path: Path,
) -> None:
    root = tmp_path / "producer"
    source = tensorfs.Store.ensure(str(root))
    subject, objects = _link(source)
    first = checkpoint.page(subject, tensorfs_root=root)
    assert not first.safe_code, first.safe_detail
    assert len(first.objects) == 1 and first.has_more
    subject.offset = first.next_offset
    last = checkpoint.page(subject, tensorfs_root=root)
    assert len(last.objects) == 1 and not last.has_more
    subject.offset = 0
    wrong = pb.CheckpointPageRequest()
    wrong.CopyFrom(subject)
    wrong.subject.source.slot = "other"
    assert "cross_subject" in checkpoint.page(wrong, tensorfs_root=root).safe_code

    with _origin() as (base, remote, requests):
        for digest, raw in objects.items():
            req = _transfer(subject, _ref(raw))
            req.upload_grant.CopyFrom(
                pb.WeightsUploadGrant(
                    object_id="sha256:" + digest,
                    length=len(raw),
                    url=base + "/" + digest,
                    required_headers=[
                        pb.WeightsUploadHeader(name="if-none-match", value="*"),
                        pb.WeightsUploadHeader(
                            name="x-amz-checksum-sha256",
                            value=base64.b64encode(bytes.fromhex(digest)).decode(),
                        ),
                    ],
                )
            )
            result = checkpoint.transfer(req, tensorfs_root=root, allow_local=True)
            assert result.state == pb.WEIGHTS_TRANSFER_STATE_UPLOADED, result
            assert result.transferred_bytes == len(raw)
        assert remote == objects

        destination = tmp_path / "replacement"
        restored = tensorfs.Store.ensure(str(destination))
        ordered = [
            subject.head.digest.hex(),
            *(key for key in objects if key != subject.head.digest.hex()),
        ]
        for digest in ordered:
            req = _transfer(subject, _ref(objects[digest]))
            req.download_url = base + "/" + digest + "?capability=do-not-log"
            result = checkpoint.transfer(req, tensorfs_root=destination, allow_local=True)
            assert result.state == pb.WEIGHTS_TRANSFER_STATE_HELD, result
            assert result.transferred_bytes == len(objects[digest])
            assert restored.contains(digest)
            before = len(requests)
            replay = checkpoint.transfer(req, tensorfs_root=destination, allow_local=True)
            assert replay.state == pb.WEIGHTS_TRANSFER_STATE_HELD, replay
            assert replay.transferred_bytes == 0
            assert len(requests) == before

        alien = _transfer(subject, _ref(b"not in the checkpoint"))
        alien.download_url = base + "/alien?capability=do-not-log"
        before = len(requests)
        refused = checkpoint.transfer(alien, tensorfs_root=destination, allow_local=True)
        assert refused.state == pb.WEIGHTS_TRANSFER_STATE_FAILED
        assert "object_ungranted" in refused.safe_code
        assert "capability" not in str(refused)
        assert len(requests) == before


def test_corrupt_head_never_authorizes_members_and_never_exposes_get_capability(
    tmp_path: Path,
) -> None:
    source = tensorfs.Store.ensure(str(tmp_path / "source"))
    subject, objects = _link(source)
    with _origin() as (base, remote, requests):
        remote.update(objects)
        remote[subject.head.digest.hex()] = b"wrong bytes"
        req = _transfer(subject, subject.head)
        req.download_url = base + "/" + subject.head.digest.hex() + "?capability=do-not-log"
        result = checkpoint.transfer(req, tensorfs_root=tmp_path / "fresh", allow_local=True)
        assert result.state == pb.WEIGHTS_TRANSFER_STATE_FAILED
        assert "do-not-log" not in str(result)
        assert ("GET", subject.head.digest.hex()) in requests
        assert not tensorfs.Store.open(str(tmp_path / "fresh")).contains(subject.head.digest.hex())
