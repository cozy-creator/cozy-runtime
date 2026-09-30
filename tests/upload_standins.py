"""Loopback stand-ins for an upload's peers: HuggingFace and Civitai origins, Tensorhub's
publication API with its object store, and the pod supervisor's disk admission.

Each keeps the byte accounting a test asserts on. The Hub stand-in implements the frozen
object-set publication protocol and verified-object custody the real Hub enforces.
"""

from __future__ import annotations

import array
import base64
import contextlib
import hashlib
import http.client
import json
import os
import socket
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cozy_runtime.internal.worker.machine_publication import PublicationRefusal


@contextmanager
def serve(handler: type[BaseHTTPRequestHandler]) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class HuggingFace:
    """Serves one repository's listing and ranged member bodies, counting body bytes. As a
    Civitai origin it serves a model version whose first file is the primary SafeTensors."""

    # TensorFS asks for at most this much of a carrier when it only wants the header.
    PROBE = 64 << 10

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.served: Counter[str] = Counter()  # body bytes per carrier, header probes apart
        self.probed = 0
        self.requests = 0  # every request, listings included
        self.authorization: list[str] = []  # transient fixture observation only
        self.touched: set[str] = set()  # members read at all, header or body
        self.lock = threading.Lock()
        self.watch: Callable[[int], None] = lambda _total: None

    def total(self) -> int:
        """Carrier body bytes moved: what a resumed download must not repeat."""
        with self.lock:
            return sum(self.served.values())

    @contextmanager
    def running(self) -> Iterator[str]:
        origin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def do_GET(self) -> None:
                with origin.lock:
                    origin.requests += 1
                    origin.authorization.append(self.headers.get("Authorization", ""))
                listing: object = None
                if self.path.startswith("/api/models/"):
                    listing = [
                        {
                            "type": "file",
                            "path": path,
                            "size": len(body),
                            "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)},
                        }
                        for path, body in origin.files.items()
                    ]
                elif self.path.startswith("/api/v1/model-versions/"):
                    base = f"http://{self.headers['Host']}/files/"
                    listing = {
                        "id": int(self.path.rsplit("/", 1)[1]),
                        "files": [
                            {
                                "id": index + 1,
                                "name": path,
                                "primary": index == 0,
                                "hashes": {"SHA256": hashlib.sha256(body).hexdigest()},
                                "downloadUrl": base + path,
                            }
                            for index, (path, body) in enumerate(origin.files.items())
                        ],
                    }
                if listing is not None:
                    data = json.dumps(listing).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                member = next(path for path in origin.files if self.path.endswith("/" + path))
                with origin.lock:
                    origin.touched.add(member)
                body = origin.files[member]
                start, end = 0, len(body) - 1
                if "Range" in self.headers:
                    first, _, last = self.headers["Range"].removeprefix("bytes=").partition("-")
                    start, end = int(first), min(int(last or end), end)
                data = body[start : end + 1]
                self.send_response(206 if "Range" in self.headers else 200)
                if "Range" in self.headers:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                probe = member.endswith(".json") or (
                    start == 0 and len(data) == min(len(body), HuggingFace.PROBE)
                )
                try:
                    for offset in range(0, len(data), 1 << 20):
                        self.wfile.write(data[offset : offset + (1 << 20)])
                        with origin.lock:
                            if probe:
                                origin.probed += min(1 << 20, len(data) - offset)
                            else:
                                origin.served[member] += min(1 << 20, len(data) - offset)
                        if not probe:
                            origin.watch(origin.total())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        with serve(Handler) as url:
            yield url


class Civitai:
    """Serves one model version whose primary SafeTensors file is ``body``, ranged."""

    def __init__(self, version: int, file_id: int, body: bytes) -> None:
        self.version, self.file_id, self.body = version, file_id, body
        self.requests = 0

    @contextmanager
    def running(self) -> Iterator[str]:
        origin = self
        listing = json.dumps(
            {
                "id": self.version,
                "files": [
                    {
                        "id": self.file_id,
                        "name": "model.safetensors",
                        "primary": True,
                        "hashes": {"SHA256": hashlib.sha256(self.body).hexdigest().upper()},
                    }
                ],
            }
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def do_GET(self) -> None:
                origin.requests += 1
                if self.path == f"/api/v1/model-versions/{origin.version}":
                    data = listing
                    self.send_response(200)
                else:
                    assert self.path == (
                        f"/api/download/models/{origin.version}?fileId={origin.file_id}"
                    ), self.path
                    start, end = 0, len(origin.body) - 1
                    if spec := self.headers.get("Range"):
                        first, _, last = spec.removeprefix("bytes=").partition("-")
                        start, end = int(first), min(int(last or end), end)
                        self.send_response(206)
                        self.send_header("Content-Range", f"bytes {start}-{end}/{len(origin.body)}")
                    else:
                        self.send_response(200)
                    data = origin.body[start : end + 1]
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        with serve(Handler) as url:
            yield url


class Hub:
    """Tensorhub's model publication routes over one in-memory object store."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stored: dict[str, bytes | None] = {}  # bodies over 1 MiB keep only presence
        self.verified: dict[str, int] = {}
        self.uploads: Counter[str] = Counter()
        self.uploaded_bytes = 0
        self.publications: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, bytes] = {}
        self.abandoned: set[str] = set()
        self.origin = ""

    def client(self) -> HubClient:
        return HubClient(self.origin)

    def _view(self, operation: str, selected: list[str] | None = None) -> dict[str, Any]:
        publication = self.publications[operation]
        ids = selected if selected is not None else sorted(publication["objects"])
        return {
            "operation": operation,
            "state": publication["state"],
            "objects": [
                {
                    "object_id": key,
                    "length": publication["objects"][key],
                    "state": "accepted" if key in self.verified else "claimed",
                }
                for key in ids
            ],
        }

    def _verify(self, key: str, length: int) -> bool:
        if key in self.verified:
            return True
        if key in self.stored:
            self.verified[key] = length
            return True
        return False

    @contextmanager
    def running(self) -> Iterator[str]:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def answer(self, status: int, value: Any = None, raw: bytes | None = None) -> None:
                data = raw if raw is not None else json.dumps(value or {}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length", "0")))

            def do_PUT(self) -> None:
                if self.path.startswith("/r2/"):
                    key = self.path.removeprefix("/r2/")
                    data = self.body()
                    digest = base64.b64encode(hashlib.sha256(data).digest()).decode()
                    with hub.lock:
                        if key in hub.stored:
                            return self.answer(412)
                        if (
                            "sha256:" + hashlib.sha256(data).hexdigest() != key
                            or self.headers.get("x-amz-checksum-sha256") != digest
                        ):
                            return self.answer(400)
                        hub.stored[key] = data if len(data) <= 1 << 20 else None
                        hub.uploads[key] += 1
                        hub.uploaded_bytes += len(data)
                    return self.answer(200)
                operation = self.path.rsplit("/publications/", 1)[1]
                objects = {
                    row["object_id"]: row["length"] for row in json.loads(self.body())["objects"]
                }
                with hub.lock:
                    publication = hub.publications.get(operation)
                    if publication is not None and publication["objects"] != objects:
                        return self.answer(409, {"code": "publication.object_set_changed"})
                    created = publication is None
                    if created:
                        hub.publications[operation] = {"objects": objects, "state": "open"}
                    return self.answer(
                        200, {"created": created, "publication": hub._view(operation)}
                    )

            def do_POST(self) -> None:
                path, _, action = self.path.rpartition("/")
                operation = path.rsplit("/publications/", 1)[1]
                request = json.loads(self.body())
                with hub.lock:
                    publication = hub.publications.get(operation)
                    if publication is None:
                        return self.answer(404)
                    if publication["state"] != "open":
                        return self.answer(409, {"code": "publication.not_open"})
                    if action == "grants":
                        grants, held = [], []
                        for key in request["object_ids"]:
                            length = publication["objects"][key]
                            if hub._verify(key, length):
                                held.append(
                                    {"object_id": key, "length": length, "state": "accepted"}
                                )
                                continue
                            grants.append(
                                {
                                    "object_id": key,
                                    "length": length,
                                    "url": f"{hub.origin}/r2/{key}",
                                    "required_headers": {
                                        "if-none-match": "*",
                                        "x-amz-checksum-sha256": base64.b64encode(
                                            bytes.fromhex(key[7:])
                                        ).decode(),
                                    },
                                    "expires_at_unix": 2_000_000_000,
                                }
                            )
                        return self.answer(
                            200, {"grants": grants, "held": held, "server_time_unix": 1_900_000_000}
                        )
                    if action == "verify":
                        for key in request["object_ids"]:
                            hub._verify(key, publication["objects"][key])
                        return self.answer(
                            200, {"publication": hub._view(operation, request["object_ids"])}
                        )
                    if action == "finalize":
                        manifest = request["manifest_id"]
                        if not all(hub._verify(k, n) for k, n in publication["objects"].items()):
                            return self.answer(409, {"code": "publication.object_unverified"})
                        body = hub.stored.get(manifest)
                        if body is None or len(body) != request["manifest_length"]:
                            return self.answer(409, {"code": "publication.manifest_absent"})
                        publication["state"] = "checkpointed"
                        hub.checkpoints[manifest] = body
                        total = sum(publication["objects"].values())
                        publication["result"] = {
                            "publish_id": operation,
                            "checkpoint_id": manifest,
                            "manifest": {"sha256": manifest[7:], "length": len(body)},
                            "state": "checkpointed",
                            "objects": len(publication["objects"]) - 1,
                            "bytes": total - len(body),
                        }
                        return self.answer(
                            202,
                            {
                                "operation": operation,
                                "state": "completed",
                                "status_url": path + "/finalization",
                                "result": publication["result"],
                            },
                        )
                return self.answer(404)

            def do_GET(self) -> None:
                with hub.lock:
                    if "/checkpoints/" in self.path:
                        body = hub.checkpoints.get(self.path.rsplit("/", 1)[1])
                        return self.answer(404) if body is None else self.answer(200, raw=body)
                    if self.path.endswith("/finalization"):
                        operation = self.path.rsplit("/", 2)[1]
                        publication = hub.publications.get(operation)
                        if publication is None or "result" not in publication:
                            return self.answer(404)
                        return self.answer(
                            200,
                            {
                                "operation": operation,
                                "state": "completed",
                                "status_url": self.path,
                                "result": publication["result"],
                            },
                        )
                return self.answer(404)

            def do_DELETE(self) -> None:
                operation = self.path.rsplit("/publications/", 1)[1]
                with hub.lock:
                    publication = hub.publications.get(operation)
                    if publication is None:
                        return self.answer(404)
                    if publication["state"] == "open":
                        publication["state"] = "abandoned"
                    hub.abandoned.add(operation)
                    return self.answer(200, {"publication": hub._view(operation)})

        with serve(Handler) as url:
            self.origin = url
            yield url


class HubClient:
    """The PublicationClient protocol over plain loopback HTTP."""

    def __init__(self, origin: str) -> None:
        parsed = urlsplit(origin)
        self.host, self.port = parsed.hostname or "", parsed.port or 80

    def request(self, method: str, path: str, body: bytes | None = None) -> bytes:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=120)
        try:
            connection.request(method, path, body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read()
        finally:
            connection.close()
        if not 200 <= response.status < 300:
            raise PublicationRefusal("publication.http_refused", status=response.status)
        return data


def _allocated(root: Path) -> int:
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            with contextlib.suppress(FileNotFoundError):
                total += os.lstat(os.path.join(directory, name)).st_blocks * 512
    return total


class Supervisor:
    """The pod supervisor's admission over a disk of ``capacity`` bytes holding ``root``.

    Free space is capacity minus what ``root`` allocates, so the Runtime sees exactly the
    disk a pod of that size would report. A sampler records the peak allocation.
    """

    def __init__(self, root: Path, capacity: int, reserve: int) -> None:
        self.root, self.capacity, self.reserve = root, capacity, reserve
        self.host, self.channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.exclusion = threading.Lock()
        self.peak = 0
        self.refusals = 0
        self.stopping = threading.Event()

    def _sample(self) -> None:
        while not self.stopping.wait(0.05):
            self.peak = max(self.peak, _allocated(self.root))

    def _serve(self) -> None:
        while True:
            try:
                body, ancillary, _, _ = self.host.recvmsg(4096, socket.CMSG_SPACE(9 * 4))
            except OSError:
                return
            if not body:
                return
            descriptors = array.array("i")
            descriptors.frombytes(ancillary[0][2])
            for descriptor in descriptors[1:]:
                os.close(descriptor)
            operation = socket.socket(fileno=descriptors[0])
            threading.Thread(
                target=self._admit, args=(json.loads(body), operation), daemon=True
            ).start()

    def _admit(self, rows: list[dict[str, Any]], operation: socket.socket) -> None:
        with operation, self.exclusion:
            used = _allocated(self.root)
            self.peak = max(self.peak, used)
            required = sum(row["bytes"] for row in rows)
            available = self.capacity - used
            if required > available - self.reserve:
                self.refusals += 1
                facts = {
                    "available": available,
                    "required": required,
                    "reserve": self.reserve,
                    "reserved": 0,
                    "available_inodes": 1 << 30,
                    "required_inodes": sum(row["inodes"] for row in rows),
                    "reserve_inodes": 0,
                    "reserved_inodes": 0,
                }
                refusal = {
                    "ok": False,
                    "code": "insufficient_storage",
                    "reason": "full",
                    "facts": facts,
                }
                operation.sendall(json.dumps(refusal).encode() + b"\n")
                return
            operation.sendall(b'{"ok":true}\n')
            # The exclusion holds until the admitted writer closes its operation socket.
            operation.recv(1)

    @contextmanager
    def running(self) -> Iterator[socket.socket]:
        threads = [
            threading.Thread(target=self._serve, daemon=True),
            threading.Thread(target=self._sample, daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            yield self.channel
        finally:
            self.stopping.set()
            self.channel.close()
            self.host.close()
            for thread in threads:
                thread.join(5)
