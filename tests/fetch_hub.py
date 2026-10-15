"""A loopback hub serving one checkpoint, and the owner's half of each fetch channel over a
real Store, as TensorD serves them (`tensorfs._ext.serve_fetch`)."""

from __future__ import annotations

import contextlib
import io
import json
import socket
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import numpy as np
import tensorfs
from tensorfs import Store, _ext
from tensorfs.derived import Config, Derivation, Part, Target, Tensor
from tensorfs.fetch import Fetch

from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.fetch_plan import Fetches
from cozy_runtime.internal.weights_sink import weights_transaction_id


def write(root: Path, components: Mapping[str, Mapping[str, np.ndarray]]) -> tuple[Store, str]:
    """`components`' f16 tensors as one checkpoint in a new Store; its manifest id."""
    store = Store.ensure(str(root))
    targets = {
        name: Target(
            add={
                key: Tensor("f16", value.shape, SPEC_PLAIN, {"value": Part("f16", value.shape)})
                for key, value in tensors.items()
            }
        )
        for name, tensors in components.items()
    }
    writer = store.begin_derived(
        weights_transaction_id("fetch-test", "create", "sha256:" + "1" * 64, "model"),
        1,
        *Derivation(
            sources={},
            targets=targets,
            configs={name: Config("add") for name in components},
            order=tuple((name, key) for name, tensors in components.items() for key in tensors),
        ).native_arguments(1 << 30),
        work_fingerprint="sha256:" + "2" * 64,
    )
    for name, tensors in components.items():
        for key, value in tensors.items():
            writer.add_part(name, key, "value", io.BytesIO(value.astype(np.float16).tobytes()))
        writer.add_config(name, io.BytesIO(b"{}"))
    return store, "sha256:" + writer.commit()["manifest"]["sha256"]


class Hub:
    """Closure, presign and object GETs for `producer`'s `manifest`, counted per object.
    Each digest in `held` is answered only once `gate` is set; `wrong` gets other bytes."""

    def __init__(self, producer: Store, manifest: str) -> None:
        self.manifest = manifest
        row = producer.manifest(manifest)
        self.bodies: dict[str, bytes] = {manifest.removeprefix("sha256:"): bytes(row["manifest"])}
        self.objects: list[dict[str, object]] = []
        with producer.acquire_cozytensors(manifest) as closure:
            for sha, length in closure.objects:
                digest = sha.removeprefix("sha256:")
                path = Path(producer.root, "blobs", digest[:2], digest[2:4], digest)
                self.bodies[digest] = path.read_bytes()
                self.objects.append({"sha256": digest, "length": length})
        self.gets: Counter[str] = Counter()
        self.first: dict[str, float] = {}
        self.asked = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.held: set[str] = set()
        self.wrong: set[str] = set()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def answer(self, data: object) -> None:
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                request = cast(dict[str, object], json.loads(self.rfile.read(length)))
                if self.path.endswith("/closure"):
                    hub.gets["closure"] += 1
                    hex_id = manifest.removeprefix("sha256:")
                    self.answer(
                        {"model": "test/model", "release": "1.0.0", "lane": "plain"}
                        | {"scope": "runtime", "complete": True, "presign_max_digests": 4096}
                        | {"manifest": {"sha256": hex_id, "length": len(hub.bodies[hex_id])}}
                        | {"objects": sorted(hub.objects, key=lambda o: str(o["sha256"]))}
                    )
                elif self.path.endswith("/presign"):
                    now = int(time.time())
                    digests = cast(list[str], request["digests"])
                    self.answer(
                        {"expires_at_unix": now + 3600, "server_time_unix": now}
                        | {"urls": {d: f"{hub.origin}/object/{d}" for d in digests}}
                    )
                else:
                    self.send_error(404)

            def do_GET(self) -> None:
                digest = self.path.rsplit("/", 1)[-1]
                body = hub.bodies[digest]
                if digest in hub.wrong:
                    body = bytes(len(body))
                requested = self.headers.get("Range")
                if requested is None or requested.startswith("bytes=0-"):
                    hub.gets[digest] += 1
                    hub.first.setdefault(digest, time.monotonic())
                if digest in hub.held:
                    hub.asked.set()
                    hub.gate.wait()
                if requested:
                    first, last = (int(v) for v in requested.removeprefix("bytes=").split("-"))
                    part = body[first : last + 1]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {first}-{first + len(part) - 1}/{len(body)}"
                    )
                    body = part
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.served: list[threading.Thread] = []

    def grant(self) -> list[dict[str, object]]:
        hex_id = self.manifest.removeprefix("sha256:")
        return [
            {"hub": self.origin, "ref": "test/model@1.0.0", "lane": "plain"}
            | {"manifest": self.manifest, "manifest_length": len(self.bodies[hex_id])}
            | {"allow_local": True}
        ]

    def channel(self, store: Store) -> int:
        """One fresh channel's executor end, its owner end served on a thread."""
        owner, executor = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        served = threading.Thread(
            target=_ext.serve_fetch,
            args=(store, owner.detach()),
            kwargs={"origins": self.grant(), "run": 1},
            daemon=True,
        )
        served.start()
        self.served.append(served)
        return executor.detach()

    def structure(self, store: Store) -> None:
        """Land the checkpoint's structure only, as TensorD's prepare does."""
        with Fetch.from_fd(self.channel(store)) as fetch:
            fetch.wait(self.manifest, [])

    def count(self, refs: object) -> list[int]:
        return [self.gets[sha] for sha, _ in cast(list[tuple[str, int]], refs)]

    def close(self) -> None:
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()
        for served in self.served:
            served.join(timeout=30)


def objects_of(store: Store, manifest: str, component: str) -> list[tuple[str, int]]:
    """A component's objects, from the header alone."""
    header = store.manifest(manifest)["header"]
    assert header is not None
    keys = list(tensorfs.parse_header(header)["components"][component])
    plan = tensorfs.plan(header, [(component, key) for key in keys], [component], window=4 << 20)
    return sorted(
        {
            (i["object"].removeprefix("sha256:"), i["object_length"])
            for i in plan.items()
            if i["source"] == "object"
        }
    )


HIDDEN, BLOCKS = 256, 4


def tensors(seed: int, blocks: int) -> dict[str, np.ndarray]:
    """A small block model's weights: `embed`, then each block's `proj`, `out` and `norm`."""
    rng = np.random.default_rng(seed)
    values = {"embed": rng.random(HIDDEN, dtype=np.float32) + 0.5}
    for index in range(blocks):
        for name in ("proj", "out"):
            values[f"blocks.{index}.{name}.weight"] = rng.normal(0, 0.06, (HIDDEN, HIDDEN))
        values[f"blocks.{index}.norm"] = rng.random(HIDDEN, dtype=np.float32) + 0.5
    return {key: value.astype(np.float16) for key, value in values.items()}


#: encoder < decoder < denoiser in bytes; `spare` is in the checkpoint and never constructed.
COMPONENTS = {
    "encoder": tensors(1, 1),
    "denoiser": tensors(2, BLOCKS),
    "decoder": tensors(3, 2),
    "spare": tensors(4, 2),
}


@contextlib.contextmanager
def served(root: Path) -> Iterator[tuple[Hub, Store]]:
    """`COMPONENTS` at a loopback hub, and a Store holding only its structure."""
    producer, manifest = write(root / "producer", COMPONENTS)
    hub = Hub(producer, manifest)
    destination = Store.init(root / "destination")
    try:
        hub.structure(destination)
        yield hub, destination
    finally:
        hub.close()


def fetches(hub: Hub, store: Store, lines: list[str] | None = None) -> Fetches:
    """A planner whose every ask is a fresh channel on `store`; its waits' lines in `lines`."""

    def progress(line: str, _fraction: float) -> None:
        if lines is not None:
            lines.append(line)

    return Fetches(lambda: hub.channel(store), progress)
