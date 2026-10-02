"""A loopback Hub (`closure`, `presign`) and object store that inject faults on chosen
routes and asks: the Runtime's twin of TensorFS's `crates/tensorfs-core/tests/fetch_faults.rs`.

Presigned URLs carry the instant they die and the store answers 403 after it. `Watched` and
`FaultHub.until` bound a test by measured progress (`liveness.Pace`), never by a wall clock.
"""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import tensorfs

from cozy_runtime.internal import liveness
from test_model_reader import model as written
from test_model_reader import plain
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot

#: An observer outside the pull must not call it wedged before TensorFS's own ledger has had
#: its chance: twice the noise floor its stall and hub-retry patience start from.
FLOOR = 2 * liveness.noise_floor()


@dataclass(frozen=True)
class Status:
    """This status, these headers, this body (bytes, or built when asked)."""

    code: int
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes | Callable[[], bytes] = b""


@dataclass(frozen=True)
class Reset:
    """After `after` body bytes, a TCP reset (SO_LINGER 0)."""

    after: int


@dataclass(frozen=True)
class Truncate:
    """After `after` body bytes, a clean close short of Content-Length."""

    after: int


@dataclass(frozen=True)
class Stall:
    """After `after` body bytes, silence until the client leaves."""

    after: int


@dataclass(frozen=True)
class Corrupt:
    """The declared length with its last byte changed."""


@dataclass(frozen=True)
class Delay:
    """Wait this long, then answer with `then` (honestly when None)."""

    seconds: float
    then: Fault | None = None


@dataclass(frozen=True)
class Gate:
    """Hold the answer until `opened` is set (or the hub closes), then answer with `then`."""

    opened: threading.Event
    then: Fault | None = None


@dataclass(frozen=True)
class Drip:
    """A whole GET answered application-limited, `rate` bytes a second in 4 KiB pieces: bytes
    keep arriving, so no silence rule fires. A ranged ask is answered honestly, as R2 serves
    a fresh range apart from a stalled read (run 2322)."""

    rate: int


Fault = Status | Reset | Truncate | Stall | Corrupt | Delay | Gate | Drip


def enveloped(code: int, error: str, message: str = "injected") -> Status:
    """The Hub's own typed refusal, as Tensorhub writes it."""
    body = {"error": {"code": error, "message": message, "remedy": ""}}
    return Status(code, (("Content-Type", "application/json"),), json.dumps(body).encode())


def tunnel_404() -> Status:
    """A dead tunnel's page in front of the Hub: a 404 the Hub never wrote."""
    return Status(404, (("Content-Type", "text/html"),), b"<html>ERR_NGROK_3200 offline</html>")


def rate_limited(seconds: int) -> Status:
    """A 429 whose envelope states its wait on the Hub's own clock."""

    def body() -> bytes:
        now = int(time.time())
        return json.dumps(
            {
                "error": {"code": "rate_limited", "message": "slow down", "remedy": ""},
                "retry_after_unix": now + seconds,
                "server_time_unix": now,
            }
        ).encode()

    return Status(429, (("Content-Type", "application/json"),), body)


@dataclass(frozen=True)
class Rule:
    """`fault` answers asks `asks` (1-based) of `route`: `closure`, `presign`, an object's
    hex, or `*` for every object."""

    route: str
    asks: range
    fault: Fault


def on(route: str, first: int, last: int, fault: Fault) -> Rule:
    return Rule(route, range(first, last + 1), fault)


ALWAYS = 1 << 30
#: the checkpoint's two object routes
HEADER, ASSET = (hashlib.sha256(body).hexdigest() for body in (_HEADER, _ASSET))


@dataclass(frozen=True)
class Checkpoint:
    model: str
    manifest: str
    length: int
    #: every servable body by bare hex, the manifest's included
    bodies: dict[str, bytes]


def checkpoint(
    tmp_path: Path, model: str = "proof/model", path: str = "model.cozytensors"
) -> Checkpoint:
    """`test_model_runtime_closure`'s checkpoint: a manifest, its header and one asset. Another
    `path` names the same bytes under a distinct manifest."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _, manifest, _, origin = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    raw = origin.manifest(manifest)["manifest"].replace(b"model.cozytensors", path.encode())
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET, raw)}
    return Checkpoint(model, "sha256:" + hashlib.sha256(raw).hexdigest(), len(raw), bodies)


def weights(tmp_path: Path, tensors: int, size: int, model: str = "proof/model") -> Checkpoint:
    """A real checkpoint of `tensors` same-size tensor objects, written by TensorFS's own
    derived writer: each big enough to have a body rate, and alike, so one's pace judges
    another's."""
    root = tmp_path / "store"
    root.mkdir(parents=True)
    store = tensorfs.Store.init(str(root))
    values = {
        f"blocks.{index}.weight": plain(np.full(size // 4, index + 1, dtype=np.float32))
        for index in range(tensors)
    }
    manifest, length = written(store, "slow-tail", values)
    bodies = {manifest.removeprefix("sha256:"): store.manifest(manifest)["manifest"]}
    for row in store.walk_cozytensors(manifest):
        hex_ = row["id"].removeprefix("sha256:")
        bodies[hex_] = (root / "blobs" / hex_[:2] / hex_[2:4] / hex_).read_bytes()
    return Checkpoint(model, manifest, length, bodies)


@dataclass
class Ledger:
    asks: dict[str, int] = field(default_factory=dict)
    #: body bytes written for each object's latest ask
    sent: dict[str, int] = field(default_factory=dict)
    #: URLs presented after they died
    expired: int = 0
    log: list[str] = field(default_factory=list)


class FaultHub:
    def __init__(self, checkpoint: Checkpoint, *rules: Rule, lifetime: int = 3600):
        self.checkpoint, self.rules, self.lifetime = checkpoint, rules, lifetime
        self.ledger = Ledger()
        self.changed = threading.Condition()
        self.held: set[socket.socket] = set()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        #: the Hub origin as a development machine names it
        self.origin = f"http://localhost:{self.server.server_port}"

    def __enter__(self) -> FaultHub:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        with self.changed:
            for held in self.held:
                held.shutdown(socket.SHUT_RDWR)  # a stalled answer ends with its hub
        for rule in self.rules:
            if isinstance(rule.fault, Gate):
                rule.fault.opened.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def asks(self, route: str) -> int:
        with self.changed:
            return self.ledger.asks.get(route, 0)

    def sent(self, route: str) -> int:
        with self.changed:
            return self.ledger.sent.get(route, 0)

    def log(self) -> str:
        with self.changed:
            return "\n".join(self.ledger.log)

    def reading(self) -> object:
        """Everything this hub has seen: a meter for `liveness.Pace`."""
        with self.changed:
            ledger = self.ledger
            return (sorted(ledger.asks.items()), sorted(ledger.sent.items()), ledger.expired)

    def until(self, done: Callable[[], bool], meter: Callable[[], object] = lambda: None) -> None:
        """Wait for `done`; fail only when neither this hub nor `meter` has moved for
        `liveness` patience."""
        pace = liveness.Pace()
        with self.changed:
            while not done():
                pace.observe((self.reading(), meter()))
                assert not pace.wedged(FLOOR), pace.verdict(FLOOR) + "\n" + self.log()
                self.changed.wait(liveness.SAMPLE_SECONDS)

    def _count(self, route: str, object_: bool) -> Fault | None:
        with self.changed:
            ask = self.ledger.asks[route] = self.ledger.asks.get(route, 0) + 1
            fault = next(
                (
                    rule.fault
                    for rule in self.rules
                    if (rule.route == route or (object_ and rule.route == "*")) and ask in rule.asks
                ),
                None,
            )
            if object_:
                self.ledger.sent[route] = 0
            self.ledger.log.append(f"{time.monotonic():.3f} {route} ask {ask}: {fault}")
            self.changed.notify_all()
        return fault

    def _sent(self, route: str, count: int) -> None:
        with self.changed:
            self.ledger.sent[route] = self.ledger.sent.get(route, 0) + count
            self.changed.notify_all()

    def _closure(self, asked: dict[str, Any]) -> dict[str, Any]:
        point = self.checkpoint
        assert asked["ref"].partition("@")[0] == point.model, asked
        manifest = point.manifest.removeprefix("sha256:")
        return {
            "complete": True,
            "lane": "bf16",
            "model": point.model,
            "manifest": {"sha256": manifest, "length": point.length},
            "objects": [
                {"length": len(body), "sha256": hex_}
                for hex_, body in sorted(point.bodies.items())
                if hex_ != manifest
            ],
            "presign_max_digests": 10,
            "release": "1.0.0",
            "scope": "runtime",
            "server_time_unix": int(time.time()),
        }

    def _presign(self, asked: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        dies = int((now + self.lifetime) * 1000)
        port = self.server.server_port
        return {
            "expires_at_unix": int(now) + self.lifetime,
            "server_time_unix": int(now),
            "urls": {
                hex_: f"http://127.0.0.1:{port}/obj/{hex_}?dies={dies}" for hex_ in asked["digests"]
            },
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def do_POST(self) -> None:
                asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                route = self.path.removeprefix("/v1/tensorfs/")
                if route not in ("closure", "presign"):
                    self.answer(Status(404, body=b"no such route"))
                    return
                fault = hub._count(route, object_=False)
                answer = hub._closure(asked) if route == "closure" else hub._presign(asked)
                self.answer(fault, json.dumps(answer, sort_keys=True).encode())

            def do_GET(self) -> None:
                path, _, query = self.path.partition("?")
                route = path.removeprefix("/obj/")
                body = hub.checkpoint.bodies.get(route)
                if not path.startswith("/obj/") or body is None:
                    self.answer(Status(404, body=b"NoSuchKey"))
                    return
                fault = hub._count(route, object_=True)
                if time.time() * 1000 > int(query.removeprefix("dies=")):
                    with hub.changed:
                        hub.ledger.expired += 1
                    denied = b"<Error><Code>AccessDenied</Code><Message>Request has expired"
                    self.answer(Status(403, body=denied + b"</Message></Error>"))
                    return
                self.answer(fault, body, route)

            def answer(self, fault: Fault | None, body: bytes = b"", route: str = "") -> None:
                if isinstance(fault, Delay):
                    time.sleep(fault.seconds)
                    fault = fault.then
                elif isinstance(fault, Gate):
                    fault.opened.wait()
                    fault = fault.then
                if isinstance(fault, Status):
                    raw = fault.body() if callable(fault.body) else fault.body
                    self.head(fault.code, len(raw), fault.headers)
                    self.write(raw)
                    return
                code, ranged = 200, tuple[tuple[str, str], ...]()
                spec = self.headers.get("Range", "").removeprefix("bytes=")
                if spec:
                    first, _, last = spec.partition("-")
                    start, end = int(first), min(int(last or len(body) - 1), len(body) - 1)
                    ranged = (("Content-Range", f"bytes {start}-{end}/{len(body)}"),)
                    code, body = 206, body[start : end + 1]
                self.head(code, len(body), ranged)
                if isinstance(fault, Drip) and not spec:
                    for at in range(0, len(body), 4096):
                        time.sleep(4096 / fault.rate)
                        if not self.write(body[at : at + 4096], route):
                            return
                    return
                if isinstance(fault, Corrupt):
                    body = body[:-1] + bytes([body[-1] ^ 0xFF])
                if not isinstance(fault, Reset | Truncate | Stall):
                    self.write(body, route)
                elif not self.write(body[: fault.after], route):
                    return
                if isinstance(fault, Reset):
                    self.connection.setsockopt(
                        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                    )
                    self.connection.close()
                elif isinstance(fault, Stall):
                    self.hold()

            def head(self, code: int, length: int, headers: tuple[tuple[str, str], ...]) -> None:
                self.send_response(code)
                self.send_header("Content-Length", str(length))
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()

            def write(self, body: bytes, route: str = "") -> bool:
                """Count bytes against their object BEFORE they leave."""
                if route:
                    hub._sent(route, len(body))
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                except OSError:
                    return False
                return True

            def hold(self) -> None:
                with hub.changed:
                    hub.held.add(self.connection)
                try:
                    while self.connection.recv(64):
                        pass
                except OSError:
                    pass
                finally:
                    with hub.changed:
                        hub.held.discard(self.connection)
                        hub.changed.notify_all()

        return Handler


class Watched[T]:
    """`work` on its own thread. `result` waits for its end, failing only when neither `hub`
    nor `meter` moved for `liveness` patience, and re-raises its exception."""

    def __init__(
        self, hub: FaultHub, work: Callable[[], T], meter: Callable[[], object] = lambda: None
    ):
        self.hub, self.meter = hub, meter
        self.value: list[T] = []
        self.error: list[BaseException] = []

        def run() -> None:
            try:
                self.value.append(work())
            except BaseException as error:
                self.error.append(error)

        self.thread = threading.Thread(target=run, name="watched", daemon=True)
        self.thread.start()

    def result(self) -> T:
        pace = liveness.Pace()
        while self.thread.is_alive():
            pace.observe((self.hub.reading(), self.meter()))
            assert not pace.wedged(FLOOR), pace.verdict(FLOOR) + "\n" + self.hub.log()
            self.thread.join(liveness.SAMPLE_SECONDS)
        if self.error:
            raise self.error[0]
        return self.value[0]


def temps(store_root: Path) -> list[str]:
    """Admission temps left behind: none may outlive an ensure, whatever it returned."""
    tmp = store_root / "tmp"
    return (
        sorted(p.name for p in tmp.iterdir() if p.name.startswith("put-")) if tmp.is_dir() else []
    )
