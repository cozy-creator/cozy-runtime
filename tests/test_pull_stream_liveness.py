"""One black-holed stream must cost one object, not the whole pull.

The transport's socket timeout is what is LEFT of the caller's deadline, so a presigned GET
that accepts a connection and then goes silent used to hold a pool slot until that deadline
expired — which is how a closure that moves at 470 MB/s died `deadline_exceeded` with fifteen
streams idle. These arms run the REAL `pull.pull` against a real hub and a real object store
on a real loopback socket, through the real SSRF guard, the real `urllib` transport and the
real `_Streams` ledger, and assert the three outcomes that matter: a clean pull is untouched,
a stall that clears is invisible to the caller, and a stall that never clears refuses by name
long before the deadline it no longer depends on.

WHAT IS STOOD IN FOR, AND WHY. `tensorfs` is an optional extra (`model-execution`) that
`uv sync` does not install, so the border is the one thing here that cannot be real. It is
INJECTED rather than imported — that is `pull`'s own design, so the caller owns the facade
lookup — and the stand-in does the only thing this leg's liveness depends on: it drains the
reader chunk by chunk exactly as the Rust door does, because the blocking read inside that
drain is the thing under test. Nothing about the transport, the retry rule or the ledger is
doubled; if any of them were, these arms would prove nothing.

The sampling constants are turned down because they are RESOLUTION, not contract: the rule
under test is "silent for far longer than this pull's own worst measured gap", and it is the
same rule at any sampling rate.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import pull

OBJECT = b"x" * 4096
OBJECT_IDS = tuple("sha256:" + f"{n:064x}" for n in range(1, 6))
MANIFEST = b"manifest-bytes"
MANIFEST_ID = f"{9:064x}"
#: The one object whose GET goes silent after its headers. It is not the first in the plan,
#: so its peers are moving bytes while it is not — which is the whole premise of the rule.
STALLED = OBJECT_IDS[2]

MODEL = "org/name"
RELEASE = "r1"
LANE = "default"

#: Far longer than any arm may take, and the number a refusal must NOT have waited for: an
#: arm that finishes near it finished on the deadline, which is the defect, not the fix.
DEADLINE = 120.0
PROMPTLY = 30.0


class Hub:
    """The hub's read routes and the object store, on one loopback port.

    `stall_for` is how many times the stalled object's GET writes its headers and then says
    nothing. A count no retry budget can outlast is a link that drops streams rather than
    slowing them; a count of one is a blip.
    """

    def __init__(self) -> None:
        self.stall_for = 0
        self.served: dict[str, int] = {}
        self.release = threading.Event()
        self.lock = threading.Lock()

    def asked(self, object_id: str) -> int:
        with self.lock:
            seen = self.served.get(object_id, 0)
            self.served[object_id] = seen + 1
            return seen


class Border:
    """The TensorFS border, stood in for down to exactly what this leg touches.

    `fetch_admit` DRAINS the reader the way the real door does and holds the reader to the
    length the plan declared, so a stream that was cut short is refused here rather than
    quietly accepted — which is what makes a killed socket visible to `_fetch` at all.
    """

    def __init__(self) -> None:
        self.admitted: set[str] = set()
        self.lock = threading.Lock()
        self.completed = False

    def manifest_max_bytes(self) -> int:
        return 1 << 20

    def fetch_plan(
        self, _store: Any, session: str, manifest_id: str, length: int, declared: list[Any]
    ) -> dict[str, Any]:
        return {
            "session": session,
            "snapshot": manifest_id,
            "document": b"plan:" + manifest_id.encode() + str(length).encode(),
            "held": [],
            "held_bytes": 0,
            "wanted": list(declared),
            "wanted_bytes": sum(int(n) for _, n in declared),
            "presence": {},
        }

    def fetch_admit(self, _store: Any, _plan: bytes, object_id: str, source: Any) -> None:
        want = len(MANIFEST) if object_id == "sha256:" + MANIFEST_ID else len(OBJECT)
        total = 0
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
        if total != want:
            raise RuntimeError(f"LENGTH_MISMATCH: {object_id} streamed {total} of {want}")
        with self.lock:
            self.admitted.add(object_id)

    def fetch_complete(self, _store: Any, _plan: bytes) -> None:
        self.completed = True


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root


def handler(hub: Hub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path.startswith("/v1/models/resolve"):
                return self._json(
                    f'{{"manifest_id":"sha256:{MANIFEST_ID}","model":"{MODEL}",'
                    f'"release":"{RELEASE}","lane":"{LANE}"}}'
                )
            if self.path.endswith("/manifest"):
                return self._body(MANIFEST)
            if self.path.startswith("/object/"):
                return self._object(self.path.removeprefix("/object/"))
            self.send_error(404)

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path.endswith("/reads"):
                host = self.headers["Host"]
                rows = ",".join(
                    f'{{"object_id":"{oid}","length":{length},"url":"http://{host}/object/{oid}"}}'
                    for oid, length in (
                        [("sha256:" + MANIFEST_ID, len(MANIFEST))]
                        + [(oid, len(OBJECT)) for oid in OBJECT_IDS]
                    )
                )
                return self._json('{"reads":[' + rows + "]}")
            self.send_error(404)

        def _json(self, body: str) -> None:
            self._body(body.encode(), content_type="application/json")

        def _body(self, raw: bytes, *, content_type: str = "application/octet-stream") -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _object(self, object_id: str) -> None:
            if object_id == "sha256:" + MANIFEST_ID:
                return self._body(MANIFEST)
            seen = hub.asked(object_id)
            self.send_response(200)
            self.send_header("Content-Length", str(len(OBJECT)))
            self.end_headers()
            if object_id == STALLED and seen < hub.stall_for:
                # Headers, one byte, then silence: a live TCP connection with no bytes on it,
                # which is exactly what a black-holed presigned GET looks like from here.
                with contextlib.suppress(OSError):
                    self.wfile.write(OBJECT[:1])
                    self.wfile.flush()
                hub.release.wait(DEADLINE)
                return
            with contextlib.suppress(OSError):
                self.wfile.write(OBJECT)

    return Handler


@pytest.fixture
def hub() -> Iterator[tuple[Hub, str]]:
    state = Hub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        state.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def fine_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sample the ledger every 100 ms with a 500 ms noise floor, not every 5 s with 30 s.

    Resolution only. The verdict still comes from the measured worst gap of this same pull,
    and the floor still sits orders of magnitude above the loopback gaps the healthy streams
    show — so nothing here is judged by a clock a test invented.
    """
    monkeypatch.setattr(pull, "STALL_SAMPLE_SECONDS", 0.1)
    monkeypatch.setattr(pull, "STALL_SAMPLES", 5)


def pulled(base: str, root: Path, border: Border) -> pull.PullResult:
    return pull.pull(
        border, Store(root), base, MODEL, timeout=DEADLINE, streams=len(OBJECT_IDS) + 1
    )


def test_a_pull_with_nothing_stalled_lands_every_object(
    hub: tuple[Hub, str], tmp_path: Path
) -> None:
    """The control: no stall, so a later refusal is about the stall and not about this leg."""
    state, base = hub
    border = Border()

    result = pulled(base, tmp_path, border)

    assert border.completed, "a clean pull did not reach fetch_complete"
    assert result.fetched == len(OBJECT_IDS) + 1
    assert border.admitted == {*OBJECT_IDS, "sha256:" + MANIFEST_ID}
    assert state.served[STALLED] == 1, "an unstalled object was asked for more than once"


def test_one_stream_that_goes_silent_costs_one_object(hub: tuple[Hub, str], tmp_path: Path) -> None:
    """A stall that clears is invisible to the caller: the pull SUCCEEDS, having re-asked.

    Twice, not four times — the retry budget is spent per failure, not eagerly — and in a
    time the sampling rule explains rather than the deadline.
    """
    state, base = hub
    state.stall_for = 1
    border = Border()

    started = time.monotonic()
    result = pulled(base, tmp_path, border)
    elapsed = time.monotonic() - started

    assert border.completed, "a pull with one recoverable stall did not complete"
    assert result.fetched == len(OBJECT_IDS) + 1
    assert border.admitted == {*OBJECT_IDS, "sha256:" + MANIFEST_ID}
    assert state.served[STALLED] == 2, f"the stalled object was asked for {state.served[STALLED]}x"
    assert elapsed < PROMPTLY, f"the recovery took {elapsed:.1f}s: it waited on the deadline"


def test_a_stream_that_never_recovers_refuses_by_name(hub: tuple[Hub, str], tmp_path: Path) -> None:
    """A permanent stall spends the object's budget and refuses `transfer_failed` naming it.

    The refusal must arrive long before the caller's deadline. That gap IS the fix: the leg
    used to have no answer other than the deadline, and it was the whole pull's.
    """
    state, base = hub
    state.stall_for = pull.FETCH_ATTEMPTS + 1
    border = Border()

    started = time.monotonic()
    with pytest.raises(pull.PullRefusal) as caught:
        pulled(base, tmp_path, border)
    elapsed = time.monotonic() - started

    assert caught.value.code == "transfer_failed"
    assert "moved no bytes while its peers" in caught.value.message
    assert STALLED in caught.value.message
    assert state.served[STALLED] == pull.FETCH_ATTEMPTS, (
        f"asked {state.served[STALLED]} times, not {pull.FETCH_ATTEMPTS}"
    )
    assert elapsed < PROMPTLY, f"the refusal took {elapsed:.1f}s: it waited on the deadline"
    assert not border.completed
    assert STALLED not in border.admitted
