"""One black-holed stream must cost one object, not the whole pull.

The transport's socket timeout is what is LEFT of the caller's deadline, so a presigned GET
that accepts a connection and then goes silent used to hold a pool slot until the deadline
expired — which is how a 96 GiB closure that moves at 470 MB/s died `readiness.deadline_exceeded`
with fifteen streams idle. These arms run the REAL `pull.pull` against a real HTTP server on
loopback and assert the three outcomes that matter: a stream that stops moving while its peers
move is named and re-asked, a stall that clears is invisible to the caller, and a stall that
never clears refuses by name long before the deadline it no longer depends on.

The sampling constants are turned down because they are resolution, not contract: the RULE
under test is "no bytes while peers moved", and it is the same rule at any sampling rate.
"""

from __future__ import annotations

import http.server
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

from cozy_runtime.internal import pull


class ProofFailure(RuntimeError):
    """The pull's stream-liveness contract did not hold."""


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ProofFailure(detail)


OBJECT_BYTES = b"x" * 4096
OBJECT_IDS = tuple("sha256:" + f"{n:064x}" for n in range(1, 6))
SNAPSHOT = f"{9:064x}"
MANIFEST = b"manifest-bytes"
STALLED = OBJECT_IDS[2]


class _Store:
    def __init__(self, root: Path) -> None:
        self.root = root


class _TensorFS:
    """The border, faked down to what this leg touches: a plan, a door, and a proof.

    `fetch_admit` DRAINS the reader exactly as the real Rust door does, because the blocking
    read inside it is the thing under test.
    """

    def __init__(self) -> None:
        self.admitted: set[str] = set()
        self.lock = threading.Lock()
        self.completed = False

    def fetch_plan(
        self, _store: Any, session: str, snapshot: str, length: int, declared: list[Any]
    ) -> dict[str, Any]:
        return {
            "session": session,
            "snapshot": snapshot,
            "document": b"plan:" + snapshot.encode() + str(length).encode(),
            "held": [],
            "held_bytes": 0,
            "wanted": list(declared),
            "wanted_bytes": sum(int(n) for _, n in declared),
            "presence": {},
        }

    def fetch_admit(self, _store: Any, _plan: bytes, object_id: str, source: Any) -> None:
        total = 0
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
        if total != len(OBJECT_BYTES):
            raise RuntimeError(f"LENGTH_MISMATCH: {object_id} streamed {total}")
        with self.lock:
            self.admitted.add(object_id)

    def fetch_complete(self, _store: Any, _plan: bytes) -> None:
        self.completed = True


class _Hub(http.server.BaseHTTPRequestHandler):
    """The hub's read routes and the object store, on one loopback port.

    `stall_for` is how many times the stalled object's GET goes silent after its headers.
    A value that is never reached is a link that drops streams rather than slowing them.
    """

    stall_for: ClassVar[int] = 0
    served: ClassVar[dict[str, int]] = {}
    release: ClassVar[threading.Event] = threading.Event()

    def log_message(self, *_: Any) -> None:
        return

    def do_GET(self) -> None:  # BaseHTTPRequestHandler names the verb, not this file
        if self.path.startswith("/v1/resolve"):
            return self._json(f'{{"checkpoint_id":"sha256:{SNAPSHOT}","repo":"org/name"}}')
        if self.path.endswith("/manifest"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(MANIFEST)))
            self.end_headers()
            self.wfile.write(MANIFEST)
            return
        if self.path.startswith("/object/"):
            return self._object(self.path.removeprefix("/object/"))
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path.endswith("/reads"):
            host = self.headers["Host"]
            rows = ",".join(
                f'{{"object_id":"{oid}","length":{len(OBJECT_BYTES)}'
                f',"url":"http://{host}/object/{oid}"}}'
                for oid in OBJECT_IDS
            )
            return self._json('{"reads":[' + rows + "]}")
        self.send_error(404)

    def _json(self, body: str) -> None:
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _object(self, object_id: str) -> None:
        served = type(self).served
        seen = served.get(object_id, 0)
        served[object_id] = seen + 1
        self.send_response(200)
        self.send_header("Content-Length", str(len(OBJECT_BYTES)))
        self.end_headers()
        if object_id == STALLED and seen < type(self).stall_for:
            # Headers, one byte, then silence: a live TCP connection with no bytes on it.
            self.wfile.write(OBJECT_BYTES[:1])
            self.wfile.flush()
            type(self).release.wait()
            return
        self.wfile.write(OBJECT_BYTES)


def _serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _pull(base: str, root: Path, tensorfs: _TensorFS) -> pull.PullResult:
    return pull.pull(
        tensorfs,
        _Store(root),
        base,
        "org/name",
        timeout=120,
        streams=len(OBJECT_IDS),
    )


def run(root: Path) -> None:
    sample, samples = pull.STALL_SAMPLE_SECONDS, pull.STALL_SAMPLES
    pull.STALL_SAMPLE_SECONDS, pull.STALL_SAMPLES = 0.05, 3
    server, base = _serve()
    try:
        # 1. Nothing stalls: the control, so a later refusal is about the stall and not the leg.
        _Hub.stall_for, _Hub.served, _Hub.release = 0, {}, threading.Event()
        tensorfs = _TensorFS()
        outcome = _pull(base, root, tensorfs)
        _require(tensorfs.completed, "a clean pull did not reach fetch_complete")
        _require(outcome.fetched == len(OBJECT_IDS), f"fetched {outcome.fetched} objects")

        # 2. One stream goes silent once. The pull must SUCCEED, having re-asked for that one
        #    object, and it must do so in a time the sampling rule explains — not the deadline.
        _Hub.stall_for, _Hub.served, _Hub.release = 1, {}, threading.Event()
        tensorfs = _TensorFS()
        started = time.monotonic()
        outcome = _pull(base, root, tensorfs)
        elapsed = time.monotonic() - started
        _require(tensorfs.completed, "a pull with one recoverable stall did not complete")
        _require(outcome.fetched == len(OBJECT_IDS), f"fetched {outcome.fetched} after a stall")
        _require(
            _Hub.served[STALLED] == 2,
            f"the stalled object was asked for {_Hub.served[STALLED]} times, not twice",
        )
        _require(elapsed < 20, f"the recovery took {elapsed:.1f}s: it waited on the deadline")
        _Hub.release.set()

        # 3. The stream never recovers. The pull must refuse BY NAME, after FETCH_ATTEMPTS
        #    asks, and long before the 120 s deadline it used to be pinned to.
        _Hub.stall_for, _Hub.served, _Hub.release = 99, {}, threading.Event()
        tensorfs = _TensorFS()
        started = time.monotonic()
        try:
            _pull(base, root, tensorfs)
        except pull.PullRefusal as exc:
            elapsed = time.monotonic() - started
            _require(exc.code == "transfer_failed", f"a permanent stall refused {exc.code}")
            _require("moved no bytes while its peers" in exc.message, exc.message)
            _require(elapsed < 20, f"the refusal took {elapsed:.1f}s: it waited on the deadline")
            _require(
                _Hub.served[STALLED] == pull.FETCH_ATTEMPTS,
                f"asked {_Hub.served[STALLED]} times, not {pull.FETCH_ATTEMPTS}",
            )
        else:
            raise ProofFailure("a permanently silent stream was accepted")
        _Hub.release.set()
    finally:
        _Hub.release.set()
        server.shutdown()
        pull.STALL_SAMPLE_SECONDS, pull.STALL_SAMPLES = sample, samples


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cozy-pull-liveness.") as tmp:
        run(Path(tmp))
    print("pull stream liveness: 3/3 arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
