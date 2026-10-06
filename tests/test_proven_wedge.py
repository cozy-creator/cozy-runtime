"""Real loopback PUT retries when the peer loses or never sends its answer."""

from __future__ import annotations

import hashlib
import http.server
import io
import re
import sys
import time
import os
import secrets
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, child_env, egress, liveness, native_wheel, spawn
from test_end_to_end import NO_EXECUTOR

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

SAMPLE = 0.1


# ------------------------------------------------------------------ rows 8 and 20: the seam


# ------------------------------------------------------------- row 9: the operator child


# ------------------------------------------------------------------ row 25: the PUT retry


class _Store:
    """A loopback object store whose first `fail_first` PUTs die after the body arrived."""

    def __init__(self) -> None:
        self.fail_first = 0
        self.puts = 0
        self.bodies: list[bytes] = []
        self.lock = threading.Lock()


def _handler(store: _Store) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def do_PUT(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with store.lock:
                store.puts += 1
                store.bodies.append(body)
                nth = store.puts
            if nth <= store.fail_first:
                # The body landed and the answer is lost: a store that finalized and then
                # dropped the connection, which is what a silent socket looks like here.
                self.close_connection = True
                self.wfile.close()
                return
            if len(store.bodies) > 1 and store.bodies[-1] == store.bodies[0]:
                # The key already holds these bytes: the immutable-write answer.
                self.send_response(egress.HTTP_PRECONDITION_FAILED)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("ETag", '"landed"')
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


@pytest.fixture
def store() -> Iterator[tuple[_Store, str]]:
    state = _Store()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_address[1]}/object/{secrets.token_hex(4)}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


POLICY = egress.EgressPolicy(allowed_hosts=("127.0.0.1",), allow_private=True)


def test_a_put_whose_answer_is_lost_is_re_sent_and_lands(store: tuple[_Store, str]) -> None:
    """One object, re-sent: the first send's body landed, its answer did not, and the
    re-send's 412 is read as that first send having landed — digest and all."""
    state, url = store
    state.fail_first = 1
    payload = os.urandom(1 << 16)

    receipt = egress.put_from(
        url,
        POLICY,
        io.BytesIO(payload),
        expected_length=len(payload),
        media_type="application/octet-stream",
        what="output 'image'",
    )

    assert state.puts == 2, f"{state.puts} send(s): the object was not re-sent once"
    assert receipt.length == len(payload)
    assert receipt.sha256 == hashlib.sha256(payload).digest()
    assert receipt.http_status == egress.HTTP_PRECONDITION_FAILED


def test_a_put_that_never_gets_an_answer_refuses_after_its_sends(
    store: tuple[_Store, str],
) -> None:
    """The budget is per object and finite: past it the refusal names how many sends."""
    state, url = store
    state.fail_first = egress.PUT_ATTEMPTS + 1
    payload = os.urandom(1 << 12)

    with pytest.raises(egress.EgressRefusal) as caught:
        egress.put_from(
            url,
            POLICY,
            io.BytesIO(payload),
            expected_length=len(payload),
            media_type="application/octet-stream",
            what="output 'image'",
        )

    assert caught.value.code == "egress_unreachable"
    assert f"after {egress.PUT_ATTEMPTS} send(s)" in caught.value.detail
    assert state.puts == egress.PUT_ATTEMPTS


SAMPLE = 0.1


@pytest.fixture(autouse=True)
def fine_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", SAMPLE)


def floor() -> float:
    return liveness.noise_floor()


def _still(verdict: str) -> float:
    """The stillness a wedge verdict reports it measured before ending the subject."""
    found = re.search(r"no measurable progress for ([0-9.]+) s", verdict)
    assert found, verdict
    return float(found.group(1))


def _base_env() -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, v) for k, v in os.environ.items() if not child_env.erased(k)))


def _operator(code: str) -> dict[str, object]:
    """One real fixture run: `exec(code)` in the runner, under its rlimits, answering null."""
    return native_wheel.qualify_operator(
        python=Path(sys.executable),
        fixture="builtins:exec",
        expected_result_digest=canonical.digest(None),
        environment=RUNNER_ENV,
        request=code,
    )


def test_an_operator_child_that_keeps_working_runs_to_its_answer() -> None:
    """Steady work far past the noise floor is not a wedge: the fixture runs to exit 0."""
    seconds = 4 * floor()
    started = time.monotonic()
    result = _operator(
        "import hashlib, time\n"
        "digest = b'x'\n"
        f"until = time.monotonic() + {seconds}\n"
        "while time.monotonic() < until:\n"
        "    digest = hashlib.sha256(digest).digest()\n"
    )
    elapsed = time.monotonic() - started
    assert result["result_digest"] == canonical.digest(None)
    assert elapsed >= seconds, f"{elapsed:.1f}s: the fixture did not run its course"


def test_an_operator_child_that_stops_dead_is_ended_on_the_measurement() -> None:
    """`signal.pause()` burns nothing forever; the child is killed with its group, by name."""
    started = time.monotonic()
    with pytest.raises(native_wheel.NativeWheelRefusal) as caught:
        _operator("import signal\nsignal.pause()\n")
    elapsed = time.monotonic() - started
    assert caught.value.code == "package_environment_native_operator_wedged"
    assert "no measurable progress for" in caught.value.detail
    assert elapsed > floor(), f"{elapsed:.1f}s: ended before the noise floor"
    # The measured stillness, not wall time: interpreter startup under load is not stillness.
    assert _still(caught.value.detail) < 6 * floor(), "the wait was a clock"


RUNNER_ENV = spawn.seal(_base_env(), {})
