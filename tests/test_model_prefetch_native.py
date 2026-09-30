"""The shared preparation path moves real TensorFS bytes once, without GPU activation."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author._observations import MAX_EVENTS, EventRing
from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.executor_replies import ObservationRow
from cozy_runtime.internal.worker import store_gc, triage
from cozy_runtime.internal.worker.attempts import AttemptEngine
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_serving import Serving, arguments
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot


@pytest.mark.parametrize("preempt", [False, True])
def test_real_model_pull_joins_across_future_payloads_and_reuses_cpu_binding(
    tmp_path: Path,
    preempt: bool,
) -> None:
    source_root = tmp_path / "origin"
    source_root.mkdir()
    _, manifest, length, source = _snapshot(source_root, include_asset=True, checkpoint_only=True)
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
    objects = [{"length": len(body), "sha256": digest} for digest, body in sorted(bodies.items())]
    bodies[manifest[7:]] = source.manifest(manifest)["manifest"]
    started, release = threading.Event(), threading.Event()
    fetched: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def answer(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/tensorfs/closure":
                response = {
                    "complete": True,
                    "lane": "bf16",
                    "model": "proof/model",
                    "manifest": {"sha256": manifest[7:], "length": length},
                    "objects": objects,
                    "presign_max_digests": 10,
                    "release": "1.0.0",
                    "scope": "runtime",
                    "server_time_unix": int(time.time()),
                }
            else:
                assert self.path == "/v1/tensorfs/presign"
                response = {
                    "expires_at_unix": int(time.time()) + 3600,
                    "server_time_unix": int(time.time()),
                    "urls": {
                        digest: f"http://127.0.0.1:{server.server_port}/{digest}"
                        for digest in request["digests"]
                    },
                }
            self.answer(json.dumps(response, sort_keys=True, separators=(",", ":")).encode())

        def do_GET(self) -> None:
            fetched.append(self.path)
            started.set()
            assert release.wait(5)
            # A paused native pull closes the old request before its replacement.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.answer(bodies[self.path.removeprefix("/")])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    store_root = tmp_path / "destination"
    store = tensorfs.Store.init(str(store_root))
    if preempt:
        # Already landed immutable bytes must survive pause/restart. The missing
        # asset still uses a real, blocked HTTP transfer and PullCancellation.
        header = tmp_path / "landed-header"
        header.write_bytes(_HEADER)
        store.put_file(str(header), "sha256:" + hashlib.sha256(_HEADER).hexdigest(), len(_HEADER))
    checkpoint = {"digest": manifest, "length": length}
    placement = {
        "installation_id": "installed-model",
        "package_interface": base64.b64encode(b"exact-interface").decode(),
        "models": [{"id": "model", "manifest": checkpoint}],
        "entrypoints": [
            {
                "name": "generate",
                "slots": [
                    {
                        "slot": "model",
                        "reference_model_id": "model",
                        "components": [{"component": "unet", "model_id": "model"}],
                        "stamps": [],
                    }
                ],
            }
        ],
    }
    capture = {
        "model_defaults": [
            {
                "callee_installation_id": "installed-model",
                "entrypoint": "generate",
                "parameter": "model",
                "public_origin": f"http://localhost:{server.server_port}",
                "rungs": [{"gpu": "*", "repository": "proof/model", "manifest": checkpoint}],
            }
        ]
    }
    progress: list[dict[str, Any]] = []
    worker = cast(
        Worker,
        SimpleNamespace(
            workspace=Workspace(store_root),
            executions=SimpleNamespace(capture_root=lambda *_: "root", capture=lambda *_: capture),
            options=SimpleNamespace(
                accelerator_backend="none", publication_authority=None, hubs=()
            ),
            host_facts=lambda: hostfacts.measure("none"),
            lanes=SimpleNamespace(entries=()),
            config=SimpleNamespace(object_storage_hosts=("127.0.0.1",)),
            control_lock=threading.RLock(),
            model_transfer_lock=threading.Lock(),
            model_transfers={},
            prepared_installations={
                "model": SimpleNamespace(document=documents.from_body(placement, pb.Placement))
            },
            calls=SimpleNamespace(progress_label=lambda *_: ""),
            emit_progress=lambda _parent, _ordinal, frame: progress.append(frame),
            # This host harness has no GPUs; the process warm is `test_startup_warm`'s.
            prespawns=SimpleNamespace(request=lambda *_: None),
        ),
    )
    serving = Serving(worker)
    target = Target(
        "installed-model",
        "generate",
        {"models": [{"path": "generate.models.model"}]},
        {"placement": placement},
        None,
        "",
    )
    call = Call("parent", 0, 1, 1, b"", b"", "child", b"", False, b"", "", "")
    request = pb.ChildCallRequest(parent_attempt_ordinal=1)
    try:
        first = serving._shared("owner", call, target, request, {}, speculative=True)
        assert started.wait(5)
        if preempt:
            other = Call("other-parent", 0, 1, 1, b"", b"", "other-child", b"", False, b"", "", "")
            with serving.demand(other):
                # Let the current HTTP read yield so the native pull observes its
                # cancellation. This fixture does not assume it aborts response headers.
                release.set()
                # Wait for native cancellation, not for an arbitrary amount of runtime.
                deadline = time.monotonic() + 5
                while serving.preparations.started():
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
                assert first is not None and not first.done()
                assert store.contains(hashlib.sha256(_HEADER).hexdigest())
            # This succeeds only if the resumed load got a NEW PullCancellation.
            assert first.result(5).placement == placement
        # A's image becomes available later. It must not change the preparation key.
        _, models = arguments(
            target, canonical_json.encode({"image": "now-available", "model": None})
        )
        joined = serving._shared("owner", call, target, request, models, speculative=False)
        assert joined is first and joined is not None
        release.set()
        prepared = joined.result(5)
        assert prepared.placement == placement
        store.verify_checkpoint_source("proof/model", manifest, length)
        if preempt:
            assert "/" + hashlib.sha256(_HEADER).hexdigest() not in fetched
        else:
            assert sorted(fetched) == sorted("/" + digest for digest in bodies)
        assert not any(row.get("kind") == "progress" for row in progress)
        observations = [row.get("fields", {}) for row in progress]
        assert any(
            row.get("position") == row.get("total") and row.get("total", 0) > 0
            for row in observations
        )
        # The prefetch's byte samples say they count bytes; its pull is one fetch record
        # at each end, naming the model, and the end has its bytes, time and rate.
        assert all(
            row.get("unit") == "bytes" for row in observations if row.get("position") is not None
        )
        fetch = [row["fields"] for row in progress if row.get("name") == "model fetch"]
        assert [row["event"] for row in fetch] == ["start", "end"] * (2 if preempt else 1)
        if preempt:
            assert fetch[1]["completed"] is False
        start, end = fetch[-2:]
        assert start["model"] == end["model"] == "proof/model" and start["manifest"] == manifest
        assert (
            start["prefetch"] is True and start["step"] == "" and start["entrypoint"] == "generate"
        )
        assert end["completed"] is True and end["bytes"] == end["total_bytes"] > 0
        if preempt:
            assert end["moved_bytes"] <= end["bytes"]
        else:
            assert end["moved_bytes"] == end["bytes"]
        assert end["elapsed_ms"] > 0
        assert end["rate_bytes_per_second"] > 0
        # The exact finished selection serves every demand; nothing is prepared again.
        completed_fetches = len(fetched)
        again = serving._shared("owner", call, target, request, {}, speculative=False)
        assert again is not None and again is first
        assert again.result(5).placement == placement
        assert len(fetched) == completed_fetches
        # No persistent read lease prevents legitimate cache eviction. Demanding the
        # completed metadata after a collection must refill bytes, not fail admission.
        store_gc.collect(store_root, keep_manifests=[])
        refilled = serving._shared("owner", call, target, request, {}, speculative=False)
        assert refilled is not first
        assert refilled is not None and refilled.result(5).placement == placement
        store.verify_checkpoint_source("proof/model", manifest, length)
        assert len(fetched) > completed_fetches
        # No lanes, engine, GPU admission, or executor exists in this host harness.
        # The real byte preparation completed using only the CPU/store interfaces.
    finally:
        release.set()
        serving.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_a_prefetch_narrates_each_hundredth_of_its_download_once() -> None:
    """Run 1510's 100 GB prefetch logged 4,065 positions: 4,065 of the run's 4,096 events.
    A hundredth a watcher can read; a demanded download's positions stay its live progress,
    on the lossy lane."""
    rows: list[dict[str, Any]] = []
    worker = cast(
        Worker,
        SimpleNamespace(workspace=None, emit_progress=lambda _p, _o, frame: rows.append(frame)),
    )
    serving = Serving(worker)
    request = pb.ChildCallRequest(parent_attempt_ordinal=1)
    stage = "generate / Downloading model weights"
    try:
        for index, child in (((1 << 32) - 1, "model-preparation.hint"), (0, "demand")):
            call = Call("parent", index, 1, 0, b"", b"", child, b"", False, b"", "", "")
            for landed in range(1001):
                frame = {"kind": "progress", "stage": stage, "position": landed, "total": 1000}
                serving._progress(call, request, frame)
    finally:
        serving.close()
    hinted, demanded = rows[:101], rows[101:]
    assert [row["fields"]["position"] for row in hinted] == list(range(0, 1001, 10))
    assert {row["name"] for row in hinted} == {"model-prefetch"}
    assert [row["position"] for row in demanded] == list(range(1001))
    assert serving.narrated == {}


def test_narrated_positions_never_shed_a_record() -> None:
    """Run 1510's prefetch narrated every position: 242 of its 256 triage rows, and its nine
    segments' records were shed. A row carrying a position samples one stream whose latest
    sample supersedes the last, in the attempt's ring and in its triage bundle, whatever
    its rate."""
    rows: list[dict[str, Any]] = []
    worker = cast(
        Worker,
        SimpleNamespace(workspace=None, emit_progress=lambda _p, _o, frame: rows.append(frame)),
    )
    serving = Serving(worker)
    call = Call("parent", (1 << 32) - 1, 1, 0, b"", b"", "hint", b"", False, b"", "", "")
    frame = {"kind": "progress", "stage": "turbo / Downloading model weights", "total": 4065}
    try:
        serving._progress(call, pb.ChildCallRequest(parent_attempt_ordinal=1), frame)
    finally:
        serving.close()
    (narrated,) = rows
    ring = EventRing()
    for index in range(MAX_EVENTS):
        ring.emit("stage", f"segment {index}", 1.0)
    for position in range(4065):
        # `AttemptEngine.forward`: each narrated row the worker emits reaches the ring.
        sample = {**narrated, "fields": {**narrated["fields"], "position": position}}
        ring.admit(AttemptEngine._row(msgspec.convert(sample, ObservationRow)))
    assert [row["name"] for row in ring.rows()] == [f"segment {i}" for i in range(MAX_EVENTS)]
    (latest,) = (msgspec.convert(row, ObservationRow) for row in ring.stream_rows())
    assert latest.name == "model-prefetch" and latest.fields["position"] == 4064
    caps = ring.caps()
    assert (caps["dropped"], caps["coalesced"], caps["streams"]) == (0, 4064, 1)
    bundle = triage.Assembly(
        subject_id="trb-" + "0" * 24,
        attempt={},
        plan={},
        posture={},
        terminal={},
        faults=[],
        measurements={},
        confessions=[],
        liveness=[],
        events=cast(Any, ring.rows()),
        streams=cast(Any, ring.stream_rows()),
        caps=cast(Any, caps),
    )
    # A bundle over its cap sheds the narration before any record.
    kept = msgspec.json.decode(bundle.build(len(bundle.build()) - 1), type=triage.Assembly)
    assert (kept.streams, len(kept.events), kept.caps["shed_to_fit"]) == ([], MAX_EVENTS, 1)


def test_a_call_held_for_weights_says_which_step_waits_and_for_how_long() -> None:
    """Run 1560's five references read 5m35s each: 28 s of work behind a 4m34s download of
    weights another call started. A held call names its step, so the step is timed without
    the wait and the wait is the download's."""
    rows: list[dict[str, Any]] = []
    worker = cast(
        Worker,
        SimpleNamespace(
            workspace=None,
            emit_progress=lambda _p, _o, frame: rows.append(frame),
            calls=SimpleNamespace(progress_label=lambda *_: "Creating reference Subject-4"),
        ),
    )
    serving = Serving(worker)
    call = Call("parent", 3, 1, 0, b"", b"", "call-4", b"", False, b"", "", "")
    target = Target("installed-model", "generate_image", {}, {}, None, "")
    try:
        released = serving.holding(call, pb.ChildCallRequest(parent_attempt_ordinal=1), target)
        time.sleep(0.01)
        released()
    finally:
        serving.close()
    start, end = (row["fields"] for row in rows)
    assert {row["name"] for row in rows} == {"model wait"}
    assert (start["event"], end["event"]) == ("start", "end")
    assert start["step"] == end["step"] == "Creating reference Subject-4"
    assert start["call"] == "call-4" and start["entrypoint"] == "generate_image"
    assert end["waited_ms"] >= 10
