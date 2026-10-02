"""The shared preparation path moves real TensorFS bytes once, without GPU activation."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import msgspec
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author._observations import MAX_EVENTS, EventRing
from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.executor_replies import ObservationRow
from cozy_runtime.internal.worker import store_gc, triage
from cozy_runtime.internal.worker.attempts import AttemptEngine
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_serving import Prepared, Serving, arguments
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from fault_hub import ALWAYS, ASSET, HEADER, Checkpoint, FaultHub, Gate, Watched, checkpoint, on
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot


def _serving(
    store_root: Path, *served: tuple[str, Checkpoint], label: str = ""
) -> tuple[Serving, list[Target], list[dict[str, Any]], list[dict[str, Any]]]:
    """A Serving over a real Workspace with one installation per `served` (Hub origin,
    checkpoint) whose one Model defaults to that checkpoint; its parents' progress frames land
    in the returned list. A call's step is `label`."""
    placements: list[dict[str, Any]] = []
    defaults: list[dict[str, Any]] = []
    targets: list[Target] = []
    for index, (origin, point) in enumerate(served):
        installation = f"installed-{index}"
        ref = {"digest": point.manifest, "length": point.length}
        placement = {
            "installation_id": installation,
            "package_interface": base64.b64encode(b"exact-interface").decode(),
            "models": [{"id": "model", "manifest": ref}],
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
        placements.append(placement)
        defaults.append(
            {
                "callee_installation_id": installation,
                "entrypoint": "generate",
                "parameter": "model",
                "public_origin": origin,
                "rungs": [{"gpu": "*", "repository": point.model, "manifest": ref}],
            }
        )
        declaration = {"models": [{"path": "generate.models.model"}]}
        targets.append(
            Target(installation, "generate", declaration, {"placement": placement}, None, "")
        )
    capture = {"model_defaults": defaults}
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
            model_preparation_progress={},
            model_preparation_observers={},
            prepared_installations={
                row["installation_id"]: SimpleNamespace(
                    document=documents.from_body(row, pb.Placement)
                )
                for row in placements
            },
            calls=SimpleNamespace(progress_label=lambda *_: label),
            emit_progress=lambda _parent, _ordinal, frame: progress.append(frame),
            # This host harness has no GPUs; the process warm is `test_startup_warm`'s.
            prespawns=SimpleNamespace(request=lambda *_: None),
        ),
    )
    return Serving(worker), targets, placements, progress


def test_real_model_pull_joins_across_future_payloads_and_reuses_cpu_binding(
    tmp_path: Path,
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
            self.answer(bodies[self.path.removeprefix("/")])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    store_root = tmp_path / "destination"
    store = tensorfs.Store.init(str(store_root))
    point = Checkpoint("proof/model", manifest, length, bodies)
    serving, (target,), (placement,), progress = _serving(
        store_root, (f"http://localhost:{server.server_port}", point)
    )
    call = Call("parent", 0, 1, 1, b"", b"", "child", b"", False, b"", "", "")
    request = pb.ChildCallRequest(parent_attempt_ordinal=1)
    try:
        first = serving._shared("owner", call, target, request, {}, speculative=True)
        assert first is not None
        assert started.wait(5), first.exception() if first.done() else "native pull not started"
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
        assert [row["event"] for row in fetch] == ["start", "end"]
        start, end = fetch
        assert start["model"] == end["model"] == "proof/model" and start["manifest"] == manifest
        assert (
            start["prefetch"] is True and start["step"] == "" and start["entrypoint"] == "generate"
        )
        assert end["completed"] is True and end["bytes"] == end["total_bytes"] > 0
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


def _ask(serving: Serving, parent: str, target: Target, *, speculative: bool) -> Future[Prepared]:
    """`parent`'s preparation of `target`: a hint, or a demand."""
    call = Call(parent, 0, 1, 1, b"", b"", parent + "-call", b"", False, b"", "", "")
    request = pb.ChildCallRequest(parent_attempt_ordinal=1)
    future = serving._shared("owner", call, target, request, {}, speculative=speculative)
    assert future is not None
    return future


def _fetches(progress: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """`model`'s `model fetch` records."""
    return [
        row["fields"]
        for row in progress
        if row.get("name") == "model fetch" and row["fields"]["model"] == model
    ]


def test_a_hint_keeps_pulling_while_a_demand_has_its_own_worker(tmp_path: Path) -> None:
    """Run 2322: real calls paused the H3 hint 0.2 s into its ~100 GB though each pulled on the
    pool's other worker, and the hint sat out the demand's 20-minute tail. A demand with a
    worker of its own leaves a hint's transfer alone."""
    hinted, demanded = (checkpoint(tmp_path / n, "proof/" + n, n + ".cozytensors") for n in "hd")
    opened = threading.Event()
    with (
        FaultHub(hinted, on(hinted.manifest[7:], 1, ALWAYS, Gate(opened))) as hint_hub,
        FaultHub(demanded) as demand_hub,
    ):
        store_root = tmp_path / "destination"
        tensorfs.Store.init(str(store_root))
        serving, (hint_target, demand_target), placements, progress = _serving(
            store_root, (hint_hub.origin, hinted), (demand_hub.origin, demanded)
        )
        try:
            hint = _ask(serving, "parent", hint_target, speculative=True)
            hint_hub.until(lambda: hint_hub.asks(hinted.manifest[7:]) >= 1)  # mid-GET
            demand = _ask(serving, "caller", demand_target, speculative=False)
            done = Watched(demand_hub, demand.result, lambda: len(progress)).result()
            assert done.placement == placements[1]
            opened.set()
            Watched(hint_hub, hint.result, lambda: len(progress)).result()
        finally:
            serving.close()
    fetched = [
        (row["event"], row.get("completed"), row.get("paused_for"))
        for row in _fetches(progress, hinted.model)
    ]
    assert fetched == [("start", None, None), ("end", True, None)], fetched


def test_a_hint_yields_its_worker_to_a_demand_left_waiting(tmp_path: Path) -> None:
    """The one pause left: a hint and a demand hold both workers and a second demand waits.
    The hint's fetch ends `completed: false` saying why and for whom, and it resumes with a
    fresh cancellation once a worker is free, its landed objects kept."""
    hinted, first, second = (
        checkpoint(tmp_path / n, "proof/" + n, n + ".cozytensors") for n in ("h", "a", "b")
    )
    # TensorFS asks a silent object again, so a gate holds every ask until it opens.
    parked, held = threading.Event(), threading.Event()
    with (
        FaultHub(hinted, on(hinted.manifest[7:], 1, ALWAYS, Gate(parked))) as hint_hub,
        FaultHub(first, on(first.manifest[7:], 1, ALWAYS, Gate(held))) as first_hub,
        FaultHub(second) as second_hub,
    ):
        store_root = tmp_path / "destination"
        store = tensorfs.Store.init(str(store_root))
        serving, targets, placements, progress = _serving(
            store_root,
            (hint_hub.origin, hinted),
            (first_hub.origin, first),
            (second_hub.origin, second),
        )
        try:
            hint = _ask(serving, "parent", targets[0], speculative=True)
            hint_hub.until(
                lambda: (
                    hint_hub.asks(hinted.manifest[7:]) >= 1
                    and store.contains(HEADER)
                    and store.contains(ASSET)
                )
            )
            running = _ask(serving, "first", targets[1], speculative=False)
            first_hub.until(lambda: first_hub.asks(first.manifest[7:]) >= 1)
            waiting = _ask(serving, "second", targets[2], speculative=False)
            Watched(second_hub, waiting.result, lambda: len(progress)).result()
            parked.set()
            resumed = Watched(hint_hub, hint.result, lambda: len(progress)).result()
            assert resumed.placement == placements[0]
            held.set()
            Watched(first_hub, running.result, lambda: len(progress)).result()
        finally:
            serving.close()
    paused, ended = (row for row in _fetches(progress, hinted.model) if row["event"] == "end")
    assert paused["completed"] is False and paused["paused_for"] == "second", paused
    assert "cancel" in paused["reason"], paused
    assert ended["completed"] is True and "reason" not in ended, ended
    # Only the cut GET is asked again.
    assert (hint_hub.asks(HEADER), hint_hub.asks(ASSET)) == (1, 1), hint_hub.log()
    assert hint_hub.asks(hinted.manifest[7:]) >= 2, hint_hub.log()


def test_a_called_download_reports_its_bytes_on_its_steps_line(tmp_path: Path) -> None:
    """The download phase is named for the call's step; its byte samples were named for the
    entrypoint, so a client drew them as a second line."""
    point = checkpoint(tmp_path / "origin")
    step = "Creating reference Subject-4"
    with FaultHub(point) as hub:
        store_root = tmp_path / "destination"
        tensorfs.Store.init(str(store_root))
        serving, (target,), (placement,), progress = _serving(
            store_root, (hub.origin, point), label=step
        )
        call = Call("parent", 3, 1, 1, b"", b"", "call-4", b"", False, b"", "", "")
        request = pb.ChildCallRequest(parent_attempt_ordinal=1)
        try:
            prepared = serving._shared("owner", call, target, request, {}, speculative=False)
            assert prepared is not None
            assert Watched(hub, prepared.result).result().placement == placement
        finally:
            serving.close()
    samples = [row for row in progress if row.get("kind") == "progress"]
    assert any(row.get("position") == row.get("total") for row in samples), samples
    assert {row["stage"] for row in samples} == {step + " / Downloading model weights"}


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
