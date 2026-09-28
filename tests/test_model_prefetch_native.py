"""The shared preparation path moves real TensorFS bytes once, without GPU activation."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import tensorfs

from cozy_runtime.internal import hostfacts
from cozy_runtime.internal.worker import store_gc
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_serving import Serving
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import worker_pb2 as pb
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot


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
    checkpoint = {"digest": manifest, "length": length}
    placement = {
        "installation_id": "installed-model",
        "package_interface": "exact-interface",
        "models": [{"id": "model", "manifest": checkpoint}],
        "entrypoints": [
            {
                "name": "generate",
                "slots": [
                    {
                        "slot": "model",
                        "reference_model_id": "model",
                        "components": [{"component": "unet", "model_id": "model"}],
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
            options=SimpleNamespace(accelerator_backend="none", publication_authority=None),
            host_facts=lambda: hostfacts.measure("none"),
            lanes=SimpleNamespace(entries=()),
            config=SimpleNamespace(object_storage_hosts=("127.0.0.1",)),
            control_lock=threading.RLock(),
            model_transfer_lock=threading.Lock(),
            model_transfers={},
            prepared_installations={"model": SimpleNamespace(document=placement)},
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
        # A's image becomes available later. It must not change the preparation key.
        joined = serving._shared(
            "owner",
            call,
            target,
            request,
            {"image": "now-available", "model": None},
            speculative=False,
        )
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
        # The exact finished selection serves every demand; nothing is prepared again.
        again = serving._shared("owner", call, target, request, {}, speculative=False)
        assert again is not None and again is first
        assert again.result(5).placement == placement
        assert len(fetched) == len(bodies)
        # No persistent read lease prevents legitimate cache eviction. Demanding the
        # completed metadata after a collection must refill bytes, not fail admission.
        store_gc.collect(store_root, keep_manifests=[])
        refilled = serving._shared("owner", call, target, request, {}, speculative=False)
        assert refilled is not first
        assert refilled is not None and refilled.result(5).placement == placement
        store.verify_checkpoint_source("proof/model", manifest, length)
        assert len(fetched) > len(bodies)
        # No lanes, engine, GPU admission, or executor exists in this host harness.
        # The real byte preparation completed using only the CPU/store interfaces.
    finally:
        release.set()
        serving.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_a_prefetch_narrates_each_tenth_of_its_download_once() -> None:
    """Run 1510's 100 GB prefetch logged 4,065 positions: 4,065 of the run's 4,096 events.
    A demanded download's positions stay its live progress, on the lossy lane."""
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
    hinted, demanded = rows[:11], rows[11:]
    assert [row["fields"]["position"] for row in hinted] == list(range(0, 1001, 100))
    assert {row["name"] for row in hinted} == {"model-prefetch"}
    assert [row["position"] for row in demanded] == list(range(1001))
    assert serving.narrated == {}
