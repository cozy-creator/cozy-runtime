"""The Runtime's models-only lane owns real TensorFS transfer and scoped credentials."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tensorfs

from cozy_runtime.internal.worker import machine_materialization
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot


def test_models_only_prepare_fetches_exact_checkpoint_then_works_offline(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _, manifest, length, origin = _snapshot(source, include_asset=True, checkpoint_only=True)
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
    objects = [{"sha256": key, "length": len(body)} for key, body in bodies.items()]
    bodies[manifest.removeprefix("sha256:")] = origin.manifest(manifest)["manifest"]
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def answer(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer execution-access"
            assert "X-Cozy-Worker-ID" not in self.headers
            seen.append(self.path)
            if self.path == "/v1/tensorfs/closure":
                assert request["ref"] == "proof/adapter@" + manifest
                response = {
                    "complete": True,
                    "lane": "bf16",
                    "model": "proof/adapter",
                    "manifest": {"sha256": manifest.removeprefix("sha256:"), "length": length},
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
            self.answer(json.dumps(response).encode())

        def do_GET(self) -> None:
            assert "Authorization" not in self.headers
            self.answer(bodies[self.path.removeprefix("/")])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "destination"
    store = tensorfs.Store.init(str(root))
    hub = f"http://localhost:{server.server_port}"
    authority = PublicationAuthority(
        hub,
        object_storage_hosts=("127.0.0.1",),
        access_token="execution-access",
        expires_at=int(time.time()) + 60,
    )
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=root,
            hubs=(authority,),
            accelerator_backend="none",
            devices="",
            sole_supervisor=True,
        ),
        InMemoryControlHost(),
    )
    request = pb.PreparePackageSetRequest(
        download_delegation=documents.canonical_bytes(
            pb.DownloadDelegation(
                models=[pb.DownloadModelRef(model="proof/adapter", manifest=manifest)]
            )
        ),
        hub=hub,
    )
    try:
        result = worker.prepare_package_set(request)
        assert result.HasField("placement_set")
        store.verify_checkpoint_source("proof/adapter", manifest, length)
        assert seen == ["/v1/tensorfs/closure", "/v1/tensorfs/presign"]
        server.shutdown()
        thread.join()
        seen.clear()
        worker.options = dataclasses.replace(worker.options, hubs=())
        assert worker.prepare_package_set(request) == result
        assert seen == []
    finally:
        worker.shutdown()
        server.shutdown()
        server.server_close()
        thread.join()


def test_preparation_closure_includes_base_and_native_adapters_once() -> None:
    base = pb.DownloadModelRef(model="org/base", manifest="sha256:" + "a" * 64)
    adapter = pb.DownloadAdapterRef(model="org/adapter", manifest="sha256:" + "b" * 64)
    base.adapters.append(adapter)
    raw = documents.canonical_bytes(pb.DownloadDelegation(models=[base]))
    selected = machine_materialization.models(raw, [pb.NativeModelBinding(adapters=[adapter])])
    assert {(row.model, row.manifest) for row in selected} == {
        (base.model, base.manifest),
        (adapter.model, adapter.manifest),
    }
