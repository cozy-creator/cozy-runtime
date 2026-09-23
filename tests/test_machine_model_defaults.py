from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, hostfacts
from cozy_runtime.internal.worker import grants, machine_checkpoint_inputs
from cozy_runtime.internal.worker import machine_model_defaults as defaults
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_serving import Serving
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import _config
from test_group_lanes import _placement
from test_machine_execution import ack, complete, offer
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot

REVISION = "sha256:" + "11" * 32
MANIFEST = {"digest": "sha256:" + "22" * 32, "length": 321}


def captured() -> dict[str, Any]:
    return {
        "bindings": [{"callee_revision_digest": REVISION, "entrypoint": "generate"}],
        "model_defaults": [
            {
                "callee_revision_digest": REVISION,
                "entrypoint": "generate",
                "parameter": "adapter",
                "public_origin": "https://catalog.example",
                "rungs": [
                    {"gpu": "H100", "repository": "proof/adapter", "manifest": MANIFEST},
                    {"gpu": "*", "repository": "proof/adapter", "manifest": MANIFEST},
                ],
            },
            {
                "callee_revision_digest": REVISION,
                "entrypoint": "generate",
                "parameter": "model",
                "unavailable_code": "model_default_unbound",
            },
        ],
    }


def peer(capture: dict[str, Any]) -> Worker:
    return cast(
        Worker,
        SimpleNamespace(
            config=SimpleNamespace(bootstrap_credential="local-bootstrap"),
            options=SimpleNamespace(accelerator_backend="none"),
            executions=SimpleNamespace(capture_root=lambda *_: "root", capture=lambda *_: capture),
        ),
    )


def target() -> Target:
    return Target(
        revision=REVISION,
        entrypoint="generate",
        interface_digest=REVISION,
        declaration={
            "models": [{"path": "generate.models.adapter"}, {"path": "generate.models.model"}]
        },
        prepared_revision={},
        binding=None,
        digest=REVISION,
    )


def test_default_capture_requires_complete_exact_inventory_and_capable_peer() -> None:
    body = captured()
    interfaces = {REVISION: {"entrypoints": [{"name": "generate", **target().declaration}]}}
    request = pb.MachineExecutionSubmit(claim=pb.Claim(wire_minor=56))
    defaults.verify(peer(body), request, body, interfaces)
    mutations: list[Callable[[dict[str, Any]], Any]] = [
        lambda value: value["model_defaults"].pop(),
        lambda value: value["model_defaults"].reverse(),
        lambda value: value["model_defaults"][0].update(parameter="foreign"),
        lambda value: value["model_defaults"][0].update(
            public_origin="https://user:secret@catalog.example"
        ),
        lambda value: value["model_defaults"][0].update(
            public_origin="https://catalog.example/foreign"
        ),
        lambda value: value["model_defaults"][0].update(
            unavailable_code="model_default_unavailable"
        ),
        lambda value: value["model_defaults"][0]["rungs"][0].update(repository="local/forged"),
        lambda value: value["model_defaults"][0]["rungs"][0].update(gpu="*"),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(body)
        mutate(changed)
        with pytest.raises(WorkspaceRefusal):
            defaults.verify(peer(changed), request, changed, interfaces)
    request.claim.wire_minor = 55
    with pytest.raises(WorkspaceRefusal, match="wire56"):
        defaults.verify(peer(body), request, body, interfaces)
    assert defaults.valid_origin("http://127.0.0.1:8819", local=True)
    assert not defaults.valid_origin("http://127.0.0.1:8819", local=False)
    assert not defaults.valid_origin("http://169.254.169.254", local=True)


def test_unused_or_overridden_defaults_do_not_touch_catalog_or_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("unused default reached TensorFS")

    monkeypatch.setattr(fill, "tensorfs_module", forbidden)
    empty_peer = cast(Worker, SimpleNamespace())
    call = cast(Call, SimpleNamespace(parent_request="root"))
    assert defaults.select(empty_peer, "owner", call, target(), {"model": {}, "adapter": {}}) == []
    defaults.materialize(empty_peer, [], ExitStack())
    worker = peer(captured())
    selected = defaults.select(worker, "owner", call, target(), {"model": {}, "adapter": None})
    assert (
        len(selected) == 1 and selected[0]["parameter"] == "adapter" and selected[0]["gpu"] == "*"
    )
    with pytest.raises(WorkspaceRefusal, match="unavailable for model"):
        defaults.select(worker, "owner", call, target(), {"adapter": {}})
    assert defaults.matches("H100", "NVIDIA H100 80GB HBM3")
    assert not defaults.matches("H100", "NVIDIA H200")
    assert not defaults.matches("H100", "")


@pytest.mark.parametrize("reuse", [False, True])
def test_serving_default_uses_real_catalog_input_pin_without_derived_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reuse: bool
) -> None:
    root, manifest, length, store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    operation = store.begin_operation("default-fixture", "proof", "adapter")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1.0.0", "bf16", manifest, length)
    workspace = Workspace(root)
    row = {
        "parameter": "adapter",
        "public_origin": "https://unreachable.example",
        "repository": "proof/adapter",
        "manifest": {"digest": manifest, "length": length},
    }
    worker = peer({})
    worker.workspace = workspace

    def no_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an already verified exact native checkpoint was downloaded")

    monkeypatch.setattr(tensorfs, "pull", no_download)
    with ExitStack() as read_access:
        leases = set((root / "tmp/leases").glob("read-*.lease"))
        defaults.materialize(worker, [row], None if reuse else read_access)
        assert bool(set((root / "tmp/leases").glob("read-*.lease")) - leases) is not reuse
        bindings, accesses = defaults.inputs([row])
        spec = pb.InvocationSpec(
            inputs=bindings,
            serving=pb.ServingInvocationSpec(
                entrypoint_binding_digest=REVISION,
                attempt_binding_id=REVISION,
                bindings_digest=REVISION,
            ),
        )
        raw, digest = documents.identity(spec)
        offer = pb.AttemptOffer(
            request_id="serving-child",
            attempt_ordinal=1,
            invocation_spec_canonical_bytes=raw,
            invocation_spec_digest=digest,
            grant=pb.DeliveryGrant(invocation_spec_digest=digest, inputs=accesses),
        )
        entries = grants.model_inputs(
            grants.bind(documents.read(raw, pb.InvocationSpec), offer.grant, digest)
        )
        with pytest.raises(WorkspaceRefusal, match="not retained"), workspace.locked() as db:
            Workspace.accept_in(db, "owner", offer)
        with workspace.locked() as db:
            assert db.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0
        with machine_checkpoint_inputs.admission(workspace, "owner", offer.request_id):
            machine_checkpoint_inputs.preflight(workspace, "owner", offer, entries)
            for entry in entries.values():
                machine_checkpoint_inputs.retain(workspace, "owner", offer, entry)
            executions = Executions(workspace)
            executions.submit(
                "owner",
                "serving-input",
                b"x" * 32,
                offer,
                expected_execution_workspace_id=executions.workspace_id,
            )
    with workspace.locked() as db:
        pin = db.execute(
            "SELECT * FROM execution_checkpoint_inputs WHERE recipient='serving-child'"
        ).fetchone()
        assert pin["input_id"] == "model:adapter" and pin["repository"] == "proof/adapter"
        assert pin["state"] == "held"
        for table in ("weights", "native_calls", "holds", "execution_model_holds"):
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    tensorfs.gc(store.root)
    restored = Executions(Workspace(root))
    result = complete(restored, "serving-child", 17)
    restored.acknowledge_collection("owner", ack(result))
    assert store.checkpoint_root(pin["native_owner"])["released"]


def test_unavailable_default_refuses_before_accelerator_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_: str) -> Any:
        raise AssertionError("unavailable default must not measure an accelerator")

    monkeypatch.setattr(hostfacts, "measure", forbidden)
    call = cast(Call, SimpleNamespace(parent_request="root"))
    captures: list[dict[str, Any]] = [captured(), {"model_defaults": []}]
    for capture in captures:
        with pytest.raises(WorkspaceRefusal, match="captured Model default is unavailable"):
            defaults.select(peer(capture), "owner", call, target(), {"adapter": {}})


@pytest.mark.parametrize("mutation", ["repository", "asset"])
def test_registered_serving_binding_skips_read_lease_but_rechecks_native_source(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, manifest, length, store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    operation = store.begin_operation("registered-default", "proof", "adapter")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1.0.0", "bf16", manifest, length)
    worker = Worker(
        _config(tmp_path / "home"),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=root, accelerator_backend="none"),
        InMemoryControlHost(),
    )
    serving = Serving(worker)
    worker.fence.record_owner_id = "owner"
    assert worker.executions is not None
    source = {"digest": manifest, "length": length}
    capture = captured()
    capture["format"] = "cozy.worker.v1.MachineExecutionCapture/1"
    capture["model_defaults"] = [
        {
            "callee_revision_digest": REVISION,
            "entrypoint": "generate",
            "parameter": "adapter",
            "public_origin": "https://unreachable.example",
            "rungs": [{"gpu": "*", "repository": "proof/adapter", "manifest": source}],
        }
    ]
    worker.executions.submit(
        "owner",
        "parent",
        b"c" * 32,
        offer(),
        capture_document=canonical_json.encode(capture),
        expected_execution_workspace_id=worker.executions.workspace_id,
    )
    document = {
        "development": {"local_revision_digest": REVISION},
        "environment_digest": REVISION,
        "package_interface": MANIFEST,
        "models": [{"id": "default", "manifest": source}],
        "entrypoints": [
            {
                "name": "generate",
                "slots": [{"slot": "adapter", "reference_model_id": "default"}],
            }
        ],
    }
    placement = _placement("registered")
    placement.document = document
    worker.prepared_revisions[(placement.placement_id, REVISION)] = placement
    selected = dataclasses.replace(
        target(),
        declaration={"models": [{"path": "generate.models.adapter"}]},
        prepared_revision={"placement": document},
    )
    call = Call("root", 0, 1, 1, b"i" * 32, b"{}", "child", b"", False, b"", "", "")
    request = pb.ChildCallRequest(
        parent_attempt_ordinal=1, request_canonical_bytes=b'{"payload":{},"models":{}}'
    )
    try:
        with serving._prepare("owner", call, selected, request).access:
            assert not list((root / "tmp/leases").glob("read-*.lease"))
        # Exact registered metadata is not byte custody. A change after the fast path
        # must still be refused by the child's ordinary native admission.
        bindings, accesses = defaults.inputs(
            [{"parameter": "adapter", "repository": "proof/adapter", "manifest": source}]
        )
        spec = pb.InvocationSpec(
            inputs=bindings,
            serving=pb.ServingInvocationSpec(
                entrypoint_binding_digest=REVISION,
                attempt_binding_id=REVISION,
                bindings_digest=REVISION,
            ),
        )
        raw, digest = documents.identity(spec)
        child = pb.AttemptOffer(
            request_id="child",
            attempt_ordinal=1,
            invocation_spec_canonical_bytes=raw,
            invocation_spec_digest=digest,
            grant=pb.DeliveryGrant(invocation_spec_digest=digest, inputs=accesses),
        )
        entries = grants.model_inputs(
            grants.bind(documents.read(raw, pb.InvocationSpec), child.grant, digest)
        )
        if mutation == "repository":
            (root / "repos/proof/adapter.json").unlink()
            reason = "REPOSITORY_ABSENT"
        else:
            asset = next(
                path
                for path in (root / "blobs").rglob("*")
                if path.is_file() and path.read_bytes() == _ASSET
            )
            asset.chmod(0o600)
            asset.write_bytes(b"wrong\n")
            reason = "OBJECT_CORRUPT"
        assert worker.workspace is not None
        with pytest.raises(WorkspaceRefusal, match=reason):
            machine_checkpoint_inputs.preflight(worker.workspace, "owner", child, entries)
        with worker.workspace.locked() as db:
            assert not db.execute("SELECT 1 FROM execution_checkpoint_inputs").fetchone()
    finally:
        serving.close()
        worker.supervision.close()
        worker.weights.close()


def test_cold_default_native_pull_requires_deployment_object_storage_host(
    tmp_path: Path,
) -> None:
    origin_root = tmp_path / "origin"
    origin_root.mkdir()
    _, manifest, length, origin = _snapshot(origin_root, include_asset=True, checkpoint_only=True)
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
    objects = [{"length": len(body), "sha256": digest} for digest, body in sorted(bodies.items())]
    bodies[manifest.removeprefix("sha256:")] = origin.manifest(manifest)["manifest"]
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
            self.answer(json.dumps(response, sort_keys=True, separators=(",", ":")).encode())

        def do_GET(self) -> None:
            fetched.append(self.path)
            self.answer(bodies[self.path.removeprefix("/")])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "destination"
    store = tensorfs.Store.init(str(root))
    worker = peer({})
    worker.workspace = Workspace(root)
    row = {
        "public_origin": f"http://localhost:{server.server_port}",
        "repository": "proof/adapter",
        "manifest": {"digest": manifest, "length": length},
    }
    try:
        worker.config = cast(
            Any, SimpleNamespace(bootstrap_credential="local", object_storage_hosts=())
        )
        with ExitStack() as access, pytest.raises(WorkspaceRefusal, match="SOURCE_NOT_ALLOWED"):
            defaults.materialize(worker, [row], access)
        assert fetched == []
        worker.config = cast(
            Any, SimpleNamespace(bootstrap_credential="local", object_storage_hosts=("127.0.0.1",))
        )
        with ExitStack() as access:
            defaults.materialize(worker, [row], access)
            store.verify_checkpoint_source("proof/adapter", manifest, length)
        assert set(fetched) == {"/" + digest for digest in bodies}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
