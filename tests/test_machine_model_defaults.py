from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.machine_serving import Serving
from cozy_runtime.internal.worker.session import Placement, Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from fault_hub import ALWAYS, HEADER, FaultHub, Gate, Stall, Watched, checkpoint, on, temps
from test_device_lanes import _config
from test_machine_execution import ack, complete, offer
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot

INSTALLATION = "local-" + "11" * 16
BINDING = "sha256:" + "11" * 32
MANIFEST = {"digest": "sha256:" + "22" * 32, "length": 321}


def captured() -> dict[str, Any]:
    return {
        "bindings": [{"callee_installation_id": INSTALLATION, "entrypoint": "generate"}],
        "model_defaults": [
            {
                "callee_installation_id": INSTALLATION,
                "entrypoint": "generate",
                "parameter": "adapter",
                "public_origin": "https://catalog.example",
                "rungs": [
                    {"gpu": "H100", "repository": "proof/adapter", "manifest": MANIFEST},
                    {"gpu": "*", "repository": "proof/adapter", "manifest": MANIFEST},
                ],
            },
            {
                "callee_installation_id": INSTALLATION,
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
            config=SimpleNamespace(),
            options=SimpleNamespace(
                accelerator_backend="none", publication_authority=None, hubs=()
            ),
            host_facts=lambda: hostfacts.measure("none"),
            model_transfer_lock=threading.Lock(),
            model_preparation_progress={},
            model_preparation_observers={},
            executions=SimpleNamespace(
                capture_root=lambda *_: "root",
                capture=lambda *_: capture,
                prepared=lambda *_: SimpleNamespace(hub=""),
            ),
            lanes=SimpleNamespace(entries=()),
        ),
    )


def parsed(capture: dict[str, Any]) -> pb.MachineExecutionCapture:
    tag = documents.doc_format(pb.MachineExecutionCapture.DESCRIPTOR.full_name)
    raw = canonical_json.encode({"format": tag, **capture})
    return documents.parse(raw, pb.MachineExecutionCapture)


def target() -> Target:
    return Target(
        installation_id=INSTALLATION,
        entrypoint="generate",
        declaration={
            "models": [{"path": "generate.models.adapter"}, {"path": "generate.models.model"}]
        },
        prepared_installation={},
        binding=None,
        operation_identity="",
    )


def test_default_capture_admits_any_order_and_open_slots_but_not_forged_rungs() -> None:
    body = captured()
    defaults.verify(peer(body), parsed(body))
    # The Creator's order, a slot left open (resolved at the machine's Hub) and a row for a
    # slot this interface lacks are all an inventory this machine can serve.
    tolerated_changes: list[Callable[[dict[str, Any]], Any]] = [
        lambda value: value["model_defaults"].pop(),
        lambda value: value["model_defaults"].reverse(),
        lambda value: value["model_defaults"][0].update(parameter="foreign"),
    ]
    for tolerated in tolerated_changes:
        changed = copy.deepcopy(body)
        tolerated(changed)
        defaults.verify(peer(changed), parsed(changed))
    mutations: list[Callable[[dict[str, Any]], Any]] = [
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
            defaults.verify(peer(changed), parsed(changed))
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
    assert (
        defaults.select(
            empty_peer, "owner", call, target(), {"model": {}, "adapter": {}}, model_choices={}
        )
        == []
    )
    defaults.materialize(empty_peer, [])
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


def test_serving_default_uses_real_catalog_input_pin_without_derived_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    leases = set((root / "tmp/leases").glob("read-*.lease"))
    defaults.materialize(worker, [row])
    # Materializing leases nothing: admission's native root is what holds the bytes.
    assert set((root / "tmp/leases").glob("read-*.lease")) == leases
    bindings, accesses = defaults.inputs([row])
    spec = pb.InvocationSpec(
        inputs=bindings,
        serving=pb.ServingInvocationSpec(
            entrypoint_binding_digest=BINDING,
            attempt_binding_id=BINDING,
            bindings_digest=BINDING,
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
    with pytest.raises(WorkspaceRefusal, match="captured Model default is unavailable"):
        defaults.select(peer(captured()), "owner", call, target(), {"adapter": {}})
    # An open slot is read at the machine's own Hub; a machine without a grant has none.
    with pytest.raises(WorkspaceRefusal, match="names no Hub"):
        defaults.select(peer({"model_defaults": []}), "owner", call, target(), {"adapter": {}})


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
    capture["format"] = "cozy.worker.v1.MachineExecutionCapture/2"
    capture["model_defaults"] = [
        {
            "callee_installation_id": INSTALLATION,
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
        "development": {"installation_id": INSTALLATION},
        "installation_id": INSTALLATION,
        "package_interface": "e30=",
        "models": [{"id": "default", "manifest": source}],
        "entrypoints": [
            {
                "name": "generate",
                "slots": [{"slot": "adapter", "reference_model_id": "default"}],
            }
        ],
    }
    placement = Placement(documents.from_body(document, pb.Placement), b"")
    worker.prepared_installations[placement.prepared_key] = placement
    selected = dataclasses.replace(
        target(),
        declaration={"models": [{"path": "generate.models.adapter"}]},
        prepared_installation={"placement": document},
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
                entrypoint_binding_digest=BINDING,
                attempt_binding_id=BINDING,
                bindings_digest=BINDING,
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
        # A stale held index skips only the early check; retention refuses it the same.
        held = {documents.spell(entry.digest) for entry in entries.values()}
        with (
            pytest.raises(WorkspaceRefusal, match=reason),
            machine_checkpoint_inputs.admission(worker.workspace, "owner", "child"),
        ):
            machine_checkpoint_inputs.preflight(
                worker.workspace, "owner", child, entries, held=held
            )
            for entry in entries.values():
                machine_checkpoint_inputs.retain(worker.workspace, "owner", child, entry)
        with worker.workspace.locked() as db:
            assert not db.execute(
                "SELECT 1 FROM execution_checkpoint_inputs WHERE state<>'released'"
            ).fetchone()
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
        "parameter": "adapter",
        "public_origin": f"http://localhost:{server.server_port}",
        "repository": "proof/adapter",
        "manifest": {"digest": manifest, "length": length},
    }
    try:
        worker.config = cast(Any, SimpleNamespace(object_storage_hosts=()))
        with pytest.raises(WorkspaceRefusal, match="SOURCE_NOT_ALLOWED"):
            defaults.materialize(worker, [row])
        assert fetched == []
        worker.config = cast(Any, SimpleNamespace(object_storage_hosts=("127.0.0.1",)))
        defaults.materialize(worker, [row])
        store.verify_checkpoint_source("proof/adapter", manifest, length)
        assert set(fetched) == {"/" + digest for digest in bodies}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_a_cold_preparation_pulls_its_slots_at_once(tmp_path: Path) -> None:
    """Run 1510 pulled H3's turbo adapter only after its 100 GB base had landed. Here each
    slot's closure waits for the other's, which a one-after-another pull never shows."""
    refs: dict[str, tuple[str, int]] = {}
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
    objects = [{"length": len(body), "sha256": digest} for digest, body in sorted(bodies.items())]
    _, base, _, origin = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    for name in ("base", "adapter"):
        # Same bytes under a different path: a second, distinct checkpoint manifest.
        raw = origin.manifest(base)["manifest"].replace(b"model.cozytensors", name.encode())
        manifest = "sha256:" + hashlib.sha256(raw).hexdigest()
        refs[f"proof/{name}@{manifest}"] = (manifest, len(raw))
        bodies[manifest.removeprefix("sha256:")] = raw
    asked = {ref: threading.Event() for ref in refs}
    together: list[bool] = []

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
                asked[request["ref"]].set()
                together.append(all(event.wait(10) for event in asked.values()))
                manifest, length = refs[request["ref"]]
                response = {
                    "complete": True,
                    "lane": "bf16",
                    "model": request["ref"].partition("@")[0],
                    "manifest": {"sha256": manifest.removeprefix("sha256:"), "length": length},
                    "objects": objects,
                    "presign_max_digests": 10,
                    "release": "1.0.0",
                    "scope": "runtime",
                    "server_time_unix": int(time.time()),
                }
            else:
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
            self.answer(bodies[self.path.removeprefix("/")])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "destination"
    store = tensorfs.Store.init(str(root))
    worker = peer({})
    worker.workspace = Workspace(root)
    worker.config = cast(Any, SimpleNamespace(object_storage_hosts=("127.0.0.1",)))
    rows = [
        {
            "parameter": ref.partition("@")[0].removeprefix("proof/"),
            "public_origin": f"http://localhost:{server.server_port}",
            "repository": ref.partition("@")[0],
            "manifest": {"digest": manifest, "length": length},
        }
        for ref, (manifest, length) in refs.items()
    ]
    reported: list[tuple[int, int]] = []
    try:
        defaults.materialize(
            worker, rows, progress=lambda done, whole: reported.append((done, whole))
        )
        for ref, (manifest, length) in refs.items():
            store.verify_checkpoint_source(ref.partition("@")[0], manifest, length)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert together == [True, True]
    # One position for the preparation: both slots' bytes, landed.
    assert reported[-1][0] == reported[-1][1] == max(whole for _, whole in reported)


def test_a_private_checkpoint_is_pulled_with_the_worker_capability_at_its_own_hub_only(
    tmp_path: Path,
) -> None:
    """The machine's own Hub serves its owner's unpublished checkpoint only to the worker
    capability; another origin never sees it, and a refused or absent grant is surfaced
    typed with no anonymous retry."""
    origin_root = tmp_path / "origin"
    origin_root.mkdir()
    _, manifest, length, origin = _snapshot(origin_root, include_asset=True, checkpoint_only=True)
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
    objects = [{"length": len(body), "sha256": digest} for digest, body in sorted(bodies.items())]
    bodies[manifest.removeprefix("sha256:")] = origin.manifest(manifest)["manifest"]
    token = "t" * 43
    servers: list[ThreadingHTTPServer] = []

    def hub(private: bool) -> tuple[str, list[tuple[str, str, str]]]:
        seen: list[tuple[str, str, str]] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def answer(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                presented = (
                    self.headers.get("X-Cozy-Worker-ID", ""),
                    self.headers.get("X-Cozy-Worker-Token", ""),
                )
                seen.append((self.path, *presented))
                if private and presented != ("worker-1", token):
                    code = "worker.ended" if presented[0] else "tensorfs.closure_denied"
                    error = {"error": {"code": code, "message": code}}
                    self.answer(401 if presented[0] else 404, json.dumps(error).encode())
                    return
                if self.path == "/v1/tensorfs/closure":
                    response: dict[str, Any] = {
                        "complete": True,
                        "lane": "bf16",
                        "model": "fidika/private",
                        "manifest": {"sha256": manifest.removeprefix("sha256:"), "length": length},
                        "objects": objects,
                        "presign_max_digests": 10,
                        "release": "1.0.0",
                        "scope": "runtime",
                        "server_time_unix": int(time.time()),
                    }
                else:
                    response = {
                        "expires_at_unix": int(time.time()) + 3600,
                        "server_time_unix": int(time.time()),
                        "urls": {
                            digest: f"http://127.0.0.1:{server.server_port}/{digest}"
                            for digest in request["digests"]
                        },
                    }
                self.answer(200, json.dumps(response, sort_keys=True).encode())

            def do_GET(self) -> None:
                seen.append(
                    (
                        "GET",
                        self.headers.get("X-Cozy-Worker-ID", ""),
                        self.headers.get("X-Cozy-Worker-Token", ""),
                    )
                )
                self.answer(200, bodies[self.path.removeprefix("/")])

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://localhost:{server.server_port}", seen

    own, own_seen = hub(private=True)
    other, other_seen = hub(private=False)

    def materialize(
        name: str,
        public_origin: str,
        authority: PublicationAuthority | None,
        hubs: tuple[PublicationAuthority, ...] = (),
        hosts: tuple[str, ...] = ("127.0.0.1",),
    ) -> None:
        root = tmp_path / name
        tensorfs.Store.init(str(root))
        worker = peer({})
        worker.workspace = Workspace(root)
        worker.options = cast(
            Any,
            SimpleNamespace(accelerator_backend="none", publication_authority=authority, hubs=hubs),
        )
        worker.config = cast(Any, SimpleNamespace(object_storage_hosts=hosts))
        row = {
            "parameter": "model",
            "public_origin": public_origin,
            "repository": "fidika/private",
            "manifest": {"digest": manifest, "length": length},
        }
        defaults.materialize(worker, [row])
        tensorfs.Store.open(str(root)).verify_checkpoint_source("fidika/private", manifest, length)

    grant = PublicationAuthority(own, "worker-1", token)
    try:
        materialize("own", own, grant)
        posts = [row for row in own_seen if row[0] != "GET"]
        assert posts and all(row[1:] == ("worker-1", token) for row in posts)
        assert all(row[1:] == ("", "") for row in own_seen if row[0] == "GET")

        materialize("other", other, grant)
        assert other_seen and all(row[1:] == ("", "") for row in other_seen)

        # Registered at the other Hub too, the machine presents that registration there and
        # reaches that Hub's object hosts, not its default Hub's.
        other_seen.clear()
        there = PublicationAuthority(other, "worker-2", token, None, ("127.0.0.1",))
        materialize("registered", other, grant, (there,), ())
        posts = [row for row in other_seen if row[0] != "GET"]
        assert posts and all(row[1:] == ("worker-2", token) for row in posts)

        for name, authority, code in (
            ("ended", PublicationAuthority(own, "worker-1", "e" * 43), "CREDENTIAL_REQUIRED"),
            ("absent", None, "REF_NOT_FOUND"),
        ):
            own_seen.clear()
            with pytest.raises(WorkspaceRefusal, match="materialization refused: " + code):
                materialize(name, own, authority)
            presented = ("", "") if authority is None else ("worker-1", "e" * 43)
            assert own_seen == [("/v1/tensorfs/closure", *presented)], own_seen

        # A row naming no Hub cannot be taken anywhere: refused as that, with nothing asked.
        own_seen.clear()
        with pytest.raises(WorkspaceRefusal, match="model_access_absent: fidika/private@sha256:"):
            materialize("unnamed", "", grant)
        assert own_seen == []
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_a_second_pull_of_one_checkpoint_waits_in_tensorfs_and_can_leave(tmp_path: Path) -> None:
    """Two preparations of one checkpoint share TensorFS's flight. The one that waits shows
    TensorFS's waiting samples and leaves when canceled, while the first still holds it."""
    point = checkpoint(tmp_path / "origin")
    # TensorFS asks a silent object again: every later ask is held until the pulls have left.
    released = threading.Event()
    stalled = on(HEADER, 1, 1, Stall(after=10)), on(HEADER, 2, ALWAYS, Gate(released))
    with FaultHub(point, *stalled) as hub:
        root = tmp_path / "store"
        tensorfs.Store.init(str(root))
        worker = peer({})
        worker.workspace = Workspace(root)
        worker.config = cast(Any, SimpleNamespace(object_storage_hosts=("127.0.0.1",)))
        row = {
            "parameter": "model",
            "public_origin": hub.origin,
            "repository": point.model,
            "manifest": {"digest": point.manifest, "length": point.length},
        }
        first, second = tensorfs.PullCancellation(), tensorfs.PullCancellation()
        leading = Watched(hub, lambda: defaults.materialize(worker, [row], cancellation=first))
        hub.until(lambda: hub.sent(HEADER) == 10)
        waited: list[tuple[int, int]] = []
        waiting = Watched(
            hub,
            lambda: defaults.materialize(
                worker, [row], cancellation=second, progress=lambda *at: waited.append(at)
            ),
        )
        hub.until(lambda: any(whole for _, whole in waited), lambda: len(waited))
        second.cancel()
        with pytest.raises(WorkspaceRefusal):
            waiting.result()
        assert leading.thread.is_alive(), hub.log()
        first.cancel()
        with pytest.raises(WorkspaceRefusal):
            leading.result()
        released.set()
        asked = hub.asks(HEADER)
        defaults.materialize(worker, [row])
        tensorfs.Store.open(str(root)).verify_checkpoint_source(
            point.model, point.manifest, point.length
        )
        assert temps(root) == [] and hub.asks(HEADER) == asked + 1, hub.log()


def test_a_preparation_reports_its_whole_size_only_once_every_pull_has_one(
    tmp_path: Path,
) -> None:
    """The adapter's Hub answers only after the base has its size: a sum of the base alone
    would tell a watcher the whole download is the base."""
    base = checkpoint(tmp_path / "base", "proof/base")
    adapter = checkpoint(tmp_path / "adapter", "proof/adapter", path="adapter.cozytensors")
    sized = threading.Event()

    @contextmanager
    def downloading(row: defaults.Selected) -> Iterator[defaults.Sample]:
        yield lambda _, total: sized.set() if row.repository == base.model and total else None

    root = tmp_path / "store"
    tensorfs.Store.init(str(root))
    worker = peer({})
    worker.workspace = Workspace(root)
    worker.config = cast(Any, SimpleNamespace(object_storage_hosts=("127.0.0.1",)))
    reported: list[tuple[int, int]] = []
    with FaultHub(base) as first, FaultHub(adapter, on("closure", 1, 1, Gate(sized))) as second:
        rows = [
            {
                "parameter": point.model.removeprefix("proof/"),
                "public_origin": hub.origin,
                "repository": point.model,
                "manifest": {"digest": point.manifest, "length": point.length},
            }
            for point, hub in ((base, first), (adapter, second))
        ]
        Watched(
            second,
            lambda: defaults.materialize(
                worker, rows, downloading=downloading, progress=lambda *at: reported.append(at)
            ),
            first.reading,
        ).result()
    whole = reported[-1][1]
    assert reported[-1][0] == whole and {size for _, size in reported} == {whole}, reported
    assert [done for done, _ in reported] == sorted(done for done, _ in reported), reported
