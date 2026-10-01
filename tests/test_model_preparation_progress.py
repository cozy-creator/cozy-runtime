"""Native model availability stays observable beside a blocked shared transfer."""

from __future__ import annotations

import hashlib
import json
import queue
import struct
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc
import pytest
import tensorfs

import signed_claims
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker import machine_materialization, machine_model_defaults
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_machine_execution import settle
from test_model_runtime_closure import _ASSET, _HEADER, _snapshot


@dataclass
class ProgressMachine:
    worker: Worker
    prepare: rpc.RuntimePreparationStub
    read: rpc.WorkerControlStub
    request: pb.PreparePackageSetRequest
    manifest: str
    manifest_length: int
    alternate_manifest: str
    first_adapter_manifest: str
    second_adapter_manifest: str
    total: int
    source_blocked: threading.Event
    release_source: threading.Event
    gets: list[str]
    channel: grpc.Channel
    address: str

    def rows(self) -> list[pb.PrepareModelProgress]:
        answer = self.read.DescribeMachine(pb.DescribeMachineQuery(claim=signed_claims.claim()))
        return list(answer.runtime.preparation_progress)


def running_progress_machine(tmp_path: Path) -> Generator[ProgressMachine, None, None]:
    source = tmp_path / "source"
    source.mkdir()
    _, manifest, length, origin = _snapshot(source, include_asset=True, checkpoint_only=True)
    bodies = {hashlib.sha256(body).hexdigest(): body for body in (_HEADER, _ASSET)}
    objects = [{"sha256": key, "length": len(body)} for key, body in bodies.items()]
    bodies[manifest.removeprefix("sha256:")] = origin.manifest(manifest)["manifest"]
    primary_total = sum(map(len, bodies.values()))
    revisions = {manifest: objects}

    def variant(value: float) -> str:
        header = _HEADER[:-4] + struct.pack("<f", value)
        header_id = hashlib.sha256(header).hexdigest()
        raw = json.dumps(
            {
                "entries": [
                    {
                        "blob": {"sha256": header_id, "length": len(header)},
                        "kind": "cozytensors",
                        "path": "model.cozytensors",
                    }
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        identity = "sha256:" + hashlib.sha256(raw).hexdigest()
        bodies[header_id] = header
        bodies[identity.removeprefix("sha256:")] = raw
        revisions[identity] = [
            {"sha256": header_id, "length": len(header)},
            {"sha256": hashlib.sha256(_ASSET).hexdigest(), "length": len(_ASSET)},
        ]
        return identity

    first_adapter_manifest, second_adapter_manifest = variant(1.0), variant(2.0)
    alternate_manifest = variant(3.0)
    blocked, release = threading.Event(), threading.Event()
    gets: list[str] = []

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
            if self.path == "/v1/tensorfs/closure":
                repository, selected = request["ref"].split("@", 1)
                assert repository in {
                    "proof/adapter",
                    "proof/first-adapter",
                    "proof/second-adapter",
                }
                assert selected in revisions
                response = {
                    "complete": True,
                    "lane": "bf16",
                    "model": repository,
                    "manifest": {
                        "sha256": selected.removeprefix("sha256:"),
                        "length": len(bodies[selected.removeprefix("sha256:")]),
                    },
                    "objects": revisions[selected],
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
            digest = self.path.removeprefix("/")
            gets.append(digest)
            if bodies[digest] == _ASSET:
                blocked.set()
                assert release.wait(30), "test did not release its blocked final object"
            self.answer(bodies[digest])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    source_thread = threading.Thread(target=server.serve_forever, daemon=True)
    source_thread.start()
    destination = tmp_path / "destination"
    tensorfs.Store.init(destination)
    hub = f"http://localhost:{server.server_port}"
    host = GrpcControlHost("127.0.0.1:0", tmp_path / "address")
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        WorkerOptions(
            **signed_claims.IDENTITY,
            root=tmp_path / "worker",
            tensorfs_root=destination,
            hubs=(
                PublicationAuthority(
                    hub,
                    object_storage_hosts=("127.0.0.1",),
                    access_token="execution-access",
                    expires_at=int(time.time()) + 60,
                ),
            ),
            accelerator_backend="none",
            devices="",
        ),
        host,
    )
    host.preparer = worker.prepare_package_set
    runtime_thread = threading.Thread(target=worker.run, daemon=True)
    runtime_thread.start()
    settle(lambda: (tmp_path / "address").exists() or not runtime_thread.is_alive())
    address = (tmp_path / "address").read_text().strip()
    channel = grpc.insecure_channel(address)
    inbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
    inbound.put(pb.RecordOwnerFrame(claim=signed_claims.claim()))
    control = rpc.WorkerControlStub(channel).Control(iter(inbound.get, None))
    assert next(control).claim_ack.accepted
    request = pb.PreparePackageSetRequest(
        download_delegation=documents.canonical_bytes(
            pb.DownloadDelegation(
                models=[pb.DownloadModelRef(model="proof/adapter", manifest=manifest, lane="bf16")]
            )
        ),
        hub=hub,
    )
    try:
        yield ProgressMachine(
            worker,
            rpc.RuntimePreparationStub(channel),
            rpc.WorkerControlStub(channel),
            request,
            manifest,
            length,
            alternate_manifest,
            first_adapter_manifest,
            second_adapter_manifest,
            primary_total,
            blocked,
            release,
            gets,
            channel,
            address,
        )
    finally:
        release.set()
        inbound.put(None)
        control.cancel()
        channel.close()
        worker.request_stop()
        host.stop()
        runtime_thread.join(30)
        assert not runtime_thread.is_alive()
        server.shutdown()
        server.server_close()
        source_thread.join(30)


@pytest.fixture
def progress_machine(tmp_path: Path) -> Iterator[ProgressMachine]:
    yield from running_progress_machine(tmp_path)


def test_shared_preparation_progress_survives_observer_detach(
    progress_machine: ProgressMachine,
) -> None:
    machine = progress_machine
    first = machine.prepare.PreparePackageSet.future(machine.request)
    assert machine.source_blocked.wait(10)
    settle(lambda: bool(machine.rows()) and machine.rows()[0].transferred_bytes > 0)
    before = machine.rows()[0]
    assert 0 < before.transferred_bytes < before.total_bytes == machine.total
    assert machine.worker.preparation_lock.locked()
    assert first.cancel()  # detach observation of the unary call; native work keeps ownership
    with grpc.insecure_channel(machine.address) as reconnected:
        second = rpc.RuntimePreparationStub(reconnected).PreparePackageSet.future(machine.request)
        rows = (
            rpc.WorkerControlStub(reconnected)
            .DescribeMachine(pb.DescribeMachineQuery(claim=signed_claims.claim()))
            .runtime.preparation_progress
        )
        assert rows[0].model.manifest == machine.manifest
        assert rows[0].transferred_bytes >= before.transferred_bytes
        assert not machine.worker.stop.is_set()
        machine.release_source.set()
        assert second.result(timeout=30).HasField("placement_set")
    completed = machine.rows()[0]
    assert completed.transferred_bytes == completed.total_bytes == machine.total
    assert len(machine.gets) == len(set(machine.gets)) == 3
    assert (
        machine.read.ListMachineExecutions(
            pb.MachineExecutionListQuery(claim=signed_claims.claim())
        ).head_number
        == 0
    )


def test_model_defaults_use_the_same_observable_native_availability(
    progress_machine: ProgressMachine,
) -> None:
    machine = progress_machine
    failures: list[BaseException] = []

    def materialize() -> None:
        try:
            machine_model_defaults.materialize(
                machine.worker,
                [
                    {
                        "parameter": "model",
                        "repository": "proof/adapter",
                        "manifest": {"digest": machine.manifest, "length": machine.manifest_length},
                        "public_origin": machine.request.hub,
                    }
                ],
                progress=None,
            )
        except BaseException as exc:
            failures.append(exc)

    # The source manifest's measured length travels with a Model default, before it lands.
    raw = documents.parse(machine.request.download_delegation, pb.DownloadDelegation)
    assert raw.models[0].manifest == machine.manifest
    writer = threading.Thread(target=materialize)
    writer.start()
    try:
        assert machine.source_blocked.wait(10), failures
        settle(lambda: bool(machine.rows()) and machine.rows()[0].transferred_bytes > 0)
        observed = machine.rows()[0]
        assert (
            not observed.model.slot and not observed.model.package and not observed.model.adapters
        )
        assert 0 < observed.transferred_bytes < observed.total_bytes == machine.total
    finally:
        machine.release_source.set()
        writer.join(30)
    assert not writer.is_alive() and not failures
    assert machine.rows()[0].transferred_bytes == machine.total


def test_malformed_optional_counters_do_not_refuse_native_preparation(
    progress_machine: ProgressMachine,
) -> None:
    machine = progress_machine
    prepared = machine.prepare.PreparePackageSet.future(machine.request)
    assert machine.source_blocked.wait(10)
    settle(lambda: bool(machine.rows()) and machine.rows()[0].transferred_bytes > 0)
    before = machine.rows()[0]
    invalid: tuple[object, ...] = (None, -1, True, "many", 1.5, 1 << 64, [], {})
    for value in invalid:
        machine_materialization.observe(machine.worker, before.model, value, value, begin=True)
        machine_materialization.completed(machine.worker, before.model, {"bytes_total": value})

        def rejected_callback(_: int, __: int) -> None:
            pytest.fail("a malformed native sample reached the observer callback")

        machine_materialization.forward(
            rejected_callback, {"bytes_done": value, "bytes_total": value}
        )
    machine_materialization.forward(lambda *_: pytest.fail("missing counters were forwarded"), {})
    machine_materialization.completed(machine.worker, before.model, None)
    machine_materialization.completed(machine.worker, before.model, {})
    machine_materialization.observe(
        machine.worker,
        before.model,
        before.transferred_bytes,
        before.total_bytes,
        result={"bytes_fetched": -1, "bytes_cached": "bad", "cache_written_bytes": 1 << 64},
    )
    assert machine.rows()[0] == before
    machine.release_source.set()
    assert prepared.result(timeout=30).HasField("placement_set")
    assert machine.rows()[0].transferred_bytes == machine.total


def test_both_native_routes_share_active_progress_without_reset(
    progress_machine: ProgressMachine,
) -> None:
    machine = progress_machine
    prepared = machine.prepare.PreparePackageSet.future(machine.request)
    assert machine.source_blocked.wait(10)
    settle(lambda: bool(machine.rows()) and machine.rows()[0].transferred_bytes > 0)
    before = machine.rows()[0]
    failures: list[BaseException] = []

    def joined() -> None:
        try:
            machine_model_defaults.materialize(
                machine.worker,
                [
                    {
                        "parameter": "second",
                        "repository": "proof/adapter",
                        "manifest": {"digest": machine.manifest, "length": machine.manifest_length},
                        "public_origin": machine.request.hub,
                    }
                ],
                progress=None,
            )
        except BaseException as error:
            failures.append(error)

    waiter = threading.Thread(target=joined)
    waiter.start()
    key = ("proof/adapter", machine.manifest)
    try:
        settle(lambda: machine.worker.model_preparation_observers.get(key, 0) == 2)
        after = machine.rows()[0]
        assert (
            after.transferred_bytes >= before.transferred_bytes
            and after.total_bytes == before.total_bytes
        )
        machine_materialization.observe(machine.worker, before.model, before.total_bytes + 1, 0)
        assert machine.rows()[0].transferred_bytes <= machine.rows()[0].total_bytes
    finally:
        machine.release_source.set()
        waiter.join(30)
    assert not failures and not waiter.is_alive()
    assert prepared.result(timeout=30).HasField("placement_set")
    assert not machine.worker.model_preparation_observers
    assert machine.rows()[0].transferred_bytes == machine.total


def test_ended_incomplete_observation_cannot_impersonate_a_live_transfer(
    progress_machine: ProgressMachine,
) -> None:
    machine = progress_machine
    model = pb.DownloadModelRef(model="proof/adapter", manifest=machine.manifest)
    with pytest.raises(RuntimeError, match="explicit test refusal"):
        with machine_materialization.observing(machine.worker, model):
            machine_materialization.observe(machine.worker, model, 10, 100)
            assert machine.rows()[0].transferred_bytes == 10
            raise RuntimeError("explicit test refusal")
    assert not machine.rows() and not machine.worker.model_preparation_observers
