"""A machine answers for its runs, packages, weights and environment (wire 66), from a real
Worker over its gRPC listener, and keeps delivered products until its store's GC evicts them."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import platform
import queue
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import grpc
import pytest
import tensorfs

import signed_claims
from cozy_runtime.internal import fill
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker import machine_models, products
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import NativeHold, Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_machine_calls import request, running
from test_machine_execution import complete, offer, settle
from test_model_runtime_closure import _snapshot
from test_workspace_byte_outputs import producing

CLAIM = signed_claims.claim()


@dataclass
class Machine:
    worker: Worker
    stub: rpc.WorkerControlStub
    executions: Executions
    store: Path

    def submit(self, name: str) -> pb.MachineExecutionReceipt:
        receipt = self.executions.submit(
            "owner",
            name,
            b"c" * 32,
            offer(name),
            expected_execution_workspace_id=self.executions.workspace_id,
        )
        return pb.MachineExecutionReceipt(number=receipt.number, request_id=receipt.request_id)

    def runs(
        self,
        *,
        after_number: int = 0,
        newest_first: bool = False,
        before_number: int = 0,
        limit: int = 0,
        states: tuple[str, ...] = (),
    ) -> pb.MachineExecutionList:
        listed: pb.MachineExecutionList = self.stub.ListMachineExecutions(
            pb.MachineExecutionListQuery(
                claim=CLAIM,
                after_number=after_number,
                newest_first=newest_first,
                before_number=before_number,
                limit=limit,
                states=states,
            )
        )
        return listed


@pytest.fixture
def machine(tmp_path: Path) -> Iterator[Machine]:
    """A Worker on a TensorFS store holding one committed release, claimed by `owner`."""
    store_root, manifest, length, store = _snapshot(
        tmp_path, include_asset=True, checkpoint_only=True
    )
    operation = store.begin_operation("reads-fixture", "proof", "adapter")
    operation.hold_manifest(manifest, length)
    operation.commit_release(None, "1.0.0", "bf16", manifest, length)
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        WorkerOptions(
            **signed_claims.IDENTITY,
            root=tmp_path / "worker",
            devices="",
            accelerator_backend="none",
            tensorfs_root=store_root,
            install_root=tmp_path / "installs",
        ),
        GrpcControlHost("127.0.0.1:0", tmp_path / "address"),
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    inbound: queue.Queue[pb.RecordOwnerFrame | None] = queue.Queue()
    thread.start()
    settle(lambda: (tmp_path / "address").exists() or not thread.is_alive())
    channel = grpc.insecure_channel((tmp_path / "address").read_text().strip())
    inbound.put(pb.RecordOwnerFrame(claim=CLAIM))
    stream = rpc.WorkerControlStub(channel).Control(iter(inbound.get, None))
    try:
        assert next(stream).claim_ack.accepted
        assert worker.executions is not None
        yield Machine(worker, rpc.WorkerControlStub(channel), worker.executions, store_root)
    finally:
        inbound.put(None)
        stream.cancel()
        channel.close()
        worker.request_stop()
        worker.host.stop()
        thread.join(30)
        assert not thread.is_alive()


def test_runs_are_numbered_at_acceptance_and_listed_by_number(machine: Machine) -> None:
    assert machine.runs().head_number == 0
    assert [machine.submit(name).number for name in ("first", "second")] == [1, 2]
    # A call the second run makes is part of that run: it has no number of its own.
    child = Calls(machine.executions.workspace).accept(
        "owner", request(running(machine.executions, "second"))
    )
    machine.executions.submit(
        "owner",
        child.child_request,
        b"c" * 32,
        offer(child.child_request),
        expected_execution_workspace_id=machine.executions.workspace_id,
    )
    assert machine.submit("third").number == 3
    complete(machine.executions, "first", {"ok": True})

    listed = machine.runs()
    assert [(run.number, run.request_id) for run in listed.executions] == [
        (1, "first"),
        (2, "second"),
        (3, "third"),
    ]
    assert listed.head_number == 3
    assert listed.execution_workspace_id == machine.executions.workspace_id
    first, second, third = listed.executions
    assert (first.state, second.state, third.state) == ("succeeded", "running", "queued")
    assert first.finished_at_ms >= first.accepted_at_ms > 0
    assert second.finished_at_ms == third.finished_at_ms == 0
    assert first.target.job and first.target.installation_id == "local-" + "11" * 16

    newest = machine.runs(newest_first=True, limit=2)
    assert [run.number for run in newest.executions] == [3, 2]
    older = machine.runs(newest_first=True, before_number=newest.executions[-1].number)
    assert [run.number for run in older.executions] == [1]
    assert [run.number for run in machine.runs(after_number=1, limit=1).executions] == [2]
    assert [run.number for run in machine.runs(states=("succeeded",)).executions] == [1]

    def state(name: str) -> pb.MachineExecutionState:
        answer: pb.MachineExecutionState = machine.stub.GetMachineExecution(
            pb.MachineExecutionQuery(
                claim=CLAIM,
                request_id=name,
                expected_execution_workspace_id=machine.executions.workspace_id,
            )
        )
        return answer

    assert state("first").number == 1 and state("first").finished_at_ms > 0
    assert state(child.child_request).number == 0

    # A follower holds one read open until the next run is accepted.
    waiting = machine.stub.ListMachineExecutions.future(
        pb.MachineExecutionListQuery(claim=CLAIM, after_number=3, wait=True)
    )
    time.sleep(0.2)  # the read is parked, not answered empty
    assert not waiting.done()
    machine.submit("fourth")
    assert [(run.number, run.request_id) for run in waiting.result(60).executions] == [
        (4, "fourth")
    ]


def test_roots_an_older_runtime_accepted_are_numbered_when_the_journal_opens(
    tmp_path: Path,
) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    for name in ("a", "b", "c"):
        executions.submit(
            "owner",
            name,
            b"c" * 32,
            offer(name),
            expected_execution_workspace_id=executions.workspace_id,
        )
    del executions
    gc.collect()
    # Runtime 0.18.75 and older never number: its journal has no numbers table at all.
    with sqlite3.connect(tmp_path / "store" / ".cozy-workspace" / "journal.sqlite3") as db:
        db.execute("DROP TABLE execution_numbers")
    reopened = Executions(Workspace(tmp_path / "store"))
    runs, head = reopened.runs("owner")
    assert [(run.number, run.request) for run in runs] == [(1, "a"), (2, "b"), (3, "c")]
    assert head == 3
    receipt = reopened.submit(
        "owner", "d", b"c" * 32, offer("d"), expected_execution_workspace_id=reopened.workspace_id
    )
    assert receipt.number == 4


def test_the_machine_describes_its_runtime_packages_and_weights(machine: Machine) -> None:
    described = machine.stub.DescribeMachine(pb.DescribeMachineQuery(claim=CLAIM))
    runtime = described.runtime
    assert (described.worker_id, described.worker_boot_id) == (
        signed_claims.WORKER_ID,
        signed_claims.BOOT_ID,
    )
    assert not described.HasField("host") and not described.runtime_absent
    assert runtime.version == importlib.metadata.version("cozy-runtime")
    assert runtime.tensorfs_version == importlib.metadata.version("tensorfs")
    assert (runtime.wire_minor, runtime.minimum_wire_minor) == (
        WIRE_MINOR,
        MIN_COMPATIBLE_WIRE_MINOR,
    )
    assert runtime.python_version == platform.python_version()
    assert runtime.accelerator_backend == "none" and not runtime.devices
    assert runtime.execution_workspace_id == machine.executions.workspace_id
    assert runtime.store.path == str(machine.store) and runtime.store.total_bytes > 0
    assert runtime.resources.platform and runtime.started_at_unix_ms > 0

    models = machine.stub.ListModels(pb.ModelListQuery(claim=CLAIM))
    assert [(m.kind, m.repository, m.version, m.lane) for m in models.models] == [
        ("release", "proof/adapter", "1.0.0", "bf16")
    ]
    assert models.models[0].manifest.length > 0
    assert [row.repository for row in models.repositories] == ["proof/adapter"]
    assert models.store.bytes_total > 0 and models.store.filesystem.path == str(machine.store)

    assert not machine.stub.ListPackages(pb.PackageListQuery(claim=CLAIM)).packages
    # The built-in operations install through their ordinary path, as a release root's first
    # call to them does.
    assert machine.worker.machine_calls is not None
    installed = machine.worker.machine_calls.builtins.capture()
    packages = machine.stub.ListPackages(pb.PackageListQuery(claim=CLAIM)).packages
    assert [(p.installation_id, p.package, p.origin) for p in packages] == [
        (installed.installation_id, "runtime/operations", "runtime")
    ]
    assert packages[0].release == installed.runtime_version and packages[0].entrypoints
    assert packages[0].installed_at_ms > 0


def test_the_reads_refuse_a_caller_that_is_not_the_owner(machine: Machine) -> None:
    stranger = signed_claims.claim(owner="stranger")
    for call, query in (
        (machine.stub.ListMachineExecutions, pb.MachineExecutionListQuery(claim=stranger)),
        (machine.stub.ListPackages, pb.PackageListQuery(claim=stranger)),
        (machine.stub.ListModels, pb.ModelListQuery(claim=stranger)),
        (machine.stub.DescribeMachine, pb.DescribeMachineQuery(claim=stranger)),
    ):
        with pytest.raises(grpc.RpcError) as refused:
            call(query)
        assert refused.value.code() == grpc.StatusCode.FAILED_PRECONDITION


@pytest.mark.parametrize("delivery", [True, False])
def test_an_acknowledged_product_stays_until_gc_where_the_store_tracks_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delivery: bool
) -> None:
    """The run log's hold on a product, as `Products` takes it; then the owner's ack."""
    if not delivery:
        monkeypatch.setattr(fill, "capabilities", frozenset)
    elif "deliver/1" not in fill.capabilities():
        pytest.skip("the installed TensorFS does not track delivery")
    workspace, spec, manifest, path = producing(tmp_path)
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "video", manifest, [("payload", path)], 200000
    )
    data = path.read_bytes()
    content = pb.Ref(digest=hashlib.sha256(data).digest(), length=len(data))
    grant = machine_models.retain_bytes(
        workspace,
        "owner",
        products.recipient("producer"),
        products.hold_path(content, source),
        source,
    )
    store = tensorfs.Store.open(str(workspace.store_root))
    assert store.tree_root(grant.retention_id) is not None

    products.Products(workspace, Executions(workspace), None).deliver("owner", "producer")
    tensorfs.gc(str(workspace.store_root))  # no pressure: only unreferenced bytes go
    with workspace.locked() as db:
        held = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (grant.retention_id,))
    assert held is not None and held.state == ("held" if delivery else "released")
    root = store.tree_root(grant.retention_id)
    if delivery:
        assert root is not None and not root["released"]
        # A replayed acknowledgement delivers again, the root already delivered.
        products.Products(workspace, Executions(workspace), None).deliver("owner", "producer")
    else:
        assert root is None or root["released"]
