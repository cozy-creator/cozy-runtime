"""A request done with its GPUs hands them to the next one while it finishes its CPU work.

Policy tests drive `GpuScheduler` alone. Worker tests use `test_gpu_scheduler.Machine`: a
real Worker with four device entries, its durable journal and drive ticks; the release
reaches it through the worker's own executor-exchange handler. The end-to-end test runs the
workflow's Python in a real executor, so `ctx.release_gpus()` crosses the real seam. The only
fake is the hardware memory seam, `accel.device_memory`.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import grpc
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal import accel, child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.gpu_scheduler import Demand, GpuScheduler
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import OWNER, GiB, Machine

RELEASED = {"ordinals": [], "cause": "released_by_root"}


# ------------------------------------------------------------------------------ policy


def test_a_released_root_hands_its_gpus_on_and_a_later_call_waits_its_turn() -> None:
    gpus = GpuScheduler(4)
    roots = {"A": 1, "B": 2}
    assert gpus.sync(roots, [Demand("a1#1", "A", 4)]) == {"a1#1": (0, 1, 2, 3)}
    gpus.release("a1#1")
    assert gpus.sync(roots, [Demand("b1#1", "B", 4)]) == {}  # A's gap holds its lease
    gpus.release_root("A")
    gpus.release_root("A")  # idempotent
    assert gpus.view()["leases"] == {}
    assert gpus.sync(roots, [Demand("b1#1", "B", 4)]) == {"b1#1": (0, 1, 2, 3)}
    # A's GPU call after its release waits at A's priority; B keeps its grant and its gap
    live = [Demand("b1#1", "B", 4), Demand("a2#1", "A", 4)]
    assert gpus.sync(roots, live) == {"b1#1": (0, 1, 2, 3)}
    gpus.release("b1#1")
    assert gpus.sync(roots, live[1:]) == {}
    assert gpus.view()["waiting"] == {"a2#1": ["B"]}
    assert gpus.sync({"A": 1}, live[1:]) == {"a2#1": (0, 1, 2, 3)}  # B ended
    gpus.release("a2#1")  # the new lease holds A's next gap again
    assert gpus.sync({"A": 1, "C": 3}, [Demand("c#1", "C", 1)]) == {}
    leases = [(root, body) for root, kind, body in gpus.drain() if kind == "gpu.lease"]
    assert leases.count(("A", RELEASED)) == 1


def test_a_call_in_flight_keeps_its_gpu_until_it_exits() -> None:
    gpus = GpuScheduler(4)
    roots = {"A": 1, "B": 2}
    refs = [Demand("r0#1", "A", 1), Demand("r1#1", "A", 1)]
    assert gpus.sync(roots, refs) == {"r0#1": (0,), "r1#1": (1,)}
    gpus.release("r0#1")
    gpus.release_root("A")  # r1 is still on GPU 1
    assert gpus.view()["leases"] == {"A": [1]}
    wide = Demand("b#1", "B", 4)
    assert gpus.sync(roots, [refs[1], wide]) == {"r1#1": (1,)}
    gpus.release("r1#1")  # GPU 1 goes to the pool, not back to A
    assert gpus.view()["leases"] == {"B": [0, 2, 3]}
    assert gpus.sync(roots, [wide]) == {"b#1": (0, 1, 2, 3)}


def test_a_root_that_is_its_own_gpu_call_releases_at_device_exit() -> None:
    gpus = GpuScheduler(4)
    roots = {"S": 1, "B": 2}
    demands = [Demand("S#1", "S", 4), Demand("b#1", "B", 1)]
    assert gpus.sync(roots, demands) == {"S#1": (0, 1, 2, 3)}
    gpus.release("S#1")  # S now encodes and sends its outputs; it is still an open root
    assert gpus.view()["leases"] == {}
    assert gpus.sync(roots, demands) == {"b#1": (0,)}
    events = [(root, kind, body) for root, kind, body in gpus.drain()]
    assert ("S", "gpu.lease", {"ordinals": [], "cause": "device_exit"}) in events


def test_a_release_ends_with_its_root_and_the_next_attempt_holds_normally() -> None:
    gpus = GpuScheduler(2)
    gpus.sync({"A": 1}, [Demand("a1#1", "A", 2)])
    gpus.release("a1#1")
    gpus.release_root("A")
    gpus.sync({}, [])  # A stopped: retry-wait, pause, cancel, or a worker restart
    assert gpus.yielded == {}
    gpus.sync({"A": 1}, [Demand("a1#2", "A", 2)])
    gpus.release("a1#2")
    assert gpus.sync({"A": 1, "B": 2}, [Demand("b#1", "B", 1)]) == {}


# ----------------------------------------------------------------------- the worker


def _measured(entry: str, kind: str) -> accel.DeviceMemory:
    return accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    monkeypatch.setattr(accel, "device_memory", _measured)
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-gpu.", dir="/tmp") as root:
        made = Machine(Path(root), "0,1,2,3", "boot-one")
        try:
            yield made
        finally:
            made.worker.shutdown()


def _release(machine: Machine, root: str, ordinal: int = 1) -> dict[str, Any]:
    """`ctx.release_gpus()` as the root's executor sends it, through the worker's handler."""
    attempt = AttemptRecord(root, ordinal, b"", {}, lane_id="orchestration", state="running")
    return machine.worker.engine.durable_exchange(attempt, {"kind": "gpu_release"})


def _running(machine: Machine, request: str) -> None:
    selected = machine.executions.offer(OWNER, request)
    accepted = pb.AttemptAccepted(
        request_id=request,
        attempt_ordinal=selected.attempt_ordinal,
        invocation_spec_digest=selected.invocation_spec_digest,
    )
    machine.executions.workspace.mark_running(OWNER, accepted)
    machine.executions.observe(OWNER, pb.WorkerFrame(attempt_accepted=accepted))


def _executor_lost(machine: Machine, request: str) -> None:
    """The attempt's executor died: a retryable outcome."""
    selected = machine.executions.offer(OWNER, request)
    body, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request,
            attempt_ordinal=selected.attempt_ordinal,
            invocation_spec_digest=documents.spell(selected.invocation_spec_digest),
            status=pb.OUTCOME_STATUS_ABANDONED,
            cause=pb.OutcomeCause(
                code=pb.CAUSE_CODE_EXECUTOR_INVALIDATED, origin=pb.CAUSE_ORIGIN_EXECUTOR
            ),
            safe_message="executor lost",
        )
    )
    outcome = pb.AttemptOutcome(
        request_id=request,
        attempt_ordinal=selected.attempt_ordinal,
        invocation_spec_digest=selected.invocation_spec_digest,
        outcome_id=f"lost-{request}-{selected.attempt_ordinal}",
        outcome_digest=digest,
        outcome_canonical_bytes=body,
    )
    machine.executions.workspace.outcome(OWNER, outcome)
    machine.executions.observe(OWNER, pb.WorkerFrame(attempt_outcome=outcome))


def test_a_released_root_lets_the_next_root_start_while_its_python_runs(
    machine: Machine,
) -> None:
    machine.root("A")
    machine.root("B")
    a1 = machine.child("A", "h3")
    b1 = machine.child("B", "h3")
    machine.tick()
    machine.finish(a1)
    view = machine.tick()
    assert view["leases"] == {"A": [0, 1, 2, 3]} and b1 + "#1" in view["waiting"]
    assert _release(machine, "A") == {"ok": True}
    assert _release(machine, "A") == {"ok": True}  # idempotent
    machine.tick()
    assert machine.granted(b1) == [0, 1, 2, 3]
    assert machine.executions.status(OWNER, "A").state == "running"  # its CPU tail runs on
    assert machine.worker.gpu_status("A").phase == "none"
    released = [
        (seq, body)
        for seq, kind, body in machine.journal("A")
        if body.get("cause") == RELEASED["cause"]
    ]
    assert [body for _, body in released] == [RELEASED]
    released_at = next(
        e.at_ms
        for e in machine.executions.events(OWNER, "A").events
        if e.sequence == released[0][0]
    )
    granted_at = next(
        e.at_ms for e in machine.executions.events(OWNER, "B").events if e.kind == "gpu.grant"
    )
    assert granted_at >= released_at
    # A calls a GPU function again: it waits behind B, and B keeps its GPUs through its gap
    a2 = machine.child("A", "h3")
    view = machine.tick()
    assert view["waiting"] == {a2 + "#1": ["B"]} and view["leases"] == {"B": [0, 1, 2, 3]}
    machine.finish(b1)
    view = machine.tick()
    assert a2 + "#1" in view["waiting"] and machine.granted(a2) == []
    machine.finish("B")
    machine.tick()
    assert machine.granted(a2) == [0, 1, 2, 3]


def test_a_serving_root_releases_its_gpus_at_device_exit(machine: Machine) -> None:
    machine.serving("A", "h3")
    machine.root("B")
    b1 = machine.child("B", "qwen")
    view = machine.tick()
    assert machine.granted("A") == [0, 1, 2, 3] and view["waiting"] == {b1 + "#1": ["A"]}
    # the lane's device exit (`Worker.run_lane`); A's encode and output transfer follow it
    machine.worker.gpu.release("A#1", ranks=[])
    assert machine.worker.gpu.view()["leases"] == {}
    assert machine.worker.gpu_status("A").phase == "none"
    machine.tick()
    assert machine.granted(b1) == [0]
    leases = [body for _, kind, body in machine.journal("A") if kind == "gpu.lease"]
    assert leases == [
        {"ordinals": [0, 1, 2, 3], "cause": "granted A#1"},
        {"ordinals": [], "cause": "device_exit"},
    ]


def test_a_retried_root_starts_with_no_lease_and_holds_its_gaps_again(machine: Machine) -> None:
    machine.submit(
        "A",
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="inst-root", job_descriptor_id="sha256:" + "12" * 32
            )
        ),
        None,
    )
    with machine.executions.workspace.locked() as db:
        db.execute("UPDATE executions SET max_attempts=2 WHERE owner=? AND request='A'", (OWNER,))
    _running(machine, "A")
    a1 = machine.child("A", "h3")
    machine.tick()
    machine.finish(a1)
    machine.tick()
    _release(machine, "A")
    assert machine.tick()["leases"] == {}
    _executor_lost(machine, "A")
    machine.executions.retry_due(OWNER, worker_boot=machine.worker.fence.worker_boot_id)
    _running(machine, "A")  # attempt 2 runs the workflow from its start
    assert machine.executions.status(OWNER, "A").attempt_ordinal == 2
    a2 = machine.child("A", "h3")
    machine.root("B")
    b1 = machine.child("B", "qwen")
    machine.tick()
    assert machine.granted(a2) == [0, 1, 2, 3]
    machine.finish(a2)
    view = machine.tick()
    assert view["leases"] == {"A": [0, 1, 2, 3]} and b1 + "#1" in view["waiting"]


def test_a_restarted_worker_forgets_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(accel, "device_memory", _measured)
    first = Machine(tmp_path, "0,1,2,3", "boot-release")
    try:
        first.root("A")
        a1 = first.child("A", "h3")
        first.tick()
        first.finish(a1)
        _release(first, "A")
        assert first.tick()["leases"] == {}
    finally:
        first.worker.shutdown()
    second = Machine(tmp_path, "0,1,2,3", "boot-release")
    try:
        second.children = {"A": 0}
        a2 = second.child("A", "h3")
        second.tick()
        second.finish(a2)
        second.root("B")
        b1 = second.child("B", "qwen")
        view = second.tick()
        assert view["leases"] == {"A": [0, 1, 2, 3]} and b1 + "#1" in view["waiting"]
    finally:
        second.worker.shutdown()


# ------------------------------------------------------------------ a real executor


WORKFLOW = """import asyncio
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@invocable
async def leaf(ctx: Context, *, n: int) -> Result:
    return Result(n * 2)
app.job(leaf)

async def gate(ctx: Context, name: str) -> None:
    Path(GATES, name + '.ready').touch()
    while not Path(GATES, name).exists():
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.02)

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    await gate(ctx, 'shots')
    ctx.release_gpus()
    ctx.release_gpus()
    await gate(ctx, 'assembly')
    return await leaf(n=6)
"""


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
def test_a_workflow_releases_from_its_executor_and_b_starts_during_its_cpu_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workflow's shots are done (its lease is held); B waits. The workflow's own
    `ctx.release_gpus()` ends the lease, B is granted while the workflow still runs, and the
    workflow then finishes its CPU work and a CPU call."""
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    monkeypatch.setattr(accel, "device_memory", _measured)
    with tempfile.TemporaryDirectory(prefix="cz-rel.", dir="/tmp") as directory:
        root = Path(directory)
        gates = root / "gates"
        gates.mkdir()
        monkeypatch.setattr(fixture, "SOURCE", WORKFLOW.replace("GATES", repr(str(gates))))
        environment, preparation = fixture.package(root, "nested")
        (root / "artifacts").mkdir()
        base = {
            k: v
            for k, v in os.environ.items()
            if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
        }
        worker = Worker(
            replace(
                RuntimeConfig(
                    cozy_home=root / "home",
                    credentials=Credentials(),
                    child_base_env=tuple(sorted(base.items())),
                ),
                bootstrap_credential="offline",
            ),
            WorkerOptions(
                root=root / "worker",
                devices="0,1,2,3",
                python=str(image_python()),
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "address"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        deadline = time.monotonic() + 300  # a hang bound on a loaded shared box, not a budget

        def until(ready: Any) -> Any:
            while not (value := ready()):
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            return value

        try:
            prepared = worker.prepare_local_package(preparation)
            installation = prepared.installed_package.installation_id
            captured, capture_digest = documents.identity(
                pb.MachineExecutionCapture(
                    root_installation_id=installation,
                    installed_packages=[prepared.installed_package],
                    bindings=[
                        pb.MachineCallableBinding(
                            caller_installation_id=installation,
                            callee_installation_id=installation,
                            module="prepare_nested",
                            export="leaf",
                            entrypoint="leaf",
                        )
                    ],
                )
            )
            if worker.supervision.current is not None:
                worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
            plans = [
                json.loads(path.read_bytes()) for path in (root / "home/job-plans").glob("*/*.json")
            ]
            binding = JobBinding.read(next(row for row in plans if row.get("job") == "nested"))
            thread.start()
            until(lambda: (root / "address").exists())
            claim = pb.Claim(
                record_owner_id=OWNER, record_owner_epoch=1, wire_minor=WIRE_MINOR, proof=b"offline"
            )
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            owner = LocalRecordOwner(
                LocalRequest(
                    entrypoint="nested",
                    payload={},
                    kind="job",
                    request_id="root",
                    job_descriptor_id=binding.job_descriptor_id,
                    timeout_ms=240_000,
                ),
                {},
                root / "grants",
                package_installation_id=binding.installation_id,
            )
            owner.host.inbound.get_nowait()
            owner._dispatch()
            offered = owner.host.inbound.get_nowait()
            assert offered is not None
            state = owner._desired_state()
            state.job.orchestration = True
            client = rpc.WorkerControlStub(
                grpc.insecure_channel((root / "address").read_text().strip())
            )
            client.SubmitMachineExecution(
                pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id="root",
                    capture_digest=capture_digest,
                    capture_canonical_bytes=captured,
                    offer=offered.attempt_offer,
                    prepared_state=state,
                    payload_canonical_bytes=canonical_json.encode({}),
                    expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                        pb.MachineExecutionWorkspaceQuery(claim=claim)
                    ).execution_workspace_id,
                )
            )
            until(lambda: (gates / "shots.ready").exists())
            # The workflow's last shot: a 4-GPU call of this root. Its template names no
            # installed package, so it is granted and then refused; the root keeps its lease.
            executions = worker.executions
            assert executions is not None and worker.machine_calls is not None
            calls = Machine.__new__(Machine)
            calls.worker, calls.executions = worker, executions
            calls.calls, calls.children = worker.machine_calls, {"root": 99}
            shot = calls.child("root", "h3")
            until(lambda: worker.execution_status(claim, shot).state == "failed")
            assert worker.gpu.view()["leases"] == {"root": [0, 1, 2, 3]}
            calls.serving("B", "h3")
            until(lambda: any(kind == "gpu.wait" for _, kind, _ in calls.journal("B")))
            (gates / "shots").touch()
            until(lambda: any(kind == "gpu.grant" for _, kind, _ in calls.journal("B")))
            assert worker.execution_status(claim, "root").state == "running"  # its CPU tail
            released = [
                seq
                for seq, kind, body in calls.journal("root")
                if body == RELEASED and kind == "gpu.lease"
            ]
            assert len(released) == 1, calls.journal("root")
            at = {e.sequence: e.at_ms for e in executions.events(OWNER, "root").events}
            granted_at = next(
                e.at_ms for e in executions.events(OWNER, "B").events if e.kind == "gpu.grant"
            )
            assert granted_at >= at[released[0]]
            (gates / "assembly").touch()
            until(lambda: worker.execution_status(claim, "root").state in ("succeeded", "failed"))
            body = documents.read(
                worker.collect_execution(claim, "root").outcome_canonical_bytes,
                pb.AttemptOutcomeBody,
            )
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            value = canonical_json.decode(base64.b64decode(body["result"]["inline_result"]))
            assert value == {"value": 12}
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            assert not thread.is_alive()
