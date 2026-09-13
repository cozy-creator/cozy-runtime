"""Captured Python composition executes on the worker without an observer."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import cast

import grpc
import pytest

from cozy_runtime import canonical_json
from cozy_runtime.internal import child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.local import LocalRecordOwner, LocalRequest
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from test_end_to_end import NO_EXECUTOR

SOURCE = """import msgspec
from cozy_runtime.author import App, Context, invocable
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int

@invocable(memoize=True)
async def leaf(ctx: Context, *, n: int) -> Result:
    return Result(n * 2)
app.job(leaf)

@invocable
async def compose(ctx: Context, *, n: int) -> Result:
    first = await leaf(n=n)
    second = await leaf(n=n)
    return Result(first.value + second.value)
app.job(compose)

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await compose(n=3)
"""


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("case", ["normal", "pause", "loss", "cycle", "capacity"])
def test_nested_capture_and_leaf_memoization_without_control_stream(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    with tempfile.TemporaryDirectory(prefix="cz-child.") as directory:
        root = Path(directory)
        source = SOURCE
        if case in ("pause", "loss"):
            source = source.replace(
                "    return Result(n * 2)",
                "    from pathlib import Path\n    import asyncio\n"
                + f"    Path({str(root / 'entered')!r}).touch()\n"
                + f"    while not Path({str(root / 'gate')!r}).exists():\n"
                + "        ctx.raise_if_cancelled()\n        await asyncio.sleep(0.02)\n"
                + "    return Result(n * 2)",
            )
        if case == "cycle":
            source = source.replace("first = await leaf(n=n)", "first = await compose(n=n)")
        if case == "capacity":
            from cozy_runtime.internal.worker import machine_slots

            monkeypatch.setattr(machine_slots, "MAX_ACTIVE_CALLS", 2)
        monkeypatch.setattr(fixture, "SOURCE", source)
        environment, preparation = fixture.package(root, "nested")
        (root / "artifacts").mkdir()
        worker = Worker(
            RuntimeConfig(
                cozy_home=root / "home",
                credentials=Credentials(),
                bootstrap_credential="offline",
                child_base_env=tuple(
                    sorted(
                        (key, value)
                        for key, value in os.environ.items()
                        if not child_env.erased(key)
                        and key not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
                    )
                ),
            ),
            WorkerOptions(
                root=root / "worker",
                devices="",
                accelerator_backend="none",
                python=str(image_python()),
                install_root=environment,
                artifact_cache=root / "artifacts",
                tensorfs_root=root / "store",
                grant_roots=(str(root),),
            ),
            GrpcControlHost("127.0.0.1:0", root / "address"),
        )
        thread = threading.Thread(target=worker.run, daemon=True)
        try:
            fixture.measure(worker, preparation)
            prepared = worker.prepare_local_package(preparation)
            placement = documents.read(
                prepared.placement_set.placement_set_canonical_bytes, pb.PlacementSet
            )["placements"][0]
            revision = pb.LocalPackageRevision(
                package=preparation.package.package,
                release=preparation.package.release,
                source_digest=preparation.package.source_digest,
                package_interface=pb.Ref(
                    digest=documents.raw(placement["package_interface"]["digest"]),
                    length=placement["package_interface"]["length"],
                ),
                files=[
                    pb.LocalPackageFileRef(
                        filename=wheel.filename,
                        digest=wheel.digest,
                        length=wheel.length,
                    )
                    for wheel in preparation.wheels
                ],
            )
            captured, capture_digest = documents.identity(
                pb.MachineExecutionCapture(
                    root_revision_digest=preparation.package.local_revision_digest,
                    revisions=[revision],
                    bindings=[
                        pb.MachineCallableBinding(
                            caller_revision_digest=preparation.package.local_revision_digest,
                            callee_revision_digest=preparation.package.local_revision_digest,
                            interface_digest=revision.package_interface.digest,
                            module="prepare_nested",
                            export=export,
                            entrypoint=export,
                        )
                        for export in ("compose", "leaf")
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
            deadline = time.monotonic() + 60
            while not (root / "address").exists():
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            claim = pb.Claim(
                record_owner_id="owner",
                record_owner_epoch=1,
                wire_minor=WIRE_MINOR,
                proof=b"offline",
            )
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            generator = LocalRecordOwner(
                LocalRequest(
                    entrypoint="nested",
                    payload={},
                    outputs=(),
                    kind="job",
                    request_id="root",
                    job_descriptor_id=binding.job_descriptor_id,
                    timeout_ms=45_000,
                ),
                {},
                root / "grants",
                package_code_digest=binding.build_id,
            )
            generator.host.inbound.get_nowait()
            generator._dispatch()
            offered = generator.host.inbound.get_nowait()
            assert offered is not None
            state = generator._desired_state()
            state.job.orchestration = True
            channel = grpc.insecure_channel((root / "address").read_text().strip())
            client = rpc.WorkerControlStub(channel)
            client.SubmitMachineExecution(
                pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id="root",
                    capture_digest=capture_digest,
                    capture_canonical_bytes=captured,
                    offer=offered.attempt_offer,
                    prepared_state=state,
                    payload_canonical_bytes=canonical_json.encode({}),
                    max_attempts=2 if case == "loss" else 1,
                )
            )
            if case == "pause":
                while not (root / "entered").exists():
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.02)
                assert len(worker.lanes.composition) == 3
                worker.control_execution(claim, "root", "pause-root", 1, "pause")
                while True:
                    assert thread.is_alive() and time.monotonic() < deadline
                    assert worker.workspace is not None
                    with worker.workspace.locked() as db:
                        states = [row[0] for row in db.execute("SELECT state FROM executions")]
                    if states == ["paused"] * 3 and not worker.lanes.composition:
                        break
                    time.sleep(0.02)
                (root / "gate").touch()
                paused = worker.execution_status(claim, "root")
                worker.control_execution(claim, "root", "resume-root", paused.generation, "resume")
            if case == "loss":
                import signal

                while not (root / "entered").exists():
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.02)
                leaf = next(
                    attempt
                    for attempt in worker.engine.live.values()
                    if attempt.job is not None and attempt.job.job == "leaf"
                )
                parent_pids = {
                    attempt.request_id: attempt.executor_pid
                    for attempt in worker.engine.live.values()
                    if attempt is not leaf
                }
                os.kill(leaf.executor_pid, signal.SIGKILL)
                while worker.execution_status(claim, leaf.request_id).attempt_ordinal != 2:
                    observed = worker.execution_status(claim, leaf.request_id)
                    assert observed.state != "failed", documents.read(
                        worker.collect_execution(claim, leaf.request_id).outcome_canonical_bytes,
                        pb.AttemptOutcomeBody,
                    )
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.02)
                for identifier, pid in parent_pids.items():
                    parent = worker.engine.live[identifier]
                    assert parent.attempt == 1 and parent.executor_pid == pid
                (root / "gate").touch()
            while worker.execution_status(claim, "root").state not in ("succeeded", "failed"):
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            outcome = worker.collect_execution(claim, "root")
            body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
            if case in ("cycle", "capacity"):
                assert body["status"] == pb.OUTCOME_STATUS_FAILED, body
                assert worker.workspace is not None
                with worker.workspace.locked() as db:
                    count = db.execute("SELECT count(*) FROM executions").fetchone()[0]
                    assert count == (2 if case == "cycle" else 3)
                    if case == "cycle":
                        assert db.execute(
                            "SELECT 1 FROM execution_calls WHERE safe_detail "
                            "LIKE '%active ancestor%'"
                        ).fetchone()
                while worker.lanes.composition:
                    assert thread.is_alive() and time.monotonic() < deadline
                    time.sleep(0.02)
                assert not worker.stop.is_set()
                return
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            assert not worker.stop.is_set()
            import base64

            assert canonical_json.decode(base64.b64decode(body["result"]["inline_result"])) == {
                "value": 12
            }
            assert not worker.connected.is_set()
            assert worker.workspace is not None
            with worker.workspace.locked() as db:
                assert db.execute("SELECT count(*) FROM executions").fetchone()[0] == 3
                assert db.execute("SELECT count(*) FROM execution_calls").fetchone()[0] == 3
                assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 1
        finally:
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            assert not thread.is_alive()


def test_managed_depth_counts_the_prospective_child(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from cozy_runtime.internal.worker.attempts import AttemptRecord
    from cozy_runtime.internal.worker.machine_calls import Calls as Dispatcher
    from cozy_runtime.internal.worker.machine_child_target import Target
    from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
    from cozy_runtime.internal.worker.workspace_calls import Calls
    from cozy_runtime.internal.worker.workspace_executions import Executions
    from test_machine_calls import request, running
    from test_machine_execution import offer

    workspace = Workspace(tmp_path / "store")
    executions, calls = Executions(workspace), Calls(workspace)
    selected = "root"
    parents = []
    for _ in range(32):
        executions.submit("owner", selected, b"c" * 32, offer(selected))
        parent = running(executions, selected)
        parents.append(selected)
        selected = calls.accept("owner", request(parent)).child_request
    dispatcher = Dispatcher(cast(Worker, SimpleNamespace(workspace=workspace)))
    target = cast(Target, SimpleNamespace(revision="different", binding=None))
    dispatcher._check_ancestors(
        "owner", cast(AttemptRecord, SimpleNamespace(request_id=parents[-2])), target, b"{}"
    )
    with pytest.raises(WorkspaceRefusal, match="depth"):
        dispatcher._check_ancestors(
            "owner", cast(AttemptRecord, SimpleNamespace(request_id=parents[-1])), target, b"{}"
        )
