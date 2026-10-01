"""A queued call is never refused because ANOTHER root's placement left the worker.

Found by the early-GPU-release proof (#790): root A's CPU call waited on its lane while root
B's replica was refused and removed; the removal moved the worker-wide admission epoch, the
lane re-checked that epoch and refused A's call `admission_epoch_stale`, and with one attempt
allowed the workflow failed. A queued call is re-checked on its own facts: its placement, the
worker's dispatchability, draining, its cancel and its deadline.

Real: the Worker on four virtual GPUs, its durable journal and drive ticks, the workflow's
Python and every call in real executors. The lane thread is held at its settle wait (the
point where a call queues behind a rebuild) until B's placement is gone; the fake is the
hardware memory seam, `accel.device_memory`.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import grpc
import pytest

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import accel, child_env
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import GrpcControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from local_owner import LocalRecordOwner, LocalRequest
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import OWNER, VIRTUAL, GiB, Machine, driverless

CPU_WORKFLOW = """import msgspec
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

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await leaf(n=6)
"""

# A serving call: its replica placement is hosted for it and the call stages on that
# placement's lane, the path a GPU call takes. It declares no Model, so no grant and no
# accelerator runtime; a model-bearing prepare imports torch, which this image lacks.
SERVING_WORKFLOW = CPU_WORKFLOW.replace("app.job(leaf)", "app.entrypoint(leaf)")


def _measured(entry: str, kind: str) -> accel.DeviceMemory:
    return accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)


@pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")
@pytest.mark.parametrize("call", ["job", "serving"])
def test_a_queued_call_survives_another_roots_placement_removal(
    monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    import test_job_preparation_isolation as fixture
    from conftest import image_python

    driverless(monkeypatch)
    monkeypatch.setattr(accel, "device_memory", _measured)
    monkeypatch.setattr(fixture, "SOURCE", SERVING_WORKFLOW if call == "serving" else CPU_WORKFLOW)
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-churn.", dir="/tmp") as directory:
        root = Path(directory)
        environment, preparation = fixture.package(root, "nested")
        (root / "artifacts").mkdir()
        base = {
            k: v for k, v in os.environ.items() if not child_env.erased(k) and k != "PYTHONPATH"
        }
        worker = Worker(
            replace(
                RuntimeConfig(
                    cozy_home=root / "home",
                    credentials=Credentials(),
                    child_base_env=tuple(sorted(base.items())),
                ),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                devices=VIRTUAL,
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

        def until(ready: Callable[[], Any]) -> Any:
            while not (value := ready()):
                assert thread.is_alive() and time.monotonic() < deadline
                time.sleep(0.02)
            return value

        # A's call reaches its lane and waits there, queued, until B's placement is gone.
        queued: list[AttemptRecord] = []
        release = threading.Event()
        settle = worker.await_settled

        def held(attempt: AttemptRecord) -> None:
            if not queued and attempt.request_id.startswith("call-") and call in attempt.spec:
                queued.append(attempt)
                release.wait(deadline - time.monotonic())
            settle(attempt)

        monkeypatch.setattr(worker, "await_settled", held)
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
            claim = signed_claims.claim(OWNER)
            worker.serve_stream(iter([pb.RecordOwnerFrame(claim=claim)]), lambda _: None)
            owner = LocalRecordOwner(
                LocalRequest(
                    entrypoint="nested",
                    payload={},
                    outputs=(),
                    kind="job",
                    request_id="A",
                    job_descriptor_id=binding.job_descriptor_id,
                    timeout_ms=240_000,
                ),
                {},
                root / "grants",
                package_installation_id=binding.installation_id,
            )
            offered = owner.offer()
            state = owner.desired_state()
            state.job.orchestration = True
            client = rpc.WorkerControlStub(
                grpc.insecure_channel((root / "address").read_text().strip())
            )
            client.SubmitMachineExecution(
                pb.MachineExecutionSubmit(
                    claim=claim,
                    submission_id="A",
                    capture_digest=capture_digest,
                    capture_canonical_bytes=captured,
                    offer=offered,
                    prepared_state=state,
                    payload_canonical_bytes=canonical_json.encode({}),
                    expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                        pb.MachineExecutionWorkspaceQuery(claim=claim)
                    ).execution_workspace_id,
                )
            )
            until(lambda: queued)
            (child,) = queued
            assert child.state == "queued"
            executions = worker.executions
            assert executions is not None and worker.machine_calls is not None
            ours = set(worker.hosted)
            epoch = worker.admission_epoch

            # Root B: a four-GPU serving request whose replica is refused (its template names
            # no installed package) and removed while A's call waits on its lane.
            other = Machine.__new__(Machine)
            other.worker, other.executions = worker, executions
            other.calls, other.children = worker.machine_calls, {}
            other.serving("B", "h3")
            until(lambda: worker.execution_status(claim, "B").state == "failed")
            until(lambda: set(worker.hosted) <= ours and worker.admission_epoch > epoch)
            assert worker.execution_status(claim, "A").state == "running"

            release.set()
            until(lambda: worker.execution_status(claim, "A").state in ("succeeded", "failed"))
            done = worker.execution_status(claim, child.request_id)
            assert (done.state, done.attempt_ordinal) == ("succeeded", 1)
            body = documents.read(
                worker.collect_execution(claim, "A").outcome_canonical_bytes,
                pb.AttemptOutcomeBody,
            )
            assert body["status"] == pb.OUTCOME_STATUS_SUCCEEDED, body
            value = canonical_json.decode(base64.b64decode(body["result"]["inline_result"]))
            assert value == {"value": 12}
        finally:
            release.set()
            worker.request_stop()
            worker.host.stop()
            if thread.ident is not None:
                thread.join(30)
            assert not thread.is_alive()
