"""One execution's failure is its own: the worker and every other execution go on.

Real Worker, real executors, real journal. Faults are injected only where no author code
can reach (a result-custody step, the journal a crash handler writes to); everything they
must not break runs unmodified. Each execution is its own supervised unit: nothing here
waits on a drive tick, because there is none.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import pytest
import tensorfs

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import canonical, child_env, liveness
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker import activity, execution_unit, machine_byte_results
from cozy_runtime.internal.worker.control import GrpcControlHost, InMemoryControlHost
from cozy_runtime.internal.worker.plan import JobBinding
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as rpc
from local_owner import LocalRecordOwner, LocalRequest
from test_end_to_end import NO_EXECUTOR
from test_workspace_custody import produced

pytestmark = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

# long_form's shape: segments in order, and the last one, having no successor, exports an
# empty continuation context (minimax-h3 1.18.20, nemuri run 1474). `deaf` never reaches a
# cancellation point and emits nothing: only a forceful reclaim ends it.
SOURCE = """import asyncio
import time
from pathlib import Path
import msgspec
from cozy_runtime.author import App, Context, FileAsset, Outputs, invocable
app = App()
class Request(msgspec.Struct):
    segments: int
    gate: str = ''
class Segment(msgspec.Struct):
    context: FileAsset
class Result(msgspec.Struct):
    sizes: list[int]

@invocable
async def leaf(ctx: Context, out: Outputs, *, index: int, last: bool, gate: str) -> Segment:
    if gate:
        with open(gate + '.entered', 'a') as entered:
            entered.write('x')
    while gate and not Path(gate).exists():
        ctx.raise_if_cancelled()
        await asyncio.sleep(0.02)
    data = b'' if last else b'context-%d' % index
    return Segment(out.save_bytes(data, media_type='application/octet-stream'))
app.job(leaf)

@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    sizes = []
    for index in range(payload.segments):
        segment = await leaf(index=index, last=index == payload.segments - 1, gate=payload.gate)
        sizes.append(len(segment.context.read_bytes()))
    return Result(sizes)

@app.job
async def deaf(ctx: Context, payload: Request) -> Result:
    Path(payload.gate + '.entered').touch()
    while not Path(payload.gate).exists():
        time.sleep(0.05)
    return Result([])
"""

TERMINAL = ("succeeded", "failed", "canceled", "paused")


@dataclass
class Machine:
    worker: Worker
    thread: threading.Thread
    client: rpc.WorkerControlStub
    claim: pb.Claim
    root: Path
    plans: dict[str, JobBinding]
    capture: tuple[bytes, bytes]
    deadline: float

    def submit(
        self,
        request_id: str,
        job: str = "nested",
        timeout_ms: int = 240_000,
        attention_kernel: str = "",
        **payload: Any,
    ) -> None:
        submit(
            self.client,
            self.claim,
            self.root,
            self.plans[job],
            self.capture,
            request_id,
            job,
            timeout_ms,
            payload,
            attention_kernel,
        )

    def wait(self, ready: Callable[[], bool]) -> None:
        while not ready():
            assert self.thread.is_alive() and time.monotonic() < self.deadline
            time.sleep(0.02)

    def state(self, request_id: str) -> str:
        return self.worker.execution_status(self.claim, request_id).state

    def settle(self, request_id: str) -> tuple[str, int, pb.AttemptOutcomeBody]:
        self.wait(lambda: self.state(request_id) in TERMINAL)
        state = self.worker.execution_status(self.claim, request_id)
        outcome = self.worker.collect_execution(self.claim, request_id)
        body = documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        return state.state, state.attempt_ordinal, body

    def children(self, parent: str) -> list[str]:
        assert self.worker.workspace is not None
        with self.worker.workspace.locked() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT child_request FROM execution_calls WHERE parent_request=? "
                    "ORDER BY call_index",
                    (parent,),
                )
            ]

    def idle(self) -> None:
        """Every unit has left: nothing owed, nothing holds the rental."""
        self.wait(lambda: not self.worker.supervisor.busy())
        assert not activity.active_work(self.worker)


def submit(
    client: rpc.WorkerControlStub,
    claim: pb.Claim,
    root: Path,
    binding: JobBinding,
    capture: tuple[bytes, bytes],
    request_id: str,
    job: str,
    timeout_ms: int,
    payload: dict[str, Any],
    attention_kernel: str = "",
) -> None:
    owner = LocalRecordOwner(
        LocalRequest(
            entrypoint=job,
            payload=payload,
            outputs=(),
            kind="job",
            request_id=request_id,
            job_descriptor_id=binding.job_descriptor_id,
            timeout_ms=timeout_ms,
            attention_kernel=attention_kernel,
        ),
        {},
        root / f"grants-{request_id}",
        package_installation_id=binding.installation_id,
    )
    offered = owner.offer()
    state = owner.desired_state()
    state.job.orchestration = True
    client.SubmitMachineExecution(
        pb.MachineExecutionSubmit(
            claim=claim,
            submission_id=request_id,
            capture_digest=capture[1],
            capture_canonical_bytes=capture[0],
            offer=offered,
            prepared_state=state,
            payload_canonical_bytes=canonical_json.encode(payload),
            expected_execution_workspace_id=client.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=claim)
            ).execution_workspace_id,
        )
    )


CLAIM = signed_claims.claim()


def new_worker(root: Path, environment: Path) -> Worker:
    from conftest import image_python

    return Worker(
        RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
            child_base_env=tuple(
                sorted(
                    (key, value)
                    for key, value in os.environ.items()
                    if not child_env.erased(key) and key != "PYTHONPATH"
                )
            ),
        ),
        WorkerOptions(
            **signed_claims.IDENTITY,
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


def prepared(
    worker: Worker, root: Path, request: pb.PrepareLocalPackageRequest
) -> tuple[dict[str, JobBinding], tuple[bytes, bytes]]:
    """Prepare the package in this worker: its job plans and its execution capture."""
    installed = worker.prepare_local_package(request).installed_package
    capture = documents.identity(
        pb.MachineExecutionCapture(
            root_installation_id=installed.installation_id,
            installed_packages=[installed],
            bindings=[
                pb.MachineCallableBinding(
                    caller_installation_id=installed.installation_id,
                    callee_installation_id=installed.installation_id,
                    module="prepare_nested",
                    export="leaf",
                    entrypoint="leaf",
                )
            ],
        )
    )
    if worker.supervision.current is not None:
        worker.supervision.retire_current(worker.supervision.current, "begin lifecycle")
    plans = {}
    for path in (root / "home/job-plans").glob("*/*.json"):
        document = canonical_json.decode(path.read_bytes())
        plans[document["job"]] = JobBinding.read(document)
    return plans, capture


@contextmanager
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    # Short: executors dial back over a unix socket under this root.
    with tempfile.TemporaryDirectory(prefix="cz-contain.") as directory:
        yield from _machine(Path(directory), monkeypatch)


def package(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    import test_job_preparation_isolation as fixture

    monkeypatch.setattr(fixture, "SOURCE", SOURCE)
    environment, preparation = fixture.package(root, "nested")
    (root / "artifacts").mkdir()
    return environment, preparation


def _machine(root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    environment, preparation = package(root, monkeypatch)
    worker = new_worker(root, environment)
    thread = threading.Thread(target=worker.run, daemon=True)
    try:
        plans, capture = prepared(worker, root, preparation)
        thread.start()
        # A hang bound, not a budget: several executors spawning on a loaded shared box.
        deadline = time.monotonic() + 300
        while not (root / "address").exists():
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.02)
        worker.serve_stream(iter([pb.RecordOwnerFrame(claim=CLAIM)]), lambda _: None)
        channel = grpc.insecure_channel((root / "address").read_text().strip())
        yield Machine(
            worker, thread, rpc.WorkerControlStub(channel), CLAIM, root, plans, capture, deadline
        )
    finally:
        worker.request_stop()
        worker.host.stop()
        if thread.ident is not None:
            thread.join(30)
        assert not thread.is_alive()


def test_last_segment_empty_context_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    with machine(monkeypatch) as m:
        m.submit("root", segments=3)
        state, ordinal, body = m.settle("root")
        assert (state, ordinal, body.status) == ("succeeded", 1, pb.OUTCOME_STATUS_SUCCEEDED), body
        assert canonical_json.decode(body.result.inline_result) == {"sizes": [9, 9, 0]}
        assert m.worker.phase != pb.WorkerPhase.WORKER_PHASE_FAILED
        m.idle()


@pytest.mark.parametrize("call", ["job", "serving"])
def test_a_runs_attention_pin_reaches_every_call_and_says_when_none_applied(
    monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    """long_form's shape: the owner pins the job, whose segments hold the attention sites.
    Every call carries the pin; a call with no site runs rather than refusing, and a run
    none of whose calls applied the pin succeeds with a warning on its root."""
    if call == "serving":
        monkeypatch.setattr(
            sys.modules[__name__], "SOURCE", SOURCE.replace("app.job(leaf)", "app.entrypoint(leaf)")
        )
    with machine(monkeypatch) as m:
        m.submit("pinned", attention_kernel="sageattention", segments=2)
        state, _, body = m.settle("pinned")
        assert state == "succeeded", body
        children = m.children("pinned")
        assert len(children) == 2
        assert m.worker.workspace is not None and m.worker.executions is not None
        with m.worker.workspace.locked() as db:
            specs = [
                documents.parse(row[0], pb.InvocationSpec)
                for row in db.execute(
                    "SELECT invocation FROM attempts WHERE request IN (?,?)", children
                )
            ]
        assert [spec.attention_kernel for spec in specs] == ["sageattention"] * 2
        events = m.worker.executions.events(m.worker.fence.record_owner_id, "pinned").events
        warnings = [canonical_json.decode(e.body) for e in events if e.kind == "warning"]
        assert warnings == [
            {
                "code": "attention_pin_unapplied",
                "message": "attention pin sageattention was never applied",
            }
        ]
        assert not [e for e in events if e.kind == "attention.applied"]
        m.idle()


def test_finish_failure_fails_only_its_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = tmp_path / "gate"
    retain = machine_byte_results.retain

    def poisoned(*args: Any, **kwargs: Any) -> None:
        if args[2] == "poisoned":  # the recipient parent
            raise KeyError("content_bytes")
        retain(*args, **kwargs)

    monkeypatch.setattr(machine_byte_results, "retain", poisoned)
    with machine(monkeypatch) as m:
        m.submit("poisoned", segments=1, gate=str(gate))
        m.submit("healthy", segments=2, gate=str(gate))
        m.wait(lambda: len(m.worker.engine.live) >= 4)  # both parents and both first leaves
        gate.touch()
        state, ordinal, body = m.settle("poisoned")
        assert (state, ordinal, body.status) == ("failed", 1, pb.OUTCOME_STATUS_FAILED), body
        assert body.execution_started
        assert "child_result_unavailable" in body.safe_message, body.safe_message
        assert "KeyError: 'content_bytes' @ " in body.safe_message, body.safe_message
        state, _, body = m.settle("healthy")
        assert (state, body.status) == ("succeeded", pb.OUTCOME_STATUS_SUCCEEDED), body
        assert canonical_json.decode(body.result.inline_result) == {"sizes": [9, 0]}
        assert m.worker.phase != pb.WorkerPhase.WORKER_PHASE_FAILED
        m.submit("after", segments=1)
        assert m.settle("after")[0] == "succeeded"
        m.idle()


def test_started_failure_ends_failed_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with machine(monkeypatch) as m:
        gate = tmp_path / "never"
        m.submit("root", segments=2, gate=str(gate))
        # The leaf is executing author code: this loss is mid-execution, not pre-dispatch.
        m.wait(lambda: gate.with_suffix(".entered").exists())
        (leaf,) = m.children("root")
        assert m.worker.engine.live["root"].executor_pid != m.worker.engine.live[leaf].executor_pid
        os.kill(m.worker.engine.live[leaf].executor_pid, signal.SIGKILL)
        state, ordinal, body = m.settle(leaf)
        assert (state, ordinal) == ("failed", 1), body
        assert body.execution_started
        assert "the executor was killed by SIGKILL during 'run_job'" in body.safe_message
        state, ordinal, body = m.settle("root")
        assert (state, ordinal, body.status) == ("failed", 1, pb.OUTCOME_STATUS_FAILED), body
        assert m.worker.phase != pb.WorkerPhase.WORKER_PHASE_FAILED
        # Never run again: the leaf entered its author code exactly once.
        assert gate.with_suffix(".entered").read_text() == "x"
        m.idle()


def test_an_executor_killed_by_a_signal_is_named_and_the_worker_serves_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with machine(monkeypatch) as m:
        gate = tmp_path / "never"
        m.submit("root", segments=1, gate=str(gate))
        m.wait(lambda: gate.with_suffix(".entered").exists())
        (leaf,) = m.children("root")
        # What a read fault on a mapped weight page does to the process that took it.
        os.kill(m.worker.engine.live[leaf].executor_pid, signal.SIGBUS)
        state, _, body = m.settle(leaf)
        assert state == "failed", body
        assert body.safe_message == (
            "executor invalidated: the executor was killed by SIGBUS during 'run_job': a read "
            "fault on mapped weights (a disk read error, or a store object changed under its "
            "mapping)"
        )
        assert m.settle("root")[0] == "failed"
        m.submit("after", segments=1)
        assert m.settle("after")[0] == "succeeded"
        m.idle()


def test_an_integrity_failure_fails_every_held_execution_and_releases_the_rental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with machine(monkeypatch) as m:
        m.submit("root", segments=2, gate=str(tmp_path / "never"))
        m.wait(lambda: len(m.worker.engine.live) >= 2)
        assert activity.active_work(m.worker)
        (leaf,) = m.children("root")
        # An execution whose failure cannot be recorded has no owner left: the one
        # worker-integrity class a unit's own crash can reach.
        queued, fail = execution_unit.Execution._queued, Executions.fail

        def crash(self: execution_unit.Execution, *args: Any) -> None:
            if self.request == "orphan":
                raise RuntimeError("injected step fault")
            queued(self, *args)

        def unrecordable(self: Executions, owner: str, request: str, why: str) -> None:
            if request == "orphan":
                raise OSError("injected journal fault")
            fail(self, owner, request, why)

        monkeypatch.setattr(execution_unit.Execution, "_queued", crash)
        monkeypatch.setattr(Executions, "fail", unrecordable)
        m.submit("orphan", segments=1)
        m.wait(lambda: m.worker.phase == pb.WorkerPhase.WORKER_PHASE_FAILED)
        assert not activity.active_work(m.worker)
        for request in ("root", leaf):
            state, ordinal, body = m.settle(request)
            assert (state, ordinal, body.status) == ("failed", 1, pb.OUTCOME_STATUS_FAILED), body
            assert "the worker failed: execution:orphan has no owner" in body.safe_message, (
                body.safe_message
            )
        assert (m.worker.root / "worker-failed.json").exists()


def test_a_deadline_ends_the_attempt_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with machine(monkeypatch) as m:
        submitted = time.monotonic()
        m.submit("root", segments=2, gate=str(tmp_path / "never"), timeout_ms=4000)
        state, ordinal, body = m.settle("root")
        elapsed = time.monotonic() - submitted
        assert state in ("canceled", "failed") and ordinal == 1, body
        assert body.cause.code == pb.CAUSE_CODE_DEADLINE_EXPIRED or "deadline" in (
            body.safe_message.lower()
        ), body
        assert elapsed >= 4.0, elapsed
        m.idle()


def test_a_cancel_escalates_only_on_measured_stillness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The patience floor is SAMPLE_SECONDS-derived; a small sample keeps this test short
    # without changing the rule: STILL_FACTOR x the worst measured gap, never below it.
    monkeypatch.setattr(liveness, "SAMPLE_SECONDS", 0.25)
    floor = liveness.noise_floor()
    with machine(monkeypatch) as m:
        gate = tmp_path / "never"
        m.submit("deaf", job="deaf", segments=0, gate=str(gate))
        m.wait(lambda: gate.with_suffix(".entered").exists())
        attempt = m.worker.engine.live["deaf"]
        state = m.worker.execution_status(m.claim, "deaf")
        # While it keeps moving (frames, as its executor would send them), nothing kills it.
        cancelled = time.monotonic()
        m.worker.control_execution(m.claim, "deaf", "cancel-1", state.generation, "cancel")
        while time.monotonic() - cancelled < 3 * floor:
            attempt.moved()
            time.sleep(floor / 4)
            assert attempt.state == "running" and attempt.executor is not None
            assert attempt.executor.alive()
        # It stops moving: after its measured patience, and not before, it is reclaimed.
        still = time.monotonic()
        state_, ordinal, body = m.settle("deaf")
        assert time.monotonic() - still >= attempt.pace.patience(floor) - 0.5
        assert (state_, ordinal, body.status) == ("canceled", 1, pb.OUTCOME_STATUS_CANCELED), body
        assert m.worker.phase != pb.WorkerPhase.WORKER_PHASE_FAILED
        m.idle()


RESTART = r"""
import sys, threading, time
from pathlib import Path
sys.path.insert(0, sys.argv[2])
root = Path(sys.argv[1])
from cozy_runtime.protocol import worker_pb2 as pb
import test_execution_containment as fixture
worker = fixture.new_worker(root, root / "environments")
request = pb.PrepareLocalPackageRequest.FromString((root / "preparation.bin").read_bytes())
plans, capture = fixture.prepared(worker, root, request)
(root / "capture.bin").write_bytes(capture[0])
(root / "capture.digest").write_bytes(capture[1])
thread = threading.Thread(target=worker.run, daemon=True)
thread.start()
while not (root / "address").exists():
    time.sleep(0.02)
worker.serve_stream(iter([pb.RecordOwnerFrame(claim=fixture.CLAIM)]), lambda _: None)
print("ready", flush=True)
sys.stdin.readline()
"""


def test_a_worker_restart_fails_dispatched_work_and_never_reruns_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(prefix="cz-restart.") as directory:
        root = Path(directory)
        _, preparation = package(root, monkeypatch)
        (root / "preparation.bin").write_bytes(preparation.SerializeToString())
        tests = str(Path(__file__).resolve().parent)

        def boot() -> subprocess.Popen[str]:
            (root / "address").unlink(missing_ok=True)
            child = subprocess.Popen(
                [sys.executable, "-c", RESTART, str(root), tests],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            assert child.stdout is not None
            for line in child.stdout:  # the worker narrates on stdout before it is ready
                if line.strip() == "ready":
                    return child
            child.kill()
            raise AssertionError(f"the worker exited before it was ready: {child.wait()}")

        first = boot()
        try:
            client = rpc.WorkerControlStub(
                grpc.insecure_channel((root / "address").read_text().strip())
            )
            capture = ((root / "capture.bin").read_bytes(), (root / "capture.digest").read_bytes())
            plans = {}
            for path in (root / "home/job-plans").glob("*/*.json"):
                document = canonical_json.decode(path.read_bytes())
                plans[document["job"]] = JobBinding.read(document)
            gate = tmp_path / "never"
            submit(
                client,
                CLAIM,
                root,
                plans["nested"],
                capture,
                "root",
                "nested",
                240_000,
                {"segments": 2, "gate": str(gate)},
            )
            deadline = time.monotonic() + 300
            while not gate.with_suffix(".entered").exists():
                assert first.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            first.kill()  # the worker dies mid-run, its leaf inside author code
            first.communicate(timeout=30)
        finally:
            if first.poll() is None:
                first.kill()
        second = boot()
        try:
            client = rpc.WorkerControlStub(
                grpc.insecure_channel((root / "address").read_text().strip())
            )
            workspace_id = client.GetMachineExecutionWorkspace(
                pb.MachineExecutionWorkspaceQuery(claim=CLAIM)
            ).execution_workspace_id

            def query(request: str) -> pb.MachineExecutionQuery:
                return pb.MachineExecutionQuery(
                    claim=CLAIM, request_id=request, expected_execution_workspace_id=workspace_id
                )

            while client.GetMachineExecution(query("root")).state not in TERMINAL:
                assert second.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            executions = Executions(Workspace(root / "store"))
            with executions.workspace.locked() as db:
                (leaf,) = [r[0] for r in db.execute("SELECT child_request FROM execution_calls")]
            for request in ("root", leaf):
                state = client.GetMachineExecution(query(request))
                assert (state.state, state.attempt_ordinal) == ("failed", 1), state
                body = documents.parse(
                    executions.outcome("owner", request).outcome_canonical_bytes,
                    pb.AttemptOutcomeBody,
                )
                assert body.execution_started, body
            # Recovered, never run again: the leaf's author code started exactly once.
            assert gate.with_suffix(".entered").read_text() == "x"
        finally:
            second.kill()
            second.communicate(timeout=30)


def test_a_boot_releases_what_a_terminal_execution_holds(tmp_path: Path) -> None:
    """A failed execution's committed bytes are released at the next boot, with no owner
    action, and the store can reclaim them."""
    value = json.loads(
        (
            Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json"
        ).read_text()
    )
    for outputs in (value["outputs"], value["job"]["publication_contract"]["outputs"]):
        for output in outputs:
            output["mime_type"] = "application/vnd.cozy.model-manifest"
    invocation = canonical.write(value)
    options = WorkerOptions(
        **signed_claims.IDENTITY, root=tmp_path / "worker", tensorfs_root=tmp_path / "store"
    )
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "producer",
        b"c" * 32,
        pb.AttemptOffer(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=hashlib.sha256(invocation).digest(),
            invocation_spec_canonical_bytes=invocation,
        ),
        expected_execution_workspace_id=executions.workspace_id,
        worker_id=options.worker_id,
    )
    store, _, receipt = produced(tmp_path)
    executions.fail("owner", "producer", "the worker failed")
    manifest = "sha256:" + json.loads(receipt.tensorfs_receipt)["manifest"]["sha256"]
    tensorfs.gc(str(store.root))
    assert store.manifest(manifest)["manifest"]  # held by its failed execution
    worker = Worker(
        RuntimeConfig(
            cozy_home=tmp_path / "home",
            credentials=Credentials(),
            record_owner_public_key=signed_claims.PUBLIC_KEY,
        ),
        options,
        InMemoryControlHost(),
    )
    try:
        worker.accept_claim(signed_claims.claim(), lambda frame: None)
        assert not executions.retention_required("owner")
        assert tensorfs.gc(str(store.root))["reclaimed_bytes"] > 0
        with pytest.raises(tensorfs.errors.ObjectAbsent):
            store.manifest(manifest)
    finally:
        worker.shutdown()
