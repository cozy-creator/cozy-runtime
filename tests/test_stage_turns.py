"""Stage turns across the real seam (`stage/1`).

Real: a model's weights in a real TensorFS store, on the weight plane's host tier (a CPU
executor computes on it in place), its `WeightResidency`, the executor's turn hooks and
durable lane over a socket pair, and on the other end the real Worker's handler, memory
manager and stage scheduler. Each component-use scope asks the Worker for its turn and
reports what it measured; the scheduler grants turns in order, learns the graph, and tells a
call past its last stage to give its device bytes back when another call waits.
"""

from __future__ import annotations

import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author._activity import observing  # noqa: E402
from cozy_runtime.author._executor_requests import (  # noqa: E402
    Answer,
    Handler,
    StageEnter,
    respond,
)
from cozy_runtime.internal import accel, plane  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.stages import StageKind, StageSample  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from cozy_runtime.internal.worker.attempts import AttemptRecord  # noqa: E402
from cozy_runtime.internal.worker.stage_scheduler import Grant, Want  # noqa: E402
from test_gpu_scheduler import VIRTUAL, GiB, Machine, driverless  # noqa: E402
from test_weight_plane import HIDDEN, Stack, load, write_checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(not plane.available(), reason="a TensorFS with the weight plane")

SCOPE = StageKind("inst-a/generate", "run", ("stack",))


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    driverless(monkeypatch)
    measured = accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)
    monkeypatch.setattr(accel, "device_memory", lambda entry, kind: measured)
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-turns.", dir="/tmp") as root:
        made = Machine(Path(root), VIRTUAL.split(",")[0], "boot-turns")  # one GPU
        try:
            yield made
        finally:
            made.close()


class Seamed:
    """An executor host on one end of a socket pair, with the plane's host tier, and the
    Worker's handler answering its durable requests on the other."""

    def __init__(self, root: Path, handler: Handler) -> None:
        left, right = socket.socketpair()
        self.worker = Channel(right)
        self.host = Executor(Channel(left), root / "executor")
        self.weights = Weights(torch, torch.device("cpu"), "cpu")
        self.weights.set_budget(-1, pinned=1 << 30)
        self.host.weights = self.weights
        checkpoint, self.values = write_checkpoint(root)
        self.model = load(self.weights, checkpoint, "construction")
        self.answering = threading.Thread(target=self._answer, args=(handler,), daemon=True)
        self.answering.start()

    def _answer(self, handler: Handler) -> None:
        while (frame := self.worker.recv()) is not None:
            answer, _ = respond(frame, handler)
            self.worker.send(answer)

    def scope(self, x: Any) -> Any:
        """One `run` scope with turns on, as an `Invoke(stages=True)` runs the package."""
        residency = self.model.residency
        with self.host._turns(residency, True):
            residency.admit("run", ("stack",))
            try:
                with torch.inference_mode():
                    return self.model.stack(x)
            finally:
                residency.release("run", ("stack",))

    def close(self) -> None:
        self.worker.sock.close()
        self.answering.join(10)


def tenant(machine: Machine, slot: str, request: str) -> tuple[Handler, AttemptRecord]:
    """A turn-taking tenant on GPU 0 and the Worker's handler for its attempt's requests."""
    lane = machine.worker.lanes.by_id["lane-0"]
    lane.row(slot)
    lane.placements.add(slot)
    attempt = AttemptRecord(request, 1, b"", {}, placement_id=slot, lane_id="lane-0")
    return machine.worker._room(lane, slot, attempt=attempt), attempt


def test_each_scope_takes_its_turn_from_the_worker_and_the_book_learns_it(
    machine: Machine, tmp_path: Path
) -> None:
    worker = machine.worker
    worker._turning["inst-a"] = True  # what its executors' hello says (`stage/1`)
    worker.stages.open_root("R", 1)
    worker.stages.want("R#1", Want(root="R", template=("inst-a", "b"), entrypoint="generate"))
    assert worker.stages.ready("R#1") and worker.stages.taking_turns("R#1")
    handler, _ = tenant(machine, "tenant-a", "R")
    executor = Seamed(tmp_path, handler)
    try:
        x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
        reference = Stack()
        reference.load_state_dict(executor.values)
        with torch.inference_mode():
            expected = reference(x)
        got = executor.scope(x)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), "bit-exact"
        assert worker.stages.view()["demands"]["R#1"]["stage"] == 1
        assert executor.model.residency.turn is None, "hooks leave with the call"
    finally:
        executor.close()
    worker.stages.release("R#1", ok=True, cell="height=4")
    graph = worker.stages.book.graph("inst-a/generate", "height=4")
    assert graph is not None and graph.observed
    assert [(s.kind.method, s.calls) for s in graph.stages] == [("run", 1)]
    cost = worker.stages.book.cost(graph.stages[0].kind, "height=4", 1)
    assert cost is not None and cost.runs == 1 and cost.wall_s > 0


def test_past_its_last_stage_a_call_gives_its_bytes_back_to_the_one_waiting(
    machine: Machine, tmp_path: Path
) -> None:
    worker = machine.worker
    worker._turning.update({"inst-a": True, "inst-b": True})
    worker.stages.book.record("inst-a/generate", "c", [StageSample(kind=SCOPE, wall_ns=10**6)])
    worker.stages.open_root("A", 1)
    worker.stages.open_root("B", 2)
    worker.stages.want(
        "A#1", Want(root="A", template=("inst-a", "b"), entrypoint="generate", cell="c")
    )
    worker.stages.want("B#1", Want(root="B", template=("inst-b", "b"), entrypoint="generate"))
    assert worker.stages.ready("A#1") and worker.stages.ready("B#1"), "B prepares ahead"
    handler, _ = tenant(machine, "tenant-a", "A")
    executor = Seamed(tmp_path, handler)
    granted: list[Grant | None] = []
    kind = worker.stages.kind("B#1", "render", ("transformer",))
    waiting = threading.Thread(
        target=lambda: granted.append(worker.stages.enter("B#1", kind, lambda: False))
    )
    waiting.start()
    try:
        bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
        while not worker.stages.view()["demands"]["B#1"]["asking"]:
            assert time.monotonic() < bound
            time.sleep(0.01)
        x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(5)).half()
        assert not granted, "B waits for A, the earlier call"
        executor.scope(x)  # A's one observed stage: then its CPU tail
        waiting.join(30)
        assert granted == [Grant((0,), True)]
        assert executor.weights.budget == 0, "A unmapped its weights at its last exit"
    finally:
        executor.close()


def test_a_scope_waiting_for_its_turn_is_not_counted_as_execution(
    machine: Machine, tmp_path: Path
) -> None:
    worker = machine.worker
    worker._turning.update({"inst-a": True, "inst-b": True})
    worker.stages.open_root("A", 1)
    worker.stages.open_root("B", 2)
    worker.stages.want("A#1", Want(root="A", template=("inst-a", "b"), entrypoint="generate"))
    worker.stages.want("B#1", Want(root="B", template=("inst-b", "b"), entrypoint="generate"))
    kind = worker.stages.kind("A#1", "render", ("transformer",))
    assert worker.stages.enter("A#1", kind, lambda: False) == Grant((0,), True)
    handler, _ = tenant(machine, "tenant-b", "B")
    executor = Seamed(tmp_path, handler)
    active: list[bool] = []
    x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(7)).half()

    def call() -> None:
        with observing(lambda _sequence, on, _known, _finished: active.append(on)):
            executor.scope(x)

    running = threading.Thread(target=call)
    running.start()
    try:
        bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
        while not worker.stages.view()["demands"]["B#1"]["asking"]:
            assert time.monotonic() < bound
            time.sleep(0.01)
        assert active == [True, False], "the wait behind A is not B's work"
        worker.stages.release("A#1", ok=True)
        running.join(30)
        assert active == [True, False, True, False]
    finally:
        executor.close()


def test_a_cancel_answers_a_scope_waiting_for_its_turn(machine: Machine, tmp_path: Path) -> None:
    worker = machine.worker
    worker._turning.update({"inst-a": True, "inst-b": True})
    worker.stages.open_root("A", 1)
    worker.stages.open_root("B", 2)
    worker.stages.want("A#1", Want(root="A", template=("inst-a", "b"), entrypoint="generate"))
    worker.stages.want("B#1", Want(root="B", template=("inst-b", "b"), entrypoint="generate"))
    handler, attempt = tenant(machine, "tenant-b", "B")
    answers: list[object] = []
    waiting = threading.Thread(
        target=lambda: answers.append(handler(StageEnter(method="render", components=())))
    )
    waiting.start()
    bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
    while not worker.stages.view()["demands"]["B#1"]["asking"]:
        assert time.monotonic() < bound
        time.sleep(0.01)
    attempt.canceling = "client"
    worker.stages.wake()
    waiting.join(30)
    assert len(answers) == 1
    answer = answers[0]
    assert isinstance(answer, Answer) and not answer.ok and answer.code == "cancelled"
