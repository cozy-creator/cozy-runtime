"""Stage turns across the real seam (`stage/1`).

Real: a model's weights in a real TensorFS store, on the weight plane's host tier (a CPU
executor computes on it in place), its `WeightResidency`, the executor's turn hooks and
durable lane over a socket pair, and on the other end the real Worker's handler, memory
manager and stage scheduler. Each component-use scope asks the Worker for its turn and
reports what it measured; the scheduler grants turns in order, learns the graph, and tells a
call past its last stage to give its device bytes back when the call waiting lacks room,
keeping the turn until they are unmapped.
"""

from __future__ import annotations

import socket
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author._activity import observing, work_clock  # noqa: E402
from cozy_runtime.author._executor_requests import (  # noqa: E402
    Answer,
    Handler,
    StageEnter,
    StageExit,
    StageGo,
    respond,
)
from cozy_runtime.internal import accel, plane, weight_policy  # noqa: E402
from cozy_runtime.internal.executor import Executor  # noqa: E402
from cozy_runtime.internal.executor_replies import PlaneFacts  # noqa: E402
from cozy_runtime.internal.seam import Channel  # noqa: E402
from cozy_runtime.internal.stages import StageKind, StageSample  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402
from cozy_runtime.internal.worker.attempts import (  # noqa: E402
    AttemptRecord,
    AttemptRefusal,
    DeviceExhausted,
)
from cozy_runtime.internal.worker.stage_scheduler import Grant, Want  # noqa: E402
from cozy_runtime.protocol import worker_pb2 as pb  # noqa: E402
from test_gpu_scheduler import VIRTUAL, GiB, Machine, driverless  # noqa: E402
from test_weight_plane import HIDDEN, Stack, load, on_card, write_checkpoint  # noqa: E402

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

    def __init__(self, root: Path, handler: Handler, device: str = "cpu") -> None:
        left, right = socket.socketpair()
        self.worker = Channel(right)
        self.host = Executor(Channel(left), root / "executor")
        self.weights = Weights(torch, torch.device(device), device)
        self.weights.set_budget(-1, pinned=1 << 30)
        self.host.weights = self.weights
        checkpoint, self.values = write_checkpoint(root)
        self.model = load(self.weights, checkpoint, "construction")
        #: the Worker's handler for the attempt this executor runs now
        self.handler = handler
        self.answering = threading.Thread(target=self._answer, daemon=True)
        self.answering.start()

    def _answer(self) -> None:
        while (frame := self.worker.recv()) is not None:
            answer, _ = respond(frame, self.handler)
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
        self.worker.sock.shutdown(socket.SHUT_RDWR)  # the answering thread reads its end
        self.answering.join(10)
        self.worker.sock.close()


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
    budgets: list[int] = []  # A's plane budget when B's turn came
    kind = worker.stages.kind("B#1", "render", ("transformer",))

    def wait() -> None:
        granted.append(worker.stages.enter("B#1", kind, lambda: False))
        budgets.append(executor.weights.budget)

    waiting = threading.Thread(target=wait)
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
        assert budgets == [0], "B's turn came after A had unmapped, so B measures real room"
    finally:
        executor.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=on_card)])
def test_a_models_next_request_finds_its_weights_mapped_and_uploads_none(
    machine: Machine, tmp_path: Path, device: str
) -> None:
    """Runs 2955-2957, three queued SDXL requests on an 8 GB card: each was told to unmap for
    the next, its own model's, and that one uploaded all 6.4 GiB again."""
    worker = machine.worker
    worker._turning["inst-a"] = True
    worker.stages.book.record("inst-a/generate", "c", [StageSample(kind=SCOPE, wall_ns=10**6)])
    # The model's executor holds device weights: its next call waits outside it for the turn.
    worker.stages.holds = lambda template, gpus: True
    want = Want(root="A", template=("inst-a", "b"), entrypoint="generate", cell="c")
    for order, root in enumerate(("A", "B"), 1):
        worker.stages.open_root(root, order)
        worker.stages.want(f"{root}#1", msgspec.structs.replace(want, root=root))
    assert worker.stages.turn_taking("A#1"), "at its device entry it goes stage by stage"
    assert not worker.stages.ready("B#1") and worker.stages.planned("B#1") == (0,)
    assert worker.stages.view()["waiting"] == {"B#1": ["A"]}
    handler, _ = tenant(machine, "tenant-a", "A")
    executor = Seamed(tmp_path, handler, device)
    try:
        x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(5)).half().to(device)
        executor.scope(x)  # A's one observed stage: past it, B is the call that waits
        assert executor.weights.budget != 0, "nobody but its own model wants the room"
        was = executor.weights.facts()
        assert device == "cpu" or was.h2d_bytes > 0, "its first request uploaded them"
        worker.stages.release("A#1", ok=True, cell="c")
        assert worker.stages.ready("B#1") and worker.stages.turn_taking("B#1")
        executor.handler, _ = tenant(machine, "tenant-a", "B")
        executor.scope(x * 2)
        now = executor.weights.facts()
        assert (now.h2d_bytes, now.evictions) == (was.h2d_bytes, was.evictions)
        assert worker.stages.view()["demands"]["B#1"]["stage"] == 1, "B took its own turn"
    finally:
        executor.close()


def test_a_nested_scope_neither_ends_the_turn_nor_unmaps_the_outer_scopes_weights(
    machine: Machine,
) -> None:
    """H3 turbo on one GPU with another call waiting: the inner scope's exit used to end the
    call's turn, and past its last stage it was answered budget 0, which unmapped the outer
    scope's weights while that scope was open. Only the outer scope's exit ends the turn."""
    worker = machine.worker
    worker._turning.update({"inst-a": True, "inst-b": True})
    outer = StageKind("inst-a/generate", "sample_turbo", ("base",))
    worker.stages.book.record("inst-a/generate", "c", [StageSample(kind=outer, wall_ns=10**6)])
    worker.stages.open_root("A", 1)
    worker.stages.open_root("B", 2)
    worker.stages.want(
        "A#1", Want(root="A", template=("inst-a", "b"), entrypoint="generate", cell="c")
    )
    worker.stages.want("B#1", Want(root="B", template=("inst-b", "b"), entrypoint="generate"))
    handler, _ = tenant(machine, "tenant-a", "A")
    first = handler(StageEnter(method="sample_turbo", components=("base",)))
    assert isinstance(first, StageGo) and first.ok
    nested = handler(StageEnter(method="sample", components=("lora",)))
    assert isinstance(nested, StageGo) and nested.ok and nested.budget_bytes == -1
    granted: list[Grant | None] = []
    kind = worker.stages.kind("B#1", "render", ("transformer",))
    waiting = threading.Thread(
        target=lambda: granted.append(worker.stages.enter("B#1", kind, lambda: False)),
        daemon=True,  # a failed assertion must not leave the run waiting on it
    )
    waiting.start()
    bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
    while not worker.stages.view()["demands"]["B#1"]["asking"]:
        assert time.monotonic() < bound
        time.sleep(0.01)
    inner = handler(StageExit(method="sample", components=("lora",)))
    assert isinstance(inner, StageGo) and inner.budget_bytes == -1, "the outer scope is open"
    assert not granted and worker.stages.view()["holders"] == {"0": "A#1"}
    last = handler(StageExit(method="sample_turbo", components=("base",)))
    assert isinstance(last, StageGo) and last.budget_bytes == 0 and not granted
    handler(StageExit(method="sample_turbo", components=("base",), yielded=True))
    waiting.join(30)
    assert granted == [Grant((0,), True)]


def _driver(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """The GPU's free bytes as the driver reports them now, for the test to move."""
    free = {"bytes": 80 * GiB}
    monkeypatch.setattr(
        accel,
        "device_memory",
        lambda entry, kind: accel.DeviceMemory("measured", free["bytes"], 80 * GiB),
    )
    return free


def test_a_device_start_waits_for_room_until_the_call_in_flight_is_past_its_stages(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs 2600/2601: SDXL filled a 6 GiB pool, the Anima executor prepared beside it could
    not create its CUDA context, and the running SDXL died. A new device context now starts
    only when the driver shows room for it. With a call in flight on a full GPU it waits until
    that call is past its stages, not for a clock, and takes nothing from it meanwhile."""
    free = _driver(monkeypatch)
    worker = machine.worker
    worker._turning["inst-a"] = True
    kind = StageKind("inst-a/generate", "denoise", ("unet",))
    worker.stages.book.record("inst-a/generate", "c", [StageSample(kind=kind, wall_ns=10**6)] * 2)
    worker.stages.open_root("A", 1)
    worker.stages.want(
        "A#1", Want(root="A", template=("inst-a", "b"), entrypoint="generate", cell="c")
    )
    handler, _ = tenant(machine, "tenant-a", "A")
    lane = worker.lanes.by_id["lane-0"]
    mib = 1 << 20
    row = lane.row("tenant-a")
    row.model_bearing, row.ledger.plane = True, PlaneFacts(context_bytes=180 * mib)
    assert worker.memory.context_need(lane) == {0: 360 * mib + weight_policy.MARGIN}

    def enter() -> None:
        go = handler(StageEnter(method="denoise", components=("unet",)))
        assert isinstance(go, Answer) and go.ok

    def step() -> None:
        enter()
        handler(StageExit(method="denoise", components=("unet",)))

    enter()  # A computes
    free["bytes"] = 10 * mib
    started: list[int] = []

    def start() -> None:
        with worker._starting(lane, "tenant-b"):
            started.append(free["bytes"])

    starting = threading.Thread(target=start, daemon=True)
    starting.start()
    bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
    while "tenant-b" not in worker.memory.wanted:
        assert time.monotonic() < bound
        time.sleep(0.01)
    handler(StageExit(method="denoise", components=("unet",)))
    assert not started, "between A's stages: A is still in flight, and the GPU is still full"
    step()  # A's last stage: past it, A's tenant is idle
    free["bytes"] = GiB  # as the driver reads once A's weights are unmapped
    starting.join(30)
    assert started == [GiB] and "tenant-b" not in worker.memory.wanted


def test_a_load_short_of_device_memory_beside_a_call_in_flight_is_loaded_again(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs 2609 and 2628: Anima's first load ran out of device memory beside SDXL, and its
    request was refused without ever starting; the same request a minute later ran. Nothing
    of it had run, so the worker loads again: once every call of another model ahead of it
    has left the GPU, or when the shortfall is one it had not met (its executor measured what
    it lacked). The same shortfall twice with nobody ahead of it is the request's own."""
    worker = machine.worker
    worker._turning["inst-b"] = True
    worker.stages.open_root("X", 1)
    worker.stages.open_root("B", 2)
    worker.stages.want("X#1", Want(root="X", template=("inst-x", "b"), entrypoint="generate"))
    worker.stages.want("B#1", Want(root="B", template=("inst-b", "b"), entrypoint="generate"))
    kind = worker.stages.kind("X#1", "denoise", ("unet",))
    assert worker.stages.enter("X#1", kind, lambda: False) is not None  # X computes
    _, attempt = tenant(machine, "tenant-b", "B")
    attempt.stage_turns = True
    loads: list[bool] = []  # each load: whether X was still in flight

    def load(_attempt: AttemptRecord) -> None:
        loads.append(bool(worker.stages.view()["demands"].get("X#1")))
        if len(loads) == 1:
            raise AttemptRefusal(
                "device_out_of_memory",
                "CUDA out of memory",
                cause=pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
            )

    monkeypatch.setattr(worker, "_ensure_once", load)
    monkeypatch.setattr(worker, "_rebuild_lane", lambda lane, slot: None)
    unmapped: list[str] = []
    monkeypatch.setattr(worker.memory, "unmap", lambda tenant, why: unmapped.append(tenant))
    loading = threading.Thread(target=worker._ensure_construction, args=(attempt,), daemon=True)
    loading.start()
    bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
    while not loads:
        assert time.monotonic() < bound
        time.sleep(0.01)
    assert loading.is_alive(), "it waits for the call ahead of it, and refuses nothing"
    worker.stages.exit("X#1", StageSample(kind=kind, wall_ns=1))
    assert loading.is_alive(), "past its stages is not enough: its attempt's tail still runs"
    worker.stages.release("X#1")
    loading.join(30)
    assert loads == [True, False], "loaded again once X was past"
    assert unmapped == ["tenant-b"], "what the refused load left mapped is unmapped first"
    # Alone on the GPU: a new shortfall is tried again (the executor measured what it
    # lacked); the same one twice is the request's own, refused with its reason.
    shortfalls = iter(["the decode ran out of memory", "a block ran out of memory"])
    tries: list[str] = []

    def short(_attempt: AttemptRecord) -> None:
        tries.append(next(shortfalls, "a block ran out of memory"))
        raise AttemptRefusal(
            "device_shortfall", tries[-1], cause=pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE
        )

    monkeypatch.setattr(worker, "_ensure_once", short)
    evicted: list[tuple[str, bool]] = []

    def make_room(
        lane: object, slot: str, need: object, why: object, contexts: bool = False
    ) -> dict[int, int]:
        evicted.append((slot, contexts))
        return {}

    monkeypatch.setattr(worker.memory, "make_room", make_room)
    with pytest.raises(DeviceExhausted, match="a block ran out of memory"):
        worker._ensure_construction(attempt)
    assert evicted == [("tenant-b", True)] * 2, "idle weights and idle contexts go before each try"
    statuses: list[int] = []
    monkeypatch.setattr(
        worker.engine, "outcome", lambda attempt, status, *rest: statuses.append(status)
    )
    worker.engine.end_held(
        attempt, DeviceExhausted("device_shortfall", "x", cause=pb.CAUSE_CODE_NO_CAPACITY)
    )
    worker.engine.end_held(
        attempt, AttemptRefusal("other", "x", cause=pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE)
    )
    assert statuses == [pb.OUTCOME_STATUS_FAILED, pb.OUTCOME_STATUS_REFUSED], (
        "never REFUSED for size"
    )
    assert tries == [
        "the decode ran out of memory",
        "a block ran out of memory",
        "a block ran out of memory",
    ]
    # An attempt holding a whole turn (an older executor, a group) never waits for others while
    # it holds the GPU, and is never refused for size either.
    attempt.stage_turns, tries[:] = False, []
    shortfalls = iter(["a block ran out of memory"])
    with pytest.raises(DeviceExhausted, match="every idle byte"):
        worker._ensure_construction(attempt)
    assert tries == ["a block ran out of memory", "a block ran out of memory"]


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

    clocks: list[tuple[float, float]] = []  # the scope's wall, and what its timers read

    def call() -> None:
        with observing(lambda _sequence, on, _known, _finished: active.append(on)):
            wall, work = time.perf_counter(), work_clock()
            executor.scope(x)
            clocks.append((time.perf_counter() - wall, work_clock() - work))

    running = threading.Thread(target=call)
    running.start()
    try:
        bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
        while not worker.stages.view()["demands"]["B#1"]["asking"]:
            assert time.monotonic() < bound
            time.sleep(0.01)
        seen = time.perf_counter()
        assert active == [True, False], "the wait behind A is not B's work"
        held = time.perf_counter() - seen
        worker.stages.release("A#1", ok=True)
        running.join(30)
        assert active == [True, False, True, False]
        ((wall, work),) = clocks
        assert wall - work >= held, "stage and step timers leave the wait out too"
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
