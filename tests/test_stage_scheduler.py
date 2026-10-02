"""The stage scheduler's shell over real threads: placement, turns, leases, prepare-ahead.

Each test drives a real `StageScheduler` the way the Worker does (roots open and close, calls
want GPUs, executors enter and exit scopes, calls release), with the machine's facts given as
data. A call whose executor takes no stage turns holds one turn from its grant to its
release, exactly as the Worker's older executors and Ulysses groups do. Worker-level
behaviour over the durable journal is `test_gpu_scheduler`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from cozy_runtime.internal.stages import CostBook, StageKind, StageSample, WeightSet
from cozy_runtime.internal.worker.stage_policy import Gpu, Host, Machine
from cozy_runtime.internal.worker.stage_scheduler import (
    Event,
    Exit,
    Grant,
    StageScheduler,
    Want,
)

GiB = 1 << 30
SDXL = ("inst-sdxl", "sha256:sdxl")
ANIMA = ("inst-anima", "sha256:anima")


class Shell:
    """One scheduler and what it told the Worker: journal rows and the units it woke."""

    def __init__(
        self,
        devices: int = 4,
        *,
        turns: bool = False,
        warm: Mapping[tuple[str, str], tuple[tuple[int, ...], ...]] | None = None,
        sets: Mapping[tuple[str, str], Mapping[str, WeightSet]] | None = None,
        book: Path | None = None,
        total: int = 80 * GiB,
        spawn_s: float = 0.0,
    ) -> None:
        self.events: list[Event] = []
        self.poked: list[str] = []
        #: whether executors take stage turns, as their hellos say once known
        self.turns = turns
        #: templates whose idle executor holds device weights
        self.holding: set[tuple[str, str]] = set()
        self.prefilled: list[str] = []
        self.gpus = [Gpu(ordinal=o, total=total) for o in range(devices)]
        self.host = Host(spawn_s=spawn_s)
        self.scheduler = StageScheduler(
            devices,
            poke=self.poked.append,
            observed=self.events.extend,
            machine=lambda: Machine(tuple(self.gpus), self.host),
            warm=lambda template: (warm or {}).get(template, ()),
            sets=lambda want: (sets or {}).get(want.template, {}),
            turns=lambda installation: self.turns,
            holds=lambda template, gpus: template in self.holding,
            prefill=lambda key, template: self.prefilled.append(key),
            fault=self._fault,
            note=lambda why: None,
            book=book,
        )

    @staticmethod
    def _fault(why: str) -> None:
        raise AssertionError(why)

    def held(self) -> dict[str, tuple[int, ...]]:
        """Each call holding a turn, and its GPUs."""
        holders: dict[str, list[int]] = {}
        for ordinal, key in self.scheduler.view()["holders"].items():
            holders.setdefault(key, []).append(int(ordinal))
        return {key: tuple(sorted(o)) for key, o in holders.items()}

    def waiting(self) -> dict[str, list[str]]:
        return dict(self.scheduler.view()["waiting"])

    def leases(self) -> dict[str, list[int]]:
        return dict(self.scheduler.view()["leases"])

    def demand(self, key: str) -> dict[str, object]:
        return dict(self.scheduler.view()["demands"][key])

    def kinds(self, kind: str) -> list[tuple[str, dict[str, object]]]:
        return [(root, body) for root, event, body in self.events if event == kind]


def until(ready: Callable[[], bool]) -> None:
    bound = time.monotonic() + 30  # a hang bound on a loaded box, not a budget
    while not ready():
        assert time.monotonic() < bound
        time.sleep(0.005)


# --------------------------------------------------------------------------- roots and leases


def test_a_root_keeps_its_lease_through_the_gap_and_an_older_waiter_takes_what_frees() -> None:
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.open_root("B", 2)
    gpus.want("a1#1", Want(root="A", widths=(4,)))
    assert shell.held() == {"a1#1": (0, 1, 2, 3)}
    gpus.release("a1#1")
    gpus.want("b1#1", Want(root="B", widths=(1,)))
    assert shell.held() == {}  # A's gap: A is between its calls
    assert shell.leases() == {"A": [0, 1, 2, 3]} and shell.waiting() == {"b1#1": ["A"]}
    gpus.want("a2#1", Want(root="A", widths=(4,)))
    assert shell.held() == {"a2#1": (0, 1, 2, 3)}  # A's next call reuses its own GPUs
    gpus.release("a2#1")
    gpus.close_root("A")
    assert shell.held() == {"b1#1": (0,)}
    # Nothing moves a younger holder: an older root waits for it, and no younger root takes
    # what the older one claims meanwhile.
    shell = Shell(4)
    gpus = shell.scheduler
    for root, priority in (("O", 1), ("Y", 5), ("Z", 9)):
        gpus.open_root(root, priority)
    gpus.want("y#1", Want(root="Y", widths=(2,)))
    assert shell.held() == {"y#1": (0, 1)}
    gpus.want("o#1", Want(root="O", widths=(4,)))
    gpus.want("z#1", Want(root="Z", widths=(1,)))
    assert shell.held() == {"y#1": (0, 1)} and set(shell.waiting()) == {"o#1", "z#1"}
    gpus.release("y#1")
    assert shell.held() == {"o#1": (0, 1, 2, 3)}


def test_two_fanned_out_roots_take_their_group_calls_in_order_and_name_no_cycle() -> None:
    """Runs 1572 and 1573: each root's references leased the GPUs they ran on, then each
    root's 4-GPU call waited behind the other's lease, forever. The older root's call takes
    the younger's idle GPUs and waits only on calls actually on them; the younger's is named
    behind its own calls while they run, then behind the older."""
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.open_root("B", 2)
    for key, root in (("a1#1", "A"), ("b1#1", "B"), ("a2#1", "A"), ("b2#1", "B")):
        gpus.want(key, Want(root=root, widths=(1,)))
    assert shell.held() == {"a1#1": (0,), "b1#1": (1,), "a2#1": (2,), "b2#1": (3,)}
    assert shell.leases() == {"A": [0, 2], "B": [1, 3]}
    gpus.release("a1#1")
    gpus.release("a2#1")
    gpus.want("sa#1", Want(root="A", widths=(4,)))
    gpus.want("sb#1", Want(root="B", widths=(4,)))
    assert shell.waiting() == {"sa#1": ["B"], "sb#1": ["B"]}
    gpus.release("b1#1")
    gpus.release("b2#1")
    assert shell.held() == {"sa#1": (0, 1, 2, 3)}
    assert shell.waiting() == {"sb#1": ["A"]} and shell.leases() == {"A": [0, 1, 2, 3]}
    gpus.release("sa#1")
    assert shell.held() == {}  # A's gap holds against the younger B
    gpus.close_root("A")
    assert shell.held() == {"sb#1": (0, 1, 2, 3)}
    yielded = [body["cause"] for root, body in shell.kinds("gpu.lease") if root == "B"]
    assert "yielded to sa#1" in yielded


def test_fifo_within_a_root_warm_first_and_never_an_excluded_gpu() -> None:
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.want("ref#1", Want(root="A", widths=(1,)))
    gpus.want("h3#1", Want(root="A", widths=(4,)))
    gpus.want("qwen#1", Want(root="A", widths=(1,)))
    # The 4-GPU call waits for the reference; the later call of the same root never
    # overtakes it onto the free GPUs.
    assert shell.held() == {"ref#1": (0,)}
    gpus.release("ref#1")
    assert shell.held() == {"h3#1": (0, 1, 2, 3)}
    # Where the call's executor is already warm, it is placed there.
    shell = Shell(4, warm={SDXL: ((2,),)})
    shell.scheduler.open_root("A", 1)
    shell.scheduler.want("q#1", Want(root="A", widths=(1,), template=SDXL))
    assert shell.held() == {"q#1": (2,)}
    # An excluded (unreadable) GPU is never taken.
    shell.scheduler.want("q#2", Want(root="A", widths=(3,), exclude=(2,)))
    gpus = shell.scheduler
    shell.scheduler.release("q#1")
    assert shell.held() == {"q#2": (0, 1, 3)}


def test_a_stopped_roots_call_keeps_its_gpus_until_it_leaves_them() -> None:
    shell = Shell(2)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.want("a#1", Want(root="A", widths=(2,)))
    assert shell.held() == {"a#1": (0, 1)}
    gpus.close_root("A")  # cancelled; its attempt is still on the GPUs
    gpus.open_root("B", 2)
    gpus.want("b#1", Want(root="B", widths=(1,)))
    assert shell.held() == {"a#1": (0, 1)}
    ranks = [{"rank": r, "pid": 70 + r, "ordinal": r, "start_us": 1, "end_us": 2} for r in (0, 1)]
    gpus.release("a#1", ranks=ranks)
    assert shell.held() == {"b#1": (0,)}
    kinds = [event for _root, event, _body in shell.events]
    assert kinds.count("gpu.grant") == 2 and "gpu.wait" in kinds
    exit_ = next(body for _root, body in shell.kinds("gpu.release"))
    assert exit_ == {"key": "a#1", "ordinals": [0, 1], "cause": "exited", "ranks": ranks}


def test_a_released_root_hands_its_gpus_on_and_a_later_call_waits_its_turn() -> None:
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.open_root("B", 2)
    gpus.want("a1#1", Want(root="A", widths=(4,)))
    gpus.release("a1#1")
    gpus.want("b1#1", Want(root="B", widths=(4,)))
    assert shell.held() == {}  # A's gap holds its lease
    gpus.release_root("A")
    gpus.release_root("A")  # idempotent
    assert "A" not in shell.leases() and shell.held() == {"b1#1": (0, 1, 2, 3)}
    # A's GPU call after its release waits at A's priority for B's call on the GPUs.
    gpus.want("a2#1", Want(root="A", widths=(4,)))
    assert shell.waiting() == {"a2#1": ["B"]}
    gpus.release("b1#1")
    assert shell.held() == {"a2#1": (0, 1, 2, 3)} and shell.leases() == {"A": [0, 1, 2, 3]}
    gpus.release("a2#1")  # the new lease holds A's next gap again
    gpus.open_root("C", 3)
    gpus.want("c#1", Want(root="C", widths=(1,)))
    assert shell.held() == {}
    released = [
        body for root, body in shell.kinds("gpu.lease") if root == "A" and not body["ordinals"]
    ]
    assert released == [{"ordinals": [], "cause": "released_by_root"}]


def test_a_call_in_flight_keeps_its_gpu_until_it_exits() -> None:
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.open_root("B", 2)
    gpus.want("r0#1", Want(root="A", widths=(1,)))
    gpus.want("r1#1", Want(root="A", widths=(1,)))
    assert shell.held() == {"r0#1": (0,), "r1#1": (1,)}
    gpus.release("r0#1")
    gpus.release_root("A")  # r1 is still on GPU 1
    gpus.want("b#1", Want(root="B", widths=(4,)))
    assert shell.held() == {"r1#1": (1,)}
    gpus.release("r1#1")  # GPU 1 goes to the pool, not back to A
    assert shell.held() == {"b#1": (0, 1, 2, 3)}


def test_a_root_that_is_its_own_gpu_call_holds_no_lease_after_it() -> None:
    """A direct serving root encodes and sends its outputs after its device exit: nothing of
    it holds a GPU then."""
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.open_root("B", 2)
    gpus.want("S#1", Want(root="S", widths=(4,)))
    gpus.want("b#1", Want(root="B", widths=(1,)))
    assert shell.held() == {"S#1": (0, 1, 2, 3)}
    gpus.release("S#1")
    assert "S" not in shell.leases() and shell.held() == {"b#1": (0,)}


def test_a_group_turn_keeps_singletons_off_its_gpus() -> None:
    shell = Shell(4)
    gpus = shell.scheduler
    gpus.open_root("H", 1)
    gpus.open_root("Q", 2)
    gpus.want("h3#1", Want(root="H", widths=(4,)))
    for n in range(3):
        gpus.want(f"q{n}#1", Want(root="Q", widths=(1,)))
    assert shell.held() == {"h3#1": (0, 1, 2, 3)}
    gpus.release("h3#1")
    gpus.close_root("H")
    assert shell.held() == {"q0#1": (0,), "q1#1": (1,), "q2#1": (2,)}


# --------------------------------------------------------------------------- stage turns


def record(book: CostBook, template: tuple[str, str], *stages: tuple[str, int, float]) -> None:
    """One successful call of `template`'s `generate` at cell `c`: (method, calls, wall)."""
    model = f"{template[0]}/generate"
    samples = [
        StageSample(kind=StageKind(model, method, (method,)), wall_ns=int(wall * 1e9))
        for method, calls, wall in stages
        for _ in range(calls)
    ]
    book.record(model, "c", samples)


def sdxl_book(path: Path) -> Path:
    book = CostBook()
    record(book, SDXL, ("encode", 2, 0.34), ("denoise", 3, 0.6), ("decode", 1, 0.13))
    record(book, ANIMA, ("render", 1, 50.0))
    book.save(path)
    return path


def ask(shell: Shell, key: str, method: str) -> tuple[threading.Thread, list[Grant | None]]:
    """`key`'s executor enters `method`'s scope on its own thread, as a seam handler does."""
    got: list[Grant | None] = []
    kind = shell.scheduler.kind(key, method, (method,))
    asking = threading.Thread(
        target=lambda: got.append(shell.scheduler.enter(key, kind, lambda: False)),
        daemon=True,  # a failed assertion must not leave the run waiting on it
    )
    asking.start()
    until(lambda: bool(got) or bool(shell.demand(key)["asking"]))
    return asking, got


def run(shell: Shell, key: str, method: str) -> Exit:
    """One whole scope of `key`: its turn, then its exit."""
    asking, got = ask(shell, key, method)
    asking.join(30)
    assert got and got[0] is not None
    kind = shell.scheduler.kind(key, method, (method,))
    return shell.scheduler.exit(key, StageSample(kind=kind, wall_ns=1))


def test_the_next_model_prepares_while_the_first_runs_its_stages_then_takes_the_gpu(
    tmp_path: Path,
) -> None:
    """The alternating benchmark's first two jobs on one 8 GB GPU: Anima is queued behind
    SDXL. Anima's call is dispatched at once (its executor starts and its construction loads
    beside SDXL's turns), and its first stage waits until SDXL is past its last observed
    stage; SDXL's CPU tail holds nothing."""
    shell = Shell(1, turns=True, book=sdxl_book(tmp_path / "stage-costs.json"), total=8 * GiB)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.open_root("A", 2)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate", cell="c"))
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate", cell="c"))
    assert gpus.ready("S#1") and gpus.ready("A#1")
    assert [body["key"] for _root, body in shell.kinds("stage.prepare")] == ["S#1", "A#1"]
    assert shell.poked == ["S#1", "A#1"]
    assert gpus.taking_turns("A#1")

    assert not run(shell, "S#1", "encode").past
    anima, got = ask(shell, "A#1", "render")  # Anima is ready before SDXL denoises
    assert shell.waiting() == {"A#1": ["S"]}
    assert not run(shell, "S#1", "encode").past
    for _ in range(3):
        assert not run(shell, "S#1", "denoise").past
        assert not shell.demand("A#1")["running"], "no bubble in an unmeasured gap"
    left = run(shell, "S#1", "decode")  # past its last stage, with Anima wanting the GPU
    assert left.past and left.claim is not None and left.claim.template == ANIMA
    # The turn stays SDXL's while its executor unmaps: Anima must not measure the GPU yet
    # (run 2521 was given the room left beside 5.8 GB still mapped, and streamed its UNet).
    assert shell.held() == {"S#1": (0,)} and not got
    gpus.yielded("S#1")
    anima.join(30)
    assert got == [Grant((0,), True)]  # Anima's first turn: a budget of its own
    assert shell.held() == {"A#1": (0,)}


def test_a_call_whose_executor_holds_weights_waits_outside_it_until_they_are_cut() -> None:
    """A call waiting inside its executor cannot be asked for its device bytes. Anima's
    executor still holds its weights from its last call, so its next call is not prepared
    ahead of SDXL's turn: its executor stays idle, where SDXL can cut it. Once it holds
    nothing the call prepares ahead."""
    shell = Shell(1, turns=True)
    shell.holding = {ANIMA}
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.open_root("A", 2)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate"))
    assert gpus.ready("S#1") and not gpus.ready("A#1")
    assert [body["key"] for _root, body in shell.kinds("stage.prepare")] == ["S#1"]
    shell.holding.clear()  # SDXL's first turn cut it
    gpus.resync()
    assert gpus.ready("A#1") and gpus.taking_turns("A#1")


def test_a_turn_belongs_to_the_calls_outermost_open_scope() -> None:
    """H3 turbo opens its LoRA sampler's scope inside its base model's. The inner scope asks
    and is answered at once, and its exit ends nothing: the call keeps its GPU until the outer
    scope closes, and only then does the next call's turn come. The book learns the outer
    scope as the stage."""
    shell = Shell(1, turns=True)
    gpus = shell.scheduler
    gpus.open_root("H", 1)
    gpus.open_root("S", 2)
    gpus.want("H#1", Want(root="H", template=ANIMA, entrypoint="generate"))
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    outer, got = ask(shell, "H#1", "sample_turbo")
    outer.join(30)
    assert got == [Grant((0,), True)]
    inner = gpus.kind("H#1", "sample", ("lora",))
    assert gpus.enter("H#1", inner, lambda: False) == Grant((0,), False)
    waiting, granted = ask(shell, "S#1", "encode")
    assert gpus.exit("H#1", StageSample(kind=inner, wall_ns=1)) == Exit()
    assert shell.held() == {"H#1": (0,)} and not granted, "the outer scope is still open"
    gpus.exit("H#1", StageSample(kind=gpus.kind("H#1", "sample_turbo", ("sample_turbo",))))
    assert shell.held() == {}, "the outer scope's exit ends the turn"
    gpus.release("H#1", ok=True, cell="c")  # a first graph keeps its claim to its release
    waiting.join(30)
    assert granted == [Grant((0,), True)]
    graph = gpus.book.graph(f"{ANIMA[0]}/generate", "c")
    assert graph is not None
    assert [(stage.kind.method, stage.calls) for stage in graph.stages] == [("sample_turbo", 1)]


def test_a_turn_on_the_same_tenant_keeps_its_budget() -> None:
    shell = Shell(1, turns=True)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    first, got = ask(shell, "S#1", "denoise")
    first.join(30)
    assert got == [Grant((0,), True)]
    kind = gpus.kind("S#1", "denoise", ("denoise",))
    gpus.exit("S#1", StageSample(kind=kind, wall_ns=1))
    again, got = ask(shell, "S#1", "denoise")
    again.join(30)
    assert got == [Grant((0,), False)], "a parked denoise stage keeps its plan"


def test_a_whole_turn_given_before_its_executor_was_known_is_given_back_at_entry() -> None:
    """A restarted machine's first call wants the GPU before any executor said `stage/1`: it
    is dispatched with a whole turn. At its device entry its executor takes stage turns, so it
    gives the whole turn back, and the next model prepares beside it."""
    shell = Shell(1)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.open_root("S", 2)
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate"))
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    assert shell.held() == {"A#1": (0,)} and not gpus.ready("S#1")
    shell.turns = True  # the prespawned executors' hellos arrive
    assert gpus.turn_taking("A#1")
    assert shell.held() == {} and gpus.ready("S#1"), "SDXL prepares while Anima runs"
    assert gpus.view()["waiting"] == {}


def test_one_executor_serves_one_call_at_a_time_and_one_more_prepares_per_gpu() -> None:
    shell = Shell(1, turns=True)
    gpus = shell.scheduler
    for n, root in enumerate(("S1", "A1", "S2", "Q1"), start=1):
        gpus.open_root(root, n)
    gpus.want("S1#1", Want(root="S1", template=SDXL, entrypoint="generate"))
    gpus.want("A1#1", Want(root="A1", template=ANIMA, entrypoint="generate"))
    gpus.want("S2#1", Want(root="S2", template=SDXL, entrypoint="generate"))
    gpus.want("Q1#1", Want(root="Q1", template=("inst-qwen", "q"), entrypoint="generate"))
    # S2 waits for SDXL's executor; Q1 waits for a seat on GPU 0 (the holder and the next).
    assert [gpus.ready(k) for k in ("S1#1", "A1#1", "S2#1", "Q1#1")] == [
        True,
        True,
        False,
        False,
    ]
    gpus.release("S1#1")
    assert gpus.ready("S2#1") and not gpus.ready("Q1#1")


def test_a_device_job_runs_alone_after_the_calls_already_preparing_leave() -> None:
    shell = Shell(2, turns=True)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.open_root("J", 2)
    gpus.open_root("A", 3)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    assert gpus.ready("S#1")
    gpus.want("J#1", Want(root="J", widths=(2,), exclusive=True))
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate"))
    assert not gpus.ready("J#1") and not gpus.ready("A#1"), "nothing new starts meanwhile"
    gpus.release("S#1")
    assert gpus.ready("J#1") and shell.held() == {"J#1": (0, 1)}
    gpus.release("J#1")
    assert gpus.ready("A#1")


def test_a_maintenance_hold_waits_for_the_running_turn_and_goes_before_the_next() -> None:
    shell = Shell(1, turns=True)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.open_root("A", 2)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate"))
    turn, got = ask(shell, "S#1", "encode")
    turn.join(30)
    held = threading.Event()
    release = threading.Event()

    def rebuild() -> None:
        with gpus.hold((0,), "rebuild"):
            held.set()
            release.wait(30)

    rebuilding = threading.Thread(target=rebuild)
    rebuilding.start()
    until(lambda: any(key.startswith("hold:") for key in gpus.view()["demands"]))
    assert not held.is_set()
    anima, got = ask(shell, "A#1", "render")
    kind = gpus.kind("S#1", "encode", ("encode",))
    gpus.exit("S#1", StageSample(kind=kind, wall_ns=1))
    until(held.is_set)
    assert not shell.demand("A#1")["running"]
    release.set()
    rebuilding.join(30)
    # SDXL is earlier than Anima and still in its gap: Anima waits for it, not the rebuild.
    assert shell.waiting() == {"A#1": ["S"]}
    gpus.release("S#1")
    anima.join(30)
    assert got and got[0] is not None


def test_a_cancel_stops_a_scope_waiting_for_its_turn() -> None:
    shell = Shell(1, turns=True)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.open_root("A", 2)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate"))
    run(shell, "S#1", "encode")
    cancelled = threading.Event()
    got: list[Grant | None] = []
    kind = gpus.kind("A#1", "render", ("render",))
    waiter = threading.Thread(target=lambda: got.append(gpus.enter("A#1", kind, cancelled.is_set)))
    waiter.start()
    until(lambda: bool(shell.demand("A#1")["asking"]))
    cancelled.set()
    gpus.wake()
    waiter.join(30)
    assert got == [None] and not shell.demand("A#1")["asking"]


# --------------------------------------------------------------------------- placement


def test_placement_takes_the_earliest_predicted_finish_and_runs_jobs_on_both_gpus(
    tmp_path: Path,
) -> None:
    """Anima on GPU 0 for another 50 s: a measured SDXL call runs now on GPU 1 rather than
    wait. With an SDXL executor warm on GPU 0 and a 60 s executor start anywhere else, the
    second SDXL call waits for GPU 0 instead."""
    book = sdxl_book(tmp_path / "stage-costs.json")
    shell = Shell(2, book=book)
    gpus = shell.scheduler
    gpus.open_root("A", 1)
    gpus.open_root("S", 2)
    gpus.want("A#1", Want(root="A", template=ANIMA, entrypoint="generate", cell="c"))
    assert shell.held() == {"A#1": (0,)}
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate", cell="c"))
    assert shell.held() == {"A#1": (0,), "S#1": (1,)}, "concurrent jobs on different GPUs"

    shell = Shell(2, book=book, warm={SDXL: ((0,),)}, spawn_s=60.0)
    gpus = shell.scheduler
    gpus.open_root("S1", 1)
    gpus.open_root("S2", 2)
    gpus.want("S1#1", Want(root="S1", template=SDXL, entrypoint="generate", cell="c"))
    assert shell.held() == {"S1#1": (0,)}
    gpus.want("S2#1", Want(root="S2", template=SDXL, entrypoint="generate", cell="c"))
    assert shell.held() == {"S1#1": (0,)} and shell.waiting() == {"S2#1": ["S1"]}
    gpus.release("S1#1")
    assert shell.held() == {"S2#1": (0,)}


def test_a_call_whose_finish_is_unknown_waits_unplaced_while_every_set_is_busy() -> None:
    shell = Shell(2)
    gpus = shell.scheduler
    for n, root in enumerate("ABC", start=1):
        gpus.open_root(root, n)
    gpus.want("A#1", Want(root="A"))
    gpus.want("B#1", Want(root="B"))
    gpus.want("C#1", Want(root="C"))
    assert gpus.planned("C#1") is None, "not stuck behind a guess"
    gpus.release("B#1")
    assert shell.held() == {"A#1": (0,), "C#1": (1,)}


def test_an_owner_placement_goes_where_a_call_of_it_would_run() -> None:
    shell = Shell(3, warm={SDXL: ((1,),)})
    assert shell.scheduler.choose((0, 1, 2), 1, SDXL) == (1,)
    assert shell.scheduler.choose((0, 2), 1, SDXL) == (0,)
    assert shell.scheduler.choose((0, 1, 2), 2, ANIMA) == (0, 1)
    assert shell.scheduler.choose((2,), 2, ANIMA) == ()


# --------------------------------------------------------------------------- admission and the book


def test_intake_refuses_only_below_a_measured_floor_with_numbers(tmp_path: Path) -> None:
    unet = WeightSet("sdxl#unet", "unet", 6 * GiB, (GiB,) * 4)
    shell = Shell(1, sets={SDXL: {"denoise": unet}}, total=8 * GiB)
    want = Want(root="S", template=SDXL, entrypoint="generate")
    assert shell.scheduler.refuse(want) is None  # streams through an 8 GB GPU
    shell.gpus[0] = Gpu(ordinal=0, total=4 * GiB)
    refused = shell.scheduler.refuse(want)
    assert refused is not None and refused.code == "NO_CAPACITY"
    assert (refused.need, refused.capacity) == (7 * GiB, 4 * GiB)
    assert str(7 * GiB) in refused.detail and "GPU 0" in refused.detail
    assert shell.scheduler.refuse(Want(root="S", widths=(2,))) is not None  # one GPU here


def test_the_book_learns_each_call_and_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "stage-costs.json"
    shell = Shell(1, turns=True, book=path)
    gpus = shell.scheduler
    gpus.open_root("S", 1)
    gpus.want("S#1", Want(root="S", template=SDXL, entrypoint="generate"))
    for method in ("encode", "encode", "denoise", "denoise", "decode"):
        run(shell, "S#1", method)
    gpus.release("S#1", ok=True, cell="height=1024,width=1024")
    book = CostBook.load(path)
    graph = book.graph("inst-sdxl/generate", "height=1024,width=1024")
    assert graph is not None and [(s.kind.method, s.calls) for s in graph.stages] == [
        ("encode", 2),
        ("denoise", 2),
        ("decode", 1),
    ]
    exits = [body["stages"] for _root, body in shell.kinds("stage.exit")]
    assert [len(stages) for stages in exits if isinstance(stages, list)] == [5]
    # A restarted Worker places the next such call by what was learned.
    again = Shell(1, turns=True, book=path)
    again.scheduler.open_root("S", 1)
    again.scheduler.want(
        "S#2", Want(root="S", template=SDXL, entrypoint="generate", cell="height=1024,width=1024")
    )
    assert again.demand("S#2")["dispatched"]


@pytest.mark.parametrize("ok", [True, False])
def test_a_call_that_takes_no_stage_turns_teaches_one_opaque_stage(
    tmp_path: Path, ok: bool
) -> None:
    path = tmp_path / "stage-costs.json"
    shell = Shell(1, book=path)
    shell.scheduler.open_root("H", 1)
    shell.scheduler.want("H#1", Want(root="H", template=SDXL, entrypoint="generate", cell="c"))
    assert shell.held() == {"H#1": (0,)}
    shell.scheduler.release("H#1", ok=ok)
    book = CostBook.load(path)
    cost = book.cost(StageKind("inst-sdxl/generate", "*", ()), "c", 1)
    assert cost is not None and (cost.runs == 1) is ok
