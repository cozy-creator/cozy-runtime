"""The stage scheduler: who runs where, one turn at a time per GPU (weight-plane.md §5).

A DEMAND is one GPU call of an execution attempt (`request#attempt`), under its scheduling
root, whose acceptance rowid is its priority. Each change runs one pass of `stage_policy`:

- PLACEMENT. A demand gets its GPU set, width included, once: the earliest predicted finish,
  from the cost book's measured stages, what is resident and pinned, and warm executors.
- TURNS. A GPU runs one turn at a time. An executor that speaks `stage/1` on one GPU asks for a
  turn at every component-use scope (`enter`/`exit`); any other (an older Runtime, a Ulysses
  group, a device job, a rebuild) holds one turn from its first device use to its release.
- PREPARE. A turn-taking demand that is next on its GPU after the running one is dispatched at
  once: its executor starts and its construction loads into the host tier while another job
  holds the GPU, and its first stage waits for its turn. One executor serves one demand at a
  time, so a second demand of the same executor waits for the first to leave.
- LEASES. A root keeps the GPUs its calls used across the gaps between them, until it ends or
  says it is done with GPUs (`release_root`); no younger root slips in between its calls.

Observations (`gpu.grant`/`gpu.wait`/`gpu.release`, `stage.turn`, `stage.exit`) are journaled
on the roots they concern after each pass, outside the policy lock.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.internal.stages import (
    CostBook,
    StageCost,
    StageGraph,
    StageKind,
    StageSample,
    WeightSet,
    static_graph,
)
from cozy_runtime.internal.worker import stage_policy as policy
from cozy_runtime.internal.worker.supervisor import fault_text

Ordinals = tuple[int, ...]
#: A JSON-shaped fact: a journaled scheduling event's body or a status document.
Document = dict[str, Any]
#: (root, event kind, body): one observation to journal on its root.
Event = tuple[str, str, Document]
#: Live demands per GPU: the one holding it and the next one, getting ready.
SEATS = 2
#: Maintenance holds (a rebuild, an unload) go before every demand.
MAINTENANCE = -1
#: Work with no durable root (an owner-placed placement's attempt) goes after every root.
UNROOTED = 1 << 62


class Want(msgspec.Struct, frozen=True, kw_only=True):
    """One demand: what its call asks of the GPUs."""

    root: str
    #: GPU counts it may run at: declared degrees the machine forms, or the owner's exact one
    widths: tuple[int, ...] = (1,)
    template: tuple[str, str] = ("", "")
    #: GPUs it can never use (unreadable)
    exclude: Ordinals = ()
    entrypoint: str = ""
    cell: str = ""
    #: a device job: every GPU, and nothing else live while it runs
    exclusive: bool = False
    #: already bound to exactly these GPUs (a placement the owner sent)
    gpus: Ordinals = ()


class Grant(msgspec.Struct, frozen=True):
    """A stage turn. `start`: the tenant needs a budget (its first turn, or after it was idle)."""

    gpus: Ordinals
    start: bool


@dataclass
class _Demand:
    key: str
    want: Want
    arrival: int
    stages: list[policy.JobStage]
    graph: StageGraph
    gpus: Ordinals = ()
    #: the unit may dispatch: it holds a whole turn, or prepares ahead of its first stage turn
    dispatched: bool = False
    #: holds one turn from its first device use to its release
    whole: bool = False
    asking: bool = False
    running: bool = False
    bubble: bool = False
    at: int = 0
    calls: int = 0
    since: float = field(default_factory=time.monotonic)
    granted: bool = False
    #: its tenant was given a budget and has not been idled since
    tenant: bool = False
    samples: list[StageSample] = field(default_factory=list)
    last_exit: float = 0.0
    refusal: policy.Refusal | None = None
    waiting: tuple[int, str] | None = None
    #: the thread that holds its whole turn, for re-entry
    owner: int = 0
    started_at: float = 0.0
    #: when it last asked for a stage turn: the CPU gap ends there, the wait begins
    asked: float = 0.0
    #: the turn it asks for or holds is one of its call's stages (a load's warm is not)
    counted: bool = True


class StageScheduler:
    """The policy, one pass per change, over every live demand and root lease."""

    def __init__(
        self,
        devices: int,
        *,
        poke: Callable[[str], None],
        observed: Callable[[list[Event]], None],
        machine: Callable[[], policy.Machine],
        warm: Callable[[tuple[str, str]], tuple[Ordinals, ...]],
        sets: Callable[[Want], Mapping[str, WeightSet]],
        turns: Callable[[str], bool],
        prefill: Callable[[str, tuple[str, str]], None],
        fault: Callable[[str], None],
        note: Callable[[str], None],
        book: Path | None = None,
    ) -> None:
        self.devices = devices
        self.poke, self.observed, self.fault, self.note = poke, observed, fault, note
        #: the machine's facts now (GPUs, residency, host tier)
        self.machine = machine
        #: GPU sets where a template's executor is live
        self.warm = warm
        #: a call's weight sets as its template's executor registered them, by component
        self.sets = sets
        #: whether an installation's executor takes stage turns (`stage/1`)
        self.turns = turns
        #: fill a waiting demand's host tier, off the pass
        self.prefill = prefill
        self.book_path = book
        self.book = CostBook.load(book) if book is not None else CostBook()
        self.lock = threading.Condition()
        self.changes = threading.Lock()
        self.roots: dict[str, tuple[int, int]] = {}
        self.leases: dict[str, set[int]] = {}
        self.demands: dict[str, _Demand] = {}
        #: GPU -> the demand holding its turn
        self.holder: dict[int, str] = {}
        self.events: list[Event] = []
        self.prefilled: set[str] = set()
        #: roots that said they are done with GPUs: their calls add to no lease until the next
        self.released: set[str] = set()
        #: the book changed since it was saved
        self.dirty = False
        self._arrival = itertools.count()
        self._holds = itertools.count()

    # ------------------------------------------------------------------ roots and demands

    def open_root(self, root: str, priority: int) -> None:
        with self.lock:
            if self.roots.get(root, (None,))[0] == priority:
                return
            self.roots[root] = (priority, next(self._arrival))
        self.resync()

    def close_root(self, root: str) -> None:
        with self.lock:
            if self.roots.pop(root, None) is None:
                return
            self.leases.pop(root, None)
        self.resync()

    def release_root(self, root: str) -> None:
        """`root` is done with GPUs: its lease ends now. A call still on its GPUs keeps them to
        its release; its next call waits at its priority like any other. Idempotent."""
        with self.lock:
            self.leases.pop(root, None)
            self.released.add(root)
        self.resync()

    def want(self, key: str, want: Want) -> None:
        """A call asks for GPUs: it is placed on the next pass."""
        with self.lock:
            held = self.demands.get(key)
            if held is not None and (held.want == want or held.dispatched):
                return
            graph = self._graph(want)
            self.demands[key] = _Demand(
                key, want, next(self._arrival), self._stages(want, graph), graph, gpus=want.gpus
            )
            self.released.discard(want.root)
        self.resync()

    def known(self, key: str) -> bool:
        with self.lock:
            return key in self.demands

    def refuse(self, want: Want) -> policy.Refusal | None:
        """Whether a call like `want` could never run here: some stage's measured floor exceeds
        every GPU at every width it may run at. Pure: nothing is queued."""
        with self.lock:
            probe = _Demand(
                "intake", want, 0, self._stages(want, self._graph(want)), StageGraph("", "", ())
            )
            machine = self.machine()
            job = self._job(probe, {g.ordinal for g in machine.gpus}, 0.0)
        return policy.refusal(job, machine)

    def ready(self, key: str) -> bool:
        """The unit may dispatch: it holds its whole turn, or prepares ahead of its first."""
        with self.lock:
            demand = self.demands.get(key)
            return demand is not None and demand.dispatched

    def planned(self, key: str) -> Ordinals | None:
        """Where the demand is placed, dispatched or not; None while it waits unplaced."""
        with self.lock:
            demand = self.demands.get(key)
            return demand.gpus if demand is not None and demand.gpus else None

    def placed(self, key: str) -> Ordinals | None:
        with self.lock:
            demand = self.demands.get(key)
            return demand.gpus if demand is not None and demand.dispatched else None

    def refused(self, key: str) -> policy.Refusal | None:
        with self.lock:
            demand = self.demands.get(key)
            return demand.refusal if demand is not None else None

    def taking_turns(self, key: str) -> bool:
        """Whether this demand runs stage by stage (it holds no whole turn)."""
        with self.lock:
            demand = self.demands.get(key)
            return demand is not None and demand.dispatched and not demand.whole

    # ------------------------------------------------------------------ whole turns

    def take(self, key: str, stop: Callable[[], bool] = lambda: False) -> bool:
        """Hold one turn on the demand's GPUs until its release: an executor that takes no stage
        turns. Re-entrant. False when `stop()` came first."""
        with self.lock:
            demand = self.demands.get(key)
            if demand is None or not demand.gpus:
                return True
            if demand.whole and demand.running:
                demand.owner = threading.get_ident()
                return True
            demand.whole, demand.asking = True, True
        self.resync()
        with self.lock:
            self.lock.wait_for(lambda: self._running(key) or stop())
            demand = self.demands.get(key)
            if demand is None or not demand.running:
                if demand is not None:
                    demand.asking = False
                return False
            demand.owner = threading.get_ident()
            return True

    @contextmanager
    def hold(self, ordinals: Ordinals, why: str) -> Iterator[None]:
        """A maintenance turn (a rebuild, a replacement, an unload) on `ordinals`, before every
        demand; a no-op on a thread whose own whole turn already covers them."""
        me = threading.get_ident()
        with self.lock:
            covered = any(
                d.whole and d.running and d.owner == me and set(ordinals) <= set(d.gpus)
                for d in self.demands.values()
            )
        if covered or not ordinals:
            yield
            return
        key = f"hold:{next(self._holds)}:{why}"[:120]
        with self.lock:
            self.demands[key] = _Demand(
                key,
                Want(root=key, widths=(len(ordinals),)),
                next(self._arrival),
                [],
                StageGraph("", "", ()),
                gpus=tuple(sorted(ordinals)),
                dispatched=True,
                whole=True,
                asking=True,
            )
        self.resync()
        try:
            with self.lock:
                self.lock.wait_for(lambda: self._running(key))
                self.demands[key].owner = me
            yield
        finally:
            with self.lock:
                self.demands.pop(key, None)
                self._free(key)
            self.resync()

    def quiet(self, ordinals: Ordinals) -> bool:
        """No whole turn runs on `ordinals`: an executor that recovers from an out-of-memory at a
        block boundary holds them, if anyone does, so a new device context may start beside."""
        with self.lock:
            return not any(
                (d := self.demands.get(self.holder[o])) is not None and d.whole
                for o in ordinals
                if o in self.holder
            )

    # ------------------------------------------------------------------ stage turns

    def enter(
        self,
        key: str,
        kind: StageKind,
        stop: Callable[[], bool],
        *,
        counted: bool = True,
        cell: str = "",
    ) -> Grant | None:
        """A `stage/1` executor's scope asks for its turn; blocks until granted. None when
        `stop()` came first (a cancel, a dead executor). An uncounted scope (a load's warm)
        takes a turn the call's graph and book never see. `cell`, once the request is
        resolved, refines the graph the call was placed by."""
        with self.lock:
            demand = self.demands.get(key)
            if demand is None or not demand.gpus:
                return None
            if demand.whole and demand.running:
                return Grant(demand.gpus, False)  # inside its own whole turn
            if cell and not demand.want.cell:
                self._refine(demand, cell)
            if counted:
                self._locate(demand, kind)
            demand.counted = counted
            demand.asking = True
            demand.asked = time.monotonic()
        self.resync()
        with self.lock:
            self.lock.wait_for(lambda: self._running(key) or stop())
            demand = self.demands.get(key)
            if demand is None or not demand.running:
                if demand is not None:
                    demand.asking = False
                return None
            start = not demand.tenant
            demand.tenant = True
            return Grant(demand.gpus, start)

    def exit(self, key: str, sample: StageSample, *, counted: bool = True) -> bool:
        """A scope ended: its turn ends and its sample is kept for the book. True when the
        demand is past its last observed stage, so its tenant may be idled."""
        with self.lock:
            demand = self.demands.get(key)
            if demand is None:
                return False
            if demand.whole:
                if counted:
                    demand.samples.append(sample)
                return False
            if not counted:
                self._free(key)
                demand.since = time.monotonic()
                past = False
            else:
                past = self._advance(demand, sample)
        self.resync()
        return past

    def _advance(self, demand: _Demand, sample: StageSample) -> bool:
        """Its sample with the CPU gap before it, its position past this call, its turn over."""
        now = time.monotonic()
        gap = int((demand.asked - demand.last_exit) * 1e9) if demand.last_exit else 0
        demand.samples.append(msgspec.structs.replace(sample, gap_ns=max(gap, 0)))
        demand.last_exit = demand.since = now
        demand.calls += 1
        if demand.at < len(demand.stages) and demand.calls >= demand.stages[demand.at].calls:
            demand.at, demand.calls = demand.at + 1, 0
        self._free(demand.key)
        past = demand.at >= len(demand.stages)
        if past:
            demand.tenant = False
        return past

    def contended(self, key: str) -> bool:
        """Another call wants a GPU of this one's set: it asks, holds, prepares or waits there."""
        with self.lock:
            demand = self.demands.get(key)
            if demand is None:
                return False
            return any(
                other is not demand
                and set(other.gpus) & set(demand.gpus)
                and (other.asking or other.dispatched or other.waiting is not None)
                for other in self.demands.values()
            )

    def idle(self, key: str) -> None:
        """The demand's tenant lost its budget (an idle cut): its next turn needs a new one."""
        with self.lock:
            demand = self.demands.get(key)
            if demand is not None:
                demand.tenant = False

    def give(self, key: str) -> None:
        """End a stage turn granted but not used (its budget was refused)."""
        with self.lock:
            demand = self.demands.get(key)
            if demand is not None:
                demand.since = time.monotonic()
                demand.tenant = False
            self._free(key)
        self.resync()

    def release(self, key: str, *, ok: bool = False, cell: str = "", **facts: object) -> None:
        """The demand left its devices, or never reached them: it ends, and a successful run
        teaches the book its stages (under its request's shape `cell`)."""
        with self.lock:
            demand = self.demands.pop(key, None)
            if demand is None:
                return
            if cell and not demand.want.cell:
                demand.want = msgspec.structs.replace(demand.want, cell=cell)
            self._free(key)
            self.prefilled.discard(key)
            root = demand.want.root
            if demand.granted:
                body = {"key": key, "ordinals": list(demand.gpus), "cause": "exited", **facts}
                self.events.append((root, "gpu.release", body))
                self._learn(demand, ok)

        self.resync()

    def wake(self) -> None:
        """Something a waiter's `stop()` reads changed (a cancel, an executor exit)."""
        with self.lock:
            self.lock.notify_all()

    # ------------------------------------------------------------------ the pass

    def resync(self) -> None:
        with self.changes:
            try:
                fresh: list[str] = []
                prefills: list[tuple[str, tuple[str, str]]] = []
                with self.lock:
                    self._pass(fresh, prefills)
                    events, self.events = self.events, []
                    save, self.dirty = self.dirty, False
                    self.lock.notify_all()
                self.observed(events)
                if save and self.book_path is not None:
                    try:
                        self.book.save(self.book_path)
                    except OSError as exc:  # advisory: relearned, never refused
                        self.note(f"the stage cost book was not saved: {exc}"[:400])
            except Exception as exc:
                self.fault(f"the stage scheduling pass raised {fault_text(exc)}")
                raise
        for key in fresh:
            self.poke(key)
        for key, template in prefills:
            self.prefill(key, template)

    def _pass(self, fresh: list[str], prefills: list[tuple[str, tuple[str, str]]]) -> None:
        machine = self.machine()
        readable = {g.ordinal for g in machine.gpus}
        now = time.monotonic()
        jobs = [self._job(d, readable, now) for d in self.demands.values()]
        jobs += [
            policy.Job(
                key="lease:" + root,
                root=root,
                priority=self.roots[root][0],
                arrival=self.roots[root][1],
                phase="gap",
                gpus=tuple(sorted(held)),
                lease=True,
            )
            for root, held in self.leases.items()
            if held and root in self.roots
        ]
        plan = policy.decide(machine, jobs)
        alone = self._alone()
        for refusal in plan.refusals:
            demand = self.demands[refusal.job]
            if demand.refusal is None:
                demand.refusal = refusal
                fresh.append(demand.key)
        for prepare in plan.prepares:
            self.demands[prepare.job].gpus = self.demands[prepare.job].gpus or prepare.gpus
        for turn in plan.turns:
            demand = self.demands[turn.job]
            if not demand.dispatched and alone is not None and alone is not demand:
                continue  # a device job waits to run alone: nothing new starts meanwhile
            if demand.want.exclusive and self._others(demand):
                continue
            demand.gpus = demand.gpus or turn.gpus
            self._grant(demand, turn, now, fresh)
        if alone is None:
            self._dispatch(plan, fresh)
        waits = {wait.job: wait for wait in plan.waits}
        for demand in self.demands.values():
            wait = waits.get(demand.key)
            if wait is None or demand.running or demand.key.startswith("hold:"):
                demand.waiting = None
                continue
            width = len(demand.gpus) or max(demand.want.widths, default=1)
            if demand.waiting != (width, wait.blocked_by):
                demand.waiting = (width, wait.blocked_by)
                self.events.append(
                    (
                        demand.want.root,
                        "gpu.wait",
                        {
                            "key": demand.key,
                            "width": width,
                            "ordinals": list(demand.gpus),
                            "blocked_by": [wait.blocked_by or demand.want.root],
                        },
                    )
                )
        for prefill in plan.prefills:
            demand = self.demands.get(prefill.job)
            if demand is not None and prefill.job not in self.prefilled and demand.gpus:
                self.prefilled.add(prefill.job)
                prefills.append((prefill.job, demand.want.template))

    def _job(self, demand: _Demand, readable: set[int], now: float) -> policy.Job:
        want = demand.want
        root = self.roots.get(want.root)
        maintenance = demand.key.startswith("hold:")
        if demand.running:
            phase: policy.Phase = "running"
        elif demand.asking:
            phase = "ready"
        elif not demand.dispatched:
            phase = "preparing" if self._early(demand) else "ready"
        elif demand.last_exit:
            phase = "gap"
        else:
            phase = "preparing"
        allowed = {o for o in readable if o not in want.exclude}
        return policy.Job(
            key=demand.key,
            root=want.root,
            priority=MAINTENANCE if maintenance else root[0] if root else UNROOTED,
            arrival=demand.arrival,
            stages=tuple(demand.stages),
            phase=phase,
            at=demand.at,
            elapsed_s=now - demand.since,
            widths=tuple(w for w in want.widths if w <= len(allowed)) or want.widths,
            gpus=demand.gpus,
            warm=self.warm(want.template) if want.template[0] else (),
            exclude=want.exclude,
        )

    def _early(self, demand: _Demand) -> bool:
        """Whether this demand prepares ahead of its turn: its executor on its set takes stage
        turns. Unplaced, it asks about the one GPU a turn-taker would get."""
        want = demand.want
        if want.exclusive or not want.template[0] or demand.whole:
            return False
        one = len(demand.gpus) == 1 if demand.gpus else max(want.widths, default=1) == 1
        return one and self.turns(want.template[0])

    def _others(self, demand: _Demand) -> bool:
        return any(
            d is not demand and d.dispatched and not d.key.startswith("hold:")
            for d in self.demands.values()
        )

    def _alone(self) -> _Demand | None:
        """The first device job still waiting to run alone, if any."""
        waiting = [d for d in self.demands.values() if d.want.exclusive and not d.dispatched]
        return min(waiting, key=lambda d: d.arrival) if waiting else None

    def _grant(self, demand: _Demand, turn: policy.Turn, now: float, fresh: list[str]) -> None:
        demand.running, demand.asking, demand.bubble = True, False, turn.bubble
        demand.since = now
        for o in demand.gpus:
            self.holder[o] = demand.key
        if demand.key.startswith("hold:"):
            return
        root = demand.want.root
        if root in self.roots and root not in self.released and _child(demand.key, root):
            self.leases.setdefault(root, set()).update(demand.gpus)
        first = not demand.granted
        if first:
            demand.granted = True
            demand.started_at = now
            self.events.append(
                (root, "gpu.grant", {"key": demand.key, "ordinals": list(demand.gpus)})
            )
        if not demand.dispatched:
            demand.dispatched = demand.whole = True  # it holds this turn to its release
            fresh.append(demand.key)
        if first or turn.bubble:
            stage = demand.stages[demand.at].kind if demand.at < len(demand.stages) else None
            self.events.append(
                (
                    root,
                    "stage.turn",
                    {
                        "key": demand.key,
                        "ordinals": list(demand.gpus),
                        "stage": stage.method if stage else "",
                        "bubble": turn.bubble,
                        "whole": demand.whole,
                    },
                )
            )

    def _dispatch(self, plan: policy.Plan, fresh: list[str]) -> None:
        """Prepare, in order, the next demand on each GPU whose executor is free: it starts
        and loads while the GPU's turn holder runs, and asks for its first turn when ready."""
        if any(
            d.want.exclusive and not d.dispatched for d in self.demands.values()
        ):  # a waiting device job lets nothing else get ready
            return
        live: dict[int, int] = {}
        engaged: set[tuple[tuple[str, str], Ordinals]] = set()
        for demand in self.demands.values():
            if demand.dispatched and not demand.key.startswith("hold:"):
                engaged.add((demand.want.template, demand.gpus))
                for o in demand.gpus:
                    live[o] = live.get(o, 0) + 1
        for prepare in plan.prepares:
            demand = self.demands[prepare.job]
            if demand.dispatched or not self._early(demand):
                continue
            seat = (demand.want.template, demand.gpus)
            if seat in engaged or any(live.get(o, 0) >= SEATS for o in demand.gpus):
                continue
            demand.dispatched = True
            engaged.add(seat)
            for o in demand.gpus:
                live[o] = live.get(o, 0) + 1
            fresh.append(demand.key)

    def _running(self, key: str) -> bool:
        demand = self.demands.get(key)
        return demand is not None and demand.running

    def _free(self, key: str) -> None:
        demand = self.demands.get(key)
        if demand is not None:
            demand.running = demand.asking = False
        for o in [o for o, k in self.holder.items() if k == key]:
            del self.holder[o]

    # ------------------------------------------------------------------ stages and costs

    def _graph(self, want: Want) -> StageGraph:
        model = _model(want)
        graph = self.book.graph(model, want.cell)
        return graph if graph is not None else static_graph(model, want.cell, model, ())

    def _stages(self, want: Want, graph: StageGraph) -> list[policy.JobStage]:
        sets = self.sets(want) if want.template[0] else {}
        return [
            policy.JobStage(
                kind=stage.kind,
                sets=tuple(sets[c] for c in (stage.kind.components or tuple(sets)) if c in sets),
                calls=stage.calls,
                cost=self._costs(want, stage.kind),
            )
            for stage in graph.stages
        ]

    def _refine(self, demand: _Demand, cell: str) -> None:
        """The request's shape is known: its graph and costs are this cell's, where measured,
        at the same position."""
        demand.want = msgspec.structs.replace(demand.want, cell=cell)
        graph = self.book.graph(_model(demand.want), cell)
        if graph is None or demand.samples or demand.at:
            demand.stages = [
                msgspec.structs.replace(stage, cost=self._costs(demand.want, stage.kind))
                for stage in demand.stages
            ]
            return
        demand.graph = graph
        demand.stages = self._stages(demand.want, graph)

    def _costs(self, want: Want, kind: StageKind) -> dict[int, StageCost]:
        return {
            w: cost for w in want.widths if (cost := self.book.cost(kind, want.cell, w)) is not None
        }

    def _locate(self, demand: _Demand, kind: StageKind) -> None:
        """Point the demand at the stage its executor entered: the next observed one of that
        kind, or a stage the graph did not predict, appended where it runs."""
        if demand.at < len(demand.stages) and demand.stages[demand.at].kind == kind:
            return
        for i in range(demand.at, len(demand.stages)):
            if demand.stages[i].kind == kind:
                demand.at, demand.calls = i, 0
                return
        if demand.stages and demand.stages[0].kind.method == "*" and not demand.samples:
            demand.stages = []  # the opaque stage: its first scope replaces it
        sets = self.sets(demand.want) if demand.want.template[0] else {}
        demand.stages.insert(
            demand.at,
            policy.JobStage(kind=kind, sets=tuple(sets[c] for c in kind.components if c in sets)),
        )
        demand.calls = 0

    def kind(self, key: str, method: str, components: Sequence[str]) -> StageKind:
        with self.lock:
            demand = self.demands.get(key)
            model = _model(demand.want) if demand is not None else ""
        return StageKind(model, method, tuple(components))

    def _learn(self, demand: _Demand, ok: bool) -> None:
        """A finished call's samples into the book; a call that took no stage turns is one
        opaque stage, its whole turn. Advisory: a write that fails is noted, never raised."""
        want = demand.want
        if not want.template[0]:
            return
        samples = demand.samples
        width = len(demand.gpus)
        if demand.whole and not samples and demand.started_at:
            wall = int((time.monotonic() - demand.started_at) * 1e9)
            kind = demand.stages[0].kind if demand.stages else StageKind(_model(want), "*", ())
            samples = [StageSample(kind=kind, width=width, wall_ns=wall, ok=ok)]
        samples = [msgspec.structs.replace(s, width=width) for s in samples]
        if not ok:
            samples = [msgspec.structs.replace(s, ok=False) for s in samples]
        if samples:
            self.book.record(_model(want), want.cell, samples)
            self.events.append(
                (
                    want.root,
                    "stage.exit",
                    {
                        "key": demand.key,
                        "ok": ok,
                        "stages": [
                            {
                                "method": s.kind.method,
                                "components": list(s.kind.components),
                                "wall_ms": round(s.wall_ns / 1e6, 3),
                                "gap_ms": round(s.gap_ns / 1e6, 3),
                                "growth_bytes": s.growth_bytes,
                            }
                            for s in samples
                        ],
                    },
                )
            )
            self.dirty = True

    # ------------------------------------------------------------------ reading

    def status(self, request: str, on_device: Callable[[str], bool]) -> Document:
        """One execution and, for a root, its calls: waiting, executing, granted, holding."""
        with self.lock:

            def mine(demand: _Demand) -> bool:
                return demand.want.root == request or demand.key.rpartition("#")[0] == request

            status: Document = {"phase": "none", "width": 0, "ordinals": [], "blocked_by": []}
            for demand in self.demands.values():
                if mine(demand) and demand.waiting is not None:
                    return {
                        **status,
                        "phase": "waiting",
                        "width": demand.waiting[0],
                        "ordinals": list(demand.gpus),
                        "blocked_by": [demand.waiting[1]] if demand.waiting[1] else [],
                    }
            granted = [d for d in self.demands.values() if mine(d) and d.granted]
            if granted:
                return {
                    **status,
                    "phase": "executing" if any(on_device(d.key) for d in granted) else "granted",
                    "width": max(len(d.gpus) for d in granted),
                    "ordinals": sorted({o for d in granted for o in d.gpus}),
                }
            if self.leases.get(request):
                return {**status, "phase": "holding", "ordinals": sorted(self.leases[request])}
            return status

    def view(self) -> Document:
        with self.lock:
            return {
                "leases": {root: sorted(held) for root, held in self.leases.items()},
                "holders": {str(o): key for o, key in sorted(self.holder.items())},
                "demands": {
                    key: {"gpus": list(d.gpus), "dispatched": d.dispatched, "whole": d.whole}
                    for key, d in self.demands.items()
                },
            }

    def predict(self, want: Want) -> Ordinals | None:
        """Where a call like `want` would be placed now, for a start ahead of its submission."""
        with self.lock:
            machine = self.machine()
            readable = {g.ordinal for g in machine.gpus}
            probe = _Demand("probe", want, next(self._arrival), [], StageGraph("", "", ()))
            jobs = [self._job(d, readable, time.monotonic()) for d in self.demands.values()]
            jobs.append(msgspec.structs.replace(self._job(probe, readable, 0.0), phase="ready"))
            plan = policy.decide(machine, jobs)
        for turn in plan.turns:
            if turn.job == "probe":
                return turn.gpus
        return next((p.gpus for p in plan.prepares if p.job == "probe"), None)

    def choose(self, candidates: Ordinals, width: int, template: tuple[str, str]) -> Ordinals:
        """The set an owner-placed placement binds among `candidates`: what a call of its
        template would be placed on, else the first `width` candidates."""
        if width <= 0 or len(candidates) < width:
            return ()
        excluded = tuple(o for o in range(self.devices) if o not in candidates)
        chosen = self.predict(
            Want(root="choose", widths=(width,), template=template, exclude=excluded)
        )
        return chosen if chosen and set(chosen) <= set(candidates) else candidates[:width]


def _child(key: str, root: str) -> bool:
    """A call under another root: its root may call again, so it keeps the GPUs between."""
    return key.rpartition("#")[0] != root


def _model(want: Want) -> str:
    """The identity costs are banked under: the installation and entrypoint, never the
    checkpoint or the prompt."""
    return f"{want.template[0]}/{want.entrypoint}" if want.template[0] else want.entrypoint
