"""The stage scheduler's decision core: pure, typed, one pass per change.

Inputs are facts. The machine: each GPU's capacity and what is resident on it, the host tier
and the link, cache and disk rates the weight plane measured. The jobs: priority, arrival,
their predicted stage graph with measured costs, where they are placed and where they are in
the graph. The output is one `Plan`: which stage turns start now on which GPUs, where an
unplaced job goes and at what width, which jobs prepare early (executor, meta construction,
host fill), which stages get their first regions prefetched, and typed refusals with
numbers. The shell applies a plan through the memory policy; this module applies nothing.

1. Order is (priority, arrival). A job runs one stage at a time on its whole GPU set, and a
   stage is never preempted: turns change hands only at stage boundaries.
2. A GPU's turn goes to the earliest-ordered job that wants it. A later job gets a turn there
   only inside a bubble: the GPU is idle, the earlier job is in a CPU gap, and the later
   stage (with its load) is predicted to end before that gap does, beside the earlier job's
   next stage rather than over it.
3. An unplaced job takes the GPU set, width included, with the earliest predicted finish:
   when the set frees up, plus the load of bytes not resident there from the fastest tier
   holding them, plus an executor start where none is warm, plus its run at that width.
   Unmeasured work predicts nothing, so the widest set the machine forms wins (today's rule).
4. A job waiting for a set prepares on it and is prefilled into the host tier; each GPU's next
   turn holder has its next stage's first regions prefetched into spare VRAM.
5. A job is refused only when no GPU could ever hold some stage's floor, on measured facts.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import msgspec

from cozy_runtime.internal.stages import GB, RegionSet, StageCost, StageKind, floor_bytes

INF = math.inf
Phase = Literal["ready", "running", "gap", "released"]


class Gpu(msgspec.Struct, frozen=True, kw_only=True):
    ordinal: int
    total: int
    context: int = 0
    """Bytes no tenant can budget: CUDA contexts and processes outside this Worker."""
    link_gbps: float
    """Achieved pinned host-to-device rate (plane EWMA)."""
    resident: Mapping[str, int] = {}
    """Weight-set content id -> bytes in VRAM here."""


class Host(msgspec.Struct, frozen=True, kw_only=True):
    pinned: Mapping[str, int] = {}
    cached: Mapping[str, int] = {}
    """Page-cache bytes per content (sampled)."""
    cache_gbps: float
    disk_gbps: float
    spawn_s: float = 0.0
    """Measured executor start: spawn, imports and meta construction."""


class Machine(msgspec.Struct, frozen=True):
    gpus: tuple[Gpu, ...]
    host: Host


class JobStage(msgspec.Struct, frozen=True, kw_only=True):
    kind: StageKind
    sets: tuple[RegionSet, ...]
    calls: int = 1
    cost: Mapping[int, StageCost] = {}
    """Width -> measured cost at this request's cell. Empty: never measured."""


class Job(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    priority: int
    arrival: int
    stages: tuple[JobStage, ...]
    phase: Phase = "ready"
    """ready: wants its next turn (an unplaced job wants placement). gap: between stages on the
    CPU. released: no further GPU work (`ctx.release_gpus()` or done)."""
    at: int = 0
    """The stage running, wanted, or next after the gap. A job in its gap past the last
    observed stage claims nothing: its GPUs are free, and a stage it did not predict waits for
    its turn in order like any other (the shell appends it before asking)."""
    elapsed_s: float = 0.0
    """Time already spent in the running stage or the current gap."""
    widths: tuple[int, ...] = (1,)
    gpus: tuple[int, ...] = ()
    """The placed set, fixed for the job's life."""
    warm: tuple[tuple[int, ...], ...] = ()
    """GPU sets where this job's installation already has a live executor."""


class Turn(msgspec.Struct, frozen=True):
    job: str
    gpus: tuple[int, ...]
    stage: int
    bubble: bool = False


class Prepare(msgspec.Struct, frozen=True):
    job: str
    gpus: tuple[int, ...]


class Prefetch(msgspec.Struct, frozen=True):
    job: str
    gpus: tuple[int, ...]
    stage: int


class Prefill(msgspec.Struct, frozen=True):
    job: str
    contents: tuple[str, ...]


class Wait(msgspec.Struct, frozen=True):
    job: str
    blocked_by: str


class Refusal(msgspec.Struct, frozen=True):
    job: str
    code: str
    detail: str
    need: int
    capacity: int


class Plan(msgspec.Struct, frozen=True, kw_only=True):
    turns: tuple[Turn, ...] = ()
    prepares: tuple[Prepare, ...] = ()
    prefetches: tuple[Prefetch, ...] = ()
    prefills: tuple[Prefill, ...] = ()
    waits: tuple[Wait, ...] = ()
    refusals: tuple[Refusal, ...] = ()


# --------------------------------------------------------------------------- predictions


def _measured(stage: JobStage) -> dict[int, StageCost]:
    return {w: c for w, c in stage.cost.items() if c.runs}


def run_s(stage: JobStage, width: int) -> float | None:
    """Warm run of all the stage's calls at `width`. An unmeasured width scales from the
    nearest measured one by width (ideal sequence parallelism): a ranking guess only."""
    measured = _measured(stage)
    if not measured:
        return None
    nearest = min(measured, key=lambda w: (abs(w - width), w))
    return measured[nearest].wall_s * stage.calls * nearest / width


def gap_s(stage: JobStage, width: int) -> float:
    """Measured CPU gap before the stage; unmeasured gaps are taken as zero (no bubble)."""
    cost = _measured(stage).get(width)
    return cost.gap_s if cost is not None else 0.0


def growth(stage: JobStage, width: int) -> int:
    """The stage's largest measured working growth: what a bubble must leave room for."""
    cost = stage.cost.get(width)
    return cost.growth if cost is not None else 0


def floor_growth(stage: JobStage, width: int) -> int:
    cost = stage.cost.get(width)
    return cost.floor_growth if cost is not None else 0


def load_s(sets: Iterable[RegionSet], gpu: Gpu, host: Host) -> float:
    """Upper bound: bytes not resident on `gpu`, moved from the fastest tier holding them,
    with no overlap credited."""
    seconds = 0.0
    for s in sets:
        missing = max(0, s.total - gpu.resident.get(s.content, 0))
        if host.pinned.get(s.content, 0) >= s.total:
            rate = gpu.link_gbps
        elif host.cached.get(s.content, 0) >= s.total:
            rate = min(gpu.link_gbps, host.cache_gbps)
        else:
            rate = min(gpu.link_gbps, host.disk_gbps)
        seconds += missing / (max(rate, 1e-9) * GB)
    return seconds


def remaining_s(job: Job, width: int) -> float:
    """Predicted time until `job` is done with its GPUs; INF when any part is unmeasured."""
    total = 0.0
    for i, stage in enumerate(job.stages[job.at :], start=job.at):
        run = run_s(stage, width)
        if run is None:
            return INF
        if i == job.at and job.phase == "running":
            run = max(0.0, run - job.elapsed_s)
        elif i == job.at and job.phase == "gap":
            run += max(0.0, gap_s(stage, width) - job.elapsed_s)
        else:
            run += gap_s(stage, width)
        total += run
    return total


def refusal(job: Job, machine: Machine) -> Refusal | None:
    """Refuse only when, at every allowed width, some stage's floor exceeds what the `width`-th
    largest GPU can hold. Activations count only at the cheapest rung the memory policy has
    measured; an estimate never refuses."""
    usable = sorted((g.total - g.context for g in machine.gpus), reverse=True)
    worst: tuple[int, int, int, str] | None = None
    for width in job.widths:
        capacity = usable[width - 1] if len(usable) >= width else 0
        needs = [(floor_bytes(s.sets, floor_growth(s, width), 0), s.kind) for s in job.stages]
        need, kind = max(needs, key=lambda row: row[0])
        if need <= capacity:
            return None
        if worst is None or need - capacity < worst[0]:
            named = f"{kind.method} ({', '.join(kind.components)})"
            worst = (need - capacity, need, capacity, named)
    if worst is None:
        return None
    _, need, capacity, stage = worst
    return Refusal(
        job.key,
        "NO_CAPACITY",
        f"stage {stage} needs {need} bytes per GPU (every common region, the largest block and "
        f"activations at the cheapest rung); the largest usable GPU holds {capacity}",
        need,
        capacity,
    )


# --------------------------------------------------------------------------- the pass


def decide(machine: Machine, jobs: Sequence[Job]) -> Plan:
    gpus = {g.ordinal: g for g in machine.gpus}
    order = sorted(
        (j for j in jobs if j.phase != "released"), key=lambda j: (j.priority, j.arrival, j.key)
    )
    turns: list[Turn] = []
    prepares: list[Prepare] = []
    waits: list[Wait] = []
    refusals: list[Refusal] = []
    idle = dict.fromkeys(gpus, True)
    #: gpu -> (the earliest-ordered job that wants it, seconds until it does: 0 now, else its gap)
    claim: dict[int, tuple[Job, float]] = {}
    #: gpu -> predicted seconds until a newly placed job could start there
    free_at = dict.fromkeys(gpus, 0.0)
    #: job -> the GPU set it holds, is placed on, or prepares on
    target: dict[str, tuple[int, ...]] = {}
    busy: set[str] = set()

    for job in order:
        if job.phase == "running":
            busy.add(job.key)
            _take(job.gpus, job, idle, claim)
            for o in job.gpus:
                free_at[o] += remaining_s(job, len(job.gpus))

    for job in order:
        if job.phase == "running":
            target[job.key] = job.gpus
            continue
        if not job.gpus:
            if (refusing := refusal(job, machine)) is not None:
                refusals.append(refusing)
                continue
            placed = _place(job, machine, idle, claim, free_at)
            if placed is None:
                continue
            chosen, bubble = placed
            target[job.key] = chosen
            if bubble is not None:
                turns.append(Turn(job.key, chosen, job.at, bubble))
                busy.add(job.key)
                _take(chosen, job, idle, claim)
                for o in chosen:
                    free_at[o] += remaining_s(job, len(chosen))
            else:
                prepares.append(Prepare(job.key, chosen))
                waits.append(Wait(job.key, _blocker(chosen, claim, job)))
            continue
        target[job.key] = job.gpus
        width = len(job.gpus)
        if job.at >= len(job.stages):
            continue  # past its last observed stage: it claims nothing until it asks again
        if job.phase == "ready":
            fits = _bubble(job, job.gpus, gpus, machine.host, claim)
            if all(idle[o] for o in job.gpus) and fits is not None:
                turns.append(Turn(job.key, job.gpus, job.at, fits))
                busy.add(job.key)
                _take(job.gpus, job, idle, claim)
            else:
                waits.append(Wait(job.key, _blocker(job.gpus, claim, job)))
        for o in job.gpus:
            claim.setdefault(o, (job, 0.0 if job.phase == "ready" else gap_left(job, width)))
            free_at[o] += remaining_s(job, width)

    refused = {r.job for r in refusals}
    return Plan(
        turns=tuple(turns),
        prepares=tuple(prepares),
        prefetches=_prefetches(order, target, busy),
        prefills=tuple(
            Prefill(j.key, missing)
            for j in order
            if j.key not in refused and (missing := _unpinned(j, machine.host))
        ),
        waits=tuple(waits),
        refusals=tuple(refusals),
    )


def gap_left(job: Job, width: int) -> float:
    return max(0.0, gap_s(job.stages[job.at], width) - job.elapsed_s)


def _take(
    chosen: Sequence[int], job: Job, idle: dict[int, bool], claim: dict[int, tuple[Job, float]]
) -> None:
    for o in chosen:
        idle[o] = False
        claim.setdefault(o, (job, 0.0))


def _blocker(chosen: Sequence[int], claim: Mapping[int, tuple[Job, float]], job: Job) -> str:
    return next((claim[o][0].key for o in chosen if o in claim and claim[o][0] is not job), "")


def _bubble(
    job: Job,
    chosen: Sequence[int],
    gpus: Mapping[int, Gpu],
    host: Host,
    claim: Mapping[int, tuple[Job, float]],
) -> bool | None:
    """False: `job` may start its stage on `chosen` outright. True: only as a bubble inside an
    earlier job's CPU gap, which it fits in time and beside that job's next stage in VRAM.
    None: it may not start now."""
    holders = [claim[o] for o in chosen if o in claim and claim[o][0] is not job]
    if not holders:
        return False
    stage = job.stages[job.at]
    width = len(chosen)
    run = run_s(stage, width)
    if run is None or any(wait <= 0.0 for _, wait in holders):
        return None
    load = max(load_s(stage.sets, gpus[o], host) for o in chosen)
    if run + load > min(wait for _, wait in holders):
        return None
    need = floor_bytes(stage.sets, growth(stage, width), 0)
    for o in chosen:
        g = gpus[o]
        kept = sum(
            g.resident.get(s.content, 0)
            for holder, _ in holders
            for s in holder.stages[holder.at].sets
        )
        if need + kept > g.total - g.context:
            return None
    return True


def _place(
    job: Job,
    machine: Machine,
    idle: Mapping[int, bool],
    claim: Mapping[int, tuple[Job, float]],
    free_at: Mapping[int, float],
) -> tuple[tuple[int, ...], bool | None] | None:
    """The set with the earliest predicted finish and whether it starts now (as a bubble or
    outright; None: it waits). Unmeasured jobs take the widest width the machine forms."""
    gpus = {g.ordinal: g for g in machine.gpus}
    widths = [w for w in job.widths if w <= len(gpus)]
    if not widths:
        return None
    measured = any(run_s(s, w) is not None for s in job.stages for w in widths)
    if not measured:
        widths = [max(widths)]
    needed = {s.content: s for st in job.stages[job.at :] for s in st.sets}.values()
    best: tuple[tuple[float, int, float, tuple[int, ...]], tuple[int, ...], bool | None] | None
    best = None
    for width in widths:
        for chosen in itertools.combinations(sorted(gpus), width):
            now = all(idle[o] for o in chosen)
            bubble = _bubble(job, chosen, gpus, machine.host, claim) if now else None
            start = 0.0 if bubble is not None else max(free_at[o] for o in chosen)
            spawn = 0.0 if chosen in job.warm else machine.host.spawn_s
            load = max(load_s(needed, gpus[o], machine.host) for o in chosen)
            ready = max(start, spawn) + load
            finish = ready + (remaining_s(job, width) if measured else 0.0)
            rank = (finish, -width, spawn + load, chosen)
            if best is None or rank < best[0]:
                best = (rank, chosen, bubble)
    return None if best is None else (best[1], best[2])


def _prefetches(
    order: Sequence[Job], target: Mapping[str, tuple[int, ...]], busy: set[str]
) -> tuple[Prefetch, ...]:
    """Per GPU, the next turn holder after whatever runs there now: the earliest-ordered job
    wanting it (its next stage if it is the one running). Its stage's first regions go into
    spare VRAM; the memory policy never evicts a running working set for them."""
    chosen: dict[int, Prefetch] = {}
    for job in order:
        stage = job.at + 1 if job.key in busy else job.at
        if stage >= len(job.stages) or job.key not in target:
            continue
        for o in target[job.key]:
            if o not in chosen:
                chosen[o] = Prefetch(job.key, target[job.key], stage)
    return tuple(dict.fromkeys(chosen.values()))


def _unpinned(job: Job, host: Host) -> tuple[str, ...]:
    """The content of the job's next two stages not yet fully in the host tier, in use order."""
    return tuple(
        dict.fromkeys(
            s.content
            for stage in job.stages[job.at : job.at + 2]
            for s in stage.sets
            if host.pinned.get(s.content, 0) < s.total
        )
    )
