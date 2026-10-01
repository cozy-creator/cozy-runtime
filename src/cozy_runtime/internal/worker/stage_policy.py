"""The stage scheduler's decision core: pure, typed, one pass per change.

Inputs are facts. The machine: each GPU's capacity and what is resident on it, the host tier
and the link, cache and disk rates the weight plane measured. The jobs: one per GPU call of an
execution, with its root's priority, its arrival, its predicted stage graph with measured
costs, where it is placed and where it is in the graph. The output is one `Plan`: which stage
turns start now on which GPUs, where an unplaced job goes and at what width, which placed jobs
prepare ahead of their turn (executor, construction, host fill), which stages get their first
regions prefetched, and typed refusals with numbers. The shell applies a plan; this module
applies nothing.

1. Order is (priority, arrival): a root's calls share its priority, so a root's calls are
   contiguous and never block each other. A job runs one stage at a time on its whole GPU set,
   and a stage is never preempted: turns change hands only at stage boundaries.
2. A GPU's turn goes to the earliest-ordered root that claims it. A later root gets a turn there
   only inside a bubble: the GPU is idle, the earlier job is in a measured CPU gap, and the
   later stage (with its load) is predicted to end before that gap does, beside the earlier
   job's next stage rather than over it. A root keeps its GPUs between its calls (a lease) until
   it ends or releases them. A job past its last observed stage claims nothing.
3. An unplaced job takes the GPU set, width included, with the earliest predicted finish: when
   the set frees up, plus the load of bytes not resident there from the fastest tier holding
   them, plus an executor start where none is warm, plus its run at that width. Unmeasured work
   takes the widest width the machine forms; while no set's finish is known and none is free,
   it waits unplaced rather than guess between sets.
4. A placed job waiting for its turn prepares there and is prefilled into the host tier; each
   GPU's next turn holder has its next stage's first regions prefetched into spare VRAM.
5. A job is refused only when no GPU could ever hold some stage's floor, on measured facts.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import msgspec

from cozy_runtime.internal.stages import GB, StageCost, StageKind, WeightSet

INF = math.inf
Phase = Literal["preparing", "ready", "running", "gap", "released"]


class Gpu(msgspec.Struct, frozen=True, kw_only=True):
    ordinal: int
    total: int
    context: int = 0
    """Bytes no tenant can budget: CUDA contexts and processes outside this Worker."""
    link_gbps: float = 0.0
    """Achieved pinned host-to-device rate (plane EWMA); 0 is unmeasured."""
    resident: Mapping[str, int] = {}
    """Weight-set content id -> bytes in VRAM here."""


class Host(msgspec.Struct, frozen=True, kw_only=True):
    pinned: Mapping[str, int] = {}
    cached: Mapping[str, int] = {}
    """Page-cache bytes per content (sampled)."""
    cache_gbps: float = 0.0
    disk_gbps: float = 0.0
    spawn_s: float = 0.0
    """Measured executor start: spawn, imports and meta construction; 0 is unmeasured."""


class Machine(msgspec.Struct, frozen=True):
    gpus: tuple[Gpu, ...]
    host: Host


class JobStage(msgspec.Struct, frozen=True, kw_only=True):
    kind: StageKind
    sets: tuple[WeightSet, ...] = ()
    calls: int = 1
    cost: Mapping[int, StageCost] = {}
    """Width -> measured cost at this request's cell. Empty: never measured."""


class Job(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    root: str = ""
    priority: int
    arrival: int
    stages: tuple[JobStage, ...] = ()
    phase: Phase = "ready"
    """preparing: placed and getting ready (executor, construction), not asking for a turn yet.
    ready: wants its next turn (an unplaced job wants placement). running: holds a turn. gap:
    between stages on the CPU. released: no further GPU work."""
    at: int = 0
    """The stage running, wanted, or next after the gap."""
    elapsed_s: float = 0.0
    """Time already spent in the running stage or the current gap."""
    widths: tuple[int, ...] = (1,)
    gpus: tuple[int, ...] = ()
    """The placed set, fixed for the job's life."""
    warm: tuple[tuple[int, ...], ...] = ()
    """GPU sets where this job's installation already has a live executor."""
    lease: bool = False
    """A root's hold between its calls: it claims its GPUs at its order, with no measured gap."""
    exclude: tuple[int, ...] = ()
    """GPUs it can never use (unreadable to it)."""


class Turn(msgspec.Struct, frozen=True):
    job: str
    gpus: tuple[int, ...]
    stage: int
    bubble: bool = False


class Prepare(msgspec.Struct, frozen=True):
    """A placed job waiting for its turn on `gpus`, in order: it may get ready there now."""

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
    """The root whose claim it waits behind; "" when nothing names one."""


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


#: gpu -> (the earliest-ordered job claiming it, seconds until it wants it: 0 now, else its gap)
Claims = dict[int, tuple[Job, float]]


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


def floor(sets: Iterable[WeightSet], activation: int) -> int:
    """The least VRAM a stage runs in: each weight set's floor at its finest grain and the
    activation it needs at the memory policy's cheapest rung."""
    return sum(s.floor for s in sets) + activation


def load_s(sets: Iterable[WeightSet], gpu: Gpu, host: Host) -> float:
    """Upper bound: bytes not resident on `gpu`, moved from the fastest tier holding them,
    with no overlap credited. A rate never measured prices nothing: ranking only."""
    seconds = 0.0
    for s in sets:
        missing = max(0, s.total - gpu.resident.get(s.content, 0))
        if host.pinned.get(s.content, 0) >= s.total:
            rate = gpu.link_gbps
        elif host.cached.get(s.content, 0) >= s.total:
            rate = min(gpu.link_gbps, host.cache_gbps)
        else:
            rate = min(gpu.link_gbps, host.disk_gbps)
        if rate > 0:
            seconds += missing / (rate * GB)
    return seconds


def remaining_s(job: Job, width: int) -> float:
    """Predicted time until `job` is done with its GPUs; INF when any part is unmeasured."""
    if job.lease:
        return INF
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
    measured; an estimate never refuses. No width the machine forms refuses too."""
    usable = sorted(
        ((g.total - g.context, g.ordinal) for g in machine.gpus if g.ordinal not in job.exclude),
        reverse=True,
    )
    if not any(w <= len(usable) for w in job.widths):
        return Refusal(
            job.key,
            "NO_CAPACITY",
            f"needs {' or '.join(map(str, job.widths))} GPUs; this machine reads {len(usable)}",
            max(job.widths, default=0),
            len(usable),
        )
    worst: tuple[int, int, int, str] | None = None
    for width in job.widths:
        if width > len(usable):
            continue
        capacity, ordinal = usable[width - 1]
        for stage in job.stages:
            need = floor(stage.sets, floor_growth(stage, width))
            if need <= capacity:
                continue
            kind = stage.kind
            named = f"stage {kind.method} ({', '.join(kind.components)}) needs {need} bytes per GPU"
            if worst is None or need - capacity < worst[0]:
                worst = (
                    need - capacity,
                    need,
                    capacity,
                    f"{named}; GPU {ordinal} holds {capacity}",
                )
            break
        else:
            return None
    if worst is None:
        return None
    _, need, capacity, detail = worst
    return Refusal(
        job.key,
        "NO_CAPACITY",
        f"{detail} (every weight set's floor and activations at the cheapest rung)",
        need,
        capacity,
    )


# --------------------------------------------------------------------------- the pass


def _rank(job: Job) -> tuple[int, int, str]:
    return (job.priority, job.arrival, job.key)


def decide(machine: Machine, jobs: Sequence[Job]) -> Plan:
    gpus = {g.ordinal: g for g in machine.gpus}
    order = sorted((j for j in jobs if j.phase != "released"), key=_rank)
    turns: list[Turn] = []
    prepares: list[Prepare] = []
    waits: list[Wait] = []
    refusals: list[Refusal] = []
    idle = dict.fromkeys(gpus, True)
    claim: Claims = {}
    #: gpu -> the job whose turn holds it now
    holder: dict[int, Job] = {}
    #: gpu -> predicted seconds until a newly placed job could start there
    free_at = dict.fromkeys(gpus, 0.0)
    #: job -> the GPU set it holds, is placed on, or prepares on
    target: dict[str, tuple[int, ...]] = {}
    busy: set[str] = set()

    for job in order:
        if job.phase == "running":
            busy.add(job.key)
            _take(job.gpus, job, idle, claim, holder)
            for o in job.gpus:
                if o in free_at:  # a device job also holds GPUs the driver cannot read
                    free_at[o] += remaining_s(job, len(job.gpus))

    for job in order:
        if job.phase == "running":
            target[job.key] = job.gpus
            continue
        if job.lease:
            for o in job.gpus:
                if o in gpus:
                    claim.setdefault(o, (job, 0.0))
            continue
        if not job.gpus:
            if job.phase not in ("ready", "preparing"):
                continue
            if (refusing := refusal(job, machine)) is not None:
                refusals.append(refusing)
                continue
            placed = _place(job, machine, idle, claim, holder, free_at)
            if isinstance(placed, Wait):
                waits.append(placed)
                continue
            chosen, bubble = placed
            target[job.key] = chosen
            if bubble is not None and job.phase == "ready":
                turns.append(Turn(job.key, chosen, job.at, bubble))
                busy.add(job.key)
                _take(chosen, job, idle, claim, holder)
            else:
                prepares.append(Prepare(job.key, chosen))
                if job.phase == "ready":
                    waits.append(Wait(job.key, _blocker(chosen, claim, holder, job)))
            for o in chosen:
                claim.setdefault(o, (job, 0.0))
                free_at[o] += remaining_s(job, len(chosen))
            continue
        target[job.key] = job.gpus
        width = len(job.gpus)
        if job.at >= len(job.stages) and job.phase == "gap":
            continue  # past its last observed stage: it claims nothing until it asks again
        if job.phase == "ready":
            fits = _bubble(job, job.gpus, gpus, machine.host, claim)
            if all(idle.get(o, False) for o in job.gpus) and fits is not None:
                turns.append(Turn(job.key, job.gpus, job.at, fits))
                busy.add(job.key)
                _take(job.gpus, job, idle, claim, holder)
            else:
                waits.append(Wait(job.key, _blocker(job.gpus, claim, holder, job)))
        elif job.phase == "preparing":
            prepares.append(Prepare(job.key, job.gpus))
        wait = gap_left(job, width) if job.phase == "gap" else 0.0
        for o in job.gpus:
            claim.setdefault(o, (job, wait))
            if o in free_at:
                free_at[o] += remaining_s(job, width)

    refused = {r.job for r in refusals}
    return Plan(
        turns=tuple(turns),
        prepares=tuple(prepares),
        prefetches=_prefetches(order, target, busy),
        prefills=tuple(
            Prefill(j.key, missing)
            for j in order
            if j.key not in refused and not j.lease and (missing := _unpinned(j, machine.host))
        ),
        waits=tuple(waits),
        refusals=tuple(refusals),
    )


def gap_left(job: Job, width: int) -> float:
    if job.at >= len(job.stages):
        return 0.0
    return max(0.0, gap_s(job.stages[job.at], width) - job.elapsed_s)


def _take(
    chosen: Sequence[int],
    job: Job,
    idle: dict[int, bool],
    claim: Claims,
    holder: dict[int, Job],
) -> None:
    for o in chosen:
        idle[o] = False
        holder[o] = job
        claim.setdefault(o, (job, 0.0))


def _ahead(holder: Job, job: Job) -> bool:
    """Whether `holder`'s claim keeps `job` off a GPU: another root's always; its own root's
    only while that earlier call still waits for its turn (FIFO within a root). A root's own
    gaps and holds never block its next call."""
    if holder is job:
        return False
    if holder.root != job.root:
        return True
    return not holder.lease and holder.phase in ("ready", "preparing")


def _blocker(chosen: Sequence[int], claim: Claims, holder: Mapping[int, Job], job: Job) -> str:
    """The root `job` is named behind: one whose turn holds a GPU it needs (its own root when
    only its own calls do), else the earliest root that claims one. Naming who holds before
    who merely waits keeps two waiting roots from naming each other."""
    held = [holder[o] for o in chosen if o in holder and holder[o] is not job]
    if held:
        return next((h.root for h in held if h.root != job.root), job.root)
    return next((claim[o][0].root for o in chosen if o in claim and _ahead(claim[o][0], job)), "")


def _bubble(
    job: Job, chosen: Sequence[int], gpus: Mapping[int, Gpu], host: Host, claim: Claims
) -> bool | None:
    """False: `job` may start its stage on `chosen` outright. True: only as a bubble inside an
    earlier root's CPU gap, which it fits in time and beside that job's next stage in VRAM.
    None: it may not start now."""
    holders = [claim[o] for o in chosen if o in claim and _ahead(claim[o][0], job)]
    if not holders:
        return False
    if job.at >= len(job.stages) or any(wait <= 0.0 for _, wait in holders):
        return None
    stage = job.stages[job.at]
    width = len(chosen)
    run = run_s(stage, width)
    if run is None:
        return None
    load = max(load_s(stage.sets, gpus[o], host) for o in chosen)
    if run + load > min(wait for _, wait in holders):
        return None
    need = floor(stage.sets, growth(stage, width))
    for o in chosen:
        g = gpus[o]
        kept = sum(
            g.resident.get(s.content, 0)
            for holder, _ in holders
            if holder.at < len(holder.stages)
            for s in holder.stages[holder.at].sets
        )
        if need + kept > g.total - g.context:
            return None
    return True


def _place(
    job: Job,
    machine: Machine,
    idle: Mapping[int, bool],
    claim: Claims,
    holder: Mapping[int, Job],
    free_at: Mapping[int, float],
) -> tuple[tuple[int, ...], bool | None] | Wait:
    """The set with the earliest predicted finish and whether it starts now (as a bubble or
    outright; None: it waits there). Unmeasured jobs take the widest width the machine forms.
    While no set's finish is known and none can start, the job waits unplaced."""
    gpus = {g.ordinal: g for g in machine.gpus}
    usable = sorted(o for o in gpus if o not in job.exclude)
    widths = [w for w in job.widths if w <= len(usable)]
    if not widths:
        return Wait(job.key, "")
    measured = any(run_s(s, w) is not None for s in job.stages for w in widths)
    if not measured:
        widths = [max(widths)]
    needed = {s.content: s for st in job.stages[job.at :] for s in st.sets}.values()
    rows = []
    for width in widths:
        for chosen in itertools.combinations(usable, width):
            now = all(idle[o] for o in chosen)
            bubble = _bubble(job, chosen, gpus, machine.host, claim) if now else None
            start = (
                0.0 if bubble is not None else max(_free(o, job, claim, free_at) for o in chosen)
            )
            spawn = 0.0 if chosen in job.warm else machine.host.spawn_s
            load = max(load_s(needed, gpus[o], machine.host) for o in chosen)
            finish = max(start, spawn) + load + (remaining_s(job, width) if measured else 0.0)
            rank = (finish, -width, spawn + load, chosen not in job.warm, chosen)
            rows.append((rank, chosen, bubble, start))
    _, chosen, bubble, start = min(rows, key=lambda row: row[0])
    if bubble is None and start == INF and len(rows) > 1:
        return Wait(job.key, _blocker(chosen, claim, holder, job))
    return chosen, bubble


def _free(o: int, job: Job, claim: Claims, free_at: Mapping[int, float]) -> float:
    """When GPU `o` frees up for `job`: never, while another root leases it."""
    held = claim.get(o)
    if held is not None and held[0].lease and held[0].root != job.root:
        return INF
    return free_at[o]


def _prefetches(
    order: Sequence[Job], target: Mapping[str, tuple[int, ...]], busy: set[str]
) -> tuple[Prefetch, ...]:
    """Per GPU, the next turn holder after whatever runs there now: the earliest-ordered job
    wanting it (its next stage if it is the one running). Its stage's first regions go into
    spare VRAM; the memory policy never evicts a running working set for them."""
    chosen: dict[int, Prefetch] = {}
    for job in order:
        stage = job.at + 1 if job.key in busy else job.at
        if job.lease or stage >= len(job.stages) or job.key not in target:
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
