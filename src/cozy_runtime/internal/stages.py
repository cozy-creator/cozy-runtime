"""Stage graphs, their measured costs, and the cost book the scheduler predicts with.

A stage is one component-use scope call (`@uses_components`), the one declaration a package
already makes. Its weight sets come from the plane's layouts; its width from
`@sequence_parallel`. The order of stages, their call counts, activation growth and CPU gaps
are measured from real runs and kept per shape cell (`PreparedRequest.cell`), so nothing is
authored twice and nothing is estimated that a run can measure. A graph nobody has observed
yet is one opaque stage.

Pure and torch-free: `worker/stage_policy.py` decides, the Worker fills `StageSample` from each
`StageExit` a `stage/1` executor sends, and `weight_policy` prices a stage's streaming.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import msgspec

from cozy_runtime.internal import weight_policy

GB = 1e9
#: An executor capability: the worker may grant it a turn per component-use scope.
STAGE_TURNS = "stage/1"


class WeightSet(msgspec.Struct, frozen=True):
    """One component's weight set as the plane registered it: its content id (the plane key a
    tier and residency are reported under) and its layout's region sizes."""

    content: str
    component: str
    common: int
    blocks: tuple[int, ...] = ()
    #: (common, largest region) at the sub-block grain, when finer than `blocks`
    fine: tuple[int, int] | None = None

    @property
    def layout(self) -> weight_policy.Layout:
        return weight_policy.Layout(self.common, self.blocks, self.fine)

    @property
    def total(self) -> int:
        return self.common + sum(self.blocks)

    @property
    def floor(self) -> int:
        """Common and one ring of the largest region, at the finest grain offered."""
        return weight_policy.floor(self.layout)


class StageKind(msgspec.Struct, frozen=True):
    """One scoped method of one model. `model` is the identity costs are banked under: the
    model class and delivery variant, never the prompt, seed or checkpoint path."""

    model: str
    method: str
    components: tuple[str, ...]


class Stage(msgspec.Struct, frozen=True):
    kind: StageKind
    calls: int = 1
    """Consecutive calls of the same scope, collapsed: a per-step scoped method is one stage."""


class StageGraph(msgspec.Struct, frozen=True):
    entrypoint: str
    cell: str
    stages: tuple[Stage, ...]
    observed: bool = False


class StageSample(msgspec.Struct, frozen=True, kw_only=True):
    """What the executor measured for one scope call, sent at its exit."""

    kind: StageKind
    width: int = 1
    passes: int = 1
    """Cycles over the stage's regions: steps for a whole-loop scope, 2 for serial CFG."""
    region_ns: tuple[int, ...] = ()
    """Mean compute per region per pass, in the stage's region order (plane timing events)."""
    wall_ns: int = 0
    gap_ns: int = 0
    """CPU time between this request's previous scope exit and this entry."""
    growth_bytes: int = 0
    """Allocator peak during the call minus reserved at entry."""
    floor_growth: int = 0
    """Activation bytes the stage needs at the memory policy's cheapest rung (tiled decode,
    chunked attention), once that policy ran there or fell short there. 0: not known."""
    stall_ns: int = 0
    ok: bool = True


class StageCost(msgspec.Struct, frozen=True, kw_only=True):
    """Measured cost of one stage kind at one cell and width. Times are smoothed means over
    successful calls (`runs`; 0 means only failures were seen). `growth` is the largest
    successful peak, never averaged (memory-manager rule 3), and only admission's floor reads
    `floor_growth`: a mode-specific peak such as an untiled decode must never refuse a GPU
    where the cheapest rung fits."""

    runs: int
    wall_s: float
    gap_s: float
    passes: float
    region_s: tuple[float, ...] = ()
    growth: int = 0
    floor_growth: int = 0
    stall_s: float = 0.0


class _CostRow(msgspec.Struct, frozen=True):
    model: str
    method: str
    cell: str
    width: int
    cost: StageCost


class _Book(msgspec.Struct, frozen=True):
    costs: tuple[_CostRow, ...] = ()
    graphs: tuple[StageGraph, ...] = ()


def observed_graph(entrypoint: str, cell: str, samples: Iterable[StageSample]) -> StageGraph:
    stages: list[Stage] = []
    for sample in samples:
        if stages and stages[-1].kind == sample.kind:
            stages[-1] = Stage(sample.kind, stages[-1].calls + 1)
        else:
            stages.append(Stage(sample.kind))
    return StageGraph(entrypoint, cell, tuple(stages), observed=True)


def static_graph(entrypoint: str, cell: str, model: str, components: Iterable[str]) -> StageGraph:
    """Before any run: one opaque stage over every component the call's models declare. It
    interleaves with nothing until measured; an executor before `stage/1` stays one."""
    return StageGraph(
        entrypoint, cell, (Stage(StageKind(model, "*", tuple(dict.fromkeys(components)))),)
    )


class CostBook:
    """Measured stage costs and last observed graphs, persisted per machine. Advisory: a lost
    or stale book costs efficiency, never correctness, so writes are never fsynced."""

    def __init__(self) -> None:
        self.costs: dict[tuple[str, str, str, int], StageCost] = {}
        self.graphs: dict[tuple[str, str], StageGraph] = {}

    def record(self, entrypoint: str, cell: str, samples: Sequence[StageSample]) -> None:
        """One finished request's scope calls, in order. A failed call forgets its growth."""
        for s in samples:
            key = (s.kind.model, s.kind.method, cell, s.width)
            prior = self.costs.get(key) or StageCost(runs=0, wall_s=0.0, gap_s=0.0, passes=0.0)
            floor = max(prior.floor_growth, s.floor_growth)
            if not s.ok:
                self.costs[key] = msgspec.structs.replace(prior, growth=0, floor_growth=floor)
                continue
            w = 1.0 / min(prior.runs + 1, 4)
            regions = tuple(n / 1e9 for n in s.region_ns)
            if len(prior.region_s) == len(regions):
                pairs = zip(prior.region_s, regions, strict=True)
                regions = tuple(o + w * (n - o) for o, n in pairs)
            self.costs[key] = StageCost(
                runs=prior.runs + 1,
                wall_s=prior.wall_s + w * (s.wall_ns / 1e9 - prior.wall_s),
                gap_s=prior.gap_s + w * (s.gap_ns / 1e9 - prior.gap_s),
                passes=prior.passes + w * (s.passes - prior.passes),
                region_s=regions,
                growth=max(prior.growth, s.growth_bytes),
                floor_growth=floor,
                stall_s=prior.stall_s + w * (s.stall_ns / 1e9 - prior.stall_s),
            )
        if samples and all(s.ok for s in samples):
            self.graphs[(entrypoint, cell)] = observed_graph(entrypoint, cell, samples)

    def cost(self, kind: StageKind, cell: str, width: int) -> StageCost | None:
        return self.costs.get((kind.model, kind.method, cell, width))

    def graph(self, entrypoint: str, cell: str) -> StageGraph | None:
        return self.graphs.get((entrypoint, cell))

    def save(self, path: Path) -> None:
        book = _Book(
            tuple(_CostRow(*key, cost) for key, cost in sorted(self.costs.items())),
            tuple(self.graphs[key] for key in sorted(self.graphs)),
        )
        scratch = path.with_name(f".{path.name}.{os.getpid()}")
        scratch.write_bytes(msgspec.json.encode(book))
        os.replace(scratch, path)

    @classmethod
    def load(cls, path: Path) -> CostBook:
        """A missing or unreadable book is an empty one: costs are relearned, never refused."""
        book = cls()
        try:
            held = msgspec.json.decode(path.read_bytes(), type=_Book)
        except (OSError, msgspec.DecodeError, msgspec.ValidationError):
            return book
        book.costs = {(r.model, r.method, r.cell, r.width): r.cost for r in held.costs}
        book.graphs = {(g.entrypoint, g.cell): g for g in held.graphs}
        return book
