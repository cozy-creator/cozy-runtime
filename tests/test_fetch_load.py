"""Load asks for what the code builds, in stage and learned block order, and a fill waits on
its objects (one-request-kind §2.6). Real: the loopback hub and structure-only Store of
`fetch_hub`, TensorFS's own channel owner, the runtime's `PlaneBackend` on a host-only
plane, and torch."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author import Model, uses_components  # noqa: E402
from cozy_runtime.author._loader import census  # noqa: E402
from cozy_runtime.author._model import component_use  # noqa: E402
from cozy_runtime.internal.derive import Observations, serving_substrate  # noqa: E402
from cozy_runtime.internal.fetch_plan import (  # noqa: E402
    Fetches,
    in_definition_order,
    stage_components,
)
from cozy_runtime.internal.fill import Checkpoint  # noqa: E402
from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights  # noqa: E402
from fetch_hub import BLOCKS, HIDDEN, fetches, objects_of, served  # noqa: E402


class Block(torch.nn.Module):  # type: ignore[misc,name-defined]
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.out = torch.nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float16)
        self.norm = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))


class Small(torch.nn.Module):  # type: ignore[misc,name-defined]
    _no_split_modules = ("Block",)

    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.embed = torch.nn.Parameter(torch.empty(HIDDEN, dtype=torch.float16))
        self.blocks = torch.nn.ModuleList([Block() for _ in range(blocks)])


class Pipe(Model[Any]):
    """Its stages in pipeline order; `component_use` lists them sorted."""

    @uses_components("encoder")
    def encode(self) -> None: ...

    @uses_components("denoiser")
    def denoise(self) -> None: ...

    @uses_components("decoder")
    def decode(self) -> None: ...


BUILT = {"encoder": 1, "denoiser": BLOCKS, "decoder": 2}


def load(weights: Weights, checkpoint: Checkpoint) -> SimpleNamespace:
    """Construct on meta and register with the plane, as the executor's Load does."""
    rows = [row for name in BUILT for row in checkpoint.rows(name)]
    backend = PlaneBackend.for_script(
        dict.fromkeys(BUILT, checkpoint),
        rows,
        weights=weights,
        construction="pipe",
        release="fetch-test/1",
    )
    scopes = in_definition_order(Pipe, component_use(Pipe))
    backend.order = tuple(stage_components(scopes, scopes))
    with serving_substrate("cpu", Observations()):
        holder = SimpleNamespace(components={n: Small(b) for n, b in BUILT.items()})
    walked = census(holder)
    backend.expect(
        {n: [d.key for d in walked.destinations if d.key.startswith(n + ".")] for n in BUILT}
    )
    live = backend.materialize(holder, walked)
    for destination in walked.destinations:
        backend.fill(destination.key, destination.spec, live[destination.key])
    backend.commit()
    residency = WeightResidency(weights, backend.components, scopes, refine=backend.refine)
    return SimpleNamespace(backend=backend, residency=residency)


def cpu(planner: Fetches) -> Weights:
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.fetch = planner
    return weights


def test_load_asks_for_what_the_code_builds_in_stage_and_learned_block_order(
    tmp_path: Path,
) -> None:
    with served(tmp_path) as (hub, store):
        planner = fetches(hub, store)
        model = load(cpu(planner), Checkpoint.from_store(store, hub.manifest))
        # Stage order is the class's definition order (encode, denoise, decode), not sorted.
        assert [sorted(refs) for _, refs in planner.groups] == [
            objects_of(store, hub.manifest, name) for name in ("encoder", "denoiser", "decoder")
        ]
        for name in BUILT:
            built = objects_of(store, hub.manifest, name)
            assert hub.count(built) == [1] * len(built), name
        spare = objects_of(store, hub.manifest, "spare")
        assert hub.count(spare) == [0] * len(spare), "a component the code never builds"

        # What this process learned orders the next attempt: decode ran first here, and the
        # denoiser's blocks ran last to first.
        residency = model.residency
        residency.admit("decode", ("decoder",))
        residency.release("decode", ("decoder",))
        denoiser = residency.components["denoiser"]
        denoiser.order = [block.index for block in reversed(denoiser.blocks)]
        groups = residency.fetch_plan()
        assert [sorted(refs) for _, refs in groups] == [
            objects_of(store, hub.manifest, name) for name in ("decoder", "encoder", "denoiser")
        ]
        common = [denoiser.common.index] if denoiser.common is not None else []
        regions = [*common, *denoiser.order]
        assert groups[2][1] == list(
            dict.fromkeys(ref for index in regions for ref in denoiser.objects[index])
        )
        planner.end()


def test_a_fill_waits_on_its_objects_and_never_fetches_them_itself(tmp_path: Path) -> None:
    with served(tmp_path) as (hub, store):
        gated = objects_of(store, hub.manifest, "denoiser")[-1][0]
        lines: list[str] = []
        hub.held = {gated}
        hub.gate.clear()
        planner = fetches(hub, store, lines)
        loaded: list[SimpleNamespace] = []
        checkpoint = Checkpoint.from_store(store, hub.manifest)
        loading = threading.Thread(
            target=lambda: loaded.append(load(cpu(planner), checkpoint)), daemon=True
        )
        loading.start()
        assert hub.asked.wait(30)
        assert loading.is_alive() and not store.contains(gated)
        hub.gate.set()
        loading.join(timeout=60)
        assert loaded and set(loaded[0].backend.components) == set(BUILT)
        assert any(line.startswith("waiting for weights") for line in lines), lines
        assert hub.gets[gated] == 1
        planner.end()
