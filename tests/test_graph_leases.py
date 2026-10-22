"""Real native/Runtime graph replay across weight eviction; run in its own GPU process."""

from __future__ import annotations

import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tensorfs = pytest.importorskip("tensorfs")
from tensorfs.varena import Level  # noqa: E402
from tensorfs.varena.torch import Declared, Memory, Weights  # noqa: E402

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.author._model import PLACED  # noqa: E402
from cozy_runtime.internal.encoding import (  # noqa: E402
    Capabilities,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.probe import QualificationResult  # noqa: E402
from cozy_runtime.internal.resolution import ConstructionFacts, ResolvedModelPlan  # noqa: E402
from cozy_runtime.internal.weights import Backend, Placement  # noqa: E402

pytestmark = pytest.mark.real_gpu
WIDTH, DEPTH, CAP = 4096, 20, 1 << 30


class Network(torch.nn.Module):
    _no_split_modules = ("Linear",)

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(DEPTH)
        )

    def forward(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x


def test_runtime_replays_same_graphs_after_backing_eviction(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("native graph residency requires CUDA")
    memory = Memory.open(0, limit=CAP)
    assert memory.native.context_measurable(), "a hard cap proof requires measured context bytes"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(19)
    plain = Network().eval()
    source = torch.randn(2, WIDTH)
    with torch.inference_mode():
        for parameter in plain.parameters():
            parameter.mul_(3**0.5)
        expected = plain(source)
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("network", cls=("torch", "", "Module"), config={})
    for key, value in plain.state_dict().items():
        component.put(
            key,
            value.contiguous().view(torch.uint8).numpy().tobytes(),
            dtype="F32",
            shape=list(value.shape),
        )
    checkpoint = writer.commit()
    del plain, writer, component, value, parameter
    gc.collect()
    with torch.device("meta"):
        module = Network().eval()
    vars(module)["_varena"] = Declared(Weights(checkpoint, "network"))
    model = memory.load(module, "ram")
    facts = measure_device(torch, 0)
    runtime = measure_runtime(torch, "graph-leases-test")
    plan = ResolvedModelPlan(
        variant="test",
        store="test",
        snapshots={},
        reference_variant="test",
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="block_paging",
        objective="latency",
        construction=ConstructionFacts("test", "Network", ("network",)),
        encoded_leaves="refuse",
    )
    backend = Backend(
        {},
        [],
        torch=torch,
        plan=plan,
        memory=memory,
        device=memory.device,
        construction="test",
        release="test",
        device_facts=facts,
        qualified=(Capabilities(), QualificationResult("test", facts, "", "test", runtime, 0, [])),
    )
    backend.models["network"] = model
    placement = Placement(memory, backend)
    installed = graphs.install(torch, object(), {"network": module})
    assert len(installed) == DEPTH
    assert WIDTH * WIDTH * 4 * DEPTH > CAP
    try:
        recordings: list[Any] = []
        for turn in range(4):
            placement.open_attempt()
            with (
                torch.inference_mode(),
                memory.request(f"runtime-graphs-{turn}", [module]) as request,
            ):
                result = module(source.to(memory.device)).cpu()
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)
                stats = request.stats()
            if turn == 1:
                recordings = [
                    next(iter(graphs._RECORDERS[layer].recordings.values())).graph
                    for layer in module.layers
                ]
                assert len(recordings) == DEPTH
            if turn >= 2:
                assert all(
                    next(iter(graphs._RECORDERS[layer].recordings.values())).graph is saved
                    for layer, saved in zip(module.layers, recordings, strict=True)
                )
            domain = placement.graph_domain("network")
            facts = placement.document()["graphs"]["network"]
            if turn >= 1:
                assert facts["active_graphs"] == facts["captures"] == DEPTH
                assert facts["domains"] == 1 and facts["retired_graphs"] == 0
            pool_bytes = domain.capture_bytes()
            print(
                json.dumps(
                    {
                        "turn": turn,
                        "graph_pools": pool_bytes,
                        "domain_workspace": domain.nbytes(),
                        "tier_bytes": stats.moved.tier.bytes,
                        "graphs": len(recordings),
                        "context_measurable": memory.native.context_measurable(),
                        "native_sha256": hashlib.sha256(
                            Path(tensorfs._ext.__file__).read_bytes()
                        ).hexdigest(),
                    }
                ),
                flush=True,
            )
            torch.cuda.synchronize()
            memory.native.load(model.set, Level.RAM)

        def concurrent(scale: float) -> None:
            stream = torch.cuda.Stream()
            with (
                torch.inference_mode(),
                torch.cuda.stream(stream),
                memory.request(f"runtime-concurrent-{scale}", [module]),
            ):
                returned = module((source * scale).to(memory.device))
                torch.cuda._sleep(1 << 20)
                actual = returned.square().cpu()
            torch.testing.assert_close(actual, (expected * scale).square(), atol=1e-5, rtol=1e-4)

        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(concurrent, [1.0, 2.0]))
        assert all(
            next(iter(graphs._RECORDERS[layer].recordings.values())).graph is saved
            for layer, saved in zip(module.layers, recordings, strict=True)
        )
    finally:
        graphs.release(torch, module)
        PLACED.pop(module, None)
        model.close()


def test_a_whole_component_replay_holds_its_blocks_for_the_request(tmp_path: Path) -> None:
    """A region that is the whole component holds every block bound from its first replay to
    the next request: no replay waits on binding again (speed2 13522's idle before each
    replay). A new request takes a new hold; shedding the graphs lets it go."""
    if not torch.cuda.is_available():
        pytest.skip("native graph residency requires CUDA")
    memory = Memory.open(0, limit=CAP)
    torch.manual_seed(23)
    width = 256

    def build() -> Any:
        stack = torch.nn.Sequential(*(torch.nn.Linear(width, width) for _ in range(4)))
        stack._no_split_modules = ("Linear",)  # each Linear a block
        return stack

    plain = build().eval()
    source = torch.randn(2, width)
    with torch.inference_mode():
        expected = plain(source)
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("stack", cls=("torch", "", "Module"), config={})
    for key, value in plain.state_dict().items():
        component.put(
            key,
            value.contiguous().view(torch.uint8).numpy().tobytes(),
            dtype="F32",
            shape=list(value.shape),
        )
    checkpoint = writer.commit()
    with torch.device("meta"):
        module = build()
    vars(module)["_varena"] = Declared(Weights(checkpoint, "stack"))
    model = memory.load(module, "gpu")
    facts = measure_device(torch, 0)
    runtime = measure_runtime(torch, "graph-hold-test")
    plan = ResolvedModelPlan(
        variant="test",
        store="test",
        snapshots={},
        reference_variant="test",
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="block_paging",
        objective="latency",
        construction=ConstructionFacts("test", "Sequential", ("stack",)),
        encoded_leaves="refuse",
    )
    backend = Backend(
        {},
        [],
        torch=torch,
        plan=plan,
        memory=memory,
        device=memory.device,
        construction="test",
        release="test",
        device_facts=facts,
        qualified=(Capabilities(), QualificationResult("test", facts, "", "test", runtime, 0, [])),
    )
    backend.models["stack"] = model
    placement = Placement(memory, backend)
    assert graphs.install(torch, object(), {"stack": module})[0] == "stack"
    try:
        holds = []
        for turn in range(2):
            placement.open_attempt()
            with torch.inference_mode(), memory.request(f"hold-{turn}", [module] * 4):
                for _ in range(4):
                    got = module(source.to(memory.device)).cpu()
                    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
                    domain = placement.graph_domain("stack")
                    held = domain.holds.get(module)
                    if graphs._RECORDERS[module].replays:
                        assert held is not None and held.attempt == placement.attempts
                assert held is not None
                holds.append(held.hold)
        assert graphs._RECORDERS[module].replays == 7  # call 1 eager, then every one replays
        assert holds[0] is not holds[1]  # a request takes its own hold
        torch.cuda.synchronize()
        assert graphs._shed(torch, domain) > 0 and module not in domain.holds
    finally:
        graphs.release(torch, module)
        PLACED.pop(module, None)
        model.close()


def test_a_devices_components_share_one_capture_pool(tmp_path: Path) -> None:
    """Ruling 2 (v6 13590: the VAE's own pool took 2.98 GB beside the UNet's): components run
    one after another on a device, so their graphs share the device's one domain and pool,
    registered as one cache serving both, and counted once."""
    if not torch.cuda.is_available():
        pytest.skip("native graph residency requires CUDA")
    memory = Memory.open(0, limit=CAP)
    torch.manual_seed(29)
    width, names = 256, ("first", "second")

    def build() -> Any:
        stack = torch.nn.Sequential(*(torch.nn.Linear(width, width) for _ in range(2)))
        stack._no_split_modules = ("Linear",)
        return stack

    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    plain = {name: build().eval() for name in names}
    for name, module in plain.items():
        component = writer.component(name, cls=("torch", "", "Module"), config={})
        for key, value in module.state_dict().items():
            data = value.contiguous().view(torch.uint8).numpy().tobytes()
            component.put(key, data, dtype="F32", shape=list(value.shape))
    checkpoint = writer.commit()
    source = torch.randn(2, width)
    with torch.inference_mode():
        expected = plain["second"](plain["first"](source))
    modules, models = {}, {}
    for name in names:
        with torch.device("meta"):
            modules[name] = build()
        vars(modules[name])["_varena"] = Declared(Weights(checkpoint, name))
        models[name] = memory.load(modules[name], "ram")
    facts = measure_device(torch, 0)
    runtime = measure_runtime(torch, "graph-shared-pool-test")
    plan = ResolvedModelPlan(
        variant="test",
        store="test",
        snapshots={},
        reference_variant="test",
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="block_paging",
        objective="latency",
        construction=ConstructionFacts("test", "Sequential", names),
        encoded_leaves="refuse",
    )
    backend = Backend(
        {},
        [],
        torch=torch,
        plan=plan,
        memory=memory,
        device=memory.device,
        construction="test",
        release="test",
        device_facts=facts,
        qualified=(Capabilities(), QualificationResult("test", facts, "", "test", runtime, 0, [])),
    )
    backend.models.update(models)
    placement = Placement(memory, backend)
    assert sorted(graphs.install(torch, object(), modules)) == sorted(names)
    try:
        for turn in range(3):
            placement.open_attempt()
            steps = [modules["first"], modules["second"]]
            with torch.inference_mode(), memory.request(f"shared-{turn}", steps):
                hidden = modules["first"](source.to(memory.device))
                got = modules["second"](hidden).cpu()
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        domain = placement.graph_domain("first")
        assert domain is placement.graph_domain("second") and domain.captures is not None
        assert [ref() for ref in domain.served] == [modules["first"], modules["second"]]
        assert all(graphs._RECORDERS[m].replays == 2 for m in modules.values())
        described = placement.document()["graphs"]
        assert described["first"]["graph_pool_bytes"] == domain.capture_bytes() > 0
        assert described["second"]["graph_pool_bytes"] == 0  # the shared pool counts once
        assert domain.cache is not None
    finally:
        for name, module in modules.items():
            graphs.release(torch, module)
            PLACED.pop(module, None)
            models[name].close()
