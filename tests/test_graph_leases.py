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
from cozy_runtime.internal.encoding import measure_device, measure_runtime  # noqa: E402
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
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="block_paging",
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
    )
    backend.models["network"] = model
    placement = Placement(memory, backend)
    installed = graphs.install(torch, object(), {"network": module})
    assert len(installed) == DEPTH + 1  # the root, and each block for while it streams
    assert WIDTH * WIDTH * 4 * DEPTH > CAP

    def graphed() -> dict[Any, Any]:
        return {
            layer: next(iter(graphs._RECORDERS[layer].recordings.values())).graph
            for layer in module.layers
            if graphs._RECORDERS[layer].recordings
        }

    try:
        recordings: dict[Any, Any] = {}
        for turn in range(4):
            placement.open_attempt()
            with (
                torch.inference_mode(),
                memory.request(f"runtime-graphs-{turn}", [module]) as request,
            ):
                result = module(source.to(memory.device)).cpu()
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-4)
                stats = request.stats()
            domain = placement.graph_domain("network")
            facts = placement.document()["graphs"]["network"]
            assert facts["captures"] == facts["capture_attempts"] == 0  # none inside a request
            if turn >= 2:  # the graphs the idle period recorded replay over evicted backing
                assert graphed() == recordings
                assert facts["replays"] == len(recordings) and facts["retired_graphs"] == 0
            torch.cuda.synchronize()
            # Idle: the layers measured twice whose pages stayed resident record, in free bytes.
            assert graphs.capture_idle(lambda: False)
            if turn == 1:
                recordings = graphed()
                assert recordings, placement.document()["graphs"]["network"]["regions"]
            pool_bytes = domain.capture_bytes()
            print(
                json.dumps(
                    {
                        "turn": turn,
                        "graph_pools": pool_bytes,
                        "store_bytes": stats.moved.store.bytes,
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
        assert graphed() == recordings
    finally:
        graphs.release(torch, module)
        PLACED.pop(module, None)
        model.close()


def test_a_step_holds_its_components_blocks_and_replays_bind_nothing(tmp_path: Path) -> None:
    """A package method's scope holds its component's blocks bound from free bytes for the
    plan step (`Placement.using`), so no replay binds again (speed2 13522's idle before each
    replay; v7: a CLIP replay bound and unbound its whole block, ~6 ms a call). Each request
    takes its own hold and its end lets it go. The first request runs as written; the idle
    period after it records."""
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
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="block_paging",
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
    )
    backend.models["stack"] = model
    placement = Placement(memory, backend)
    assert graphs.install(torch, object(), {"stack": module})[0] == "stack"
    try:
        holds = []
        for turn in range(3):
            placement.open_attempt()
            with torch.inference_mode(), memory.request(f"hold-{turn}", [module] * 4):
                with placement.using("generate", ("stack",)):  # the package method's scope
                    held = placement.holds.get("stack")
                    assert held is not None and placement.graph_held("stack") is held[1]
                    holds.append(held[0])
                    for _ in range(4):
                        got = module(source.to(memory.device)).cpu()
                        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
                placement.release()  # the request's end
                assert not placement.holds
            torch.cuda.synchronize()
            assert graphs.capture_idle(lambda: False)
        assert graphs._RECORDERS[module].replays == 8  # the first request as written
        # the receipt is the request's own: the third request's four replays
        facts = placement.document()["graphs"]["stack"]
        assert facts["replays"] == 4 and facts["regions"][""]["prepare_ms"] < 1, facts
        assert len({id(hold) for hold in holds}) == 3  # a request takes its own hold
        torch.cuda.synchronize()
        assert graphs._shed(torch, placement.graph_domain("stack")) > 0
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
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="block_paging",
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
    )
    backend.models.update(models)
    placement = Placement(memory, backend)
    assert {"first", "second"} <= set(graphs.install(torch, object(), modules))
    try:
        for turn in range(4):  # two requests measure; the idle period after the second records
            placement.open_attempt()
            steps = [modules["first"], modules["second"]]
            with torch.inference_mode(), memory.request(f"shared-{turn}", steps):
                with placement.using("generate", ("first",)):  # held: each its root's graph
                    hidden = modules["first"](source.to(memory.device))
                with placement.using("generate", ("second",)):
                    got = modules["second"](hidden).cpu()
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
            torch.cuda.synchronize()
            assert graphs.capture_idle(lambda: False)
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


def test_a_capture_that_fails_to_begin_leaves_the_allocator_serving() -> None:
    """The verifier's HIGH: torch ends no capture whose begin raised after its pool's begin hook
    ran, and varena kept routing the thread into the pool, so its next allocation waited
    forever. Here the domain's control stream is already capturing, so the capture's begin fails
    after the hook: varena is told, and an allocation on this thread is served at once."""
    memory = Memory.open(0, limit=CAP)
    owner = graphs.capture_device(torch, memory)
    owner.initialize()
    pool = owner.capture_pool()
    outer = torch.cuda.CUDAGraph()
    with torch.cuda.stream(owner.control):
        outer.capture_begin(capture_error_mode="thread_local")
        try:
            graph, why = graphs.capture(torch, lambda: None, pool, owner)
        finally:
            outer.capture_end()
    assert graph is None and why, why
    served = torch.empty(1 << 20, device="cuda")  # not routed into the failed capture's pool
    torch.cuda.synchronize()
    assert served.numel() == 1 << 20
    outer.reset()
