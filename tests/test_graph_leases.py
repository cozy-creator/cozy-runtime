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
from cozy_runtime.internal.encoding import Capabilities, measure_device, measure_runtime  # noqa: E402
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
            pool_bytes = sum(
                entry.nbytes
                for layer in module.layers
                for entry in graphs._RECORDERS[layer].recordings.values()
            )
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
                actual = module((source * scale).to(memory.device)).cpu()
            torch.testing.assert_close(actual, expected * scale, atol=1e-5, rtol=1e-4)

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
