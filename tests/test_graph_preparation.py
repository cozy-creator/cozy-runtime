"""Real allocator refusal before capture, and the real Varena hold behind graph tokens."""

from __future__ import annotations

import gc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tensorfs = pytest.importorskip("tensorfs")

from tensorfs.varena.torch import Memory, Placeholder  # noqa: E402

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.author._model import PLACED  # noqa: E402
from cozy_runtime.internal.encoding import measure_device, measure_runtime  # noqa: E402
from cozy_runtime.internal.resolution import ConstructionFacts, ResolvedModelPlan  # noqa: E402
from cozy_runtime.internal.weights import Backend, Residency  # noqa: E402

pytestmark = pytest.mark.real_gpu


@pytest.fixture
def placed(tmp_path: Path) -> Iterator[tuple[Any, Any, Residency, list[int]]]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA allocation and event-ordered holds need a GPU")
    calls: list[int] = []
    base: Any = torch.nn.Module

    class Identity(base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        def forward(self, x: Any) -> Any:
            calls.append(1)
            return x

    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("identity", cls=("torch", "", "Identity"), config={})
    component.put("weight", torch.ones(1).numpy().tobytes(), dtype="F32", shape=[1])
    checkpoint = writer.commit()
    device = torch.device("cuda", 0)
    memory = Memory.open(device)
    with torch.device("meta"):
        module = Identity()
    model = memory.load(module, ck=checkpoint)
    facts, runtime = measure_device(torch, 0), measure_runtime(torch, "graph-preparation-test")
    # An empty real backend supplies the ownership boundary. The public standalone loader
    # above already registered this model; no kernel selection or fake device is involved.
    plan = ResolvedModelPlan(
        variant="test",
        store="test",
        snapshots={},
        policy="test",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="all_resident",
        construction=ConstructionFacts("test", "Identity", ("identity",)),
        encoded_leaves="refuse",
    )
    backend = Backend(
        {},
        [],
        torch=torch,
        plan=plan,
        memory=memory,
        device=device,
        construction="test",
        device_facts=facts,
    )
    backend.models["identity"] = model
    residency = Residency(memory, backend)
    residency.open_attempt()
    graphs.install(torch, object(), {"identity": module})
    try:
        yield module, model, residency, calls
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.synchronize()
        recorder = graphs._RECORDERS.pop(module, None)
        if recorder is not None:
            graphs._shed(torch, recorder)
        for (name, owner), (token, _) in tuple(residency.pins.items()):
            residency.unpin(name, owner, token)
        PLACED.pop(module, None)
        model.close()
        torch.cuda.empty_cache()


def test_preparation_oom_releases_the_hold_before_one_eager_call(placed: Any) -> None:
    module, model, residency, calls = placed
    x = torch.zeros(64 << 20, dtype=torch.uint8, device="cuda")
    leaves, spec = torch.utils._pytree.tree_flatten(((x,), {}))
    key = graphs._signature(torch, leaves, spec)
    recorder = graphs._RECORDERS[module] = graphs._Recorder()
    recorder.seen.add(key)  # This signature has already had its first eager call.
    recorder.pool, recorder.stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved, total = torch.cuda.memory_reserved(), torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((reserved + (2 << 20)) / total)
    try:
        with torch.inference_mode():
            assert module(x) is x
            assert calls == [1]
            assert model.held == 0 and not residency.pins
            assert not recorder.recordings and recorder.pool is None
            assert recorder.tries[key] == 1
            assert recorder.waiting[key] == residency.attempts
            assert module(x) is x
            assert calls == [1, 1] and model.held == 0 and not residency.pins
            assert recorder.tries[key] == 1, "the same attempt must not acquire a new hold"
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)


def test_unpin_keeps_the_physical_hold_until_the_last_token_reference(placed: Any) -> None:
    _, model, residency, _ = placed
    token = residency.pin("identity", "graphs:identity", lambda: 0)
    assert token is not None and model.held == 1
    cache = residency.pins[("identity", "graphs:identity")][1]
    residency.unpin("identity", "graphs:identity", object())
    assert residency.pins, "a stale owner must not release a replacement pin"
    residency.unpin("identity", "graphs:identity", token)
    assert not residency.pins and cache not in model.memory.caches
    assert model.held == 1, "an in-flight caller still holds the returned token"
    del token
    gc.collect()
    assert model.held == 0


def test_a_scope_holds_a_blockless_component_for_every_read_in_it(placed: Any) -> None:
    """Read outside its forward (H3's turbo factors by the DiT's hooks), a component with no
    blocks binds whole per op; a scope's hold keeps it bound, so a read is a plain tensor."""
    module, model, residency, _ = placed
    assert isinstance(module.weight, Placeholder)
    release = residency.hold(["identity", "absent"])
    assert model.held == 1 and not isinstance(module.weight, Placeholder)
    assert float((module.weight * 2).sum()) == 2.0
    release()
    gc.collect()
    assert model.held == 0 and not residency.pins and isinstance(module.weight, Placeholder)
