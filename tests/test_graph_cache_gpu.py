"""Real cache ownership; CUDA cases run only on an explicitly assigned rental."""

from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import tensorfs  # noqa: E402
from tensorfs.varena.torch import Block, Declared, Fill, Memory, Model  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.internal.encoding import (  # noqa: E402
    Capabilities,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.probe import QualificationResult  # noqa: E402
from cozy_runtime.internal.resolution import ConstructionFacts, ResolvedModelPlan  # noqa: E402
from cozy_runtime.internal.weights import Backend, Residency, _Pin  # noqa: E402


def loaded_model(
    folder: Path, device: str, fill: Fill | None = None
) -> tuple[Any, Any, Memory, Model]:
    reference = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
    )
    writer = tensorfs.Writer(cache_dir=folder)
    component = writer.component("net", cls=("", "", ""), config={})
    for key, value in reference.state_dict().items():
        component.put(key, value.detach().numpy(), dtype="F32", shape=list(value.shape))
    checkpoint = writer.commit()
    with torch.device("meta"):
        module = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
        )
    memory = Memory.open(torch.device(device))
    module._varena = Declared(fill=fill)
    model = memory.load(module, ck=checkpoint, component="net", level="gpu")
    return module, checkpoint, memory, model


def residency(memory: Memory, model: Model) -> Residency:
    # A real empty backend: no encoded tensors require kernel qualification.
    # The component itself was already placed through Memory.load above.
    facts = measure_device(torch, memory.gpu)
    runtime = measure_runtime(torch, "graph-cache-test")
    plan = ResolvedModelPlan(
        variant="",
        store="test",
        snapshots={},
        reference_variant="",
        policy="decode_floor",
        tensors=(),
        device=facts,
        runtime=runtime,
        placement="test",
        objective="latency",
        construction=ConstructionFacts("graph-cache-test", "", ()),
        encoded_leaves="accept",
    )
    backend = Backend(
        {},
        [],
        torch=torch,
        plan=plan,
        memory=memory,
        device=memory.device,
        construction="graph-cache-test",
        device_facts=facts,
        qualified=(
            Capabilities(),
            QualificationResult(
                tuple_key="empty",
                device=facts,
                device_predicate=facts.predicate,
                runtime_release="graph-cache-test",
                runtime=runtime,
                suite_ms=0,
                observations=[],
            ),
        ),
    )
    backend.models["net"] = model
    return Residency(memory, backend)


def test_retired_token_rejects_borrow_and_last_reference_closes_its_real_cache(
    tmp_path: Path,
) -> None:
    _, checkpoint, memory, model = loaded_model(tmp_path, "cpu")
    cache = memory.cache("retired", lambda _: 0)
    token = _Pin(model, cache)
    token.retired = True
    retained = token
    pointer = weakref.ref(token)
    try:
        # This must not initialize CUDA: a retired token cannot reacquire bindings.
        with token.serving() as ready:
            assert not ready
        del token
        gc.collect()
        assert pointer() is retained and not bool(cache.closed)
        del retained
        gc.collect()
        assert pointer() is None and bool(cache.closed) and cache not in memory.caches
    finally:
        memory.close()
        checkpoint.close()


@pytest.mark.real_gpu
def test_pressure_preserves_a_delayed_replay_then_reclaims_its_last_token(tmp_path: Path) -> None:
    assert torch.cuda.device_count() == 1
    module, checkpoint, memory, model = loaded_model(tmp_path, "cuda:0")
    placed = residency(memory, model)
    graphs.install(torch, object(), {"net": module})
    sample = torch.randn(4, 64, device="cuda")
    paused, relieved = threading.Event(), threading.Event()
    failures: list[BaseException] = []
    pressure: threading.Thread | None = None
    try:
        with torch.inference_mode(), memory.request("record", plan=[model]):
            for _ in range(3):
                module(sample)
            recorder = graphs._RECORDERS[module]
            assert recorder.recordings, "the real managed model must record a graph"
            entry = next(iter(recorder.recordings.values()))
            token = placed.pins[("net", "graphs:net")]
            cache = token.cache
            pools, inputs = graphs._allocations(recorder.recordings.values())
            pool_bytes = graphs._pool_bytes(torch, pools)
            input_bytes = graphs._cache_bytes(torch, set(), inputs)
            assert input_bytes > 0, "static inputs outside the private pool must be counted"
            assert token.nbytes == graphs._cache_bytes(torch, pools, inputs)
            assert token.nbytes == pool_bytes + input_bytes > pool_bytes > 0
            assert graphs._allocations([entry, entry]) == graphs._allocations([entry]), (
                "aliases to a recording's input storages must count only once"
            )

            def ask_for_room() -> None:
                try:
                    assert paused.wait(15)
                    with memory.request("pressure"):
                        memory.free(memory.device, memory.available() + 1)
                        assert not placed.pins[("net", "graphs:net")].retired
                        assert model.common.count >= 2, "the consumer borrowed no binding hold"
                except BaseException as exc:
                    failures.append(exc)
                finally:
                    relieved.set()

            dispatch_mode: Any = TorchDispatchMode

            class PauseCopy(dispatch_mode):  # type: ignore[misc]  # Optional Torch is untyped.
                def __torch_dispatch__(
                    self,
                    func: Any,
                    types: Any,
                    args: tuple[Any, ...] = (),
                    kwargs: dict[str, Any] | None = None,
                ) -> Any:
                    if func is torch.ops.aten.copy_.default and not paused.is_set():
                        paused.set()
                        assert relieved.wait(15), "pressure did not finish during the replay lease"
                    return func(*args, **(kwargs or {}))

            pressure = threading.Thread(target=ask_for_room)
            pressure.start()
            consumer = torch.cuda.Stream()
            consumer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(consumer), PauseCopy():
                torch.cuda._sleep(50_000_000)
                output = module(sample)
            pressure.join(20)
            assert not pressure.is_alive() and not failures, failures
            assert paused.is_set()

            # The registry can now retire its cached refs, but this local recording
            # still owns the graph allocation and the persistent binding token.
            torch.cuda.empty_cache()
            memory.free(memory.device, memory.available() + 1)
            assert token.retired and not bool(cache.closed) and model.common.count == 1
            assert graphs._pool_bytes(torch, pools) > 0
            assert graphs._cache_bytes(torch, set(), inputs) == input_bytes
            assert not recorder.recordings and token.nbytes == 0
            consumer.synchronize()
            bypass = graphs.BYPASS.set(True)
            try:
                expected = module(sample)
            finally:
                graphs.BYPASS.reset(bypass)
            assert torch.equal(output, expected)
            pointer = weakref.ref(token)
            del entry, token
            gc.collect()
            assert pointer() is None and bool(cache.closed) and model.common.count == 0
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            assert graphs._cache_bytes(torch, pools, inputs) == 0, (
                "released recordings retained their private pool or default-pool static inputs"
            )
    finally:
        relieved.set()
        if pressure is not None:
            pressure.join(20)
        memory.close()
        checkpoint.close()


@pytest.mark.real_gpu
def test_capture_fallback_releases_its_lease_before_eager_pressure(tmp_path: Path) -> None:
    assert torch.cuda.device_count() == 1
    module, checkpoint, memory, model = loaded_model(tmp_path, "cuda:0")
    placed = residency(memory, model)
    graphs.install(torch, object(), {"net": module})
    sample = torch.randn(4, 64, device="cuda")
    refused, checked = False, False

    def refuse_capture(_: Any, __: Any) -> None:
        nonlocal refused, checked
        if torch.cuda.is_current_stream_capturing() and not refused:
            refused = True
            raise torch.OutOfMemoryError("injected capture refusal before eager recovery")
        if refused and not checked:
            checked = True
            cache = placed.pins[("net", "graphs:net")].cache
            pointer = weakref.ref(placed.pins[("net", "graphs:net")])
            torch.cuda.empty_cache()
            memory.free(memory.device, memory.available() + 1)
            assert pointer() is None and cache.closed, "eager recovery retained its graph lease"

    module.register_forward_pre_hook(refuse_capture)
    try:
        with torch.inference_mode(), memory.request("fallback", plan=[model]):
            first = module(sample)
            second = module(sample)
            assert refused and checked
            assert torch.equal(first, second)
    finally:
        memory.close()
        checkpoint.close()


@pytest.mark.real_gpu
def test_changed_fill_retires_the_old_graph_before_binding_new_weights(tmp_path: Path) -> None:
    assert torch.cuda.device_count() == 1

    class Refill:
        version = 1

        def prepare(self) -> int:
            return self.version

        def __call__(self, _block: Block, parts: Any) -> None:
            for roles in parts.values():
                for value in roles.values():
                    value.fill_(self.version)

    refill = Refill()
    module, checkpoint, memory, model = loaded_model(tmp_path, "cuda:0", refill)
    placed = residency(memory, model)
    graphs.install(torch, object(), {"net": module})
    sample = torch.ones(4, 64, device="cuda")
    try:
        with torch.inference_mode():
            for index in range(3):
                with memory.request(f"original-{index}", plan=[model]):
                    first = module(sample)
            assert graphs._RECORDERS[module].recordings
            old = weakref.ref(placed.pins[("net", "graphs:net")])
            refill.version = 2
            with memory.request("changed", plan=[model]):
                changed = module(sample)
            assert old() is None, "the stale graph retained old common bindings"
            assert torch.equal(first, torch.full_like(first, 8321))
            assert torch.equal(changed, torch.full_like(changed, 33282))
            with memory.request("changed-warm", plan=[model]):
                warm = module(sample)
            assert torch.equal(warm, changed)
    finally:
        memory.close()
        checkpoint.close()
