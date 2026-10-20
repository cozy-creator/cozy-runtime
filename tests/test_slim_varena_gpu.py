"""Actual CUDA consumer checks; opt in only on an explicitly rented GPU."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import tensorfs  # noqa: E402
from cozy_runtime.graphs import capture  # noqa: E402
from tensorfs.varena.torch import Block, Declared, Memory, Source  # noqa: E402
from torch import Tensor, nn  # noqa: E402


def burst(nbytes: int, child: Callable[[], object] | None = None) -> nn.Module:
    model = nn.Module()
    model.register_parameter("weight", nn.Parameter(torch.ones(())))

    def forward() -> Tensor:
        temporary = torch.empty(nbytes, dtype=torch.uint8, device=model.weight.device)
        temporary.zero_()
        del temporary
        if child is not None:
            child()
        return model.weight + 1

    model.forward = forward
    return model


def weights(folder: Path, names: tuple[str, ...]) -> tensorfs.Checkpoint:
    writer = tensorfs.Writer(cache_dir=folder)
    for name in names:
        writer.component(name, cls="", config={}).put("weight", torch.ones(()))
    return writer.commit()


@pytest.mark.real_gpu
def test_nested_model_keeps_the_outer_measured_peak(tmp_path: Path) -> None:
    assert torch.cuda.device_count() == 1
    ck = weights(tmp_path, ("outer", "inner"))
    with torch.device("meta"):
        inner = burst(10 << 20)
        outer = burst(100 << 20, lambda: inner())
    memory = Memory.open(torch.device("cuda", 0))
    try:
        a = memory.load(outer, ck=ck, component="outer")
        b = memory.load(inner, ck=ck, component="inner")
        with torch.no_grad(), memory.request("nested", plan=[{a, b}]) as request:
            assert outer().item() == 2
        assert request.stats is not None
        steps = {step.model: step for step in request.stats.steps}
        assert steps[a.name].activations >= 100 << 20
        assert steps[b.name].activations >= 10 << 20
    finally:
        memory.close()
        ck.close()


@pytest.mark.real_gpu
def test_native_refill_does_not_hold_python_memory_lock(tmp_path: Path) -> None:
    assert torch.cuda.device_count() == 1
    ck = weights(tmp_path, ("model",))
    with torch.device("meta"):
        model = burst(0)
    memory = Memory.open(torch.device("cuda", 0))
    entered, acquired = threading.Event(), threading.Event()

    class Refill:
        version = 1

        def prepare(self) -> int:
            return self.version

        def __call__(self, _block: Block, parts: Mapping[str, Mapping[str, Tensor]]) -> None:
            if self.version == 2:
                entered.set()
                assert acquired.wait(10), "native refill held Memory.lock"
            for values in parts.values():
                for value in values.values():
                    value.mul_(1)

    refill = Refill()
    model._varena = Declared(Source(ck, "model", str(ck.id)), fill=refill)

    def take_lock() -> None:
        if entered.wait(10):
            with memory.lock:
                acquired.set()

    thread = threading.Thread(target=take_lock)
    try:
        loaded = memory.load(model)
        refill.version = 2
        thread.start()
        with torch.no_grad(), memory.request("refill-lock", plan=[loaded]):
            assert model().item() == 2
        assert acquired.is_set()
    finally:
        if thread.ident is not None:
            thread.join(timeout=15)
        memory.close()
        ck.close()


@pytest.mark.real_gpu
def test_refused_capture_preserves_cache_held_common_for_eager_fallback(tmp_path: Path) -> None:
    """A failed setup owns no part of the graph cache's pre-existing bindings."""
    assert torch.cuda.device_count() == 1

    class Weighted(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2))

        def forward(self, value: Tensor) -> Tensor:
            return value @ self.weight

    class Nested(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.empty(2))
            self.body = nn.Sequential(Weighted(), Weighted())

        def forward(self, value: Tensor) -> Tensor:
            return self.body(value) + self.bias

    writer = tensorfs.Writer(cache_dir=tmp_path)
    component = writer.component("nested", cls="", config={})
    component.put("bias", torch.tensor([1.0, 2.0]))
    component.put("body.0.weight", 2 * torch.eye(2))
    component.put("body.1.weight", 3 * torch.eye(2))
    ck = writer.commit()
    with torch.device("meta"):
        module = Nested()
    module._varena = Declared(blocks=("body.0", "body.1"))
    memory = Memory.open(torch.device("cuda", 0))
    cache = memory.cache("capture-regression", lambda _need: 0)
    try:
        model = memory.load(module, ck=ck, component="nested")
        value = torch.tensor([1.0, 2.0], device="cuda")
        expected = torch.tensor([7.0, 14.0], device="cuda")
        capture_stream, fallback_stream = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.no_grad(), memory.request("capture-held-common", plan=[model]):
            assert torch.equal(module(value), expected)
            hold = cache.hold(model)
            try:
                blocks = (model.common, *model.blocks)
                held_counts = tuple(block.count for block in blocks)
                assert all(count > 0 for count in held_counts)
                assert model.common.regions and model.common.pins
                assert all(block.stream is not None for block in blocks)
                with cache.serving():
                    natural_graph, _why = capture(
                        torch, lambda: module(value), torch.cuda.graph_pool_handle(), capture_stream
                    )
                    del natural_graph
                    # Natural capture refusals vary by CUDA version. Also make CUDA
                    # itself refuse this public measurement callback, proving failed
                    # setup before the forward acquires its own common binding.
                    measured_in_capture: list[bool] = []

                    def synchronize_measurement() -> None:
                        measured_in_capture.append(torch.cuda.is_current_stream_capturing())
                        torch.cuda.synchronize()

                    prior_measure = memory.measure
                    memory.measure = synchronize_measurement
                    try:
                        graph, why = capture(
                            torch,
                            lambda: module(value),
                            torch.cuda.graph_pool_handle(),
                            capture_stream,
                        )
                    finally:
                        memory.measure = prior_measure
                    assert graph is None and why, "the real CUDA capture must be refused"
                    assert measured_in_capture == [True]
                    counts_after_refusal = tuple(block.count for block in blocks)
                    streams_after_refusal = tuple(block.stream for block in blocks)
                    # Execute fallback before checking bookkeeping: the old wheel reaches
                    # wait_stream(None) here, just like the retained SDXL run12156.
                    with torch.cuda.stream(fallback_stream):
                        result = module(value)
                    torch.cuda.current_stream().wait_stream(fallback_stream)
                    assert torch.equal(result, expected)
                    del result
                assert counts_after_refusal == held_counts
                assert all(stream is not None for stream in streams_after_refusal)
                assert tuple(block.count for block in blocks) == held_counts
            finally:
                hold.close()
            assert all(block.count == 0 and block.stream is None for block in blocks)
            assert torch.equal(module(value), expected)
            assert all(block.count == 0 and block.stream is None for block in blocks)
    finally:
        cache.close()
        memory.close()
        ck.close()
