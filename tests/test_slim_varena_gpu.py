"""Actual CUDA consumer checks; opt in only on an explicitly rented GPU."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import tensorfs  # noqa: E402
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
