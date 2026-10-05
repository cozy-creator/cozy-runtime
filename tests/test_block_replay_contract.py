"""Current production block replay on CPU math, with controlled device lease/room I/O.

This exercises the imported WeightResidency wrapper and dispatch/mutation/retry
logic. Only stream/acquisition/allocator accounting are CPU boundary fixtures;
no CUDA context, device fault or model inference is involved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from cozy_runtime.internal import accel, weight_policy
from cozy_runtime.internal.weights import Component, Region, Views, WeightResidency, Weights


class StatefulBlock(torch.nn.Module):
    def __init__(self, effect: str) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.linear.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
        self.register_buffer("ticks", torch.zeros(()))
        self.calls = 0
        self.effect = effect
        self.generator = torch.Generator().manual_seed(1234)
        self.fail_next = False

    def forward(self, value: Any) -> Any:
        self.calls += 1
        if self.effect == "counter":
            extra = self.calls
        elif self.effect == "buffer":
            self.ticks.add_(1)
            extra = self.ticks
        else:
            extra = torch.rand(value.shape, generator=self.generator)
        result = self.linear(value) + extra
        if self.fail_next:
            self.fail_next = False
            raise torch.OutOfMemoryError("deliberate one-shot OOM after module state advanced")
        return result


class CpuLease:
    def __init__(self, tensor: Any, trace: dict[str, Any]) -> None:
        self.tensor, self.trace = tensor, trace
        self.ptr = tensor.data_ptr()
        self.ring_offset = None
        trace["acquired"] += 1

    def view(self) -> Any:
        return self.tensor.view(torch.uint8).flatten()

    def release(self) -> None:
        self.trace["released"] += 1


def wrapped(module: StatefulBlock, monkeypatch: Any) -> tuple[Weights, Region, dict[str, Any]]:
    # Actual CPU plane and actual production objects. Enable the nongrouped
    # device-control branch while all tensor math/lease storage remains CPU.
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.host_only = False
    weights.stream = lambda: 0
    weights.hold_host = lambda: None
    trace: dict[str, Any] = {"acquired": 0, "released": 0, "room": [], "free": 16}
    monkeypatch.setattr(
        accel,
        "allocation",
        lambda *args, **kwargs: {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "driver_free_bytes": trace["free"],
            "driver_total_bytes": 1 << 20,
        },
    )

    def shrink(need: int, floor: int) -> int:
        trace["room"].append({"need": need, "floor": floor})
        trace["free"] += max(need, weight_policy.MARGIN)
        return max(need, weight_policy.MARGIN)

    weights.shrink = shrink
    tensor = module.linear.weight.detach()
    region = Region(1, "block", module, [], span=tensor.numel() * tensor.element_size())
    # bind() uses its ordinary cached-view path over real owned CPU storage.
    region.views[tensor.data_ptr()] = Views(
        [(module.linear._parameters, "weight", tensor, module.linear.weight)], []
    )
    component = Component(
        "model",
        "model/1",
        module,
        "cpu-proof",
        0,
        [region],
        False,
        None,
        [region],
        weight_policy.Layout(0, (region.span,)),
        resident=(1,),
    )
    weights.components[component.key] = component
    residency = WeightResidency(weights, {"model": component})
    residency._acquire = lambda component, index, stream: CpuLease(tensor, trace)
    # The original _make_room and Weights.progressed execute; shrink/accounting
    # above model device I/O and report actual progress before retry is granted.
    return weights, region, trace


def state(module: StatefulBlock) -> dict[str, Any]:
    return {
        "calls": module.calls,
        "ticks": float(module.ticks),
        "rng_sha256": hashlib.sha256(module.generator.get_state().numpy().tobytes()).hexdigest(),
    }


@pytest.mark.parametrize("effect", ["counter", "buffer", "rng"])
def test_oom_cannot_replay_a_forward_based_only_on_unchanged_tensor_inputs(
    effect: str,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    control = StatefulBlock(effect)
    current = StatefulBlock(effect)
    _weights, region, trace = wrapped(current, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    before = value.clone()
    with torch.inference_mode():
        assert torch.equal(control(value), current(value))
        assert region.mutates is False  # production's first-run inference
        expected = control(value)
        current.fail_next = True
        actual = current(value)
    record = {
        "effect": effect,
        "input_exact": torch.equal(value, before),
        "inferred_input_mutation": region.mutates,
        "expected": expected.tolist(),
        "actual": actual.tolist(),
        "control_state": state(control),
        "actual_state": state(current),
        "trace": trace,
        "wrapper_source": __import__(
            "cozy_runtime.internal.weights", fromlist=["__file__"]
        ).__file__,
        "torch": torch.__version__,
        "scope": "CPU block-control flow and device I/O fixture; not CUDA or inference proof",
    }
    (tmp_path / f"{effect}.json").write_text(json.dumps(record, indent=2) + "\n")
    assert record["input_exact"]
    assert trace["acquired"] == trace["released"] == 3
    assert len(trace["room"]) == 1
    assert torch.equal(actual, expected), record


def test_successful_current_block_path_preserves_module_semantics(monkeypatch: Any) -> None:
    control, current = StatefulBlock("buffer"), StatefulBlock("buffer")
    _weights, region, trace = wrapped(current, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    with torch.inference_mode():
        for _ in range(3):
            assert torch.equal(control(value), current(value))
    assert state(current) == state(control)
    assert region.mutates is False
    assert trace["acquired"] == trace["released"] == 3 and not trace["room"]
