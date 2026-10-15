"""Block replay after an out-of-memory, on CPU math with controlled device lease/room I/O.

Owner, 2026-10-05: "I want processes to succeed and be resistant to changes, not fail." A
block that fails after its forward changed state runs again from the state it started with:
the tensors it wrote into and the generators it drew from are put back first. Only
stream/acquisition/allocator accounting are CPU fixtures; no CUDA context is involved.
"""

from __future__ import annotations

import hashlib
import sys
from types import MethodType
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import accel, weight_policy
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Load
from cozy_runtime.internal.weights import (
    Component,
    Region,
    Views,
    WeightResidency,
    Weights,
)
from test_request_isolation import MODULE, _serve

torch = pytest.importorskip("torch")
pytest_plugins = ("test_request_isolation",)


class StatefulBlock(torch.nn.Module):  # type: ignore[misc,name-defined]
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


def wrapped(
    module: Any, monkeypatch: Any, *, root: Any = None, weight_module: Any = None
) -> tuple[Weights, Region, dict[str, Any]]:
    # Actual CPU plane and actual production objects. Enable the nongrouped
    # device-control branch while all tensor math/lease storage remains CPU.
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.host_only = False
    monkeypatch.setattr(weights, "stream", lambda: 0)
    monkeypatch.setattr(weights, "hold_host", lambda: None)
    monkeypatch.setattr(weights, "limit_torch", lambda: None)
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

    def shrink(need: int, keep: int = 0) -> int:
        trace["room"].append({"need": need, "keep": keep})
        trace["free"] += max(need, weight_policy.MARGIN)
        return max(need, weight_policy.MARGIN)

    monkeypatch.setattr(weights, "shrink", shrink)
    # The allocator's own counters are the same CPU boundary: a process that initialized
    # CUDA (an earlier GPU test) would otherwise read them for a CPU device.
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: 0)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda *args, **kwargs: {})
    owner = (
        weight_module
        if weight_module is not None
        else module.linear
        if isinstance(module, StatefulBlock)
        else module
    )
    tensor = owner.weight.detach()
    region = Region(1, "block", module, [], span=tensor.numel() * tensor.element_size())
    # bind() uses its ordinary cached-view path over real owned CPU storage.
    region.views[tensor.data_ptr()] = Views(
        [(owner._parameters, "weight", tensor, owner.weight)], []
    )
    cpu_weight_set: Any = "cpu-proof"
    component = Component(
        "model",
        "model/1",
        root if root is not None else module,
        cpu_weight_set,
        0,
        [Region(0, "", module, []), region],
        False,
        None,
        [region],
        weight_policy.Layout(0, (region.span,)),
        resident=(1,),
    )
    weights.components[component.key] = component
    residency = WeightResidency(weights, {"model": component})
    trace["residency"] = residency
    monkeypatch.setattr(
        residency, "_acquire", lambda component, index, stream: CpuLease(tensor, trace)
    )
    # The original _make_room and Weights.progressed execute; shrink/accounting
    # above model device I/O and report actual progress before retry is granted.
    return weights, region, trace


def state(module: StatefulBlock) -> dict[str, Any]:
    return {
        "calls": module.calls,
        "ticks": float(module.ticks),
        "rng_sha256": hashlib.sha256(module.generator.get_state().numpy().tobytes()).hexdigest(),
    }


@pytest.mark.parametrize("effect", ["buffer", "rng"])
def test_a_block_that_fails_after_changing_state_runs_again_from_it(
    effect: str, monkeypatch: Any
) -> None:
    control = StatefulBlock(effect)
    current = StatefulBlock(effect)
    _weights, _region, trace = wrapped(current, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    before = value.clone()
    with torch.inference_mode():
        assert torch.equal(control(value), current(value))  # the first run learns its effects
        expected = control(value)
        current.fail_next = True
        actual = current(value)
    assert torch.equal(actual, expected) and torch.equal(value, before)
    assert {k: v for k, v in state(current).items() if k != "calls"} == {
        k: v for k, v in state(control).items() if k != "calls"
    }
    assert len(trace["room"]) == 1 and not trace["residency"].poisoned
    assert trace["acquired"] == trace["released"] == 3


def test_a_python_attribute_is_the_one_state_a_replay_does_not_put_back(monkeypatch: Any) -> None:
    """A counter kept as a plain Python attribute is not seen: the replay advances it once
    more. The call still succeeds, which is the owner's rule."""
    control, current = StatefulBlock("counter"), StatefulBlock("counter")
    _weights, _region, trace = wrapped(current, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    with torch.inference_mode():
        control(value), current(value)
        control(value)
        current.fail_next = True
        current(value)
    assert current.calls == control.calls + 1 and len(trace["room"]) == 1


def test_successful_current_block_path_preserves_module_semantics(monkeypatch: Any) -> None:
    control, current = StatefulBlock("buffer"), StatefulBlock("buffer")
    _weights, _region, trace = wrapped(current, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    with torch.inference_mode():
        for _ in range(3):
            assert torch.equal(control(value), current(value))
    assert state(current) == state(control)
    assert trace["acquired"] == trace["released"] == 3 and not trace["room"]


class CachedBlock(StatefulBlock):
    """A Wan VAE's causal block: it writes its input into the feature cache it is handed and
    advances the decoder's shared index into it."""

    def forward(self, value: Any, feat_cache: list[Any], feat_idx: list[int]) -> Any:
        if feat_cache[feat_idx[0]] is not None:
            raise ValueError("a first frame's convolution takes no feature cache")
        feat_cache[feat_idx[0]] = value.clone()
        feat_idx[0] += 1
        return super().forward(value)


def test_a_block_runs_again_with_the_lists_it_was_handed_as_they_were(monkeypatch: Any) -> None:
    """Run 5686's decoder: each causal block writes its input into the feature cache it is
    handed and advances their shared index. Run again after an out-of-memory with both as
    the failed run left them, it would take the next block's slot; they are put back first,
    as its tensors and generators are."""
    control, current = CachedBlock("rng"), CachedBlock("rng")
    _weights, _region, trace = wrapped(current, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    with torch.inference_mode():
        for block in (control, current):  # the first run learns its effects
            block(value, [None], [0])
        expected_cache, cache, expected_index, index = [None, None], [None, None], [0], [0]
        expected = control(value, expected_cache, expected_index)
        current.fail_next = True
        actual = current(value, cache, index)
    assert torch.equal(actual, expected) and index == expected_index == [1]
    assert torch.equal(cache[0], expected_cache[0]) and cache[1] is None
    assert len(trace["room"]) == 1


class FailAfterOp(torch.utils._python_dispatch.TorchDispatchMode):  # type: ignore[misc,name-defined]
    """Real operator execution, then a one-shot CPU exception at its return boundary."""

    def __init__(self, target: Any) -> None:
        super().__init__()
        self.target, self.calls = target, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if func is self.target:
            self.calls += 1
            if self.calls == 1:
                raise torch.OutOfMemoryError("one-shot operator return-boundary failure")
        return out


def test_a_failed_op_recovers_without_replaying_its_python_forward(
    monkeypatch: Any,
) -> None:
    module = StatefulBlock("counter")
    _weights, _region, trace = wrapped(module, monkeypatch)
    failed = FailAfterOp(torch.ops.aten.linear.default)
    value = torch.tensor([[0.25, -0.5]])
    with failed, torch.inference_mode():
        actual = module(value)
    assert torch.equal(actual, value + 1)
    assert module.calls == 1 and failed.calls == 2
    assert trace["acquired"] == trace["released"] == 1
    assert len(trace["room"]) == 1 and not trace["residency"].poisoned


@pytest.mark.parametrize("effect", ["mutable", "copy", "rng", "custom"])
def test_a_failed_op_runs_again_from_what_it_read(effect: str, monkeypatch: Any) -> None:
    """An op that fails after it ran (the harshest case: a real OOM fails before writing)
    runs again from the tensors it writes into and the generator state it drew from. A
    pointwise op allocates nothing, so it can only fail before it writes: it is not copied.
    The copy is an allocation too, and recovers like the op (run 6021)."""
    weights, _region, trace = wrapped(StatefulBlock("counter"), monkeypatch)
    value = torch.zeros(2)
    generator = torch.Generator().manual_seed(17)
    control = torch.Generator().manual_seed(17)
    calls = []
    if effect == "custom":

        @torch.library.custom_op("cozy_recovery_proof::once", mutates_args=())  # type: ignore[untyped-decorator]
        def once(value: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            calls.append(1)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("custom operation ran out of memory")
            return value + 1

        def operation() -> Any:
            return once(value)

        failed = FailAfterOp(None)
    elif effect in ("mutable", "copy"):

        def operation() -> Any:
            return value.index_put_((torch.arange(2),), torch.ones(2), accumulate=True)

        failed = FailAfterOp(
            torch.ops.aten.index_put_.default
            if effect == "mutable"
            else torch.ops.aten.clone.default
        )
    else:

        def operation() -> Any:
            return torch.rand(2, generator=generator)

        failed = FailAfterOp(torch.ops.aten.rand.generator)
    with failed, weights.recovering(), torch.inference_mode():
        out = operation()
    assert len(trace["room"]) == 1 and not trace["residency"].poisoned
    if effect == "custom":
        assert calls == [1, 1] and torch.equal(out, torch.ones(2))
    else:
        assert failed.calls == 2
    if effect in ("mutable", "copy"):
        assert torch.equal(value, torch.ones(2)), "applied once"
    if effect == "rng":
        assert torch.equal(out, torch.rand(2, generator=control)), "the same draw"


def test_before_forward_acquisition_failure_keeps_existing_recovery(monkeypatch: Any) -> None:
    module = StatefulBlock("counter")
    weights, region, trace = wrapped(module, monkeypatch)
    residency = trace["residency"]
    acquire = residency._acquire
    attempts = []
    replans = []

    def constrained(component: Any, index: int, stream: int) -> Any:
        attempts.append(1)
        if len(attempts) <= 3:
            raise residency._budget_error("before forward acquisition")
        return acquire(component, index, stream)

    monkeypatch.setattr(residency, "_acquire", constrained)
    monkeypatch.setattr(weights, "grow", lambda need: False)
    monkeypatch.setattr(residency, "_replan", lambda resume: replans.append(resume))
    value = torch.tensor([[0.25, -0.5]])
    with torch.inference_mode():
        assert torch.equal(module(value), value + 1)
    assert module.calls == 1 and len(attempts) == 4 and len(replans) == 2
    assert trace["acquired"] == trace["released"] == 1
    assert len(trace["room"]) == 1 and trace["room"][0]["need"] == region.span
    assert not residency.poisoned


@pytest.mark.parametrize("custom", ["leaf", "forward", "hook"])
def test_custom_decoder_calls_run_again_after_a_failure(custom: str, monkeypatch: Any) -> None:
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(72)
    vae = diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(8, 8),
        norm_num_groups=4,
    ).eval()
    latent = torch.randn(1, 4, 2, 3)
    before = latent.clone()
    resnet = vae.decoder.mid_block.resnets[0]
    with torch.inference_mode():
        expected = vae.decode(latent).sample
    calls: list[int] = []
    armed: list[int] = []

    def advance() -> None:
        calls.append(1)
        if armed:
            armed.clear()
            raise torch.OutOfMemoryError("custom decoder callback advanced state")

    if custom == "leaf":

        class CustomDropout(torch.nn.Module):  # type: ignore[misc,name-defined]
            def forward(self, value: Any) -> Any:
                advance()
                return value

        resnet.dropout = CustomDropout()
    elif custom == "forward":

        def forward(_self: Any, value: Any) -> Any:
            advance()
            return value

        resnet.dropout.forward = MethodType(forward, resnet.dropout)
    elif custom == "hook":

        def hook(_module: Any, _args: Any, _out: Any) -> None:
            advance()

        resnet.dropout.register_forward_hook(hook)
    _weights, _region, trace = wrapped(resnet, monkeypatch, root=vae, weight_module=resnet.conv1)
    with torch.inference_mode():
        assert torch.equal(vae.decode(latent).sample, expected)
        count = len(calls)
        armed.append(1)
        assert torch.equal(vae.decode(latent).sample, expected)
    # the callback ran again with the block: its own Python effects are not put back
    assert len(calls) == count + 2 and torch.equal(latent, before)
    assert len(trace["room"]) == 1 and not trace["residency"].poisoned
    assert trace["acquired"] == trace["released"]


def test_an_executor_request_recovers_in_process_and_its_generation_serves_on(
    host: tuple[Executor, dict[str, Any]], monkeypatch: Any
) -> None:
    runner, load = host
    assert runner.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
    assert runner.activate(Activate(construction="a"))["ok"]
    module = StatefulBlock("buffer")
    _weights, _region, trace = wrapped(module, monkeypatch)
    value = torch.tensor([[0.25, -0.5]])
    with torch.inference_mode():
        module(value)  # the first run learns that it writes its buffer
    module.fail_next = True

    def fail_in_handler() -> None:
        try:
            with torch.inference_mode():
                module(value)
        finally:
            runner.residency = trace["residency"]

    monkeypatch.setattr(sys.modules[MODULE], "contract_failure", fail_in_handler)
    served = _serve(runner, "contract", "contract")
    assert served["outcome"]["terminal"] == "succeeded" and not served["poisoned"], served
    assert float(module.ticks) == 2, "the failed attempt's write was put back, then made once"
    assert runner.load(msgspec.convert({**load, "construction": "b"}, Load))["ok"]
