"""Exhausted operator recovery preserves capacity facts but retires entered Python state."""

from __future__ import annotations

import sys
import zlib
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import image_vae, lora_bake, weight_policy
from cozy_runtime.internal import weights as source
from cozy_runtime.internal.encoding import SPEC_PLAIN, Encoded, RolePart
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Load
from cozy_runtime.internal.fill import PlanRow
from cozy_runtime.internal.weights import (
    INDIVISIBLE,
    Bake,
    Component,
    Region,
    ResidencyRefusal,
    Views,
    Weight,
    WeightResidency,
)
from test_block_replay_contract import CpuLease, FailAfterOp, StatefulBlock, state, wrapped
from test_request_isolation import MODULE, _serve

torch = pytest.importorskip("torch")
pytest_plugins = ("test_request_isolation",)


@pytest.mark.parametrize("fault", ["rounding", "copy"])
@pytest.mark.parametrize("nested", [False, True])
def test_sdk_seeded_bind_refills_its_base_before_one_forward(
    fault: str, nested: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = StatefulBlock("counter")
    module.linear.to(torch.bfloat16)
    weights, region, trace = wrapped(module, monkeypatch)
    tensor = module.linear.weight.detach()
    base = tensor.clone()  # immutable CPU source bytes standing for the TensorFS base
    down = torch.tensor([[0.17, 0.23]], dtype=torch.bfloat16)
    up = torch.tensor([[0.21], [0.19]], dtype=torch.bfloat16)
    values = (tensor, down, up)
    names = ("linear.weight", "linear.lora_A.proof.weight", "linear.lora_B.proof.weight")
    rows = []
    for name, value in zip(names, values, strict=True):
        shape = tuple(value.shape)
        nbytes = value.numel() * value.element_size()
        row = PlanRow(
            f"model.{name}",
            name,
            "model",
            "bf16",
            shape,
            nbytes,
            Encoded(SPEC_PLAIN, "plain", (RolePart("value", "bf16", shape, nbytes),)),
        )
        rows.append(Weight(row, "plain", None, {"value": (0, "bf16", shape)}))
    region.weights = rows
    views = region.views[tensor.data_ptr()]
    region.views[tensor.data_ptr()] = Views(views.writes, [{"value": v} for v in values])
    region.bakes = [Bake(0, layer=module.linear, factors=[(1, 2, 1.0)])]
    region.bake = True
    value = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
    control = base.clone()
    with torch.inference_mode():
        lora_bake.bake(
            torch,
            {"value": control},
            down,
            up,
            seed=zlib.crc32(names[0].encode()),
        )
        expected = torch.nn.functional.linear(value, control) + 1
    calls, refills = [], []
    original = lora_bake._bf16_either if fault == "rounding" else lora_bake.bake

    def fail_after_sdk_work(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)  # actual seed/draw and, for copy, stored-weight writes
        calls.append(1)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("one-shot SDK bind failure after its actual work")
        return out

    monkeypatch.setattr(
        lora_bake, "_bf16_either" if fault == "rounding" else "bake", fail_after_sdk_work
    )

    def refill(_component: Any, held: list[Region]) -> None:
        # Native refill is the only I/O boundary here. The actual _restore is called and
        # only this region's owned source bytes/bake flags change, never Python state/RNG.
        assert held == [region] and region.dirty
        refills.append(1)
        with torch.utils._python_dispatch._disable_current_modes():
            tensor.copy_(base)
        source._unbaked(region)

    monkeypatch.setattr(weights, "refill", refill)
    outer = None
    if nested:

        class Outer(StatefulBlock):
            def forward(self, value: Any) -> Any:
                self.calls += 1
                return module(value)

        outer = Outer("counter")
        owned = outer.linear.weight.detach()
        part = Region(1, "outer", outer, [], span=owned.numel() * owned.element_size())
        part.views[owned.data_ptr()] = Views(
            [(outer.linear._parameters, "weight", owned, outer.linear.weight)], []
        )
        cpu_weight_set: Any = "outer-cpu-proof"
        component = Component(
            "outer",
            "outer/1",
            outer,
            cpu_weight_set,
            0,
            [Region(0, "", outer, []), part],
            False,
            None,
            [part],
            weight_policy.Layout(0, (part.span,)),
            resident=(1,),
        )
        weights.components[component.key] = component
        residency = WeightResidency(weights, {"outer": component})
        monkeypatch.setattr(residency, "_acquire", lambda *_: CpuLease(owned, trace))
    with torch.inference_mode():
        actual = (outer or module)(value)
    assert torch.equal(actual, expected) and torch.equal(tensor, control)
    assert calls == [1, 1] and refills == [1]
    assert module.calls == 1 and (outer is None or outer.calls == 1)
    assert region.baked and not region.dirty and not trace["residency"].poisoned
    assert trace["acquired"] == trace["released"] == (4 if nested else 3)


@pytest.mark.parametrize("effect", ["buffer", "rng"])
def test_exhausted_recovery_reports_its_capacity_and_leaves_the_state_it_found(
    effect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing can make room: the call fails with the measured refusal (the only way it may
    fail), its forward's writes and draws put back, so the executor serves the next call
    exactly as if the failed one never ran."""
    control, current = StatefulBlock(effect), StatefulBlock(effect)
    weights, _region, trace = wrapped(current, monkeypatch)
    monkeypatch.setattr(weights, "shrink", lambda _need, _floor: 0)
    failed = FailAfterOp(torch.ops.aten.linear.default)
    value = torch.tensor([[0.25, -0.5]])
    before = value.clone()
    with failed, torch.inference_mode(), pytest.raises(ResidencyRefusal) as failure:
        current(value)
    assert failure.value.code == INDIVISIBLE and "aten.linear.default" in str(failure.value)
    assert torch.equal(value, before) and not trace["residency"].poisoned
    assert trace["acquired"] == trace["released"] == 1
    with torch.inference_mode():
        assert torch.equal(current(value), control(value))
    assert {k: v for k, v in state(current).items() if k != "calls"} == {
        k: v for k, v in state(control).items() if k != "calls"
    }


def test_exhausted_functional_op_without_a_python_body_keeps_its_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = StatefulBlock("counter")
    weights, _region, trace = wrapped(module, monkeypatch)
    monkeypatch.setattr(weights, "shrink", lambda _need, _floor: 0)
    value = torch.tensor([[0.25, -0.5]])
    weight = torch.eye(2)
    failed = FailAfterOp(torch.ops.aten.linear.default)
    with (
        failed,
        weights.recovering(),
        torch.inference_mode(),
        pytest.raises(ResidencyRefusal) as failure,
    ):
        torch.nn.functional.linear(value, weight)
    assert failure.value.code == INDIVISIBLE and not trace["residency"].poisoned
    assert module.calls == 0 and failed.calls == 1
    with weights.recovering(), torch.inference_mode():
        assert torch.equal(torch.nn.functional.linear(value, weight), value)


def test_exhausted_sdk_decoder_body_keeps_its_resumable_output_only_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(74)
    vae = diffusers.AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(8, 8),
        norm_num_groups=4,
    ).eval()
    latent = torch.randn(1, 4, 2, 3)
    with torch.inference_mode():
        expected = image_vae.Decode(vae, latent, band=100).run().sample
    conv = vae.decoder.conv_in
    weights, _region, trace = wrapped(conv, monkeypatch, root=vae)
    monkeypatch.setattr(weights, "shrink", lambda _need, _floor: 0)
    within = trace["residency"]._within("model")
    calls = []

    def fail_once(module: Any, body: Any) -> Any:
        if module is not conv:
            return within(module, body)

        def pass_body() -> Any:
            out = body()
            calls.append(1)
            if len(calls) == 1:
                raise torch.OutOfMemoryError("SDK output-only pass failed after its writes")
            return out

        return within(module, pass_body)

    decode = image_vae.Decode(vae, latent, band=100)
    with torch.inference_mode():
        with pytest.raises(ResidencyRefusal) as failure:
            decode.run(fail_once)
        assert failure.value.code == INDIVISIBLE and not trace["residency"].poisoned
        assert torch.equal(decode.run(fail_once).sample, expected)
    assert calls == [1, 1] and trace["acquired"] == trace["released"] == 2


def test_an_executor_capacity_failure_keeps_its_generation_serving(
    host: tuple[Executor, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, load = host
    assert runner.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
    assert runner.activate(Activate(construction="a"))["ok"]
    target = runner.constructions["a"]
    module = StatefulBlock("buffer")
    weights, _region, trace = wrapped(module, monkeypatch)
    monkeypatch.setattr(weights, "shrink", lambda _need, _floor: 0)
    value = torch.tensor([[0.25, -0.5]])

    def fail_in_handler() -> None:
        try:
            with FailAfterOp(torch.ops.aten.linear.default), torch.inference_mode():
                module(value)
        finally:
            target.residency = trace["residency"]

    monkeypatch.setattr(sys.modules[MODULE], "contract_failure", fail_in_handler)
    failed = _serve(runner, "exhausted", "contract")
    assert (failed["outcome"]["terminal"], failed["outcome"]["code"]) == (
        "failed",
        INDIVISIBLE,
    ), failed
    assert not failed["poisoned"] and not runner.poisoned
    assert float(module.ticks) == 0, "its write was put back"
    assert runner.load(msgspec.convert({**load, "construction": "b"}, Load))["ok"]
