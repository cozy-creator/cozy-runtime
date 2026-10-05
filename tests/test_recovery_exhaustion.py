"""Exhausted operator recovery preserves capacity facts but retires entered Python state."""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.internal import image_vae
from cozy_runtime.internal.executor import Executor
from cozy_runtime.internal.executor_commands import Activate, Load, Start
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights import INDIVISIBLE, ResidencyRefusal
from test_block_replay_contract import FailAfterOp, StatefulBlock, state, wrapped
from test_request_isolation import MODULE, _serve

torch = pytest.importorskip("torch")
pytest_plugins = ("test_request_isolation",)


@pytest.mark.parametrize("effect", ["counter", "buffer", "rng"])
def test_exhausted_op_recovery_retires_once_only_forward_state(
    effect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, current = StatefulBlock(effect), StatefulBlock(effect)
    weights, _region, trace = wrapped(current, monkeypatch)
    monkeypatch.setattr(weights, "shrink", lambda _need, _floor: 0)
    failed = FailAfterOp(torch.ops.aten.linear.default)
    value = torch.tensor([[0.25, -0.5]])
    before = value.clone()
    with torch.inference_mode():
        control(value)
    with failed, torch.inference_mode(), pytest.raises(ResidencyRefusal) as failure:
        current(value)
    # This is the original measured refusal from Weights.progressed, not a new floor or
    # contract error replacing the op's capacity cause. The CPU room/lease I/O is controlled.
    assert failure.value.code == INDIVISIBLE and "aten.linear.default" in str(failure.value)
    assert failure.value.shortfall is None and trace["residency"].last_shortfall is None
    assert state(current) == state(control) and torch.equal(value, before)
    assert failed.calls == 1 and trace["acquired"] == trace["released"] == 1
    assert trace["residency"].poisoned and not trace["room"]
    with pytest.raises(ResidencyRefusal, match="residency_poisoned"):
        current(value)
    assert current.calls == 1


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


def test_executor_capacity_failure_with_changed_state_refuses_generation_reuse(
    host: tuple[Executor, dict[str, Any]],
    tmp_path: Path,
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
    assert failed["poisoned"] and runner.poisoned and trace["residency"].poisoned
    assert module.calls == 1 and float(module.ticks) == 1
    refused = runner.load(msgspec.convert({**load, "construction": "b"}, Load))
    assert not refused["ok"] and refused["code"] == "poisoned_generation"
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    left, right = socket.socketpair()
    with left, right:
        replacement = Executor(Channel(left), replacement_root)
        assert replacement.start(
            msgspec.convert({**load["binding"], "devices": load["devices"]}, Start)
        )["ok"]
        assert replacement.load(msgspec.convert({**load, "construction": "a"}, Load))["ok"]
        assert replacement.activate(Activate(construction="a"))["ok"]
        served = _serve(replacement, "replacement", "ok")
    assert served["outcome"]["terminal"] == "succeeded" and not served["poisoned"]
