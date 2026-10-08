"""Actual CPU profiler/execution proof; no GPU speed or weight-transfer claim."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def recorder():
    path = (
        Path(__file__).parents[1] / "research/h3-request-profile/h3_request_profile/instrument.py"
    )
    spec = importlib.util.spec_from_file_location("h3_profile_recorder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Module()
        self.block.to_q = torch.nn.Linear(8, 8)

    def forward(self, value):
        return self.block.to_q(value).relu()


def test_bounded_real_cpu_trace_keeps_request_outputs_and_removes_hooks():
    module, model = recorder(), Transformer()
    value = torch.ones(2, 8)

    def render():
        output = value
        for _ in range(8):
            output = model(output)
        return output

    with torch.no_grad():
        expected = render()
        actual, trace, summary = module.record_request(
            model, 4, render, lambda: {}, [torch.profiler.ProfilerActivity.CPU]
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    facts = json.loads(summary)
    assert len(facts["steps"]) == 8
    assert facts["profiled_step_indices"] == [3, 4]
    assert all(row["plane_delta"] == {} for row in facts["steps"])  # no invented GPU facts
    events = json.loads(gzip.decompress(trace))["traceEvents"]
    markers = {e.get("name") for e in events if str(e.get("name", "")).startswith("h3.step.")}
    assert markers == {"h3.step.3.dense", "h3.step.4.sparse"}
    assert any(e.get("name") == "h3.module.block.to_q" for e in events)
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert not model.block.to_q._forward_pre_hooks and not model.block.to_q._forward_hooks


def test_failure_also_removes_observer_hooks():
    module, model = recorder(), Transformer()

    def fail():
        model(torch.ones(2, 8))
        raise ValueError("authored failure")

    with pytest.raises(ValueError, match="authored failure"):
        module.record_request(model, 4, fail, lambda: {}, [torch.profiler.ProfilerActivity.CPU])
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert not model.block.to_q._forward_pre_hooks and not model.block.to_q._forward_hooks
