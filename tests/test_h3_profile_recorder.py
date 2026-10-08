"""Actual CPU profiler/execution proof; no GPU speed or weight-transfer claim."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("detailed", [False, True])
def test_bounded_real_cpu_trace_keeps_request_outputs_and_removes_hooks(detailed):
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
            model, 4, render, lambda: {}, [torch.profiler.ProfilerActivity.CPU], detailed=detailed
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    facts = json.loads(summary)
    assert facts["record_shapes_and_memory"] is detailed
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


def test_whole_render_counters_and_existing_attribution_include_non_denoise_work():
    from cozy_runtime.author._observations import Attribution

    module, model = recorder(), Transformer()
    attribution = Attribution()
    counters = {"h2d_bytes": 10, "evictions": 1}

    def render():
        counters["h2d_bytes"] += 100  # before any observed transformer call
        attribution.stage("condition_text", 2.5)
        output = torch.ones(2, 8)
        for _ in range(8):
            output = model(output)
        counters["h2d_bytes"] += 200  # decoder work after the final transformer call
        counters["evictions"] += 2
        for name in ("decode_audio", "decode_video", "check_output"):
            attribution.stage(name, 1.0)
        return output

    with torch.no_grad():
        _, _, raw = module.record_request(
            model,
            4,
            render,
            lambda: dict(counters),
            [torch.profiler.ProfilerActivity.CPU],
            attribution=attribution.document,
        )
    summary = json.loads(raw)
    whole = summary["render_call"]
    assert whole["plane_delta"] == {"h2d_bytes": 300, "evictions": 2}
    assert all(row["plane_delta"] == {"h2d_bytes": 0, "evictions": 0} for row in summary["steps"])
    assert whole["attribution_before"]["stages"] == {}
    assert whole["attribution_after"] == attribution.document()
    assert 0 < whole["wall_ms"] <= whole["profiler_context_wall_ms"]


def test_output_observer_forwards_event_identity_and_records_finalize_tail():
    module = recorder()
    events = [object(), object(), object()]
    saved = object()

    def sink(source):
        assert list(source) == events
        return saved

    target = SimpleNamespace(save_video_stream=sink, video_audio_frame_samples=1024)
    observed = module.ObservedOutputs(target)
    assert observed.save_video_stream(iter(events)) is saved
    assert observed.video_audio_frame_samples == 1024
    row = observed.sinks[0]
    assert row["succeeded"] and row["events"] == 3 and row["overlaps_decode_video"]
    assert 0 <= row["source_event_wait_and_materialize_ms"] <= row["wall_ms"]
    assert 0 <= row["after_source_eof_finalize_probe_ms"] <= row["wall_ms"]


def test_output_observer_keeps_source_failures_and_does_not_invent_eof():
    module = recorder()

    def broken():
        yield object()
        raise ValueError("bad media")

    observed = module.ObservedOutputs(SimpleNamespace(save_video_stream=list))
    with pytest.raises(ValueError, match="bad media"):
        observed.save_video_stream(broken())
    assert not observed.sinks[0]["succeeded"]
    assert observed.sinks[0]["after_source_eof_finalize_probe_ms"] is None
