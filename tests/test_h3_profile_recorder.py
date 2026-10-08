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
    assert facts["events"] and facts["operator_aggregation"] == "key_averages"
    assert len(facts["steps"]) == 8
    assert facts["profiled_step_indices"] == [3, 4]
    assert all(row["plane_delta"] == {} for row in facts["steps"])  # no invented GPU facts
    events = json.loads(gzip.decompress(trace))["traceEvents"]
    markers = {e.get("name") for e in events if str(e.get("name", "")).startswith("h3.step.")}
    assert markers == {"h3.step.3.dense", "h3.step.4.sparse_eligible"}
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


def test_registered_handler_survives_public_export_proxy_replacement(monkeypatch):
    import sys

    from cozy_runtime.author import App

    module, app = recorder(), App()

    @app.entrypoint(name="fl2va")
    def original():
        return 7

    def managed_proxy():
        raise AssertionError("a diagnostic renderer must not submit a child instead")

    monkeypatch.setitem(sys.modules, "h3", SimpleNamespace(app=app, fl2va=managed_proxy))
    handler = module.retained_handler("fl2va")
    assert handler is original and handler() == 7
    app.get("fl2va").internal = True
    with pytest.raises(ValueError, match="public retained"):
        module.retained_handler("fl2va")


@pytest.mark.parametrize("step_count,dense_steps", [(8, 4), (30, 10)])
def test_continuous_trace_covers_preparation_all_steps_and_decode_without_math_change(
    step_count,
    dense_steps,
):
    module, model = recorder(), Transformer()

    def render():
        with torch.profiler.record_function("phase.text_encoder"):
            value = torch.ones(2, 8).sin()
        for _ in range(step_count):
            value = model(value)
        with torch.profiler.record_function("phase.video_decode"):
            value = value.cos()
        return value

    with torch.no_grad():
        expected = render()
        got, packed, raw = module.record_request(
            model,
            dense_steps,
            render,
            lambda: {},
            [torch.profiler.ProfilerActivity.CPU],
            full_request_trace=True,
        )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    summary, trace = json.loads(raw), json.loads(gzip.decompress(packed))
    assert summary["full_request_trace"] and not summary["record_shapes_and_memory"]
    assert summary["events"] == [] and summary["operator_aggregation"] == "offline from raw trace"
    assert summary["profiled_step_indices"] == list(range(step_count))
    events = trace["traceEvents"]
    names = {event.get("name") for event in events}
    assert {"phase.text_encoder", "phase.video_decode"} <= names
    assert all(
        f"h3.step.{step}.{'dense' if step < dense_steps else 'sparse_eligible'}" in names
        for step in range(step_count)
    )
    assert summary["trace_sizes"][0]["gzip_bytes"] == len(packed)
    assert all(event["ts"] > 0 for event in events if "ts" in event)
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_full_trace_refuses_tensor_retaining_detail_mode():
    module, model = recorder(), Transformer()
    with pytest.raises(ValueError, match="shape/memory recording off"):
        module.record_request(
            model,
            4,
            lambda: None,
            lambda: {},
            [torch.profiler.ProfilerActivity.CPU],
            full_request_trace=True,
            detailed=True,
        )


def test_file_backed_continuous_trace_retains_tree_without_changing_outputs(tmp_path):
    from cozy_runtime.author import Outputs
    from cozy_runtime.author.fakes import fake_attempt

    module, model = recorder(), Transformer()
    out = Outputs(fake_attempt(spool=tmp_path / "spool", max_output_bytes=8 << 20))

    def render():
        value = torch.ones(2, 8)
        for _ in range(8):
            value = model(value)
        return value

    with torch.no_grad():
        expected = render()
        actual, tree, raw = module.record_request(
            model,
            4,
            render,
            lambda: {},
            [torch.profiler.ProfilerActivity.CPU],
            full_request_trace=True,
            trace_sink=out.save_tree,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    summary = json.loads(raw)
    assert summary["trace_format"] == "tree/trace.json.gz"
    trace = tree.path / "trace.json.gz"
    assert trace.stat().st_size == summary["trace_sizes"][0]["gzip_bytes"]
    with gzip.open(trace, "rt") as source:
        names = {row.get("name") for row in json.load(source)["traceEvents"]}
    assert all(f"h3.step.{i}.{'dense' if i < 4 else 'sparse_eligible'}" in names for i in range(8))


def test_old_output_grant_refuses_before_gpu_or_render_work(tmp_path):
    from cozy_runtime.author import Outputs
    from cozy_runtime.author.fakes import fake_attempt

    out = Outputs(fake_attempt(spool=tmp_path / "spool"))
    with pytest.raises(ValueError, match="output grant before rendering"):
        recorder().profile_request(None, 4, None, outputs=out, telemetry=None)


def test_manual_ranges_preserve_real_telemetry_and_have_native_thread_identity(tmp_path):
    import os
    import threading

    from cozy_runtime.author import Telemetry
    from cozy_runtime.author.fakes import fake_attempt, fake_context

    module = recorder()
    attempt = fake_attempt(spool=tmp_path / "spool")
    target = Telemetry(attempt, fake_context())
    ranges = []
    observed = module.ObservedTelemetry(target, ranges)
    with observed.stage("condition_text", overall_range=(0.0, 1.0)):
        observed.metric("test", 1.0)
    assert "condition_text" in attempt.attribution.document()["stages"]
    assert len(ranges) == 1
    row = ranges[0]
    assert row["label"] == "h3.phase.condition_text"
    assert row["pid"] == os.getpid() and row["tid"] == threading.get_native_id()
    assert row["native_tid"] == threading.get_native_id()
    assert row["python_ident"] == threading.get_ident()
    assert row["pthread_tid_uint32"] == threading.get_ident() & 0xFFFFFFFF
    assert row["pthread_tid_int32"] == (
        (threading.get_ident() + 0x80000000) % 0x100000000 - 0x80000000
    )
    assert 0 < row["start_unix_ns"] <= row["end_unix_ns"]


def test_manual_mode_does_not_attach_projection_hooks_and_keeps_cpu_output():
    module, model = recorder(), Transformer()
    source = torch.ones(2, 8)

    def render():
        assert not model.block.to_q._forward_pre_hooks and not model.block.to_q._forward_hooks
        value = source
        for _ in range(8):
            value = model(value)
        return value

    expected = render()
    actual, packed, raw = module.record_request(
        model,
        4,
        render,
        lambda: {},
        [torch.profiler.ProfilerActivity.CPU],
        full_request_trace=True,
        trace_cpu_ops=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    summary = json.loads(raw)
    assert summary["trace_cpu_ops"] is False and summary["events"] == []
    assert len(summary["ranges"]) == 8
    assert {row["label"] for row in summary["ranges"]} == {
        f"h3.step.{i}.{'dense' if i < 4 else 'sparse_eligible'}" for i in range(8)
    }
    trace = json.loads(gzip.decompress(packed))
    assert not any(event.get("name", "").startswith("h3.module.") for event in trace["traceEvents"])
    assert not model._forward_pre_hooks and not model._forward_hooks
