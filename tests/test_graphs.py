"""`graphs.install`: a component root's calls run as written on a CPU; on CUDA the second call of
a signature records a graph whose first replay is checked against the call as written, later
calls replay it, and a call that depends on more than its arguments keeps running as written."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope  # noqa: E402
from cozy_runtime.author._model import WARNINGS  # noqa: E402

CUDA = torch.cuda.is_available()


def _mlp(device: str) -> Any:
    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
    ).to(device)
    graphs.install(torch, object(), {"mlp": module})
    return module


def _counting() -> Any:
    """Adds how often it ran: a call that depends on state beside its arguments."""

    base: Any = torch.nn.Module

    class Counting(base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("calls", torch.zeros(()))

        def forward(self, x: Any) -> Any:
            self.calls += 1
            return x + self.calls

    return Counting()


def test_calls_on_a_cpu_run_as_written() -> None:
    module = _mlp("cpu")
    x = torch.randn(4, 64)
    with torch.inference_mode():
        assert torch.equal(module(x), torch.nn.Sequential.__call__(module, x))
    assert type(module).__name__ == "Sequential" and isinstance(module, torch.nn.Sequential)


def test_attention_layouts_bypass_graph_pins_on_a_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """H3 scopes every DiT step: this branch must not create a recording or a lifetime pin."""
    module = _mlp("cpu")
    x = torch.randn(4, 64)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.inference_mode(), attention_scope(AttentionLayout(4, 0, 0)):
        assert torch.equal(module(x), torch.nn.Sequential.__call__(module, x))
    assert module not in graphs._RECORDERS


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_recordings_replay_across_calls_and_signatures() -> None:
    module = _mlp("cuda")
    with torch.inference_mode():
        for _ in range(4):
            x = torch.randn(4, 64, device="cuda")
            assert torch.allclose(module(x), torch.nn.Sequential.__call__(module, x))
        recorder = graphs._RECORDERS[module]
        assert len(recorder.recordings) == 1 and recorder.replays == 3  # calls 2, 3, 4
        wider = torch.randn(8, 64, device="cuda")
        for _ in range(2):  # a new signature: as written first, then recorded
            assert torch.allclose(module(wider), torch.nn.Sequential.__call__(module, wider))
        assert len(recorder.recordings) == 2
        token = graphs.BYPASS.set(True)
        try:
            module(wider)
        finally:
            graphs.BYPASS.reset(token)
        assert recorder.replays == 4
        graphs._shed(torch, recorder)
        assert not recorder.recordings


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_call_that_depends_on_hidden_state_runs_as_written() -> None:
    module = _counting().cuda()
    graphs.install(torch, object(), {"counting": module})
    WARNINGS.clear()
    with torch.inference_mode():
        x = torch.zeros(3, device="cuda")
        for _ in range(4):
            module(x)
    assert not graphs._RECORDERS[module].recordings
    assert [message for message, _ in WARNINGS] == ["a CUDA graph was not used"]


def _stepped() -> Any:
    """HiDiffusion's patched blocks: a step counter in a plain attribute picks the path, and a
    window shift is drawn from the CPU generator."""

    base: Any = torch.nn.Module

    class Stepped(base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 64)
            self.timestep = 0

        def forward(self, x: Any) -> Any:
            shift = float(torch.rand(1))
            self.timestep += 1
            return self.linear(x) * self.timestep + shift

    return Stepped()


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_first_request_equals_a_warm_one_for_a_module_with_state() -> None:
    """The owner: measuring and recording never change model state, so a fresh executor's
    first request equals a warm one with the same seed (run 11226 drifted two steps)."""
    torch.manual_seed(1)
    module = _stepped().cuda()
    graphs.install(torch, object(), {"stepped": module})

    def request() -> tuple[list[Any], int]:
        module.timestep = 0
        torch.manual_seed(5)
        x = torch.ones(4, 64, device="cuda")
        return [module(x) for _ in range(3)], module.timestep

    with torch.inference_mode():
        first, steps = request()
        warm, again = request()
    assert steps == again == 3
    assert all(torch.equal(a, b) for a, b in zip(first, warm, strict=True))
    assert not graphs._RECORDERS[module].recordings


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_capture_out_of_memory_leaves_the_calls_serving() -> None:
    module = _mlp("cuda")
    armed = [True]

    def hook(*_: Any) -> None:
        if armed[0] and torch.cuda.is_current_stream_capturing():
            armed[0] = False
            raise torch.OutOfMemoryError("CUDA out of memory (injected)")

    module[1].register_forward_pre_hook(hook)
    with torch.inference_mode():
        for _ in range(5):
            x = torch.randn(4, 64, device="cuda")
            assert torch.allclose(module(x), torch.nn.Sequential.__call__(module, x))
    recorder = graphs._RECORDERS[module]
    assert not armed[0] and len(recorder.recordings) == 1 and recorder.replays >= 1
