"""`graphs.install`: a component root's calls run as written on a CPU; on CUDA the second call of
a signature records a graph whose first replay is checked against the call as written, later
calls replay it, and a call that depends on more than its arguments keeps running as written."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime import graphs  # noqa: E402
from cozy_runtime.author._model import WARNINGS  # noqa: E402

CUDA = torch.cuda.is_available()


def _mlp(device: str) -> Any:
    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
    ).to(device)
    graphs.install(object(), {"mlp": module})
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
        assert graphs.release() and not recorder.recordings


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need a GPU")
def test_a_call_that_depends_on_hidden_state_runs_as_written() -> None:
    module = _counting().cuda()
    graphs.install(object(), {"counting": module})
    WARNINGS.clear()
    with torch.inference_mode():
        x = torch.zeros(3, device="cuda")
        for _ in range(4):
            module(x)
    assert not graphs._RECORDERS[module].recordings
    assert [message for message, _ in WARNINGS] == ["a CUDA graph was not used"]


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
