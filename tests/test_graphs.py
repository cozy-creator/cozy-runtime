"""`graphs.graphed`: calls are the module's own, eagerly on a CPU; on CUDA the second call of a
signature records a graph that later `with` blocks replay, and a new signature records anew."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime import graphs  # noqa: E402


def _mlp(device: str) -> Any:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)
    ).to(device)


def test_calls_on_a_cpu_are_the_modules_own() -> None:
    module = _mlp("cpu")
    x = torch.randn(4, 64)
    with torch.inference_mode(), graphs.graphed(module) as call:
        assert torch.equal(call(x), module(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_recordings_outlive_the_with() -> None:
    module = _mlp("cuda")
    recorder_of = graphs._RECORDERS.get
    with torch.inference_mode():
        for _ in range(4):  # one `with` per step, as a package's per-step method opens it
            x = torch.randn(4, 64, device="cuda")
            with graphs.graphed(module) as call:
                out = call(x)
            assert torch.allclose(out, module(x))
        recorder = recorder_of(module)
        assert recorder is not None and len(recorder.recordings) == 1
        with graphs.graphed(module) as call:
            wider = torch.randn(8, 64, device="cuda")
            assert torch.allclose(call(wider), module(wider))  # a new signature: eager first
            assert torch.allclose(call(wider), module(wider))  # then recorded
        assert len(recorder.recordings) == 2
        assert graphs.release() and not recorder.recordings
