"""Real allocation refusal at an idle capture stays outside arbitrary package execution."""

from __future__ import annotations

import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from cozy_runtime import graphs  # noqa: E402

pytestmark = pytest.mark.real_gpu


def test_no_room_at_idle_leaves_one_eager_call_and_no_executable() -> None:
    """Under a cap a capture cannot fit beside: the request's call runs once as written and
    records nothing (nothing records on a request's path), and the idle period's capture finds
    no room for its static input, makes no executable and waits for a later period."""
    if not torch.cuda.is_available():
        pytest.skip("real CUDA allocation refusal")
    module = torch.nn.Linear(16 << 20, 1).cuda().eval()
    graphs.install(torch, object(), {"linear": module})
    with torch.inference_mode():
        x = torch.zeros((1, 16 << 20), device="cuda")
        result = module(x)  # first eager call also seeds output capacity in the default pool
        del result
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        # A 64 MiB static input cannot fit even if Torch drops every cached segment; the
        # ordinary eager result is one float and still fits the existing output segment.
        torch.cuda.set_per_process_memory_fraction(reserved / total)
        calls: list[int] = []
        original = sys.getprofile()

        def profile(frame: Any, event: str, arg: Any) -> None:
            if event == "call" and frame.f_code is torch.nn.Linear.forward.__code__:
                calls.append(1)

        try:
            sys.setprofile(profile)
            result = module(x)
            sys.setprofile(original)
            assert graphs.capture_idle(lambda: False)
        finally:
            sys.setprofile(original)
            torch.cuda.set_per_process_memory_fraction(1.0)
        assert len(calls) == 1
        recorder = graphs._RECORDERS[module]
        assert not recorder.recordings and recorder.capture_attempts == 1 and recorder.queued
        torch.testing.assert_close(result, torch.nn.Linear.__call__(module, x))
    graphs.release(torch, module)
    assert graphs.capture_idle(lambda: False)  # a released region's job goes
