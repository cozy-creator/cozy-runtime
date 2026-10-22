"""Real preparation allocation refusal stays outside arbitrary package execution."""

from __future__ import annotations

import sys
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from cozy_runtime import graphs  # noqa: E402

pytestmark = pytest.mark.real_gpu


def test_preparation_oom_leaves_one_eager_call_and_no_executable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("real CUDA allocation refusal")
    module = torch.nn.Linear(1, 1).cuda().eval()
    graphs.install(torch, object(), {"linear": module})
    with torch.inference_mode():
        x = torch.zeros((16 << 20, 1), device="cuda")
        result = module(x)  # first eager call also seeds output capacity in the default pool
        del result
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        # The domain/private capture pool cannot borrow the default pool's cached output.
        # It must refuse preparation while one ordinary eager result still fits that cache.
        torch.cuda.set_per_process_memory_fraction(reserved / total)
        calls: list[int] = []
        original = sys.getprofile()

        def profile(frame: Any, event: str, arg: Any) -> None:
            if event == "call" and frame.f_code is torch.nn.Linear.forward.__code__:
                calls.append(1)

        try:
            sys.setprofile(profile)
            result = module(x)
        finally:
            sys.setprofile(original)
            torch.cuda.set_per_process_memory_fraction(1.0)
        assert len(calls) == 1
        recorder = graphs._RECORDERS[module]
        assert not recorder.recordings and recorder.waiting
        torch.testing.assert_close(result, torch.nn.Linear.__call__(module, x))
    graphs.release(torch, module)
