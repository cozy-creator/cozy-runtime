"""Bounded ComfyUI encoding vectors through Runtime's production decode providers.

Run on one existing CUDA worker with Torch, Triton and the built TensorFS wheel:
    python -m pytest --real-gpu -q -p no:cacheprovider tests/test_comfy_gpu_consumer.py
No model download is needed. This proves the portable encoding consumer, not the
ComfyUI application or a model's visual output.
"""

from __future__ import annotations

import pytest

from cozy_runtime.internal import probe
from cozy_runtime.internal.decode_cores import HAS_TRITON
from cozy_runtime.internal.encoding import launch_providers
from cozy_runtime.internal.encoding.adapters import CoreDequant

torch = pytest.importorskip("torch")
SCHEMES = frozenset(("comfy.asym_w4a8_int8", "comfy.w6a8_int8", "comfy.convrot_w4a4"))


@pytest.mark.real_gpu
@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(), reason="CUDA and Triton")
def test_comfy_vectors_through_launched_gpu_consumers() -> None:
    providers = launch_providers()
    target = torch.device("cuda")
    checked: dict[str, int] = dict.fromkeys(SCHEMES, 0)
    for case in probe.vector_cases():
        scheme = case.encoding.scheme
        if scheme not in SCHEMES or case.refuse:
            continue
        candidates = [p for p in providers[scheme] if isinstance(p, CoreDequant)]
        assert len(candidates) == 1, scheme
        got = probe.decode_case(torch, candidates[0], case, target)
        assert got is not None and got.is_cuda, case.key
        assert probe.same_bits(torch, got, case.expect) == 0, f"{case.component}/{case.key}"
        checked[scheme] += 1
    assert all(checked.values()), checked
    assert sum(checked.values()) == 18, checked
    torch.cuda.synchronize()
