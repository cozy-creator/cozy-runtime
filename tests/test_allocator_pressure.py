"""The plane's facts say when torch's allocator ran short of room: it dropped its whole cache
to place one allocation (`alloc_retries`), or this process returned cached segments itself
(`cache_releases`). A real allocator on a real card; a stage with room shows neither."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import accel  # noqa: E402
from cozy_runtime.internal.weights import Weights  # noqa: E402

pytestmark = pytest.mark.skipif(not accel.present(torch, "cuda"), reason="a CUDA card")
MIB = 1 << 20


def test_facts_count_a_cache_torch_dropped_for_one_allocation() -> None:
    weights = Weights(torch, torch.device("cuda", 0), "cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    try:
        accel.release_cached(torch, "cuda")
        before = weights.facts()
        # 512 MiB live, then idle in torch's cache: no retry, nothing returned.
        held = torch.empty(512 * MIB, dtype=torch.uint8, device="cuda")
        del held
        calm = weights.facts()
        assert calm.alloc_retries == before.alloc_retries
        assert calm.cache_releases == before.cache_releases
        # Held to what it has reserved plus 192 MiB, torch cannot place 640 MiB beside the idle
        # 512: it drops its cache and tries again. That is the squeeze, counted.
        limit = torch.cuda.memory_reserved() + 192 * MIB
        torch.cuda.set_per_process_memory_fraction(limit / total, 0)
        larger = torch.empty(640 * MIB, dtype=torch.uint8, device="cuda")
        squeezed = weights.facts()
        assert (
            squeezed.alloc_retries > calm.alloc_retries and squeezed.alloc_ooms == calm.alloc_ooms
        )
        # This process's own release is counted apart, with its bytes.
        del larger
        accel.release_cached(torch, "cuda")
        after = weights.facts()
        assert after.cache_releases == squeezed.cache_releases + 1
        assert after.cache_released_bytes - squeezed.cache_released_bytes >= 640 * MIB
        mine = test_facts_count_a_cache_torch_dropped_for_one_allocation.__name__
        assert after.cache_released_by[mine][1] >= 640 * MIB, "named by the function that asked"
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
