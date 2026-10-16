"""Every entry of the Triton kernel sets an executor builds at start compiles, for an Ada and a
Blackwell card, with no GPU: a set that fails to build leaves its ops on their eager paths, and
nothing else says so (0.25.0's nvfp4 set typed its runtime sizes as constexprs)."""

from __future__ import annotations

from typing import Any

import pytest

triton = pytest.importorskip("triton")

from triton.backends.compiler import GPUTarget  # noqa: E402

from cozy_runtime import lm  # noqa: E402
from cozy_runtime.internal import nvfp4_linear, triton_kernels  # noqa: E402


@pytest.mark.parametrize("kernels", [nvfp4_linear.KERNELS, lm.KERNELS], ids=lambda k: k.name)
@pytest.mark.parametrize("arch", [89, 120])
def test_each_entry_compiles(kernels: Any, arch: int) -> None:
    assert kernels.entries
    for entry in kernels.entries:
        triton.compile(
            triton_kernels._source(entry),
            target=GPUTarget("cuda", arch, 32),
            options={"num_warps": entry.num_warps},
        )
