"""Each op band rule on a GPU (py-review-1 item 9): the banded op's peak live bytes
(`torch.cuda.max_memory_allocated`, its results included) below the whole op's, and its
results the whole op's. A rule whose bands save nothing has no place in `cuts`.

The shapes are a VAE decoder's at 1024 px: a 3x3 conv at 512 channels, a causal video conv's
3x3x3, a 2x nearest upsampling, a 32-group norm; by images where the batch has more than one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal.execution import catch, cuts  # noqa: E402, F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="live bytes are a GPU's")
aten = torch.ops.aten
LEVEL = 2  # four bands: a cut Rust answers on a card that is a little short


def _randn(*shape: int) -> Any:
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def _conv(x: Any, out: int, kernel: int) -> tuple[Any, tuple[object, ...]]:
    dims = x.dim() - 2
    w = _randn(out, x.shape[1], *[kernel] * dims) * 0.02
    rest = ([1] * dims, [kernel // 2] * dims, [1] * dims, False, [0] * dims, 1)
    return aten.convolution.default, (x, w, _randn(out), *rest)


CASES: dict[str, Callable[[], tuple[Any, tuple[object, ...]]]] = {
    "conv2d 512ch 128px by channels": lambda: _conv(_randn(1, 512, 128, 128), 512, 3),
    "conv2d 256ch 64px by images": lambda: _conv(_randn(4, 256, 64, 64), 256, 3),
    "conv3d 384ch 5x64px by channels": lambda: _conv(_randn(1, 384, 5, 64, 64), 384, 3),
    "upsample 2x 512ch 128px by channels": lambda: (
        aten.upsample_nearest2d.vec,
        (_randn(1, 512, 128, 128), None, [2.0, 2.0]),
    ),
    "upsample 2x 256ch 64px by images": lambda: (
        aten.upsample_nearest2d.vec,
        (_randn(4, 256, 64, 64), None, [2.0, 2.0]),
    ),
    "group norm 32 groups 512ch 128px by groups": lambda: (
        aten.native_group_norm.default,
        (_randn(1, 512, 128, 128), _randn(512), _randn(512), 1, 512, 128 * 128, 32, 1e-6),
    ),
    "group norm 32 groups 256ch 64px by images": lambda: (
        aten.native_group_norm.default,
        (_randn(4, 256, 64, 64), _randn(256), _randn(256), 4, 256, 64 * 64, 32, 1e-6),
    ),
}


def _peak(run: Callable[[], Any]) -> tuple[int, Any]:
    """`run()`'s peak live bytes above what was live before it, and its result."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before, out


@pytest.mark.parametrize("name", list(CASES))
def test_a_band_rule_lowers_the_op_s_live_bytes(name: str) -> None:
    func, args = CASES[name]()
    with torch.no_grad():
        func(*args)  # the library's handles and plans exist before anything is measured
        whole_peak, whole = _peak(lambda: func(*args))
        plan = catch.RULES[func.overloadpacket](func, args, {})
        assert plan is not None and plan.finest >= LEVEL, name
        banded_peak, got = _peak(lambda: plan(catch.Scope(name, plan.finest, LEVEL)))
    errors = [
        (a.float() - b.float()).abs().max().item()
        for a, b in zip(catch.tensors(got), catch.tensors(whole), strict=True)
    ]
    print(
        f"LIVE {name}: whole {whole_peak / 2**20:.1f} MiB, {1 << LEVEL} bands "
        f"{banded_peak / 2**20:.1f} MiB, max error {max(errors):.3g}"
    )
    assert banded_peak < whole_peak, (name, whole_peak, banded_peak)
