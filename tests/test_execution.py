"""Real kernels preserve pure-cut results; package code is never retried after failure."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from torch import nn  # noqa: E402

from cozy_runtime.author._split import install as install_split  # noqa: E402
from cozy_runtime.internal import attention_ulysses, execution  # noqa: E402
from cozy_runtime.internal.execution import catch, cuts  # noqa: E402
from execution_models import checkpoint, memory, placed, scores, transformer  # noqa: E402

BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def installed() -> Iterator[None]:
    attention_ulysses.install(execution)
    install_split(execution)
    yield
    attention_ulysses.install(None)
    install_split(None)


def tokens(seed: int, rows: int = 96) -> Any:
    return torch.randn(1, rows, 16, generator=torch.Generator().manual_seed(seed)).to(BF16)


def test_without_an_oom_the_catch_changes_nothing(tmp_path: Path) -> None:
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    x = tokens(0)
    with torch.no_grad():
        want = plain(x)
        with mem.request("whole") as r, execution.serving(r):
            got = model(x)
    assert torch.equal(got, want)
    assert r.stats().leftover == 0


def test_a_conv_cut_site_s_bands_are_the_whole_conv() -> None:
    """The module-level conv cut at levels 0-3 on real CPU kernels in bfloat16, each one's max
    error printed: level 0 is the module's own call; 2-D by images and by output channels, and
    3-D by output channels, bit-equal to the whole conv."""
    gen = torch.Generator().manual_seed(14)
    cases = {
        "conv2d by channels": (nn.Conv2d(16, 16, 3, padding=1), (1, 16, 32, 32)),
        "conv2d by images": (nn.Conv2d(16, 16, 3, padding=1), (4, 16, 16, 16)),
        "conv3d by channels": (nn.Conv3d(8, 16, 3, padding=1), (1, 8, 4, 16, 16)),
    }
    for name, (conv, shape) in cases.items():
        conv, x = conv.to(BF16), torch.randn(*shape, generator=gen).to(BF16)
        with torch.no_grad():
            whole = conv(x)
            units = x.shape[0] if x.shape[0] > 1 else conv.weight.shape[0]
            for level in range(4):
                got = cuts._bands(
                    conv._conv_forward, x, conv.weight, conv.bias, x.shape[0] > 1, units, level
                )
                error = (got.float() - whole.float()).abs().max().item()
                print(f"{name} level {level}: max error {error:.3g} against whole")
                assert torch.equal(got, whole), (name, level)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ffn_tokens_and_attention_pieces_against_the_whole(dtype: Any) -> None:
    """FFN token pieces and attention head groups and query rows at levels 1-3 against the call
    whole on real CPU kernels, each one's measured max error printed. A GEMM of fewer rows may
    sum in another order on a CPU, so pieces are equal up to rounding (the output bar)."""
    gen = torch.Generator().manual_seed(10)
    ffn = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)).to(dtype)
    x = torch.randn(2, 48, 64, generator=gen).to(dtype)
    q, k, v = (torch.randn(1, 64, 8, 16, generator=gen).to(dtype) for _ in range(3))
    tolerance = 2**-6 if dtype == torch.bfloat16 else 2**-18
    with torch.no_grad():
        whole_ffn, whole_attention = ffn(x), scores(q, k, v)
        for level in (1, 2, 3):
            heads = cuts.pieces(8, level)  # 2^level pieces of whole K/V heads
            pieces = {
                "ffn tokens": (cuts._tokens(ffn, x, 96, level), whole_ffn),
                "attention heads": (
                    cuts._attend(lambda *t: scores(*t[:3]), q, k, v, None, 1, heads, 64),
                    whole_attention,
                ),
                "attention rows": (
                    cuts._attend(lambda *t: scores(*t[:3]), q, k, v, None, 1, 8, 64 >> level),
                    whole_attention,
                ),
            }
            for name, (got, want) in pieces.items():
                assert isinstance(got, torch.Tensor)
                error = (got.float() - want.float()).abs().max().item()
                print(f"{dtype} {name} level {level}: max error {error:.3g} against whole")
                torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)


def test_arbitrary_package_failure_runs_once() -> None:
    calls: list[int] = []

    class Package(nn.Module):
        def forward(self, x: Any) -> Any:
            calls.append(1)
            # A real impossible CPU allocation, after an observable package-side effect.
            return x + torch.empty(1 << 62, dtype=torch.uint8)

    model = Package()
    execution.install(model)
    mem = memory()
    with mem.request("failure") as request, execution.serving(request), pytest.raises(RuntimeError):
        model(torch.ones(1))
    assert calls == [1]
    assert request.stats().retries == request.stats().cuts == request.stats().parks == 0


def test_non_cut_settle_does_not_replay_package_code() -> None:
    seen: list[int] = []

    def package() -> None:
        seen.append(1)
        torch.empty(1 << 62, dtype=torch.uint8)

    mem = memory()
    with mem.request("outside") as request, execution.serving(request), pytest.raises(RuntimeError):
        catch.settle(package, None, ())
    assert seen == [1] and request.stats().retries == 0


def test_custom_forward_is_not_assumed_to_be_a_pure_cut() -> None:
    class StatefulConv(nn.Conv2d):
        def _conv_forward(self, x: Any, w: Any, b: Any) -> Any:
            self.changed = getattr(self, "changed", 0) + 1
            return super()._conv_forward(x, w, b)

    model = StatefulConv(1, 2, 1)
    assert cuts.install(model) == 0
    model(torch.ones(1, 1, 2, 2))
    assert model.changed == 1
