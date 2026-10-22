"""Real kernels preserve pure-cut results; package code is never retried after failure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
AGREEMENT = Path(__file__).parent / "testdata" / "collective_agreement.py"
ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


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


def test_an_oom_at_a_cut_site_takes_rust_s_cut_and_the_next_request_starts_there(
    tmp_path: Path,
) -> None:
    """An FFN projection that cannot allocate for more than 48 tokens (an allocation the
    allocator could not make room for) runs out inside its FFN's cut site: Rust cuts the site
    finer, its token pieces give the whole FFN's result, and the next request at that shape
    starts at the level Rust kept, with nothing running out."""
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    refused: list[int] = []

    def short(module: nn.Module, args: tuple[Any, ...]) -> None:
        rows = args[0].numel() // args[0].shape[-1]
        if rows > 48:
            refused.append(rows)
            catch._refused(0, rows * 256 * 2, 0, 0)
            raise torch.OutOfMemoryError("out of memory. Tried to allocate 48.00 KiB")

    model.blocks[0].ff.net[0].proj.register_forward_pre_hook(short)
    x = tokens(1)
    with torch.no_grad():
        want = plain(x)
        with mem.request("short") as r, execution.serving(r):
            got = model(x)
        assert refused and r.stats().cuts >= 1
        torch.testing.assert_close(got, want, rtol=2**-6, atol=2**-6)
        refused.clear()
        with mem.request("short again") as r, execution.serving(r):
            again = model(x)
    assert not refused and r.stats().cuts == r.stats().retries == 0
    assert torch.equal(again, got)


def test_an_oom_in_an_exchange_becomes_the_same_cut_on_every_gpu(tmp_path: Path) -> None:
    """Two gloo ranks: in the first pass every rank's exchange runs out before any rank enters
    its collectives. None runs it again; each fails typed and tells Rust, and from the next pass
    both agree the finer level and cross the heads in halves, each pass degree 1's result byte
    for byte. The ranks' rows are gathered once, in the first pass at their shapes."""
    argv = [sys.executable, str(AGREEMENT), str(tmp_path), "--degree", "2"]
    subprocess.run(argv, env=ENV, check=True)
    for rank in range(2):
        passes = json.loads((tmp_path / f"rank-{rank}.json").read_text())["passes"]
        assert passes[0]["failed"] == "device_out_of_memory", passes
        assert all(p["byte_equal"] for p in passes[1:]), passes
        assert [p["exchanged_heads"] for p in passes] == [[], [4, 4], [4, 4], [4, 4]], passes
        assert [p["gathered"] for p in passes] == [1, 0, 0, 0], passes


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


def test_nvfp4_ffn_keeps_whole_token_quantization_domain() -> None:
    import sys

    from diffusers.models.attention import FeedForward

    from cozy_runtime.internal import nvfp4_linear
    from cozy_runtime.internal.encoding.leaves import Nvfp4NativeLeaf

    module = FeedForward(64, dim_out=64, mult=1, activation_fn="gelu").eval()
    projection = module.net[0].proj
    parts = {
        "weight": torch.full((64, 32), 0x12, dtype=torch.uint8),
        "scale": torch.ones((64, 4)).to(torch.float8_e4m3fn),
        "scale_2": torch.tensor(0.125),
    }
    module.net[0].proj = Nvfp4NativeLeaf().leaf(torch, parts, projection, torch.float32)
    forward = module.forward
    cut = cuts._ffn("nvfp4:tokens", module, forward)
    x = torch.randn(2, 24, 64)
    x[:, -1] *= 100  # tensor-global activation extrema would change across token pieces
    visited: list[str] = []
    arms: list[str] = []

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is catch.cut.__code__:
            visited.append(frame.f_code.co_name)

    token, previous = nvfp4_linear.OBSERVED.set(arms), sys.getprofile()
    try:
        sys.setprofile(profile)
        with torch.inference_mode():
            got, expected = cut(x), forward(x)
    finally:
        sys.setprofile(previous)
        nvfp4_linear.OBSERVED.reset(token)
    assert not cuts._token_local(module) and not visited
    assert arms == ["torch-w4a16-decode", "torch-w4a16-decode"]
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_attention_dropout_preserves_the_whole_calls_rng() -> None:
    import sys

    q, k, v = (torch.randn(1, 16, 4, 8) for _ in range(3))
    options = attention_ulysses.Options(
        attn_mask=None, dropout_p=0.5, is_causal=False, scale=None, enable_gqa=False
    )

    def dropout(q: Any, k: Any, v: Any, **options: Any) -> Any:
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **options
        ).transpose(1, 2)

    local = attention_ulysses.Local(dropout, lambda *_: "torch-cpu", rows=4)
    visited: list[str] = []

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code is cuts.attend.__code__:
            visited.append(frame.f_code.co_name)

    previous = sys.getprofile()
    try:
        with torch.inference_mode():
            torch.manual_seed(55)
            expected = local.forward(q, k, v, **options)
            state = torch.get_rng_state()
            torch.manual_seed(55)
            sys.setprofile(profile)
            got = attention_ulysses._call("torch", local, q, k, v, options)
    finally:
        sys.setprofile(previous)
    assert not visited and torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
