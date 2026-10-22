"""Real kernels preserve pure-cut results; package code is never retried after failure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from torch import nn  # noqa: E402

from cozy_runtime.author import Cut, split  # noqa: E402
from cozy_runtime.author._errors import is_capacity  # noqa: E402
from cozy_runtime.author._split import _finest  # noqa: E402
from cozy_runtime.author._split import install as install_split  # noqa: E402
from cozy_runtime.internal import attention_ulysses, execution  # noqa: E402
from cozy_runtime.internal.execution import catch, cuts  # noqa: E402
from execution_models import checkpoint, memory, placed, scores, transformer, vae  # noqa: E402

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


def test_an_oom_at_a_cut_site_cuts_it_and_the_next_request_starts_there(tmp_path: Path) -> None:
    """An FFN projection that cannot allocate for more than 48 tokens (an allocation the
    allocator could not make room for) runs out inside its FFN's cut site: the site goes finer,
    its token pieces give the whole FFN's result, and the next request at that shape starts at
    the level the process kept, with nothing running out."""
    plain = transformer()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "dit", plain), "dit", transformer)
    refused: list[int] = []

    def short(module: nn.Module, args: tuple[Any, ...]) -> None:
        rows = args[0].numel() // args[0].shape[-1]
        if rows > 48:
            refused.append(rows)
            raise torch.OutOfMemoryError("out of memory. Tried to allocate 48.00 KiB")

    model.blocks[0].ff.net[0].proj.register_forward_pre_hook(short)
    x = tokens(1)
    with torch.no_grad():
        want = plain(x)
        with mem.request("short") as r, execution.serving(r):
            got = model(x)
        assert refused and any(n >= 1 for (at, _), n in catch.levels() if at.endswith(":tokens"))
        torch.testing.assert_close(got, want, rtol=2**-6, atol=2**-6)
        refused.clear()
        with mem.request("short again") as r, execution.serving(r):
            again = model(x)
    assert not refused
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


def _decode_split(mem: Any, tmp_path: Path) -> tuple[Any, Any, tuple[Cut, ...], int]:
    """SDXL's tiled decode (`split` of the VAE's decode by latent rows and columns) on a small
    VAE over host pages, and the finest level of its split."""
    model = placed(mem, checkpoint(tmp_path, "vae", vae()), "vae", vae)
    z = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(7)).to(BF16)
    tiles = (Cut(2, scale=2, overlap=2), Cut(3, scale=2, overlap=2))
    return model, z, tiles, cuts.levels(-(-z.numel() // _finest(tuple(z.shape), tiles)))


def test_a_tiled_decode_that_runs_out_whole_completes_finer(tmp_path: Path) -> None:
    """v6 13598, on host pages: the VAE's last up block cannot allocate above 10x10 latent
    windows, so the whole 16x16 decode runs out inside its pass: the split goes finer at once
    (no run again at the same level) and the decode completes in tiles. 13598 re-ran the whole
    decode for 17 minutes."""
    mem = memory()
    model, z, tiles, levels = _decode_split(mem, tmp_path)
    refused: list[tuple[int, int]] = []

    def short(module: nn.Module, args: tuple[Any, ...]) -> None:
        h, w = (int(n) for n in args[0].shape[-2:])
        if h * w > 20 * 20:  # the last up block runs at twice the latent's size
            refused.append((h, w))
            # A level fails at most once (then it is cut finer); more is a loop.
            assert len(refused) <= levels + 1, f"the decode re-ran without end: {refused}"
            raise torch.OutOfMemoryError("out of memory. Tried to allocate 8.00 KiB")

    model.decoder.up_blocks[-1].register_forward_pre_hook(short)
    with torch.no_grad(), mem.request("decode") as r, execution.serving(r):
        got = split(lambda x: model.decode(x).sample, z, *tiles)
    print(f"completed finer after {refused}")
    assert tuple(got.shape) == (1, 3, 32, 32) and bool(torch.isfinite(got).all())
    assert all(refused.count(w) == 1 for w in set(refused)), refused
    assert any(at.startswith("split:") and n >= 1 for (at, _), n in catch.levels())


def test_a_decode_whose_pass_cannot_start_fails_typed(tmp_path: Path) -> None:
    """A pass start varena refuses (Exhausted: the VAE's weights alone exceed the memory) inside
    the split is the split's out-of-memory: varena is asked with the split's site, and with no
    cut able to help it answers Exhausted, typed, the pass tried a bounded number of times."""
    mem = memory(capacity=1 << 16)
    model, z, tiles, levels = _decode_split(mem, tmp_path)
    starts: list[int] = []
    model.decoder.register_forward_pre_hook(lambda *_: starts.append(1), prepend=True)
    model.post_quant_conv.register_forward_pre_hook(lambda *_: starts.append(0), prepend=True)
    with (
        torch.no_grad(),
        mem.request("decode") as r,
        execution.serving(r),
        pytest.raises(Exception) as failed,
    ):
        split(lambda x: model.decode(x).sample, z, *tiles)
    print(f"{failed.value!r} after {len(starts)} pass start(s)")
    assert is_capacity(failed.value)
    assert len(starts) <= 2 * (levels + 1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_an_rms_norm_s_token_pieces_are_the_whole_norm(dtype: Any) -> None:
    """torch's nn.RMSNorm (v7 H3 13826's site) at levels 0-3 on real CPU kernels: each row is
    normalized alone, so token pieces are the whole norm, bit for bit."""
    gen = torch.Generator().manual_seed(15)
    norm = nn.RMSNorm(48, eps=1e-6).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(48, generator=gen).to(dtype))
    cut = cuts._norm("norm:tokens", norm, norm.forward)
    x = torch.randn(2, 37, 48, generator=gen).to(dtype)
    with torch.no_grad():
        whole = norm(x)
        flat = x.reshape(-1, 48)
        for level in range(4):
            got = cast("Any", cuts._rows(norm.forward, flat, flat.shape[0], level)).view(x.shape)
            assert torch.equal(got, whole), level
        assert torch.equal(cut(x), whole)


def test_an_oom_in_an_rms_norm_cuts_it_by_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An RMS norm whose op cannot allocate for more than 32 rows runs out inside its own cut
    site: Rust cuts it finer and its token pieces give the whole norm's result (block code's
    norm, the allocation 13826 could not make). The op's refusal is torch's, in its forward."""

    def build() -> nn.Module:
        torch.manual_seed(16)
        return nn.Sequential(nn.Linear(16, 16), nn.RMSNorm(16), nn.Linear(16, 16)).to(BF16)

    plain = build().eval()
    mem = memory()
    model = placed(mem, checkpoint(tmp_path, "normed", plain), "normed", build)
    refused: list[int] = []

    norm = nn.RMSNorm.forward

    def short(module: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        if rows > 32:
            refused.append(rows)
            raise torch.OutOfMemoryError("out of memory. Tried to allocate 2.00 KiB")
        return cast("torch.Tensor", norm(module, x))

    x = tokens(3, rows=96)
    with torch.no_grad():
        want = plain(x)
    monkeypatch.setattr(nn.RMSNorm, "forward", short)
    with torch.no_grad(), mem.request("normed") as r, execution.serving(r):
        got = model(x)
    assert refused and any(n >= 1 for (at, _), n in catch.levels() if at.endswith("1:tokens"))
    assert torch.equal(got, want)


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


def test_runtime_wrappers_stand_in_for_the_class_forward_and_say_so() -> None:
    """v13: H3's glue substitutes a block only when its modules are plain (no instance forward
    but the memory's own) and refused every block Runtime had wrapped (fusion_block_shape).
    Runtime installs its wrappers after the execution plan, and marks them as TensorFS's were:
    each calls its module's class forward at each call, so a class swap still takes effect."""
    norm = nn.RMSNorm(8)
    block = nn.Sequential(norm)
    execution.install(block)
    assert vars(norm)["forward"]._varena and vars(block)["forward"]._varena

    class Swapped(nn.RMSNorm):
        def forward(self, x: Any) -> Any:
            return torch.full_like(x, 7.0)

    norm.__class__ = Swapped
    with torch.no_grad():
        assert torch.equal(block(torch.ones(2, 8)), torch.full((2, 8), 7.0))


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


def test_non_cut_settle_does_not_replay_package_code() -> None:
    seen: list[int] = []

    def package() -> None:
        seen.append(1)
        torch.empty(1 << 62, dtype=torch.uint8)

    mem = memory()
    with mem.request("outside") as request, execution.serving(request), pytest.raises(RuntimeError):
        catch.settle(package, None)
    assert seen == [1]


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
