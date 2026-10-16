"""Fused DiT blocks (`cozy_runtime.fusion`): a package's declaration, Runtime's kernels and
substitution, through the real build, load and install paths. The launch arms need a card."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from cozy_runtime import fusion  # noqa: E402
from cozy_runtime.author.kernels import Plan  # noqa: E402
from cozy_runtime.internal import kernel_cache, kernel_compile, triton_kernels  # noqa: E402

WIDTH, HEADS, HEAD_DIM = 256, 2, 128


class HeadNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(1 + 0.2 * torch.randn(HEAD_DIM))
        self.eps = 1e-6

    def forward(self, x: Any) -> Any:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q, self.to_k, self.to_v, self.to_out = (
            torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(4)
        )
        self.norm_q, self.norm_k = HeadNorm(), HeadNorm()

    def forward(self, x: Any, rotary: Any = None) -> Any:
        q, k, v = (f(x).unflatten(-1, (HEADS, HEAD_DIM)) for f in (self.to_q, self.to_k, self.to_v))
        q, k = self.norm_q(q), self.norm_k(k)
        if rotary is not None:  # interleaved pairs, complex tables
            q, k = (
                torch.view_as_real(torch.view_as_complex(t.float().unflatten(-1, (-1, 2))) * rotary)
                .flatten(3)
                .type_as(t)
                for t in (q, k)
            )
        out = torch.nn.functional.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v))
        )
        return self.to_out(out.transpose(1, 2).flatten(2))


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.up = torch.nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.down = torch.nn.Linear(3 * WIDTH, WIDTH, bias=False)


class Block(torch.nn.Module):
    """A single-stream adaLN block in plain torch: per-token rows (`index`) of a shared table."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(WIDTH, elementwise_affine=False, eps=1e-6)
        self.norm2 = torch.nn.LayerNorm(WIDTH, elementwise_affine=False, eps=1e-6)
        self.attn = Attention()
        self.mlp = MLP()

    def forward(self, hidden_states: Any, table: Any, index: Any, rotary: Any = None) -> Any:
        scale1, gate1, scale2, gate2 = (t[index][None] for t in table.chunk(4, dim=-1))
        x = hidden_states
        x = x + gate1.tanh() * self.attn(self.norm1(x) * (1 + scale1), rotary=rotary)
        normed = self.norm2(x) * (1 + scale2)
        mlp = self.mlp.down(torch.nn.functional.silu(self.mlp.gate(normed)) * self.mlp.up(normed))
        return x + gate2.tanh() * mlp


def _rows(block: Any, arguments: Any) -> fusion.Modulation:
    return fusion.Modulation(arguments["index"], *arguments["table"].chunk(4, dim=-1))


class FluxAttention(torch.nn.Module):
    """Diffusers' attention contract (to_q/k/v, to_out, heads, a processor), q/k normalized and
    rotated in fp32 by repeated (cos, sin) tables, as Flux does."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = HEADS
        self.to_q, self.to_k, self.to_v = (
            torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(3)
        )
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(WIDTH, WIDTH, bias=False)])
        self.norm_q, self.norm_k = (torch.nn.RMSNorm(HEAD_DIM, eps=1e-6) for _ in range(2))
        self.processor = SimpleNamespace(_attention_backend=None, _parallel_config=None)

    def forward(self, x: Any, rotary: Any) -> Any:
        q, k, v = (f(x).unflatten(-1, (HEADS, HEAD_DIM)) for f in (self.to_q, self.to_k, self.to_v))
        cos, sin = (t[:, None] for t in rotary)

        def rotate(t: Any) -> Any:
            real, imag = t.unflatten(-1, (-1, 2)).unbind(-1)
            return t * cos + torch.stack([-imag, real], dim=-1).flatten(-2) * sin

        q, k = (rotate(n(t.float())).to(v.dtype) for n, t in ((self.norm_q, q), (self.norm_k, k)))
        out = torch.nn.functional.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v))
        )
        return self.to_out[0](out.transpose(1, 2).flatten(2))


class FeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = torch.nn.Linear(WIDTH, 6 * WIDTH, bias=False)
        self.linear_out = torch.nn.Linear(3 * WIDTH, WIDTH, bias=False)


class ParallelBlock(torch.nn.Module):
    """Flux's single-stream block: one modulated norm read by attention and a fused gate|up
    SwiGLU, one gate on their sum; per-token rows of a (shift, scale, gate) table."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(WIDTH, elementwise_affine=False, eps=1e-6)
        self.attn = FluxAttention()
        self.ff = FeedForward()

    def forward(self, hidden_states: Any, temb_mod: Any, image_rotary_emb: Any) -> Any:
        table, index = temb_mod
        shift, scale, gate = (t[index][None] for t in table.chunk(3, dim=-1))
        normed = (1 + scale) * self.norm(hidden_states) + shift
        gated, up = self.ff.linear_in(normed).chunk(2, dim=-1)
        mlp = self.ff.linear_out(torch.nn.functional.silu(gated) * up)
        return hidden_states + gate * (self.attn(normed, image_rotary_emb) + mlp)


def _parallel_rows(block: Any, arguments: Any) -> fusion.Modulation:
    table, index = arguments["temb_mod"]
    shift, scale, gate = table.chunk(3, dim=-1)
    return fusion.Modulation(index, scale, gate, shift_msa=shift)


PARALLEL = fusion.DiTBlock(
    ParallelBlock,
    width=WIDTH,
    head_dim=HEAD_DIM,
    modulation=_parallel_rows,
    norms=("norm",),
    norm="layer",
    attention="attn",
    arguments=(),
    mlp=fusion.GatedMLP("ff", gate="linear_in", down="linear_out"),
    rope=fusion.Rope("image_rotary_emb", interleaved=True, fp32=True),
    shift=True,
)

LAYOUT = fusion.DiTBlock(
    Block,
    width=WIDTH,
    head_dim=HEAD_DIM,
    modulation=_rows,
    norms=("norm1", "norm2"),
    norm="layer",
    attention="attn",
    arguments=("rotary",),
    mlp=fusion.GatedMLP("mlp", gate="gate", up="up", down="down"),
    rope=fusion.Rope("rotary", interleaved=True),
    gate="tanh",
)


def test_a_declaration_is_a_kernel_set_its_module_rebuilds() -> None:
    kernels = LAYOUT.kernels
    assert kernels.name.startswith("cozy-dit-") and kernels.origin == __name__
    assert kernels.modules == {"cozy_runtime.fusion._kernels", __name__}
    assert [e.kernel.__name__ for e in kernels.entries] == [
        "norm_modulate_kernel",
        "norm_modulate_kernel",
        "norm_modulate_kernel",
        "swiglu_kernel",
        "head_norm_rope_kernel",
    ]
    assert fusion.DiTBlock(**_declared()).kernels.identity == kernels.identity
    assert fusion.DiTBlock(**_declared(gate="identity")).kernels.name != kernels.name
    with pytest.raises(ValueError, match="rotary"):
        fusion.DiTBlock(**_declared(rope=fusion.Rope("rotary", interleaved=True, rotary=129)))
    # A parallel block has no second norm: its residual sums attention and the MLP.
    assert [e.constexprs.get("BRANCHES") for e in PARALLEL.kernels.entries] == [0, 2, None, None]
    assert PARALLEL.kernels.entries[-1].constexprs["FP32"] == 1


def _declared(**changes: Any) -> dict[str, Any]:
    fields = ("block", "width", "head_dim", "modulation", "norms", "norm", "attention")
    rest = ("arguments", "mlp", "rope", "qk_norms", "gate", "shift", "hidden_states")
    return {name: getattr(LAYOUT, name) for name in (*fields, *rest)} | changes


def test_off_cuda_or_off_layout_the_eager_block_serves() -> None:
    never = Plan("cpu", lambda kernels: pytest.fail("loaded kernels for a CPU"))
    for layout, cls in ((LAYOUT, Block), (PARALLEL, ParallelBlock)):
        block = cls()
        assert fusion.install(never, layout, block) == ""
        assert type(block) is cls


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> kernel_cache.Store:
    store = kernel_cache.Store(tmp_path / "u")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(store.own / "triton"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    return store


def _sequential_case() -> tuple[Any, Any, tuple[Any, ...]]:
    tokens, rows = 300, 2
    x = 8 * torch.randn(1, tokens, WIDTH, device="cuda", dtype=torch.bfloat16)
    table = 0.5 * torch.randn(rows, 4 * WIDTH, device="cuda", dtype=torch.bfloat16)
    index = torch.randint(0, rows, (tokens,), device="cuda")
    angles = 100 * torch.rand(tokens, HEAD_DIM // 2, device="cuda")
    return LAYOUT, Block(), (x, table, index, torch.polar(torch.ones_like(angles), angles))


def _parallel_case() -> tuple[Any, Any, tuple[Any, ...]]:
    pytest.importorskip("diffusers")
    tokens, rows = 300, 3
    x = 8 * torch.randn(1, tokens, WIDTH, device="cuda", dtype=torch.bfloat16)
    table = 0.5 * torch.randn(rows, 3 * WIDTH, device="cuda", dtype=torch.bfloat16)
    index = torch.randint(0, rows, (tokens,), device="cuda")
    angles = (100 * torch.rand(tokens, HEAD_DIM // 2, device="cuda")).repeat_interleave(2, dim=1)
    return PARALLEL, ParallelBlock(), (x, (table, index), (angles.cos(), angles.sin()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiling and launching needs a card")
@pytest.mark.parametrize("case", [_sequential_case, _parallel_case])
def test_a_block_installs_once_its_kernels_are_built_and_matches_eager(
    case: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, monkeypatch)
    torch.manual_seed(0)
    layout, block, inputs = case()
    cls = type(block)
    block = block.to(device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        eager = block(*inputs)
    deferred: list[Any] = []

    def when_ready(kernels: Any, install: Any) -> bool:
        launch = triton_kernels.ready(kernels, device="cuda")
        if launch is None:
            deferred.append((kernels, install))
            return False
        install(launch)
        return True

    plan = Plan(torch.device("cuda"), lambda k: triton_kernels.load(k, device="cuda"), when_ready)
    identity = fusion.install(plan, layout, block)
    # In place from the start (Runtime wraps each block's forward after the plan); eager serves
    # until the build is ready.
    assert identity.startswith("sha256:") and type(block).__name__ == f"Fused{cls.__name__}"
    assert deferred
    with torch.inference_mode():
        assert torch.equal(block(*inputs), eager)
    major, minor = torch.cuda.get_device_capability()
    job = triton_kernels.job(layout.kernels, major * 10 + minor)
    assert kernel_compile.wait(store, job).state == "ready"
    kernels, install = deferred.pop()
    install(triton_kernels.ready(kernels, device="cuda"))
    with torch.inference_mode():
        fused = block(*inputs)
    distance = (fused.float() - eager.float()).norm() / eager.float().norm()
    assert math.isfinite(distance.item()) and distance.item() < 2e-3
    assert sorted(dict(block.named_parameters())) == sorted(dict(cls().named_parameters()))
