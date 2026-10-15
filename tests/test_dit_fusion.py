"""Fused DiT blocks (`cozy_runtime.fusion`): a package's declaration, Runtime's kernels and
substitution, through the real build, load and install paths. The launch arms need a card."""

from __future__ import annotations

import math
from pathlib import Path
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


def _declared(**changes: Any) -> dict[str, Any]:
    fields = ("block", "width", "head_dim", "modulation", "norms", "norm", "attention")
    rest = ("arguments", "mlp", "rope", "qk_norms", "gate", "shift", "hidden_states")
    return {name: getattr(LAYOUT, name) for name in (*fields, *rest)} | changes


def test_off_cuda_or_off_layout_the_eager_block_serves() -> None:
    block = Block()
    never = Plan("cpu", lambda kernels: pytest.fail("loaded kernels for a CPU"))
    assert fusion.install(never, LAYOUT, block) == ""
    assert type(block) is Block


def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> kernel_cache.Store:
    store = kernel_cache.Store(tmp_path / "u")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(store.own / "triton"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr("tempfile.tempdir", None)
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    return store


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiling and launching needs a card")
def test_a_block_installs_once_its_kernels_are_built_and_matches_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, monkeypatch)
    torch.manual_seed(0)
    block = Block().to(device="cuda", dtype=torch.bfloat16)
    tokens, rows = 300, 2
    x = 8 * torch.randn(1, tokens, WIDTH, device="cuda", dtype=torch.bfloat16)
    table = 0.5 * torch.randn(rows, 4 * WIDTH, device="cuda", dtype=torch.bfloat16)
    index = torch.randint(0, rows, (tokens,), device="cuda")
    angles = 100 * torch.rand(tokens, HEAD_DIM // 2, device="cuda")
    rotary = torch.polar(torch.ones_like(angles), angles)
    with torch.inference_mode():
        eager = block(x, table, index, rotary=rotary)
    deferred: list[Any] = []

    def when_ready(kernels: Any, install: Any) -> bool:
        launch = triton_kernels.ready(kernels, device="cuda")
        if launch is None:
            deferred.append((kernels, install))
            return False
        install(launch)
        return True

    plan = Plan(torch.device("cuda"), lambda k: triton_kernels.load(k, device="cuda"), when_ready)
    identity = fusion.install(plan, LAYOUT, block)
    # In place from the start (Runtime wraps each block's forward after the plan); eager serves
    # until the build is ready.
    assert identity.startswith("sha256:") and type(block).__name__ == "FusedBlock" and deferred
    with torch.inference_mode():
        assert torch.equal(block(x, table, index, rotary=rotary), eager)
    major, minor = torch.cuda.get_device_capability()
    job = triton_kernels.job(LAYOUT.kernels, major * 10 + minor)
    assert kernel_compile.wait(store, job).state == "ready"
    kernels, install = deferred.pop()
    install(triton_kernels.ready(kernels, device="cuda"))
    with torch.inference_mode():
        fused = block(x, table, index, rotary=rotary)
    distance = (fused.float() - eager.float()).norm() / eager.float().norm()
    assert math.isfinite(distance.item()) and distance.item() < 2e-3
    assert sorted(dict(block.named_parameters())) == sorted(dict(Block().named_parameters()))
