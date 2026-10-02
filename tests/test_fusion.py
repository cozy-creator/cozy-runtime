"""h3a-015: the fused glue against the eager composition it replaces, on a real card.

Tolerance, measured on the RTX 4070 (sm89) against torch 2.13 eager and recorded here as
the contract: the gate-add, SwiGLU and fp8-epilogue chains are BIT-IDENTICAL; every chain
that starts with an RMSNorm differs from torch's fused `rms_norm` by one bf16 rounding of
the norm on at most 1e-5 of its elements (fp32 variance reduction order — torch's own
kernel is at the same distance from every torch-op spelling of the formula). The one or
two roundings after the norm carry that step through the modulation (or the rotation), so
the bound is over the tensor's range, not per element: `|fused - eager| <= 2^-6 * max|eager|`
(`_RANGE`). The mismatch FRACTION is the sharp criterion — a wrong table row, a wrong
partner channel or a dropped rounding differs on ~50% of elements, not 1e-5.

Imports are at the top and guarded (torch/triton/diffusers absent skips every card arm).
The kernels compile once per session the way a worker boot compiles them: `fusion.expect`
submits the build to the machine's compile manager, whose niced background process fills
the session store's Triton cache; one arm proves a fresh process then loads and launches
without compiling anything.

The block arm runs a one-block real-width MiniMax-H3 transformer through the runtime's
installer — the same `apply_fusion_plan` the executor calls — twice: as decoded bf16
Linears and as `fp8-rowwise/1` encoded leaves, so the `PreQuantized` handoff to the leaf is
exercised on the real `_scaled_mm` route.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import ConformanceError, Model
from cozy_runtime.internal import accel, fusion, fusion_install, kernel_cache, kernel_compile
from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise

try:
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3RotaryPosEmbed,
        _apply_rotary_emb,
    )
    from triton.runtime.cache import get_cache_manager
except ImportError as _absent:
    CARD_ABSENT = f"torch, triton and diffusers must be importable: {_absent}"
else:
    CARD_ABSENT = ""


def _card() -> str:
    if CARD_ABSENT:
        return CARD_ABSENT
    if not accel.present(torch, "cuda"):
        return "a CUDA device is required"
    if torch.cuda.get_device_capability()[0] < 8:
        return "the fused glue needs sm80+"
    return ""


NO_CARD = _card()
needs_card = pytest.mark.skipif(bool(NO_CARD), reason=NO_CARD or "")

_BF16_MISMATCH_FRACTION = 1e-5
_RANGE = 2.0**-6
HIDDEN, FFN, HEADS, HEAD_DIM, ROTARY = 5376, 14336, 56, 128, 96
EPS = 1e-5


def _arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return int(major) * 10 + int(minor)


@pytest.fixture(scope="session")
def machine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[kernel_cache.Store | None]:
    """This card's kernels in a machine store, built exactly as a worker boot builds them."""
    if NO_CARD:
        yield None
        return
    root = tmp_path_factory.mktemp("machine")
    (root / "boot").mkdir()
    store = kernel_cache.Store(root / "u")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TRITON_CACHE_DIR", str(store.own / "triton"))
        patch.setenv("TMPDIR", str(root / "boot"))  # the worker boot's failure scope
        patch.setattr("tempfile.tempdir", None)
        patch.setattr(kernel_cache, "_MACHINE_STORE", [store])
        device = torch.device("cuda")
        assert fusion.expect(device) == "compiling"
        job = fusion.job(_arch())
        with store.building(job.key):  # waits on the live builder, never a clock
            pass
        state = kernel_compile.status(store, job)
        assert state.state == "ready", state
        yield store


@pytest.fixture(autouse=True)
def _no_grad(machine: kernel_cache.Store | None) -> Iterator[None]:
    if NO_CARD:
        yield
        return
    with torch.no_grad():
        yield
    # Each arm builds a real-width transformer; a development card is 8 GiB, so the
    # previous arm's is collected before the next one allocates.
    gc.collect()
    torch.cuda.empty_cache()


def _assert_bf16_close(reference: Any, actual: Any, *, exact: bool = False) -> None:
    assert actual.dtype == reference.dtype and actual.shape == reference.shape
    reference, actual = reference.float(), actual.float()
    fraction = (reference != actual).float().mean().item()
    assert fraction <= (0.0 if exact else _BF16_MISMATCH_FRACTION), f"{fraction:.2e} differ"
    worst = (reference - actual).abs().max().item()
    assert worst <= _RANGE * reference.abs().max().item(), f"{worst:.3e} apart"


def _tables(torch: Any, rows: int, device: Any) -> tuple[Any, ...]:
    table = (0.5 * torch.randn(rows, 6 * HIDDEN, device=device)).to(torch.bfloat16)
    return tuple(table.chunk(6, dim=-1))


def _tree(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*"))}


@needs_card
def test_a_fresh_process_loads_the_boot_build_and_launches_without_compiling(
    machine: kernel_cache.Store,
) -> None:
    """The session's build went through `kernel_compile.submit` to the niced background
    builder. A fresh process with `triton.compile` and the C compiler disarmed loads it from
    the store and launches; Triton's cache is afterwards exactly what the build left."""
    job = fusion.job(_arch())
    entry = machine.entry(job.key)
    assert entry is not None and [p.name for p in entry.iterdir()] == ["entry.json"]
    producer = json.loads((entry / "entry.json").read_text())["producer"]
    assert producer["entries"] == len(job.spec["entries"]) == 8
    print(f"sm{_arch()} fused kernels compiled in {producer['compile_ms'] / 1000:.1f} s")
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    before = _tree(cache)
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        import torch, triton
        from triton.compiler import compiler
        from cozy_runtime.internal import fusion, kernel_cache

        def refuse(*args, **kwargs):
            raise AssertionError("the loader reached triton.compile")

        triton.compile = compiler.compile = refuse
        kernel_cache.configure(kernel_cache.Store(Path({str(machine.own)!r})))
        torch.manual_seed(0)
        x, branch = torch.randn(2, 3, 512, {HIDDEN}, device="cuda").to(torch.bfloat16)
        gate = torch.randn(9, {HIDDEN}, device="cuda").to(torch.bfloat16)
        index = torch.randint(0, 9, (512,), device="cuda")
        eager = x + gate.index_select(0, index) * branch
        assert torch.equal(fusion.gate_add(x, gate, branch, index), eager)
        """
    )
    env = {**os.environ, "CC": "/the-c-compiler-must-not-run"}
    ran = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr[-3000:]
    assert _tree(cache) == before


@needs_card
@pytest.mark.parametrize("batch", [1, 2])
def test_row_kernels_match_eager(batch: int) -> None:
    torch.manual_seed(20260907)
    device, seq, rows = torch.device("cuda"), 1024, 9
    x = (2 * torch.randn(batch, seq, HIDDEN, device=device)).to(torch.bfloat16)
    branch = torch.randn(batch, seq, HIDDEN, device=device).to(torch.bfloat16)
    norm = torch.nn.RMSNorm(HIDDEN, eps=EPS).to(device, torch.bfloat16)
    norm.weight.data.copy_((1 + 0.1 * torch.randn(HIDDEN, device=device)).to(torch.bfloat16))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _tables(torch, rows, device)
    index = torch.randint(0, rows, (seq,), device=device)

    def modulate(value: Any, scale: Any, shift: Any) -> Any:
        return value * (1.0 + scale.index_select(0, index)) + shift.index_select(0, index)

    normed = modulate(norm(x), scale_msa, shift_msa)
    _assert_bf16_close(
        normed, fusion.rmsnorm_modulate(x, norm.weight, scale_msa, shift_msa, index, EPS)
    )
    payload, scale = quantize_activation_rowwise(torch, normed)
    quantized = fusion.rmsnorm_modulate(x, norm.weight, scale_msa, shift_msa, index, EPS, fp8=True)
    assert quantized.shape == tuple(normed.shape)
    assert torch.equal(scale, quantized.scale)
    fp8_fraction = (payload.view(torch.uint8) != quantized.payload.view(torch.uint8)).float().mean()
    assert fp8_fraction.item() <= _BF16_MISMATCH_FRACTION

    hidden = x + gate_msa.index_select(0, index) * branch
    fused_hidden, fused_normed = fusion.gate_add_rmsnorm_modulate(
        x, gate_msa, branch, index, norm.weight, scale_mlp, shift_mlp, EPS
    )
    _assert_bf16_close(hidden, fused_hidden, exact=True)
    _assert_bf16_close(modulate(norm(hidden), scale_mlp, shift_mlp), fused_normed)
    _assert_bf16_close(
        x + gate_mlp.index_select(0, index) * branch,
        fusion.gate_add(x, gate_mlp, branch, index),
        exact=True,
    )

    projected = (2 * torch.randn(batch, seq, 2 * FFN, device=device)).to(torch.bfloat16)
    value, gate = projected.chunk(2, dim=-1)
    activated = value * torch.nn.functional.silu(gate)
    _assert_bf16_close(activated, fusion.swiglu(projected), exact=True)
    payload, scale = quantize_activation_rowwise(torch, activated)
    quantized = fusion.swiglu(projected, fp8=True)
    assert torch.equal(payload.view(torch.uint8), quantized.payload.view(torch.uint8))
    assert torch.equal(scale, quantized.scale)


@needs_card
@pytest.mark.parametrize("batch", [1, 2])
def test_rope_kernel_matches_eager(batch: int) -> None:
    torch.manual_seed(20260907)
    device, seq = torch.device("cuda"), 512
    query = torch.randn(batch, seq, HEADS, HEAD_DIM, device=device).to(torch.bfloat16)
    norm = torch.nn.RMSNorm(HEAD_DIM, eps=EPS).to(device, torch.bfloat16)
    norm.weight.data.copy_((1 + 0.1 * torch.randn(HEAD_DIM, device=device)).to(torch.bfloat16))
    positions = torch.randint(0, 64, (seq, 3), device=device)
    cos, sin = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROTARY // 6).to(device)(positions)
    assert cos.shape == (seq, ROTARY) and cos.dtype is torch.float32
    _assert_bf16_close(
        _apply_rotary_emb(norm(query), cos, sin),
        fusion.rmsnorm_rope(query, norm.weight, cos, sin, EPS),
    )


class _Holder:
    """The shape the installer walks: an object whose attribute has a `components` map."""

    def __init__(self, transformer: Any) -> None:
        self.pipe = type("Pipe", (), {"components": {"fl2va_dit": transformer}})()


def _one_block_transformer(torch: Any, device: Any, **config: int) -> Any:
    transformer = MiniMaxH3Transformer3DModel(num_layers=1, num_refiner_layers=1, **config)
    with torch.no_grad():
        for parameter in transformer.parameters():
            parameter.normal_(std=0.02)
    for name, child in transformer.named_children():
        child.to(torch.float32 if name in transformer._keep_in_fp32_modules else torch.bfloat16)
    return transformer.to(device).eval()


def _packed_inputs(torch: Any, device: Any, seq: int) -> dict[str, Any]:
    torch.manual_seed(1)
    text, audio = 40, 24
    video = seq - text - audio
    return {
        "hidden_states": torch.randn(1, video, 96, device=device),
        "audio_hidden_states": torch.randn(1, audio, 32, device=device),
        "encoder_hidden_states": torch.randn(1, text, 5120, device=device),
        "timestep": torch.tensor([0.7, 0.0], device=device),
        "timestep_indices": (torch.arange(seq, device=device) % 2),
        "token_tags": torch.cat(
            [
                torch.ones(text, dtype=torch.int64),
                torch.zeros(video, dtype=torch.int64),
                torch.full((audio,), 2, dtype=torch.int64),
            ]
        ).to(device),
        "position_ids": torch.randint(0, 32, (seq, 3), device=device),
        "video_indices": torch.arange(text, text + video, device=device),
        "audio_indices": torch.arange(text + video, seq, device=device),
        "text_indices": torch.arange(text, device=device),
        "return_dict": False,
    }


def _encode_leaves(torch: Any, transformer: Any) -> int:
    """Stand the block's six Linears in as `fp8-rowwise/1` leaves, the fill's own way."""
    provider = RowwiseNativeLeaf(encoding="fp8-rowwise/1")
    block = transformer.transformer_blocks[0]
    sites = {
        "to_q": block.attn,
        "to_k": block.attn,
        "to_v": block.attn,
        "proj": block.ff.net[0],
        "2": block.ff.net,
    }
    for attribute, owner in sites.items():
        linear = getattr(owner, attribute) if attribute != "2" else owner[2]
        weight = linear.weight.detach().float()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
        payload = (weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        leaf = provider.leaf(
            torch, {"data": payload, "scale": scale.reshape(-1)}, linear, torch.bfloat16
        )
        if attribute == "2":
            owner[2] = leaf
        else:
            setattr(owner, attribute, leaf)
    return len(sites)


@needs_card
@pytest.mark.parametrize("encoded,hooked", [(False, False), (True, False), (True, True)])
def test_installed_block_matches_eager_transformer(encoded: bool, hooked: bool) -> None:
    device = torch.device("cuda")
    transformer = _one_block_transformer(torch, device)
    if encoded:
        assert _encode_leaves(torch, transformer) == 5
    hook_inputs = []
    if hooked:

        def observe_input(module: Any, args: tuple[Any, ...]) -> None:
            assert isinstance(args[0], torch.Tensor)
            hook_inputs.append(tuple(args[0].shape))

        block = transformer.transformer_blocks[0]
        for leaf in (
            block.attn.to_q,
            block.attn.to_k,
            block.attn.to_v,
            block.ff.net[0].proj,
            block.ff.net[2],
        ):
            leaf.register_forward_pre_hook(observe_input)
    inputs = _packed_inputs(torch, device, seq=1536)
    video_eager, audio_eager = transformer(**inputs)
    holder = _Holder(transformer)
    applied = fusion_install.apply_fusion_plan(holder, torch=torch, device=device)
    video_fused, audio_fused = transformer(**inputs)

    assert applied.applied and applied.reason == ""
    assert [path for path, _, _ in applied.substitutions] == [
        "fl2va_dit.transformer_blocks.0.attn.processor",
        "fl2va_dit.transformer_blocks.0.ff.net.0",
        "fl2va_dit.transformer_blocks.0",
    ]
    assert applied.fp8_sites == (3 if encoded and not hooked else 0)
    assert applied.document()["substitutions"] == {
        "MiniMaxH3AttnProcessor->FusedMiniMaxH3AttnProcessor": 1,
        "SwiGLU->FusedSwiGLU": 1,
        "MiniMaxH3TransformerBlock->FusedMiniMaxH3TransformerBlock": 1,
    }
    block = transformer.transformer_blocks[0]
    assert block._fusion_attn_fp8 is (encoded and not hooked)
    assert block.ff.net[0]._fusion_fp8 is (encoded and not hooked)
    if hooked:
        assert len(hook_inputs) == 10
    # The heads are float32; one block of one-ulp bf16 noise stays well inside 1e-2 there.
    for eager, fused in ((video_eager, video_fused), (audio_eager, audio_fused)):
        assert fused.dtype is torch.float32 and fused.shape == eager.shape
        assert torch.isfinite(fused).all()
        assert (fused - eager).abs().max().item() <= 1e-2 * eager.abs().max().item()


@needs_card
def test_a_build_not_ready_serves_eager_under_accept_and_is_awaited_under_require(
    machine: kernel_cache.Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`accept` is CONSENT: absent, compiling or failed, the plan records the typed code and
    the eager forward serves bit for bit. `require` waits for the build, then refuses. No
    kernel launches, so the transformer is narrow."""
    device = torch.device("cuda")
    transformer = _one_block_transformer(
        torch,
        device,
        num_attention_heads=2,
        hidden_size=256,
        ffn_dim=512,
        time_embed_hidden_dim=256,
        time_embed_dim=128,
    )
    inputs = _packed_inputs(torch, device, seq=1536)
    video_eager, audio_eager = transformer(**inputs)
    was = type(transformer.transformer_blocks[0])
    monkeypatch.setattr(fusion, "_LOADED", {})

    def outcome(required: bool = False) -> str:
        try:
            record = fusion_install.apply_fusion_plan(
                _Holder(transformer), torch=torch, device=device, required=required
            )
        except fusion_install.FusionRefusal as exc:
            return f"refused {exc.code}"
        assert not record.applied and record.substitutions == () and record.fp8_sites == 0
        assert record.document()["code"] == record.code
        return record.code

    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [None])  # no rooted worker
    assert outcome() == "fusion_kernels_absent"
    assert outcome(required=True) == "refused fusion_kernels_absent"

    store = kernel_cache.Store(tmp_path / "u")
    cache = store.own / "triton"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache))
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    job = fusion.job(_arch())
    held = store.claim(job.key)  # a live builder holds the key
    assert held is not None
    assert outcome() == "fusion_kernels_compiling"
    os.close(held)

    cache.chmod(0o500)  # the builder cannot write Triton's cache
    try:
        assert outcome(required=True) == "refused fusion_kernels_failed"
        assert outcome() == "fusion_kernels_failed"
    finally:
        cache.chmod(0o700)
    assert kernel_compile.status(store, job).state == "failed"
    assert type(transformer.transformer_blocks[0]) is was
    video, audio = transformer(**inputs)
    assert torch.equal(video, video_eager) and torch.equal(audio, audio_eager)


@needs_card
def test_a_ready_build_that_does_not_load_refuses_even_under_accept(
    machine: kernel_cache.Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store marks the build ready and Triton's cache lost an entry: breakage, which
    `accept` must not hide behind eager."""
    device = torch.device("cuda")
    monkeypatch.setattr(fusion, "_LOADED", {})
    job = fusion.job(_arch())
    group = Path(get_cache_manager(next(iter(job.spec["entries"].values()))).cache_dir)
    away = group.rename(group.with_name(f"{group.name}.away"))
    try:
        with pytest.raises(fusion_install.FusionRefusal) as refused:
            fusion_install.apply_fusion_plan(object(), torch=torch, device=device)
        assert refused.value.code == "fusion_kernels_corrupt"
    finally:
        group.rmdir()  # the lookup's own empty directory
        away.rename(group)


@needs_card
def test_source_drift_serves_eager_before_any_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    transformer = _one_block_transformer(torch, device)
    was = type(transformer.transformer_blocks[0])
    monkeypatch.setitem(fusion_install._SOURCE_DIGESTS, "SwiGLU.forward", "0" * 64)
    record = fusion_install.apply_fusion_plan(_Holder(transformer), torch=torch, device=device)
    assert not record.applied and record.code == "fusion_source_mismatch"
    assert type(transformer.transformer_blocks[0]) is was
    with pytest.raises(fusion_install.FusionRefusal) as refused:
        fusion_install.apply_fusion_plan(
            _Holder(transformer), torch=torch, device=device, required=True
        )
    assert refused.value.code == "fusion_source_mismatch"
    assert type(transformer.transformer_blocks[0]) is was


@needs_card
def test_absent_triton_is_recorded_and_eager_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda")
    monkeypatch.setattr(fusion, "_LOADED", {})
    monkeypatch.setattr(fusion, "TRITON_ABSENT", "triton does not import (planted)")
    record = fusion_install.apply_fusion_plan(object(), torch=torch, device=device)
    assert not record.applied and record.code == "fusion_triton_absent"
    with pytest.raises(fusion_install.FusionRefusal) as refused:
        fusion_install.apply_fusion_plan(object(), torch=torch, device=device, required=True)
    assert refused.value.code == "fusion_triton_absent"


def test_fusion_is_a_closed_class_keyword() -> None:
    class Consenting(Model[object], encoded_leaves="accept", fusion="accept"):
        pass

    class Requiring(Model[object], fusion="require"):
        pass

    class Silent(Model[object]):
        pass

    assert Consenting.__fusion__ == "accept" and Consenting.__encoded_leaves__ == "accept"
    assert Requiring.__fusion__ == "require"
    assert Silent.__fusion__ == "refuse"
    with pytest.raises(ConformanceError) as refused:

        class Loose(Model[object], fusion="maybe"):
            pass

    assert refused.value.code == "fusion_value"
