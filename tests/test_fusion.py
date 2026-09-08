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
The kernels are never compiled here: a session fixture builds the artifact for this card
with `fusion_build` (the same command the worker image's build stage runs) and the loader
resolves cubins from it; one arm proves `triton.compile` is never reached.

The block arm runs a one-block real-width MiniMax-H3 transformer through the runtime's
installer — the same `apply_fusion_plan` the executor calls — twice: as decoded bf16
Linears and as `fp8-rowwise/1` encoded leaves, so the `PreQuantized` handoff to the leaf is
exercised on the real `_scaled_mm` route.
"""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import ConformanceError, Model
from cozy_runtime.internal import fusion, fusion_build, fusion_install
from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise

try:
    import torch
    import triton
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3RotaryPosEmbed,
        _apply_rotary_emb,
    )
    from triton.compiler import compiler
    from triton.runtime.cache import get_cache_manager
except ImportError as _absent:
    CARD_ABSENT = f"torch, triton and diffusers must be importable: {_absent}"
else:
    CARD_ABSENT = ""


def _card() -> str:
    if CARD_ABSENT:
        return CARD_ABSENT
    if not torch.cuda.is_available():
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


def _point_loader_at(directory: Path) -> None:
    fusion.KERNELS_DIR = directory
    fusion.load._cached = {}  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The built kernels for this card, exactly as `fusion_build` writes them for an image."""
    if NO_CARD:
        yield Path()
        return
    directory = tmp_path_factory.mktemp("fusion-kernels")
    fusion_build.build(directory, (_arch(),))
    original = fusion.KERNELS_DIR
    _point_loader_at(directory)
    yield directory
    _point_loader_at(original)


@pytest.fixture(autouse=True)
def _no_grad(artifact: Path) -> Iterator[None]:
    if NO_CARD:
        yield
        return
    _point_loader_at(artifact)
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


def _one_block_transformer(torch: Any, device: Any) -> Any:
    transformer = MiniMaxH3Transformer3DModel(num_layers=1, num_refiner_layers=1)
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
@pytest.mark.parametrize("encoded", [False, True])
def test_installed_block_matches_eager_transformer(encoded: bool) -> None:
    device = torch.device("cuda")
    transformer = _one_block_transformer(torch, device)
    if encoded:
        assert _encode_leaves(torch, transformer) == 5
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
    assert applied.fp8_sites == (3 if encoded else 0)
    assert applied.document()["substitutions"] == {
        "MiniMaxH3AttnProcessor->FusedMiniMaxH3AttnProcessor": 1,
        "SwiGLU->FusedSwiGLU": 1,
        "MiniMaxH3TransformerBlock->FusedMiniMaxH3TransformerBlock": 1,
    }
    block = transformer.transformer_blocks[0]
    assert block._fusion_attn_fp8 is encoded and block.ff.net[0]._fusion_fp8 is encoded
    # The heads are float32; one block of one-ulp bf16 noise stays well inside 1e-2 there.
    for eager, fused in ((video_eager, video_fused), (audio_eager, audio_fused)):
        assert fused.dtype is torch.float32 and fused.shape == eager.shape
        assert torch.isfinite(fused).all()
        assert (fused - eager).abs().max().item() <= 1e-2 * eager.abs().max().item()


@needs_card
def test_loader_never_compiles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fresh cache scope, `triton.compile` armed to fail: load and launch every entry."""

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("the runtime reached triton.compile")

    monkeypatch.setattr(triton, "compile", refuse)
    monkeypatch.setattr(compiler, "compile", refuse)
    monkeypatch.setattr(triton.knobs.cache, "dir", str(tmp_path))
    fusion.load._cached = {}  # type: ignore[attr-defined]
    device = torch.device("cuda")
    fusion.warm(
        device, hidden=HIDDEN, ffn=FFN, heads=HEADS, head_dim=HEAD_DIM, rotary=ROTARY, fp8=True
    )
    manifest = json.loads((fusion.KERNELS_DIR / fusion.MANIFEST).read_text())
    utils = Path(get_cache_manager(manifest["utils"]).cache_dir).name
    entries = sorted(path.name for path in tmp_path.iterdir())
    assert entries == [utils], "only Triton's launcher module was placed; no kernel compiled"


@needs_card
def test_artifact_absent_or_stale_is_typed_at_the_loader(tmp_path: Path, artifact: Path) -> None:
    device = torch.device("cuda")
    _point_loader_at(tmp_path)
    with pytest.raises(fusion.FusionUnavailable) as absent:
        fusion.load(device)
    assert absent.value.code == "fusion_kernels_absent"

    other = tmp_path / "other-arch"
    other.mkdir()
    manifest = json.loads((artifact / fusion.MANIFEST).read_text())
    manifest["targets"] = {"sm999": manifest["targets"][f"sm{_arch()}"]}
    (other / fusion.MANIFEST).write_text(json.dumps(manifest))
    _point_loader_at(other)
    with pytest.raises(fusion.FusionUnavailable) as wrong_arch:
        fusion.load(device)
    assert wrong_arch.value.code == "fusion_kernels_absent"
    assert f"sm{_arch()}" in str(wrong_arch.value)

    stale = tmp_path / "stale"
    stale.mkdir()
    manifest = json.loads((artifact / fusion.MANIFEST).read_text())
    manifest["kernels"] = "0" * 64
    (stale / fusion.MANIFEST).write_text(json.dumps(manifest))
    _point_loader_at(stale)
    with pytest.raises(fusion.FusionUnavailable) as stale_sources:
        fusion.load(device)
    assert stale_sources.value.code == "fusion_kernels_stale"

    # A CONSENTING PACKAGE STILL SERVES on a stale artifact: recorded, not refused.
    stale_record = fusion_install.apply_fusion_plan(
        _Holder(_one_block_transformer(torch, device)), torch=torch, device=device
    )
    assert not stale_record.applied and stale_record.code == "fusion_kernels_stale"
    with pytest.raises(fusion_install.FusionRefusal) as through_plan:
        fusion_install.apply_fusion_plan(
            _Holder(_one_block_transformer(torch, device)),
            torch=torch,
            device=device,
            required=True,
        )
    assert through_plan.value.code == "fusion_kernels_stale"


@needs_card
def test_absent_artifact_serves_eager_and_records_the_fact(tmp_path: Path) -> None:
    """`fusion="accept"` is CONSENT: an image with no artifact serves the eager forward.

    The whole path the executor calls, on a real transformer: the plan does not install,
    the record carries the typed code and both fixed paths the loader looked in, no
    module's class changed, and the next forward is bit-for-bit the eager one. The
    `fusion="require"` spelling is what turns the same condition into a refusal.
    """
    device = torch.device("cuda")
    transformer = _one_block_transformer(torch, device)
    inputs = _packed_inputs(torch, device, seq=1536)
    video_eager, audio_eager = transformer(**inputs)
    was = type(transformer.transformer_blocks[0])

    nowhere = tmp_path / "an-image-baked-without-the-stage"
    _point_loader_at(nowhere)
    record = fusion_install.apply_fusion_plan(_Holder(transformer), torch=torch, device=device)

    assert not record.applied and record.code == "fusion_kernels_absent"
    assert str(nowhere) in record.reason and str(fusion.IMAGE_KERNELS_DIR) in record.reason
    assert record.document()["code"] == "fusion_kernels_absent"
    assert record.substitutions == () and record.fp8_sites == 0
    assert type(transformer.transformer_blocks[0]) is was
    video, audio = transformer(**inputs)
    assert torch.equal(video, video_eager) and torch.equal(audio, audio_eager)

    with pytest.raises(fusion_install.FusionRefusal) as required:
        fusion_install.apply_fusion_plan(
            _Holder(transformer), torch=torch, device=device, required=True
        )
    assert required.value.code == "fusion_kernels_absent"


@needs_card
def test_build_is_deterministic() -> None:
    fusion_build.check((_arch(),))


@needs_card
def test_the_built_artifact_carries_the_launcher_module(artifact: Path) -> None:
    """The NON-`--check` build path, which CI never runs: `utils=True`.

    `fusion_build.check` passes `utils=False` and so never calls `build_utils()` — which
    has Triton link AND dlopen `cuda_utils`, whose own `libcuda.so.1` NEEDED entry has to
    resolve. Every arm in this file loads through the fixture's `build(...)` (utils on),
    so the module is built, carried and placed; this names the invariant.
    """
    manifest = json.loads((artifact / fusion.MANIFEST).read_text())
    assert manifest["utils"] and manifest["utils"] == fusion.utils_key()
    carried = sorted(path.name for path in (artifact / "utils" / manifest["utils"]).iterdir())
    assert any(name.endswith(".so") for name in carried), carried


@needs_card
def test_corrupt_artifact_refuses_even_under_accept(tmp_path: Path, artifact: Path) -> None:
    """An artifact keyed to THIS runtime that does not work is a BAD BAKE, not an absence.

    `accept` serves eager past an image that was never given kernels; it must not serve
    eager past one that was given broken ones, or nobody would ever find the bake bug.
    """
    device = torch.device("cuda")
    manifest = json.loads((artifact / fusion.MANIFEST).read_text())

    # (a) the launcher module the manifest names, not carried.
    no_utils = tmp_path / "no-launcher-module"
    shutil.copytree(artifact, no_utils)
    shutil.rmtree(no_utils / "utils" / manifest["utils"])
    _point_loader_at(no_utils)
    with pytest.raises(fusion.FusionUnavailable) as broken:
        fusion.load(device)
    assert broken.value.code == "fusion_kernels_corrupt"
    with pytest.raises(fusion_install.FusionRefusal) as refused:
        fusion_install.apply_fusion_plan(
            _Holder(_one_block_transformer(torch, device)), torch=torch, device=device
        )
    assert refused.value.code == "fusion_kernels_corrupt"

    # (b) the cubin entries the manifest names, not carried.
    gutted = tmp_path / "no-entry-files"
    shutil.copytree(artifact, gutted)
    for digest in manifest["targets"][f"sm{_arch()}"].values():
        shutil.rmtree(gutted / f"sm{_arch()}" / digest)
    _point_loader_at(gutted)
    with pytest.raises(fusion.FusionUnavailable) as gone:
        fusion.load(device)
    assert gone.value.code == "fusion_kernels_corrupt"


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
    monkeypatch.setattr(fusion.load, "_cached", {}, raising=False)
    monkeypatch.setattr(fusion, "TRITON_ABSENT", "triton does not import (planted)")
    record = fusion_install.apply_fusion_plan(
        _Holder(_one_block_transformer(torch, device)), torch=torch, device=device
    )
    assert not record.applied and record.code == "fusion_triton_absent"
    with pytest.raises(fusion_install.FusionRefusal) as refused:
        fusion_install.apply_fusion_plan(
            _Holder(_one_block_transformer(torch, device)),
            torch=torch,
            device=device,
            required=True,
        )
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
