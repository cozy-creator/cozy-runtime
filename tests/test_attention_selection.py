"""The runtime picks the fastest attention kernel it has (cr-124).

The table arms run everywhere. The CUDA arm runs `testdata/attention_selection_proof.py` in a
child process against real diffusers 0.40.0 modules on the box's own card, and skips naming
whichever of those this host lacks.

This file also holds the cr-122 observation-cap arms. They were written for the attestation
this issue deletes and they OUTLIVE it: the rule they prove — telemetry may not fail the thing
it observes — is what made a 1,744-character record a truncation instead of an infinite retry
loop, and nothing about that depends on what produced the record.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._observations import (
    NAME_CAP,
    TEXT_CAP,
    EventRing,
)
from cozy_runtime.internal import (
    attention,
    attention_fp8,
    attention_upstream,
    executor_commands,
)
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor_commands import Binding
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.plan import DeclaredBinding, ModelBinding
from cozy_runtime.internal.worker.session import executor_load_command
from test_end_to_end import NO_EXECUTOR

PACKAGE = Path(__file__).resolve().parent.parent / "examples" / "marco-polo"
needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


def _device(sm: int, kind: str = "cuda") -> DeviceFacts:
    return DeviceFacts(kind=kind, name="card", sm=sm, driver="580", configuration="", index=0)


def _refusal(run: Any) -> attention.AttentionRefusal:
    with pytest.raises(attention.AttentionRefusal) as refused:
        run()
    return refused.value


# ------------------------------------------------------------------------- the static rule


RANKED = ["flash-attn3", "sdpa"]


def test_fp8_admits_a_strict_subset_of_the_bf16_entrys_sites_so_no_floor_site_moves() -> None:
    """The promotion moves ONLY sites the bf16 entry already served at head dim <= 128. Every
    site that kept the floor keeps it — H3's two VAEs (x37) hold it on dtype — because fp8's
    admission is a strict subset of flash-attn3's on every axis, and a site outside it falls
    through to the next rank rather than being forced."""
    fp8, bf16 = attention.BY_NAME["flash-attn3-fp8"], attention.BY_NAME["flash-attn3"]
    dtypes = ("bfloat16", "float16", "float32", "float8_e4m3fn", "")
    dims = (0, 64, 128, 192, 256, 512)
    admitted = {(d, h) for d in dtypes for h in dims if attention._admits(fp8, d, h)}
    assert admitted == {
        ("bfloat16", 64),
        ("bfloat16", 128),
        ("float16", 64),
        ("float16", 128),
    }
    assert all(attention._admits(bf16, d, h) for d, h in admitted)
    assert attention._admits(bf16, "bfloat16", 256) and not attention._admits(fp8, "bfloat16", 256)


def test_the_fp8_quantiser_maps_each_head_onto_e4m3_and_hands_back_its_descale() -> None:
    """cr-136. The eager path is the executable floor of the fused one and what a CPU probe
    runs: per-(batch, head) descales in FA3's shape, every code finite, the largest magnitude
    of every head landing on 440 (not 448, which a rounded product could overshoot into NaN),
    and a dequantised tensor within e4m3's own precision of the input. An all-zero head is
    zeros, not a division by zero."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    import torch

    x = torch.randn(2, 96, 3, 32, generator=torch.Generator().manual_seed(1)) * 0.6
    x[1, :, 2] = 0.0
    codes, descale = attention_fp8.quantise_eager(x.to(torch.bfloat16))
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == x.shape
    assert descale.shape == (2, 3) and descale.dtype == torch.float32
    assert torch.isfinite(codes.float()).all()
    top = codes.float().abs().amax(dim=(1, 3))
    assert torch.allclose(
        top[:, :2], torch.full_like(top[:, :2], attention_fp8.E4M3_TARGET), rtol=0.02
    )
    assert float(top[1, 2]) == 0.0
    assert float(descale[1, 2]) == pytest.approx(attention_fp8.SCALE_FLOOR)
    back = codes.float() * descale[:, None, :, None]
    error = float((back - x.to(torch.bfloat16).float()).norm() / x.norm())
    assert 1e-2 < error < 6e-2, error
    # A CUDA-less box has no fused path, and asks for it exactly once.
    assert attention_fp8.quantise(x.to(torch.bfloat16))[1].shape == (2, 3)


def test_every_nameable_kernel_has_one_name_and_one_backend() -> None:
    names = [c.name for c in (*attention.CANDIDATES, *attention.PINNABLE)]
    backends = [c.backend for c in (*attention.CANDIDATES, *attention.PINNABLE)]
    assert len(names) == len(set(names)) == len(set(backends))
    assert set(attention.BY_NAME) == set(names)
    # Both quantized-activation kernels remain explicit choices. A speed result
    # alone does not make their numerical change an automatic model default.
    for name in ("flash-attn3-fp8", "sageattention", "sageattention-fp16pv"):
        assert name in {c.name for c in attention.PINNABLE}
        assert name not in {c.name for c in attention.CANDIDATES}


@pytest.mark.parametrize(
    ("name", "dtype", "head_dim", "admits"),
    [
        ("flash-attn3-fp8", "bfloat16", 128, True),
        ("flash-attn3-fp8", "float16", 128, True),
        ("flash-attn3-fp8", "bfloat16", 256, False),
        ("flash-attn3-fp8", "bfloat16", 0, False),
        ("flash-attn3-fp8", "float32", 128, False),
        ("flash-attn3", "bfloat16", 128, True),
        ("flash-attn3", "bfloat16", 256, True),
        ("flash-attn3", "bfloat16", 512, False),
        ("flash-attn3", "float32", 128, False),
        ("cudnn", "float16", 128, True),
        ("cudnn", "float16", 256, False),
        ("cudnn", "bfloat16", 0, False),
        ("sdpa", "float32", 512, True),
    ],
)
def test_a_site_is_admitted_by_its_own_dtype_and_head_dimension(
    name: str, dtype: str, head_dim: int, admits: bool
) -> None:
    """Admission is per SITE and per KERNEL, decided once at the pin. An unknown head
    dimension admits only the unbounded floor: the runtime does not guess a shape into a
    kernel that would fault on it."""
    assert attention._admits(attention.BY_NAME[name], dtype, head_dim) is admits


def test_the_device_alone_decides_which_ranked_kernels_are_even_candidates() -> None:
    """The pure half of the rule: no import, no compile, just the card. A device outside a
    kernel's range never sees it, and every device sees the floor last."""
    for device in (_device(0, kind="cpu"), _device(70), _device(89)):
        assert [c.name for c in attention.for_device(device)] == ["sdpa"]
    assert [c.name for c in attention.for_device(_device(90))] == RANKED
    assert [c.name for c in attention.for_device(_device(100))] == ["sdpa"]
    # Resolution then drops what the image does not ship, in order and without raising.
    for device in (_device(0, kind="cpu"), _device(89), _device(90)):
        offered = [entry.name for entry in attention.available(device)]
        ranked = [candidate.name for candidate in attention.for_device(device)]
        assert offered == [name for name in ranked if name in offered]


# ----------------------------------------------------------- the range is bounded ABOVE too


def test_every_candidate_states_the_architectures_it_serves() -> None:
    """h3a-022. An OPEN-ENDED range admitted a card the binary had no device code for, and
    FA3's launcher exits the process from C++ there. Every entry built for a card answers "and
    up to what?"; the torch backends are unbounded because torch, not this runtime, builds
    them, and the floor may never be bounded above."""
    for candidate in (*attention.CANDIDATES, *attention.PINNABLE):
        assert not candidate.max_sm or candidate.max_sm >= candidate.min_sm, candidate.name
        assert not candidate.artifact or candidate.max_sm, candidate.name
    fa3 = attention.BY_NAME["flash-attn3"]
    assert (fa3.min_sm, fa3.max_sm, fa3.artifact) == (90, 90, "flash-attn3")
    sage = attention.BY_NAME["sageattention"]
    assert (sage.max_sm, sage.sms, sage.local) == (
        120,
        frozenset({89, 90, 120}),
        attention_upstream.SAGE,
    )
    assert attention.BY_NAME["sol-attn"].sms == frozenset({89, 90, 100, 120})
    fp16pv = attention.BY_NAME["sageattention-fp16pv"]
    assert (fp16pv.min_sm, fp16pv.max_sm, fp16pv.artifact) == (90, 90, "sageattention")
    assert attention.BY_NAME["cudnn"].max_sm == 0 and attention.BY_NAME["sdpa"].max_sm == 0
    assert attention.CANDIDATES[-1].name == "sdpa"
    assert attention.CANDIDATES[-1].min_sm == 0 and attention.CANDIDATES[-1].max_sm == 0
    for sm in (100, 120):
        refused = _refusal(lambda card=sm: attention.pinned("flash-attn3", _device(card)))
        assert refused.code == "attention_kernel_unsupported" and f"sm{sm}" in str(refused)


@pytest.mark.parametrize(
    ("name", "card", "says", "below"),
    [
        ("sageattention-fp16pv", 120, "sageattention-fp16pv serves sm90 only", 89),
        ("sageattention", 100, "sageattention serves SM [89, 90, 120], not sm100", 86),
    ],
)
def test_a_static_ceiling_refuses_the_card_by_name_instead_of_reaching_the_kernel(
    name: str, card: int, says: str, below: int
) -> None:
    """RED ARM, no plant and no device: a card outside the declared range is refused BY NAME
    rather than admitted and reached.

    Admitting it is what the old table did, and probing it is what would then have run the
    kernel: h3a-022 measured `cudaErrorNoKernelImageForDevice` out of this extension on an
    sm_120 card, from an image that builds it `arch=compute_90a,code=sm_90a`."""
    refused = _refusal(lambda: attention.pinned(name, _device(card)))
    assert refused.code == "attention_kernel_unsupported"
    assert says in str(refused)
    assert f"sm{card}" in str(refused)
    # The refusal is also the discoverability (cr-125): it says what this card CAN run.
    assert "This environment runs" in str(refused)
    # And the same entry on the card it does claim gets past the device gate, so the ceiling
    # is a bound and not a blanket refusal — sm90 reaches resolution and fails on absence.
    assert _refusal(lambda: attention.pinned(name, _device(90))).code == "attention_kernel_absent"
    under = _refusal(lambda: attention.pinned(name, _device(below)))
    assert under.code == "attention_kernel_unsupported" and f"sm{below}" in str(under)
    # The device gate is applied wherever a candidate is offered, not only at the pin.
    assert name not in attention.image_kernels(_device(card))


def test_a_candidate_that_fails_to_resolve_for_any_reason_drops_out_of_auto() -> None:
    """PLANTED RED ARM. Auto-selection has no refusal arm — the floor takes everything — so a
    kernel that fails to resolve for ANY reason, not only a typed one, must drop out of the
    ranking rather than fail a prepare that was only trying to go faster. The plant is the
    shape that would really happen: a baked snapshot built for another ABI, whose loader
    raises whatever it raises. A PIN is the opposite and keeps its typed refusal."""
    broken = attention.BY_NAME["flash-attn3"]
    real = attention._resolve

    def planted(candidate: attention.Candidate, device: DeviceFacts) -> attention.Ready:
        if candidate is broken:
            raise RuntimeError("ImportError: undefined symbol: _ZN3c105ErrorC1E")
        return real(candidate, device)

    attention._resolve = planted
    try:
        assert "flash-attn3" not in [entry.name for entry in attention.available(_device(90))]
        assert "flash-attn3" not in attention.image_kernels(_device(90))
        # The pin does NOT swallow it: a caller who named it is told, not quietly served
        # something else.
        with pytest.raises(RuntimeError, match="undefined symbol"):
            attention.pinned("flash-attn3", _device(90))
    finally:
        attention._resolve = real


def test_a_pin_this_runtime_does_not_know_refuses_and_names_what_it_knows() -> None:
    refused = _refusal(lambda: attention.pinned("flash_attention_2", _device(90)))
    assert refused.code == "attention_kernel_unknown"
    assert "flash-attn3" in str(refused) and "sageattention" in str(refused)


def test_a_pin_this_device_cannot_serve_refuses_naming_the_device() -> None:
    refused = _refusal(lambda: attention.pinned("flash-attn3", _device(89)))
    assert refused.code == "attention_kernel_unsupported"
    assert "sm90" in str(refused) and "sm89" in str(refused)
    assert _refusal(lambda: attention.pinned("cudnn", _device(0, "cpu"))).code == (
        "attention_kernel_unsupported"
    )


def test_the_group_boot_line_renders_from_merged_facts_alone() -> None:
    """A GROUP prepare has ONE boot note, so its `attention` fact is merged from every
    model's and rendered by the same function — not by an `Applied` it does not have. Same
    sentence, namespaced components, offered set unioned in rank order."""
    single = attention.Applied(
        hosts={"dit": {"flash-attn3": 52}},
        offered=("flash-attn3", "sdpa"),
    )
    assert single.line() == attention.line(single.totals(), single.hosts, list(single.offered), "")
    merged = attention.line(
        {"flash-attn3": 104, "sdpa": 36},
        {
            "fl2va/dit": {"flash-attn3": 52},
            "ref2va/dit": {"flash-attn3": 52},
            "fl2va/vae": {"sdpa": 36},
        },
        ["flash-attn3", "sdpa"],
        "",
    )
    assert merged.startswith("flash-attn3 x104, sdpa x36 (auto over ['flash-attn3', 'sdpa'])")
    assert "fl2va/vae: sdpa" in merged and "ref2va/dit: flash-attn3" in merged
    # A construction with no diffusers attention site says so rather than saying nothing.
    assert attention.line({}, {}, [], "").startswith("NONE")


def test_a_pinned_group_prepare_renders_pinned_from_its_models_own_facts() -> None:
    """THE RED ARM (cr-135). `_prepare_many` merges its models' `attention` facts into the
    one boot record through `attention.merge`; the merge it replaced hardcoded an empty pin,
    so a PINNED two-model prepare printed `auto over [...]` — indistinguishable in the record
    from an unpinned one, which is exactly the A/B h3a-013 cannot afford."""
    pinned = {
        parameter: attention.Applied(
            hosts={"dit": {"cudnn": 52}}, offered=("cudnn",), pin="cudnn"
        ).document()
        for parameter in ("fl2va", "ref2va")
    }
    merged = attention.merge(pinned)
    assert merged is not None
    assert merged["pin"] == "cudnn"
    assert merged["line"] == "cudnn x104 (PINNED cudnn) — fl2va/dit: cudnn; ref2va/dit: cudnn"
    assert merged["hosts"] == {"fl2va/dit": {"cudnn": 52}, "ref2va/dit": {"cudnn": 52}}
    assert merged["kernels"] == {"cudnn": 104} and merged["offered"] == ["cudnn"]

    # Unpinned models merge to `auto`, the offered set unioned in rank order.
    auto = attention.merge(
        {
            "fl2va": attention.Applied(
                hosts={"dit": {"flash-attn3": 52}}, offered=("flash-attn3", "sdpa")
            ).document(),
            "ref2va": attention.Applied(hosts={"vae": {"sdpa": 36}}, offered=("sdpa",)).document(),
        }
    )
    assert auto is not None and auto["pin"] == ""
    assert auto["line"].startswith("flash-attn3 x52, sdpa x36 (auto over ['flash-attn3', 'sdpa'])")

    # A model with no site contributes nothing; a set with no site at all is no fact.
    partial = attention.merge({"fl2va": pinned["fl2va"], "vae": {}})
    assert partial is not None
    assert partial["line"] == "cudnn x52 (PINNED cudnn) — fl2va/dit: cudnn"
    assert attention.merge({"a": {}, "b": {}}) is None


# ------------------------------------------------- the pin on REAL diffusers modules (cr-138)


def _diffusers_skip() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    try:
        installed = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        return "diffusers is not installed"
    if installed != "0.40.0":
        return f"diffusers is {installed}, the proof needs 0.40.0"
    return ""


def _ready(*names: str) -> list[attention.Ready]:
    """The ranked entries as `available` hands them to `install`, carrying diffusers' REAL
    members — the fp8 one registered into its registry on first use — without the device
    proof, which is the CUDA arm's to run."""
    return [
        attention.Ready(attention.BY_NAME[name], attention._member(attention.BY_NAME[name]))
        for name in names
    ]


def _sites_of(count: int, *, dim_head: int, heads: int, dtype: Any) -> Any:
    import torch
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    return torch.nn.ModuleList(
        [
            WanAttention(
                dim=dim_head * heads, heads=heads, dim_head=dim_head, processor=WanAttnProcessor()
            )
            for _ in range(count)
        ]
    ).to(dtype)


def test_the_attention_processor_uses_the_installed_backend() -> None:
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    import torch

    root = _sites_of(1, dim_head=8, heads=1, dtype=torch.float32)
    device = _device(0, kind="cpu")
    applied = attention.select(device, {"dit": root})
    assert applied.totals() == {"sdpa": 1}
    with torch.no_grad():
        output = root[0](torch.ones(1, 2, 8))
    assert output.shape == (1, 2, 8) and output.isfinite().all()

    # A processor that ignores the field also passes a successful forward.
    # Rejecting an invalid backend proves this forward reads the installed choice.
    root[0].processor._attention_backend = "missing-audit-backend"
    with pytest.raises(ValueError, match="missing-audit-backend"):
        root[0](torch.ones(1, 2, 8))


def _h3_shaped() -> dict[str, Any]:
    """Real H3 attention classes at reduced width, on meta; VAE dtype-only stand-ins.

    The previous helper used WanAttention for every site, which never exercised
    H3's processor despite the test name. No model weights or GPU are allocated.
    """
    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    with torch.device("meta"):
        dits = {
            name: torch.nn.ModuleList(
                [MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128) for _ in range(52)]
            ).to(torch.bfloat16)
            for name in ("fl2va_dit", "ref2va_dit")
        }
        return {
            **dits,
            "audio_vae": _sites_of(16, dim_head=64, heads=1, dtype=torch.float32),
            "video_vae": _sites_of(21, dim_head=64, heads=1, dtype=torch.float32),
        }


def _backends(roots: Mapping[str, Any]) -> dict[str, list[str]]:
    """What each component's processors would dispatch through, read back off the processors."""
    out: dict[str, set[str]] = {}
    for component, _module, processor in attention._sites(roots):
        member = getattr(processor, "_attention_backend", None)
        out.setdefault(component, set()).add(getattr(member, "value", str(member)))
    return {component: sorted(values) for component, values in out.items()}


def test_auto_preserves_h3_attention_precision_and_explicit_fp8_remains_available() -> None:
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    roots = _h3_shaped()
    weights = {name: [(id(p), p.dtype) for p in root.parameters()] for name, root in roots.items()}
    active = _AttentionBackendRegistry.get_active_backend()
    applied = attention.install(_ready(*RANKED), roots)
    assert applied.totals() == {"flash-attn3": 104, "sdpa": 37}
    assert applied.hosts == {
        "audio_vae": {"sdpa": 16},
        "fl2va_dit": {"flash-attn3": 52},
        "ref2va_dit": {"flash-attn3": 52},
        "video_vae": {"sdpa": 21},
    }
    assert _backends(roots) == {
        "audio_vae": ["_cozy_sdpa"],
        "fl2va_dit": ["_cozy_flash_attn3"],
        "ref2va_dit": ["_cozy_flash_attn3"],
        "video_vae": ["_cozy_sdpa"],
    }
    assert _AttentionBackendRegistry.get_active_backend() == active
    assert {
        name: [(id(p), p.dtype) for p in root.parameters()] for name, root in roots.items()
    } == weights

    # Explicit comparisons keep the FP8 implementation. Automatic preparation
    # after the pin restores BF16 on the same real H3 processor instances.
    dits = {name: roots[name] for name in ("fl2va_dit", "ref2va_dit")}
    pinned = attention.install(_ready("flash-attn3-fp8"), dits, "flash-attn3-fp8")
    assert pinned.totals() == {"flash-attn3-fp8": 104}
    assert all(values == ["_cozy_flash_attn3_fp8"] for values in _backends(dits).values())
    restored = attention.install(_ready(*RANKED), roots)
    assert restored.totals() == {"flash-attn3": 104, "sdpa": 37}
    assert not any("_cozy_flash_attn3_fp8" in values for values in _backends(roots).values())


def _classic(dtype: Any) -> dict[str, Any]:
    """Diffusers' own classic `Attention` sites: a narrow SDXL-style UNet and AutoencoderKL."""
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=(2, 4),
    )
    vae = AutoencoderKL(block_out_channels=(32,), latent_channels=4)
    return {"unet": unet.eval().to(dtype), "vae": vae.eval().to(dtype)}


def test_classic_attention_sites_are_managed_observed_and_bit_identical() -> None:
    """SDXL's UNet and every AutoencoderKL ran torch SDPA through `AttnProcessor2_0`, which no
    selection could reach: evidence said `unobserved` and every pin refused (run 1632)."""
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    import torch
    from diffusers.models.attention_processor import AttnProcessor2_0

    roots = _classic(torch.float32)
    sample, text, image = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 32), torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        before = (roots["unet"](sample, 10, text).sample, roots["vae"](image).sample)
    records = list(attention._attention_records(roots))
    assert records and all(type(row[3]) is AttnProcessor2_0 for row in records)
    assert not list(attention._sites(roots))

    device = _device(0, kind="cpu")
    applied = attention.select(device, roots)
    assert applied.totals() == {"sdpa": len(records)}
    assert set(applied.hosts) == {"unet", "vae"}
    assert attention.observed(roots).totals() == {"sdpa": len(records)}
    with torch.no_grad():
        after = (roots["unet"](sample, 10, text).sample, roots["vae"](image).sample)
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert attention.check(device, roots, "unet=sdpa").name == "sdpa"

    processors = [processor for _, _, processor in attention._sites(roots)]
    attention.select(device, roots)
    assert [processor for _, _, processor in attention._sites(roots)] == processors
    assert len({id(processor) for processor in processors}) == len(records)


def test_auto_keeps_sdpa_on_classic_sites_and_a_pin_moves_them() -> None:
    """Adoption changes what can be chosen, never what auto chooses: a classic site keeps the
    kernel it always ran while a native site on the same card takes the ranked one."""
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    import torch

    roots = _classic(torch.float16)
    attention.adopt(roots)
    roots["dit"] = _sites_of(2, dim_head=64, heads=1, dtype=torch.float16)
    applied = attention.install(_ready(*RANKED), roots)
    assert applied.hosts["dit"] == {"flash-attn3": 2}
    assert set(applied.hosts["unet"]) == set(applied.hosts["vae"]) == {"sdpa"}
    pinned = attention.install(_ready("flash-attn3"), {"unet": roots["unet"]}, "unet=flash-attn3")
    assert set(pinned.hosts["unet"]) == {"flash-attn3"}
    assert _backends({"unet": roots["unet"]}) == {"unet": ["_cozy_flash_attn3"]}


def test_per_request_swap_can_be_followed_by_auto_without_pin_leak() -> None:
    """A pinned attempt changes live processors only for that attempt's boundary."""
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    import torch

    roots = {"dit": _sites_of(1, dim_head=64, heads=1, dtype=torch.float32)}
    device = _device(0, kind="cpu")
    attention.install(_ready("sdpa"), roots)
    attention.plan(device, roots, "sdpa").apply()
    assert _backends(roots) == {"dit": ["_cozy_sdpa"]}
    # This is the exact reset Executor.invoke performs before an unpinned request.
    attention.select(device, roots, "")
    assert _backends(roots) == {"dit": ["_cozy_sdpa"]}


def test_explicit_fp8_refuses_unsupported_sites_while_auto_preserves_dtype() -> None:
    """Both eligible floating sites keep FA3; FP32 uses SDPA. An explicit
    FP8 selection still refuses a head dimension it cannot execute."""
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    import torch

    roots = {
        "dit": _sites_of(1, dim_head=128, heads=2, dtype=torch.bfloat16),
        "wide": _sites_of(1, dim_head=256, heads=1, dtype=torch.float16),
        "vae": _sites_of(1, dim_head=64, heads=1, dtype=torch.float32),
    }
    applied = attention.install(_ready(*RANKED), roots)
    assert applied.hosts == {
        "dit": {"flash-attn3": 1},
        "wide": {"flash-attn3": 1},
        "vae": {"sdpa": 1},
    }
    assert applied.line() == (
        "flash-attn3 x2, sdpa x1 "
        "(auto over ['flash-attn3', 'sdpa']) "
        "— dit: flash-attn3; vae: sdpa; wide: flash-attn3"
    )
    assert _backends(roots) == {
        "dit": ["_cozy_flash_attn3"],
        "wide": ["_cozy_flash_attn3"],
        "vae": ["_cozy_sdpa"],
    }
    # A PIN of fp8 on the 256-dim site is a typed refusal, never a quiet demotion.
    two = {"dit": roots["dit"], "wide": roots["wide"]}
    refused = _refusal(lambda: attention.install(_ready("flash-attn3-fp8"), two, "flash-attn3-fp8"))
    assert refused.code == "attention_kernel_unsupported"
    assert "wide's attention: float16 activations at head dimension 256" in str(refused)


def test_group_merge_preserves_automatic_precision_and_explicit_fp8() -> None:
    """A group reports the precision actually selected and retains explicit pins."""
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    roots = _h3_shaped()
    auto = attention.merge(
        {
            "fl2va": attention.install(_ready(*RANKED), {"dit": roots["fl2va_dit"]}).document(),
            "ref2va": attention.install(
                _ready(*RANKED), {"dit": roots["ref2va_dit"], "vae": roots["video_vae"]}
            ).document(),
        }
    )
    assert auto is not None and auto["pin"] == ""
    assert auto["kernels"] == {"flash-attn3": 104, "sdpa": 21}
    assert auto["line"] == (
        "flash-attn3 x104, sdpa x21 (auto over ['flash-attn3', 'sdpa']) "
        "— fl2va/dit: flash-attn3; ref2va/dit: flash-attn3; ref2va/vae: sdpa"
    )
    pinned = attention.merge(
        {
            parameter: attention.install(
                _ready("flash-attn3-fp8"), {"dit": roots[f"{parameter}_dit"]}, "flash-attn3-fp8"
            ).document()
            for parameter in ("fl2va", "ref2va")
        }
    )
    assert pinned is not None and pinned["pin"] == "flash-attn3-fp8"
    assert pinned["line"] == (
        "flash-attn3-fp8 x104 (PINNED flash-attn3-fp8) "
        "— fl2va/dit: flash-attn3-fp8; ref2va/dit: flash-attn3-fp8"
    )


def test_every_head_local_kernel_shards() -> None:
    """Runtime's Ulysses wrapper makes every head-local kernel context-parallel — the fp8,
    FA4, Sage and Kitchen refusals were adapter plumbing (audit 2026-09-26 §1), and
    FlashInfer's scalar V scale is now applied per head outside its kernel."""
    why = _diffusers_skip()
    if why:
        pytest.skip(why)
    import torch

    roots = {"dit": _sites_of(2, dim_head=128, heads=2, dtype=torch.bfloat16)}
    assert attention.install(_ready(*RANKED), roots, degree=1).totals() == {"flash-attn3": 2}
    sharded = attention.install(_ready(*RANKED), roots, degree=4)
    assert sharded.totals() == {"flash-attn3": 2}
    assert sharded.offered == tuple(RANKED)
    assert _backends(roots) == {"dit": ["_cozy_flash_attn3"]}
    for name in sorted(set(attention.BY_NAME) - set(RANKED)):
        assert attention.install(_ready(name), roots, name, degree=4).totals() == {name: 2}
        attention.admit(f"dit={name}", 4)


# ------------------------------------------------------------------------------ the CUDA arm


def _cuda_proof_skip() -> str:
    why = _diffusers_skip()
    if why:
        return why
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    return ""


def test_the_fastest_admitted_kernel_is_pinned_on_real_attention_sites() -> None:
    why = _cuda_proof_skip()
    if why:
        pytest.skip(why)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/attention_selection_proof.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    out = json.loads(child.stdout.splitlines()[-1])

    # The head dimension and dtype come off the REAL modules, not from a declaration.
    assert out["head_dims"] == {"dit": 128, "vae": 256}
    assert out["dtypes"] == {"dit": "bfloat16", "vae": "float16"}

    # AUTO lands on real processors, and the boot line says what it did in one line.
    assert out["auto"]["offered"] == out["offered"]
    assert set(out["auto"]["kernels"]) <= set(out["offered"])
    for backends in out["auto_backends"].values():
        assert backends and all(b for b in backends)
    chosen = ", ".join(f"{n} x{c}" for n, c in sorted(out["auto"]["kernels"].items()))
    assert out["auto"]["line"].startswith(chosen)

    # Every site takes the first supported precision-preserving kernel.
    expected: dict[str, dict[str, int]] = {}
    for site in ("dit", "vae"):
        for name in out["offered"]:
            if attention._admits(
                attention.BY_NAME[name], out["dtypes"][site], out["head_dims"][site]
            ):
                expected[site] = {name: 1}
                break
    assert out["auto"]["hosts"] == expected
    assert "flash-attn3-fp8" not in out["offered"]
    if "flash-attn3" in out["offered"]:
        assert out["offered"] == RANKED
        assert out["auto"]["hosts"] == {"dit": {"flash-attn3": 1}, "vae": {"flash-attn3": 1}}
    assert out["auto_degree4"]["kernels"] == out["auto"]["kernels"]
    assert out["auto_degree4"]["offered"] == out["offered"]

    # PER-SITE ADMISSION on one construction: cuDNN serves head dim 128 and not 256, so the
    # two sites take different kernels and neither falls back per call.
    assert out["mixed"]["hosts"] == {"dit": {"cudnn": 1}, "vae": {"sdpa": 1}}
    assert out["mixed_backends"] == {"dit": ["_cozy_cudnn"], "vae": ["_cozy_sdpa"]}

    # A PIN IS NEVER QUIETLY REPLACED. The same 256-dim site under a cuDNN-only pin refuses.
    assert out["pin_unserved"]["code"] == "attention_kernel_unsupported"
    assert "head dimension 256" in out["pin_unserved"]["detail"]
    assert out["pin_served"]["pin"] == "cudnn"
    assert out["pin_served"]["line"] == "cudnn x1 (PINNED cudnn) — dit: cudnn"
    assert out["pin_backends"] == {"dit": ["_cozy_cudnn"]}

    # cr-135: the pin SURVIVES THE GROUP MERGE. Two real constructions pinned to cuDNN, read
    # back off their own processors, render PINNED from the one merged record; unpinned they
    # render auto over what this card offers.
    assert out["group_pinned"]["pin"] == "cudnn"
    assert out["group_pinned"]["line"] == (
        "cudnn x2 (PINNED cudnn) — fl2va/dit: cudnn; ref2va/dit: cudnn"
    )
    assert out["group_auto"]["pin"] == ""
    assert out["group_auto"]["offered"] == out["offered"]
    assert sum(out["group_auto"]["kernels"].values()) == 2
    assert "(auto over " in out["group_auto"]["line"]

    # The refusals double as discoverability: each names what would have to be true.
    assert out["unknown"]["code"] == "attention_kernel_unknown"
    assert out["too_old"]["code"] == "attention_kernel_unsupported"
    assert out["absent"]["code"] == "attention_kernel_absent"
    assert "sdpa" in out["image_kernels"]
    # A device refusal names what this image and card DO run, so the developer who pinned a
    # kernel that cannot serve here is not left guessing (cr-125's discoverability rule).
    assert "This environment runs" in out["too_old"]["detail"]
    # h3a-022, through the real registry: the range is bounded ABOVE as well as below, and the
    # card above the ceiling is refused by name rather than reached.
    assert out["too_new"]["code"] == "attention_kernel_unsupported"
    assert "sageattention-fp16pv serves sm90 only" in out["too_new"]["detail"]

    # And the pin computes: one real forward through the pinned site.
    assert out["forward"] == {"shape": [1, 64, 1024], "dtype": "torch.bfloat16"}

    # cr-136. The fp8 path is NAMEABLE on any box, and what it does depends on the card and
    # what this machine has compiled: below sm90 the device refuses it; on sm90 without its
    # compiled artifact it is absent; with it, it is pinned and runs.
    fp8 = out["fp8"]
    if fp8["code"]:
        refusals = (
            {"attention_kernel_unsupported"}
            if out["sm"] != 90
            else {"attention_kernel_absent", "attention_kernel_compiling"}
        )
        assert fp8["code"] in refusals and "flash-attn3-fp8" in fp8["detail"], fp8
    else:
        assert fp8["backend"] == "_cozy_flash_attn3_fp8"
        assert fp8["line"] == "flash-attn3-fp8 x1 (PINNED flash-attn3-fp8) — dit: flash-attn3-fp8"
        assert fp8["forward"] == {"shape": [1, 64, 1024], "dtype": "torch.bfloat16"}
        assert fp8["sharded"] == {"flash-attn3-fp8": 1}


# ------------------------------------------------- the durable record is BOUNDED (cr-122)


def test_an_over_cap_observation_truncates_and_never_faults_the_attempt() -> None:
    """cr-122 RED ARM, KEPT. This exact value faulted the executor on the H100: the emit
    boundary raised `observation_too_large`, the refusal escaped device release, the worker
    abandoned the attempt EXECUTOR_INVALIDATED, rebuilt and ran the same 30-step generation
    again — three complete attempts before an operator cancelled the fourth. A record that
    cannot be held is truncated and counted. It is never a fault."""
    ring = EventRing(subject="req#1")
    row = ring.emit("confession", "attention", "x" * 1744, subject="req#1")
    assert isinstance(row.value, str)
    assert len(row.value) == TEXT_CAP and row.value.endswith("…")
    assert ring.caps()["truncated"] == 1 and ring.caps()["refused"] == 0
    assert ring.caps()["kept"] == 1 and ring.caps()["admitted"] == 1
    # Field values are payload too, and a field of 8 KiB is the same defect wearing a hat.
    ring.emit("log", "note", "short", detail="y" * 8192)
    assert len(str(ring.events[-1].fields["detail"])) == TEXT_CAP
    assert ring.caps()["truncated"] == 2 and ring.caps()["refused"] == 0
    # A NAME over its cap keeps a digest of the whole, so two long names never alias.
    first, second = ring.emit("log", "n" * (NAME_CAP + 1)), ring.emit("log", "n" * (NAME_CAP + 2))
    assert len(first.name) == NAME_CAP and first.name != second.name
    assert ring.caps()["refused"] == 0


def test_the_attempt_confession_of_a_huge_value_is_admitted() -> None:
    """The real worker record, through the real `confess` the release path calls."""
    attempt = AttemptRecord(request_id="req", attempt=1, digest=b"", spec={})
    attempt.confess("attention", "z" * 4096)
    assert attempt.ring.caps()["refused"] == 0 and attempt.ring.caps()["truncated"] == 1
    assert [row["kind"] for row in attempt.confessions] == ["attention"]
    assert len(str(attempt.ring.events[-1].value)) == TEXT_CAP


# ------------------------------------------ the execution-path override, threaded (cr-125)


def _binding(models: int = 1) -> DeclaredBinding:
    """One resolved binding, exactly the shape the boot path hands the prepare builder."""
    slots = tuple(
        ModelBinding(
            model_class="pkg:Dit",
            model_binding_path=f"m{index}",
            model_parameter_name=f"p{index}",
            store="store",
            variant="fp8",
            reference_snapshot="sha256:aa",
            components=("dit",),
            snapshots={"dit": "sha256:aa"},
        )
        for index in range(models)
    )
    return DeclaredBinding(
        entrypoint_binding_digest="sha256:bb",
        entrypoint="fl2va",
        model_class=slots[0].model_class,
        model_binding_path=slots[0].model_binding_path,
        model_parameter_name=slots[0].model_parameter_name,
        release="rel-1",
        models=slots if models > 1 else (),
        components=slots[0].components,
        snapshots=slots[0].snapshots,
        store=slots[0].store,
        variant=slots[0].variant,
        reference_snapshot=slots[0].reference_snapshot,
    )


def test_a_serving_prepare_carries_no_pin_at_all() -> None:
    """PRODUCTION IS KNOB-FREE. The override is absent by default and the serving prepare is
    byte-identical to what it was before cr-125 — the key is not present, not empty."""
    served = executor_load_command(
        _binding(), construction="c", devices="0", authorized_device_limit_bytes=7
    )
    assert "attention_pin" not in executor_commands.encode(served)


def test_a_pinned_prepare_names_one_kernel_for_every_model_of_the_construction() -> None:
    """The pin rides the ONE prepare funnel, and a group takes it whole: two models of one
    construction cannot serve two different execution paths and still be one measurement."""
    single = executor_load_command(
        _binding(),
        construction="c",
        devices="0",
        authorized_device_limit_bytes=7,
        attention_pin="sageattention",
    )
    assert single.attention_pin == "sageattention"
    group = executor_load_command(
        _binding(models=2),
        construction="c",
        devices="0",
        authorized_device_limit_bytes=7,
        attention_pin="sageattention",
    )
    # It sits on the COMMAND, not inside a model row: `_prepare_many` hands the same value
    # to every child prepare, the way it hands them one device authorization.
    assert executor_commands.encode(group)["attention_pin"] == "sageattention"
    assert "attention_pin" not in {f.name for f in msgspec.structs.fields(Binding)}
