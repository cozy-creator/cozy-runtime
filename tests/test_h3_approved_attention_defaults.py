"""Real Diffusers sites and Runtime policy, without launching a CUDA context."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention  # noqa: E402

from cozy_runtime.internal import attention, attention_producer  # noqa: E402
from cozy_runtime.internal.encoding import DeviceFacts  # noqa: E402
from cozy_runtime.models.minimax_h3.model import BLACKWELL_DIT_ATTENTION, H3Model  # noqa: E402


def roots() -> dict[str, Any]:
    return {
        "fl2va_dit": torch.nn.ModuleList(
            [MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128).to(torch.bfloat16)]
        )
    }


def ready() -> list[attention.Ready]:
    # Ready is the already-compiled input to installation. This tests policy and actual
    # instance mutation, not whether these GPU libraries launch on a CPU test host.
    return [
        attention.Ready(attention.BY_NAME[name], attention._member(attention.BY_NAME[name]))
        for name in dict.fromkeys(
            (*BLACKWELL_DIT_ATTENTION, "kitchen-sol-producer-fast-shared-qkv", "kitchen-sol")
        )
    ]


@pytest.mark.parametrize("sm", [80, 86, 89, 90, 100, 103, 120, 121])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_blackwell_chooses_hybrid_and_other_hardware_retains_its_policy(
    sm: int, degree: int
) -> None:
    modules = roots()
    device = DeviceFacts(kind="cuda", name=f"sm{sm}", sm=sm, driver="", configuration="", index=0)
    model = object.__new__(H3Model)
    selected = attention.install(
        ready(), modules, degree=degree, device=device, choose=model.choose_attention
    )
    if sm in {120, 121}:
        expected = (
            "kitchen-sol-producer-fp4-lowmem-shared-qkv"
            if degree == 1
            else "kitchen-sol-fp4-lowmem"
        )
    else:
        expected = "sol-attn" if sm in {89, 90, 100} else "sdpa"
    assert selected.hosts == {"fl2va_dit": {expected: 1}}
    assert attention.observed(modules).hosts == selected.hosts
    assert isinstance(modules["fl2va_dit"][0].processor, attention_producer.Processor) == (
        degree == 1 and sm in {120, 121}
    )


def test_explicit_dense_override_wins_and_restores_the_approved_default() -> None:
    modules = roots()
    device = DeviceFacts(kind="cuda", name="RTX 5090", sm=120, driver="", configuration="", index=0)
    selected = attention.install(
        ready(), modules, device=device, choose=object.__new__(H3Model).choose_attention
    )
    defaults = attention.snapshot(modules)
    pin = "fl2va_dit=sdpa"
    swap = attention.plan(device, modules, pin)
    with attention.override(defaults, modules, swap, pin):
        assert attention.observed(modules).hosts == {"fl2va_dit": {"sdpa": 1}}
    assert attention.observed(modules).hosts == selected.hosts


@pytest.mark.parametrize("fails", [False, True])
def test_temporary_producer_restores_original_processor_identity(fails: bool) -> None:
    modules = roots()
    site = modules["fl2va_dit"][0]
    attention.install([entry for entry in ready() if entry.name == "sdpa"], modules)
    original = site.processor
    defaults = attention.snapshot(modules)
    candidate = attention.BY_NAME["kitchen-sol-producer-fp4-lowmem-shared-qkv"]
    swap = attention.Swap(
        attention.Ready(candidate, attention._member(candidate)),
        (("fl2va_dit", original),),
        candidate.name,
        (("fl2va_dit", "0", site),),
    )
    with pytest.raises(RuntimeError, match="handler failed") if fails else nullcontext():
        with attention.override(defaults, modules, swap):
            assert isinstance(site.processor, attention_producer.Processor)
            assert attention.observed(modules).totals() == {candidate.name: 1}
            if fails:
                raise RuntimeError("handler failed")
    assert site.processor is original and not hasattr(site, "_cozy_sol_site")
    assert attention._is_selected(defaults, modules, "fl2va_dit=sdpa")


@pytest.mark.parametrize("sm", [75, 130])
def test_unsupported_recipe_is_reported_and_generic_policy_stays_supported(sm: int) -> None:
    modules = roots()
    device = DeviceFacts(kind="cuda", name=f"sm{sm}", sm=sm, driver="", configuration="", index=0)
    refused: dict[str, str] = {}
    selected = attention.install(
        [entry for entry in ready() if entry.name == "sdpa"],
        modules,
        device=device,
        choose=object.__new__(H3Model).choose_attention,
        skipped=refused,
    )
    assert selected.hosts == {"fl2va_dit": {"sdpa": 1}}
    assert "sol-attn" in refused and "sageattention" in refused
    assert not {c.name for c in attention.CANDIDATES} & set(BLACKWELL_DIT_ATTENTION[:2])


def test_explicit_fp4_on_hopper_and_producer_on_multiple_gpus_refuse() -> None:
    device = DeviceFacts(kind="cuda", name="H100", sm=90, driver="", configuration="", index=0)
    with pytest.raises(attention.AttentionRefusal, match="serves SM"):
        attention.check(device, roots(), "kitchen-sol-fp4-lowmem")
    with pytest.raises(attention.AttentionRefusal, match="context parallelism"):
        attention.admit("kitchen-sol-producer-fast-shared-qkv", 4)


@pytest.mark.parametrize(
    "kernel,dense",
    [
        (
            "kitchen-sol-producer-fp4-lowmem-shared-qkv",
            "sageattention3-fp4-global-lowmem; sparse-prefix=kitchen-int8",
        ),
        ("kitchen-sol-fp4-lowmem", "sageattention3-fp4-global-lowmem; sparse-prefix=kitchen-int8"),
        ("kitchen-sol", "kitchen-int8"),
    ],
)
def test_actual_attention_receipt_names_the_hybrid_dense_reference(
    tmp_path: Path, kernel: str, dense: str
) -> None:
    from cozy_runtime.author._services import Attempt
    from cozy_runtime.internal.executor import Executor

    record = Attempt("h3-approved-attention", tmp_path, 0)
    choices = attention.Applied({"fl2va_dit": {kernel: 1}}, (kernel,), "")
    Executor._emit_attention(record, choices, "attention.selected")
    row = record.ring.rows()[0]
    assert row["value"] == kernel
    assert row["fields"] == {
        "component": "fl2va_dit",
        "sites": 1,
        "pin": "",
        "dense_reference": dense,
    }
