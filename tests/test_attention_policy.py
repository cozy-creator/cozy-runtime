"""Real Diffusers processor selection; CPU execution where attention arithmetic is used."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author import AttentionContext, Model
from cozy_runtime.internal import attention
from cozy_runtime.internal.attention_registry import register
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor


def _device() -> DeviceFacts:
    return DeviceFacts(kind="cpu", name="CPU", sm=0, driver="", configuration="", index=0)


def _roots() -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor_type: Any = WanAttnProcessor

    return {
        name: torch.nn.ModuleList(
            [WanAttention(dim=8, heads=1, dim_head=8, processor=processor_type())]
        )
        for name in ("dit", "vae")
    }


def _ready(*names: str) -> list[attention.Ready]:
    return [
        attention.Ready(attention.BY_NAME[name], attention._member(attention.BY_NAME[name]))
        for name in names
    ]


def _choices(roots: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        getattr(processor, "_attention_backend", None)
        for _, _, processor in attention._sites(roots)
    )


def test_author_context_is_immutable_and_model_policy_executes() -> None:
    class Policy(Model[object]):
        def choose_attention(self, context: AttentionContext) -> str | None:
            assert context.component in ("dit", "vae") and context.module_path == "0"
            assert context.device_kind == "cpu" and context.sm == 0
            assert context.activation_dtype == "float32" and context.head_dim == 8
            assert context.eligible == ("sdpa",) and context.recommended == "sdpa"
            with pytest.raises(FrozenInstanceError):
                context.degree = 99  # type: ignore[misc]
            return "sdpa"

    policy = object.__new__(Policy)
    roots = _roots()
    applied = attention.select(_device(), roots, choose=policy.choose_attention)
    assert applied.totals() == {"sdpa": 2}
    import torch

    with torch.no_grad():
        assert roots["dit"][0](torch.ones(1, 2, 8)).isfinite().all()


def test_invalid_author_choice_leaves_every_processor_unchanged() -> None:
    roots = _roots()
    before = _choices(roots)

    def choose(context: AttentionContext) -> str:
        return "sdpa" if context.component == "dit" else "kitchen-int8"

    with pytest.raises(attention.AttentionRefusal, match="eligible"):
        attention.select(_device(), roots, choose=choose)
    assert _choices(roots) == before


def test_scoped_pin_and_snapshot_restore_different_prepared_policy() -> None:
    roots = _roots()
    # Installation-only control: real members, no claim that FA3 executes on CPU.
    for root in roots.values():
        root.to(dtype=__import__("torch").bfloat16)
    attention.install(_ready("flash-attn3"), roots)
    prepared = attention.snapshot(roots)
    before = _choices(roots)
    applied = attention.plan(_device(), roots, "dit=sdpa").apply()
    assert applied.hosts == {"dit": {"sdpa": 1}} and applied.pin == "dit=sdpa"
    assert _choices(roots)[0] == "_cozy_sdpa" and _choices(roots)[1] == before[1]
    prepared.restore()
    assert _choices(roots) == before


def test_scoped_prepare_pin_preserves_other_components_and_rejects_typos() -> None:
    roots = _roots()
    seen: list[str] = []

    def choose(context: AttentionContext) -> None:
        seen.append(context.component)

    applied = attention.select(_device(), roots, "dit=sdpa", choose=choose)
    assert applied.hosts == {"dit": {"sdpa": 1}, "vae": {"sdpa": 1}}
    assert seen == ["vae"]
    for pin in ("missing=sdpa", "=sdpa", "dit=", "dit=sdpa=bad"):
        with pytest.raises(attention.AttentionRefusal):
            attention.plan(_device(), roots, pin)


@pytest.mark.parametrize(
    "model_count,aggregate,aliases",
    [(1, False, False), (1, True, False), (2, True, False), (1, False, True)],
)
def test_model_scope_and_restore_use_actual_executor_roots(
    tmp_path: Path, model_count: int, aggregate: bool, aliases: bool
) -> None:
    children = [Executor(None, tmp_path / str(i)) for i in range(model_count)]  # type: ignore[arg-type]
    for i, child in enumerate(children):
        roots = _roots()
        attention.install(_ready("sdpa"), roots)
        child.attention_defaults = attention.snapshot(roots)
        child.models[f"model{i}"] = roots["dit"]
        if aliases:
            child.models["alternate"] = roots["dit"]
        child.device_facts = _device()
        # Only the component registry is needed here; no storage operations are
        # substituted. Selection and execution use actual Diffusers processors.
        child.backend = SimpleNamespace(
            components={"dit": roots["dit"]}, parked={"vae": roots["vae"]}
        )
    parent = children[0]
    if aggregate:
        parent = Executor(None, tmp_path / "parent")  # type: ignore[arg-type]
        parent.model_executors = children
    all_roots = parent._attention_roots()
    expected_sites = {"model0/dit=sdpa": 1, "dit=sdpa": model_count, "sdpa": 2 * model_count}
    if aliases:
        expected_sites["alternate/dit=sdpa"] = 1
    for pin, count in expected_sites.items():
        planned = parent._attention_plan(pin)
        assert planned is not None and len(planned.sites) == count
    with pytest.raises(attention.AttentionRefusal, match="matches no"):
        parent._attention_plan("absent/dit=sdpa")
    swap = parent._attention_plan("model0/dit=sdpa")
    with attention.override(
        parent._attention_snapshot(), all_roots, swap, "model0/dit=sdpa"
    ) as applied:
        component = "alternate+model0/dit" if aliases else "model0/dit"
        assert applied.hosts[component] == {"sdpa": 1}
        import torch

        with torch.no_grad():
            assert all_roots[component][0](torch.ones(1, 2, 8)).isfinite().all()
    for _, _, processor in attention._sites(all_roots):
        processor._attention_backend = "stale-request"
    parent._restore_attention()
    assert _choices(all_roots) == ("_cozy_sdpa",) * (2 * model_count)


def test_scoped_pin_preserves_compiled_and_parallel_fences() -> None:
    roots = _roots()
    roots["dit"][0]._compiled_call_impl = lambda: None
    with pytest.raises(attention.AttentionRefusal) as exc:
        attention.plan(_device(), roots, "dit=sdpa")
    assert exc.value.code == "attention_kernel_frozen"
    attention.plan(_device(), roots, "vae=sdpa")
    attention.plan(_device(), roots, "vae=sdpa", 2)  # a group swaps per request too


def test_a_compiled_construction_serves_the_pin_it_was_prepared_with(tmp_path: Path) -> None:
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    roots = _roots()
    attention.select(_device(), roots, "dit=sdpa")
    roots["dit"][0]._compiled_call_impl = lambda: None
    executor.attention_defaults = attention.snapshot(roots)
    executor.device_facts = _device()
    executor.backend = SimpleNamespace(components=roots, parked={})
    assert executor._attention_plan("dit=sdpa") is None
    with pytest.raises(attention.AttentionRefusal, match="before compilation") as refused:
        executor._attention_plan("dit=flash-attn3")
    assert refused.value.code == "attention_kernel_frozen"


def test_registration_is_idempotent_local_and_dispatches_real_cpu_attention() -> None:
    torch = pytest.importorskip("torch")
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry, dispatch_attention_fn

    def cpu_attention(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        _parallel_config: Any = None,
    ) -> Any:
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        ).transpose(1, 2)

    registry: Any = _AttentionBackendRegistry
    active = registry.get_active_backend()
    member = register("_cozy_cpu_policy_test", cpu_attention)
    assert register("_cozy_cpu_policy_test", cpu_attention) is member
    with pytest.raises(ValueError, match="another implementation"):
        register("_cozy_cpu_policy_test", lambda: None)
    q = torch.randn(1, 4, 2, 8)
    mask = torch.tensor([[True, True, False, False]])
    got = dispatch_attention_fn(q, q, q, backend=member, attn_mask=mask, scale=0.3)
    assert torch.equal(got, cpu_attention(q, q, q, attn_mask=mask, scale=0.3))
    assert not attention._context_parallel(member)
    assert registry.get_active_backend() == active


def test_new_approximate_backends_remain_explicit_and_architecture_exact() -> None:
    for name, sms in [
        ("kitchen-int8", {90, 100, 103, 120, 121}),
        ("flashinfer-bf16-fp8", {100, 103}),
        ("flash-attn4-fp8", {100, 103}),
        ("sageattention3-fp4", {120}),
        ("sageattention", {89, 90, 120}),
    ]:
        candidate = attention.BY_NAME[name]
        assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
        assert candidate.sms == sms
    assert not attention._admits(attention.BY_NAME["flashinfer-bf16-fp8"], "bfloat16", 64)
    assert attention._admits(attention.BY_NAME["flashinfer-bf16-fp8"], "bfloat16", 128)


def test_model_qualified_preparation_pin_routes_to_one_construction() -> None:
    assert (
        attention.for_model("base_model/dit=sageattention", ["base_model"]) == "dit=sageattention"
    )
    assert attention.for_model("base_model/dit=sageattention", ["adapter"]) == ""
    assert attention.for_model("dit=sageattention", ["adapter"]) == "dit=sageattention"
    assert attention.for_model("sdpa", ["base_model"]) == "sdpa"


def test_explicit_install_rejects_all_sites_before_any_change() -> None:
    roots = _roots()
    torch = pytest.importorskip("torch")
    roots["dit"].to(dtype=torch.bfloat16)
    original = _choices(roots)
    with pytest.raises(attention.AttentionRefusal):
        attention.install(_ready("flash-attn3-fp8"), roots, "flash-attn3-fp8")
    assert _choices(roots) == original


def test_failure_restores_policy_and_observations_keep_actual_choice(
    tmp_path: Path,
) -> None:
    from cozy_runtime.author._services import Attempt

    roots = _roots()
    torch = pytest.importorskip("torch")
    for root in roots.values():
        root.to(dtype=torch.bfloat16)
    prepared = attention.install(_ready("flash-attn3"), roots)
    defaults = attention.snapshot(roots)
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    executor.attention = prepared
    record = Attempt("attention-policy-test", tmp_path, 0)
    swap = attention.plan(_device(), roots, "dit=sdpa")
    with (
        pytest.raises(RuntimeError, match="handler failed"),
        attention.override(defaults, roots, swap, "dit=sdpa") as choices,
    ):
        executor._emit_attention(record, choices, "attention.selected")
        assert executor._execution_observation(choices)["kernel_symbol"] == "flash-attn3,sdpa"
        assert executor.attention.totals() == {"flash-attn3": 2}
        raise RuntimeError("handler failed")
    with attention.override(defaults, roots, None) as choices:
        assert choices.totals() == {"flash-attn3": 2}
        assert executor._execution_observation(choices)["kernel_symbol"] == "flash-attn3"
        executor._emit_attention(record, choices, "attention.restored")
    rows = record.ring.rows()
    assert any(row["name"] == "attention.selected" and row["value"] == "sdpa" for row in rows)
    assert not any(row["name"] == "attention.restored" and row["value"] == "sdpa" for row in rows)


def test_sol_sites_are_removed_after_request_and_partial_apply_failure() -> None:
    from cozy_runtime.internal import attention_sol

    roots = _roots()
    attention.install(_ready("sdpa"), roots)
    defaults = attention.snapshot(roots)
    dit, vae = roots["dit"][0], roots["vae"][0]
    entry = _ready("sol-attn")[0]
    swap = attention.Swap(entry, (("dit", dit.processor),), "dit=sol-attn", (("dit", "0", dit),))
    with attention.override(defaults, roots, swap):
        assert hasattr(dit, "_cozy_sol_site")
    assert not hasattr(dit, "_cozy_sol_site")
    assert not dit._forward_pre_hooks and not dit._forward_hooks
    attention_sol.install_site(vae, "vae", "prepared")
    conflicting = attention.Swap(
        entry,
        (("dit", dit.processor), ("vae", vae.processor)),
        "sol-attn",
        (("dit", "0", dit), ("vae", "changed", vae)),
    )
    with (
        pytest.raises(ValueError, match="another Sol site"),
        attention.override(defaults, roots, conflicting),
    ):
        pytest.fail("conflicting installation must fail before entering the request")
    assert not hasattr(dit, "_cozy_sol_site")
    assert hasattr(vae, "_cozy_sol_site")
    assert _choices(roots) == ("_cozy_sdpa", "_cozy_sdpa")
    attention_sol.remove_site(vae, "vae", "prepared")


def test_sol_can_be_installed_for_context_parallel_degree() -> None:
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor_type: Any = WanAttnProcessor

    roots = {
        "dit": torch.nn.ModuleList(
            [WanAttention(dim=512, heads=4, dim_head=128, processor=processor_type())]
        ).to(torch.bfloat16)
    }
    applied = attention.install(_ready("sol-attn"), roots, "sol-attn", degree=4)
    assert applied.totals() == {"sol-attn": 1}


def test_encoded_projection_output_contract_wins_over_norm_and_storage_dtype() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal.encoding import RowwiseNativeLeaf

    module = _roots()["dit"][0]
    provider = RowwiseNativeLeaf(encoding="fp8-rowwise/1")
    for name in ("to_q", "to_k", "to_v"):
        linear = getattr(module, name).to(dtype=torch.bfloat16)
        parts = {
            "data": torch.zeros_like(linear.weight, dtype=torch.float8_e4m3fn),
            "scale": torch.ones(linear.out_features, dtype=torch.float32),
        }
        setattr(module, name, provider.leaf(torch, parts, linear, torch.bfloat16))
    assert next(module.parameters()).dtype == torch.float32
    assert module.to_q.data.dtype == torch.float8_e4m3fn
    assert attention._dtype(module) == "bfloat16"
    module.to_v.out_dtype = torch.float16
    assert attention._dtype(module) == ""


def test_active_packed_projection_and_unknown_contract_are_distinguished() -> None:
    torch = pytest.importorskip("torch")
    module = _roots()["dit"][0]
    module.fuse_projections()
    module.to_qkv.to(dtype=torch.bfloat16)
    assert module.to_q.weight.dtype == torch.float32
    assert attention._dtype(module) == "bfloat16"
    module.fused_projections = False
    assert attention._dtype(module) == "float32"
    module.to_q = torch.nn.Identity()
    assert attention._dtype(module) == ""


def test_repeated_group_pin_requires_every_scoped_prepared_site() -> None:
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor_type: Any = WanAttnProcessor
    modules = [
        WanAttention(dim=128, heads=1, dim_head=128, processor=processor_type()) for _ in range(2)
    ]
    roots = {"base/dit": torch.nn.ModuleList(modules).to(torch.bfloat16)}
    attention.install(_ready("sdpa"), roots, "sdpa", degree=2)
    defaults = attention.snapshot(roots)
    assert attention._is_selected(defaults, roots, "base/dit=sdpa")
    assert not attention._is_selected(defaults, roots, "missing/dit=sdpa")
    assert not attention._is_selected(defaults, roots, "base/dit=absent")
    # A changed site is not a no-op, even if the other site still matches.
    modules[1].processor._attention_backend = attention._member(attention.BY_NAME["cudnn"])
    assert not attention._is_selected(defaults, roots, "base/dit=sdpa")
    changed_defaults = attention.snapshot(roots)
    defaults.restore()
    # Nor may restoration silently replace a currently matching choice with another.
    assert not attention._is_selected(changed_defaults, roots, "base/dit=sdpa")
    del modules[1].processor._attention_backend
    # Class defaults do not turn an unmanaged/missing instance choice into a match.
    assert not attention._is_selected(defaults, roots, "base/dit=sdpa")


def _h3_roots(dtype: str) -> dict[str, Any]:
    """Real Diffusers attention sites under H3's component names, 128-dimension heads."""
    torch = pytest.importorskip("torch")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor_type: Any = WanAttnProcessor
    return {
        name: torch.nn.ModuleList(
            [WanAttention(dim=256, heads=2, dim_head=128, processor=processor_type())]
        ).to(getattr(torch, dtype))
        for name in ("fl2va_dit", "ref2va_dit", "audio_vae", "video_vae")
    }


def _cuda(sm: int) -> DeviceFacts:
    """Device facts only: nothing below computes on them, so no card is needed."""
    return DeviceFacts(kind="cuda", name=f"sm{sm}", sm=sm, driver="", configuration="", index=0)


def test_h3_prefers_sol_then_sage_then_fa3_for_every_dit_and_never_fp8() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import DIT_ATTENTION, H3Model, H3TurboBase

    assert DIT_ATTENTION == ("sol-attn", "sageattention", "flash-attn3", "sdpa")
    fp8 = {"flash-attn3-fp8", "flashinfer-bf16-fp8"}
    assert not fp8 & set(DIT_ATTENTION)
    for device in (_device(), _cuda(90), _cuda(100)):
        assert not fp8 & {candidate.name for candidate in attention.for_device(device)}
    for model_type in (H3Model, H3TurboBase):
        model = object.__new__(model_type)
        for component in ("fl2va_dit", "ref2va_dit", "audio_vae", "video_vae", "text_encoder"):
            context = AttentionContext(
                component, "0", "cuda", "H100", 90, "bfloat16", 128, 4, (), "flash-attn3", ""
            )
            expected = DIT_ATTENTION if component.endswith("_dit") else None
            assert model.choose_attention(context) == expected


@pytest.mark.parametrize(("sm", "served"), [(90, "flash-attn3"), (100, "sdpa")])
def test_h3_dits_fall_through_when_sol_and_sage_cannot_serve(sm: int, served: str) -> None:
    """sm90 with Sol and Sage not compiled here walks to FA3; sm100, where FA3's hopper build
    has no device code, walks to SDPA. Each earlier preference says why it was skipped."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import H3Model

    roots = _h3_roots("bfloat16")
    skipped: dict[str, str] = {}
    applied = attention.install(
        _ready(*[c.name for c in attention.for_device(_cuda(sm))]),
        roots,
        device=_cuda(sm),
        choose=object.__new__(H3Model).choose_attention,
        skipped=skipped,
    )
    assert applied.hosts == {name: {served: 1} for name in roots}
    assert {"sol-attn", "sageattention"} <= set(skipped), skipped
    if sm != 90:
        assert f"sm{sm}" in skipped["sageattention"] and "sm100" not in skipped["sol-attn"]
    assert not {name for name in skipped if "fp8" in name}


def test_h3_preference_records_skips_through_real_cpu_selection() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import H3Model

    applied = attention.select(
        _device(), _h3_roots("float32"), choose=object.__new__(H3Model).choose_attention
    )
    assert applied.totals() == {"sdpa": 4}
    assert applied.skipped is not None
    assert {"sol-attn", "sageattention", "flash-attn3"} <= set(applied.skipped)
    assert "skipped" in applied.line()


@pytest.mark.parametrize(
    ("offered", "dense"),
    [(("sageattention", "flash-attn3"), "sageattention"), (("flash-attn3",), "flash-attn3")],
)
def test_sol_recomputes_dense_rows_with_the_next_h3_preference(
    offered: tuple[str, ...], dense: str
) -> None:
    """Sol's dense steps, prefix and paths run on the kernel after it in H3's preference, not
    the ranked floor (run 1565: SDPA at 35 s/step where Sage takes ~24 s). An unready Sage is
    recorded as skipped, so a later request boundary re-selects onto it."""
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.internal import attention_sol
    from cozy_runtime.models.minimax_h3.model import H3Model

    skipped: dict[str, str] = {}
    applied = attention.install(
        _ready("sol-attn", *offered, "sdpa"),
        _h3_roots("bfloat16"),
        device=_cuda(90),
        choose=object.__new__(H3Model).choose_attention,
        skipped=skipped,
    )
    assert applied.hosts["ref2va_dit"] == {"sol-attn": 1}
    assert attention_sol.dense_identity() == dense
    assert ("sageattention" in skipped) == (dense != "sageattention")


def test_a_pin_overrides_the_h3_preference() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import H3Model

    policy = object.__new__(H3Model)
    consulted: list[str] = []

    def choose(context: AttentionContext) -> Any:
        consulted.append(context.component)
        return policy.choose_attention(context)

    ready = _ready("flash-attn3", "sdpa")
    scoped = attention.install(
        ready, _h3_roots("bfloat16"), "ref2va_dit=sdpa", device=_cuda(90), choose=choose
    )
    assert scoped.hosts["ref2va_dit"] == {"sdpa": 1}
    assert scoped.hosts["fl2va_dit"] == {"flash-attn3": 1}
    assert "ref2va_dit" not in consulted
    consulted.clear()
    whole = attention.install(ready, _h3_roots("bfloat16"), "sdpa", device=_cuda(90), choose=choose)
    assert whole.totals() == {"sdpa": 4} and not consulted


def test_a_kernel_this_runtime_lacks_is_skipped_for_the_next_preference() -> None:
    """A package written for a newer Runtime prefers a kernel an older machine lacks; the
    load used to refuse `attention_kernel_unknown` instead of serving its next choice."""
    applied = attention.select(_device(), _roots(), choose=lambda context: ("sol-next", "sdpa"))
    assert applied.totals() == {"sdpa": 2}
