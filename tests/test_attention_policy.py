"""Real Diffusers processor selection; CPU execution where attention arithmetic is used."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author import AttentionProducer
from cozy_runtime.author import _attention as author_attention
from cozy_runtime.internal import attention, attention_sol
from cozy_runtime.internal.attention_registry import register
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor


def _device() -> DeviceFacts:
    return DeviceFacts(kind="cpu", name="CPU", sm=0, driver="", configuration="", index=0)


def _roots(dim: int = 8) -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    processor_type: Any = WanAttnProcessor

    return {
        name: torch.nn.ModuleList(
            [WanAttention(dim=dim, heads=1, dim_head=dim, processor=processor_type())]
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


def test_scoped_pin_and_snapshot_restore_different_prepared_policy() -> None:
    roots = _roots(128)  # a shape FA3's build has
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
    applied = attention.select(_device(), roots, "dit=sdpa")
    assert applied.hosts == {"dit": {"sdpa": 1}, "vae": {"sdpa": 1}} and applied.pin == "dit=sdpa"
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
        child.backend = SimpleNamespace(roots=roots, gemm_sites={})
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


def test_a_pin_falls_back_visibly_where_its_kernel_cannot_run() -> None:
    """A compiled module keeps the kernel it was compiled with, and a kernel this card cannot
    run keeps the prepared one: the request records why instead of failing or silently
    serving another kernel."""
    roots = _roots()
    attention.select(_device(), roots)
    roots["dit"][0]._compiled_call_impl = lambda: None
    swap = attention.plan(_device(), roots, "dit=cudnn")
    assert not swap.sites
    (row,) = swap.fallbacks
    assert (row["component"], row["preferred"], row["selected"]) == ("dit", "cudnn", "sdpa")
    assert "compiled" in row["reason"]
    applied = swap.apply()
    assert applied.fallbacks == swap.fallbacks
    assert "vae" not in {r["component"] for r in swap.fallbacks}
    assert "dit: selected sdpa instead of cudnn" in applied.line()
    card = attention.plan(_device(), roots, "vae=flash-attn3")
    assert not card.sites and "wgmma" in card.fallbacks[0]["reason"]
    assert len(attention.plan(_device(), roots, "vae=sdpa").sites) == 1
    assert len(attention.plan(_device(), roots, "vae=sdpa", 2).sites) == 1  # a group too


def test_a_compiled_construction_serves_the_pin_it_was_prepared_with(tmp_path: Path) -> None:
    executor = Executor(None, tmp_path)  # type: ignore[arg-type]
    roots = _roots()
    attention.select(_device(), roots, "dit=sdpa")
    roots["dit"][0]._compiled_call_impl = lambda: None
    executor.attention_defaults = attention.snapshot(roots)
    executor.device_facts = _device()
    executor.backend = SimpleNamespace(roots=roots, gemm_sites={})
    assert executor._attention_plan("dit=sdpa") is None
    other = executor._attention_plan("dit=flash-attn3")
    assert other is not None and not other.sites
    assert "compiled" in other.fallbacks[0]["reason"]


def test_a_pinned_build_is_imported_first_and_serves_the_default_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One process imports one build of a module. A pin known at load reserves its build,
    which also holds the default kernel's; a pin needing another build than the one imported
    is a fresh executor's (the machine starts one), never a refusal of the run."""
    hopper = _cuda(90)
    fa3, fp8 = attention.BY_NAME["flash-attn3"], attention.BY_NAME["flash-attn3-fp8"]
    monkeypatch.setattr(attention, "_BUILDS", {})
    attention.reserve("dit=flash-attn3-fp8", hopper)
    assert attention._build(fa3) == attention._build(fp8) == "flash-attn3-full"
    attention.reserve("sageattention-fp16pv", hopper)
    assert attention._build(attention.BY_NAME["sageattention"]) == "sageattention-full"
    monkeypatch.setattr(attention, "_BUILDS", {fa3.import_name: fa3.artifact})
    assert attention._build(fa3) == "flash-attn3"
    with pytest.raises(attention.AttentionRefusal) as needs:
        attention.plan(hopper, _roots(), "dit=flash-attn3-fp8")
    assert needs.value.code == "attention_kernel_needs_executor"
    assert "flash-attn3-full" in str(needs.value)


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

    roots = _roots(128)  # a shape FA3's build has
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
    swap = attention.Swap((("dit", "0", dit, dit.processor, entry),), "dit=sol-attn")
    with attention.override(defaults, roots, swap):
        assert hasattr(dit, "_cozy_sol_site")
    assert not hasattr(dit, "_cozy_sol_site")
    assert not dit._forward_pre_hooks and not dit._forward_hooks
    attention_sol.install_site(vae, "vae", "prepared")
    conflicting = attention.Swap(
        (("dit", "0", dit, dit.processor, entry), ("vae", "changed", vae, vae.processor, entry)),
        "sol-attn",
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


def test_a_site_that_casts_its_own_operands_is_admitted_by_its_declared_dtype() -> None:
    """A package's own attention site (SeedVR2's windows) has no Q/K/V projections to read: it
    declares the dtype its operands reach the kernel in, and is selected like any other site."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.attention import AttentionModuleMixin
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    class Processor:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, attn: Any, q: Any, k: Any, v: Any) -> Any:
            return dispatch_attention_fn(
                q, k, v, backend=self._attention_backend, parallel_config=self._parallel_config
            )

    class Site(torch.nn.Module, AttentionModuleMixin):  # type: ignore[misc,name-defined]
        activation_dtype: Any = torch.bfloat16
        head_dim = 128

        def __init__(self) -> None:
            super().__init__()
            self.set_processor(Processor())

        def forward(self, q: Any, k: Any, v: Any) -> Any:
            return self.processor(self, *(x.to(self.activation_dtype) for x in (q, k, v)))

    site = Site()
    roots = {"dit": torch.nn.ModuleList([site, torch.nn.Linear(4, 4)])}
    assert attention._dtype(site) == "bfloat16"
    assert attention.install(_ready("flash-attn3", "sdpa"), roots).totals() == {"flash-attn3": 1}
    assert attention.install(_ready("sdpa"), roots).totals() == {"sdpa": 1}
    q, k, v = torch.randn(3, 2, 7, 1, 128).unbind(0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(x.bfloat16().transpose(1, 2) for x in (q, k, v))
    ).transpose(1, 2)
    torch.testing.assert_close(site(q, k, v), expected)
    site.activation_dtype = torch.int8
    assert attention._dtype(site) == ""


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


#: H3's declaration: its DiTs tolerate 4-bit attention; its VAEs stay exact.
H3_BITS = {"fl2va_dit": 4, "ref2va_dit": 4}


def _dit_roots(dtype: str) -> dict[str, Any]:
    """Real Diffusers attention sites under two DiTs' and two VAEs' names, 128-dim heads."""
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


@pytest.mark.parametrize(("sm", "served"), [(90, "flash-attn3"), (100, "cudnn")])
def test_dits_step_up_a_level_when_their_cell_cannot_serve(sm: int, served: str) -> None:
    """H3's DiTs at level 4 walk their family's level-8 cell (Sage2 on wgmma, FA4 fp8 on
    tcgen05), absent here, to level 16, with one warning per DiT; the VAEs sit at 16."""
    pytest.importorskip("diffusers")
    roots = _dit_roots("bfloat16")
    skipped: dict[str, str] = {}
    applied = attention.install(
        _ready("flash-attn3", "cudnn", "sdpa"),
        roots,
        device=_cuda(sm),
        bits=H3_BITS,
        skipped=skipped,
    )
    assert applied.hosts == {name: {served: 1} for name in roots}
    absent = "sageattention" if sm == 90 else "flash-attn4-fp8"
    assert set(skipped) == {absent}
    assert {(row["component"], row["preferred"], row["selected"]) for row in applied.fallbacks} == {
        (dit, absent, served) for dit in H3_BITS
    }


@pytest.mark.parametrize(
    ("offered", "identity"),
    [
        (("sageattention", "flash-attn3"), "sageattention; exact=flash-attn3"),
        (("flash-attn3",), "flash-attn3"),
    ],
)
def test_sol_recomputes_dense_rows_on_the_sites_level_and_exact_paths_at_16(
    offered: tuple[str, ...], identity: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sparse site (a package producer serves it) on Hopper at level 4: Sol + fixed-Kmean
    Sage2 first, absent here, then Sol, whose dense rows take the level's Sage2 and whose
    exact paths FA3 (run 1565: SDPA at 35 s/step where Sage takes ~24 s). An unready Sage is
    skipped, so a later boundary re-selects onto it."""
    pytest.importorskip("diffusers")
    roots = _dit_roots("bfloat16")
    dits = {id(root[0]) for name, root in roots.items() if name in H3_BITS}

    def validate(module: object) -> None:
        if id(module) not in dits:
            raise ValueError("not an H3 DiT site")

    producer = AttentionProducer("h3-p", "kitchen-sol-fp4-fixed-means", lambda *_: None, validate)
    monkeypatch.setitem(author_attention._PRODUCERS, "h3-p", producer)
    held = attention_sol.binding()
    skipped: dict[str, str] = {}
    try:
        applied = attention.install(
            _ready("sol-attn", *offered, "sdpa"),
            roots,
            device=_cuda(90),
            bits=H3_BITS,
            skipped=skipped,
        )
        assert applied.hosts["ref2va_dit"] == {"sol-attn": 1}
        assert applied.hosts["video_vae"] == {"flash-attn3": 1}
        assert attention_sol.dense_identity() == identity
        assert "sol-attn-sage2-fixed-kmean" in skipped
        assert ("sageattention" in skipped) == ("sageattention" not in offered)
    finally:
        attention_sol.rebind(held)
        for name in H3_BITS:
            module = roots[name][0]
            if getattr(module, "_cozy_sol_site", None) is not None:
                attention_sol.remove_site(module, name, "0")


def test_a_pin_overrides_the_level() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    ready = _ready("flash-attn3", "sdpa")
    scoped = attention.install(
        ready, _dit_roots("bfloat16"), "ref2va_dit=sdpa", device=_cuda(90), bits=H3_BITS
    )
    assert scoped.hosts["ref2va_dit"] == {"sdpa": 1}
    assert scoped.hosts["fl2va_dit"] == {"flash-attn3": 1}
    whole = attention.install(ready, _dit_roots("bfloat16"), "sdpa", device=_cuda(90), bits=8)
    assert whole.totals() == {"sdpa": 4}
