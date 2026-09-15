"""Real Diffusers processor selection; CPU execution where attention arithmetic is used."""

from __future__ import annotations

import os
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
        attention.Ready(attention.BY_NAME[name], attention._member(attention.BY_NAME[name].backend))
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
    assert _choices(roots)[0] == "native" and _choices(roots)[1] == before[1]
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
    assert len(parent._attention_plan("model0/dit=sdpa").sites) == 1
    assert len(parent._attention_plan("dit=sdpa").sites) == model_count
    assert len(parent._attention_plan("sdpa").sites) == 2 * model_count
    if aliases:
        assert len(parent._attention_plan("alternate/dit=sdpa").sites) == 1
    with pytest.raises(attention.AttentionRefusal, match="matches no"):
        parent._attention_check("absent/dit=sdpa")
    parent._attention_check("model0/dit=sdpa")
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
    assert _choices(all_roots) == ("native",) * (2 * model_count)


def test_scoped_pin_preserves_compiled_and_parallel_fences() -> None:
    roots = _roots()
    roots["dit"][0]._compiled_call_impl = lambda: None
    with pytest.raises(attention.AttentionRefusal) as exc:
        attention.plan(_device(), roots, "dit=sdpa")
    assert exc.value.code == "attention_kernel_frozen"
    attention.plan(_device(), roots, "vae=sdpa")
    with pytest.raises(attention.AttentionRefusal, match="context-parallel"):
        attention.plan(_device(), roots, "vae=sdpa", "a context-parallel group is installed")


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
    for name, sms in [("kitchen-int8", {90, 100, 103}), ("flashinfer-bf16-fp8", {100, 103})]:
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
    assert _choices(roots) == ("native", "native")
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


def test_probe_refuses_wrong_output_dtype_even_with_perfect_values() -> None:
    torch = pytest.importorskip("torch")
    query = torch.ones(1, 2, 1, 8, dtype=torch.bfloat16)
    exact = query.float()
    entry = _ready("sdpa")[0]
    attention._validate_probe(entry, _device(), query, query, exact)
    with pytest.raises(attention.AttentionRefusal, match="must return"):
        attention._validate_probe(entry, _device(), exact, query, exact)


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


@pytest.mark.skipif(
    os.environ.get("COZY_TEST_ATTENTION_CUDA") != "1",
    reason="requires two explicitly owned CUDA devices",
)
def test_numerical_probe_runs_on_the_selected_nondefault_device() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two explicitly owned CUDA devices")
    indices: list[int] = []

    def reference(query: Any, key: Any, value: Any) -> Any:
        indices.append(query.device.index)
        assert query.device == key.device == value.device == torch.device("cuda", 1)
        assert torch.cuda.current_device() == 1
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

    member = register("_cozy_probe_device_test", reference)
    entry = attention.Ready(
        attention.Candidate(
            "index-probe", member.value, 0, frozenset({"bfloat16"}), 128, rel_l2_max=2e-2
        ),
        member,
    )
    major, minor = torch.cuda.get_device_capability(1)
    selected = DeviceFacts(
        kind="cuda", name="owned GPU 1", sm=major * 10 + minor, driver="", configuration="", index=1
    )
    with torch.cuda.device(0):
        attention._computes(entry, selected)
        assert torch.cuda.current_device() == 0
    assert indices == [1]
