"""Routing and admission checks; no local GPU is required."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author import AttentionContext
from cozy_runtime.internal import (
    attention,
    attention_kitchen_sol,
    attention_producer,
    attention_sol,
)
from cozy_runtime.internal import attention_upstream as upstream
from cozy_runtime.internal.encoding import DeviceFacts

DENSE = "sageattention3-fp4-fixed-means"
HYBRID = "kitchen-sol-producer-fp4-fixed-means-shared-qkv"


@pytest.mark.parametrize("degree", [1, 2, 4, 7, 8])
def test_explicit_dense_selector_installs_on_real_diffusers_sites(degree: int) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    # Actual small projection modules with the real H3 head geometry. No CUDA launch:
    # Ready is the kernel-resolution result, whose numerical half is run5247 evidence.
    site = MiniMaxH3Attention(hidden_size=8, heads=56, dim_head=128).to(torch.bfloat16)
    roots = {"fl2va_dit": torch.nn.ModuleList([site])}
    candidate = attention.BY_NAME[DENSE]
    resolved = attention.Ready(candidate, attention._member(candidate))
    device = DeviceFacts("cuda", "RTX 5090", 120, "", "", index=0)
    selected = attention.install([resolved], roots, degree=degree, device=device)
    assert selected.hosts == {"fl2va_dit": {DENSE: 1}}
    assert attention.observed(roots).hosts == selected.hosts
    assert not isinstance(site.processor, attention_producer.Processor)


def test_fixed_recipe_stays_sm120_bf16_head128_only() -> None:
    for name in (DENSE, HYBRID):
        candidate = attention.BY_NAME[name]
        assert candidate not in attention.CANDIDATES
        for sm in (80, 89, 90, 100, 120, 121):
            facts = DeviceFacts("cuda", "fixture", sm, "580", "", index=0)
            assert bool(attention._device_admits(candidate, facts)) == (sm != 120)
        assert attention._admits(candidate, "bfloat16", 128)
        assert not attention._admits(candidate, "float16", 128)
        assert not attention._admits(candidate, "bfloat16", 64)


def test_fixed_recipe_supports_context_parallel_diffusers_registry() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    for degree in (1, 2, 4, 7, 8):
        attention.admit(HYBRID, degree)


def test_h3_prefers_the_qualified_hybrid_without_promoting_all_dense() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import (
        BLACKWELL_DIT_ATTENTION,
        DIT_ATTENTION,
        H3Model,
    )

    assert BLACKWELL_DIT_ATTENTION[0] == HYBRID and DENSE not in BLACKWELL_DIT_ATTENTION
    assert DIT_ATTENTION == ("sol-attn", "sageattention", "flash-attn3", "sdpa")
    context = AttentionContext(
        "fl2va_dit",
        "block",
        "cuda",
        "fixture",
        120,
        "bfloat16",
        128,
        1,
        (HYBRID, DENSE, "kitchen-sol-fp4-lowmem"),
        "sdpa",
        "",
    )
    model = object.__new__(H3Model)
    assert model.choose_attention(context) == BLACKWELL_DIT_ATTENTION


def test_hybrid_changes_only_dense_callback_and_keeps_producer_policy(monkeypatch: Any) -> None:
    seen = {}

    def boundary(*args: Any, **options: Any) -> str:
        seen.update(options)
        return "value"

    monkeypatch.setattr(attention_sol, "_attention", boundary)
    assert attention_producer.materialized_fp4_fixed_means(1, 2, 3) == "value"
    assert seen["dense"] is attention_kitchen_sol.dense_fp4_fixed_means
    assert seen["sparse"] is attention_producer._missing_processor
    assert HYBRID in attention_producer.REUSE and HYBRID in attention_producer.SHARED_QKV
    assert attention_producer.CHUNK == 4096
    assert attention_sol.dense_identity(HYBRID) == DENSE + "; sparse-prefix=kitchen-int8"


def test_dense_adapter_preserves_operands_scale_and_upstream_correction(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    q, k, v = (torch.ones(1, 3, 2, 128, dtype=torch.bfloat16) for _ in range(3))
    validation = []
    monkeypatch.setattr(upstream, "_cuda_inputs", lambda *a, **kw: validation.append(kw))
    calls = []

    def forward(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return q.clone()

    monkeypatch.setattr(upstream, "import_module", lambda _: SimpleNamespace(forward=forward))
    result = upstream.sage3_fp4_fixed_means(q, k, v, scale=128**-0.5)
    assert result.shape == q.shape
    assert calls == [((q, k, v), {})]  # omitted fixed_correction keeps the upstream matmul
    assert validation[0]["admitted"] == ((12, 0),)
    for options in ({"is_causal": True}, {"enable_gqa": True}, {"scale": 1.0}):
        with pytest.raises(ValueError):
            upstream.sage3_fp4_fixed_means(q, k, v, **options)


def test_source_does_not_modify_h3_schedule_or_default_table() -> None:
    from pathlib import Path

    model = Path(__file__).resolve().parents[1] / "src/cozy_runtime/models/minimax_h3/model.py"
    tree = ast.parse(model.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "sample_fl2va_turbo",
            "sample_ref2va_turbo",
        ):
            defaults = {
                arg.arg: ast.literal_eval(default)
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
                if default is not None and arg.arg == "sol_dense_steps"
            }
            assert defaults["sol_dense_steps"] == 4


def test_full_packed_virtual_offsets_are_widened_before_stride_products() -> None:
    from pathlib import Path

    import numpy as np

    from cozy_runtime.internal import attention_sage3_fixed as candidate

    tree = ast.parse(Path(candidate.__file__).read_text())
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for name in ("partial_mean", "center_and_correct"):
        assignments = {
            target.id: node.value
            for node in functions[name].body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for index in ("token", "head"):
            expression = ast.unparse(assignments[index])
            assert expression.endswith(".to(tl.int64)"), (name, index, expression)
    # Evaluate the actual source address expression with virtual full-sequence
    # indices; the test allocates 128 integer outputs, not a giant QKV buffer.
    center = functions["center_and_correct"]
    address = next(
        node.value
        for node in center.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "source_v"
    )
    code = compile(ast.Expression(address), "actual-center-address", "eval")
    count, heads, dim = 109225, 56, 128
    context = {
        "token": np.array([count - 1], dtype=np.int64),
        "head": np.array(heads - 1, dtype=np.int64),
        "channel": np.arange(dim, dtype=np.int64),
        "vt": 3 * heads * dim,
        "vh": dim,
    }
    relative = eval(code, {"__builtins__": {}}, context)
    absolute = relative + 2 * heads * dim  # V starts after packed Q and K.
    assert absolute.dtype == np.int64
    assert int(absolute.max()) == count * 3 * heads * dim - 1
    assert int(absolute.max()) > 2**31 - 1
    assert not np.array_equal(absolute.astype(np.int32).astype(np.int64), absolute)
