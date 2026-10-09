"""Request selection and model policy tests; CUDA arithmetic has an ordinary job gate."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from cozy_runtime.internal import attention, attention_precision, attention_producer, attention_sol
from cozy_runtime.internal.encoding import DeviceFacts


def test_matrix_controls_are_request_only_and_name_their_dense_arithmetic() -> None:
    for name, local in attention_precision.NAMES.items():
        candidate = attention.BY_NAME[name]
        assert candidate.local is local
        assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
        assert candidate.sms == frozenset({120})
        with pytest.raises(attention.AttentionRefusal):
            attention.pinned(name, DeviceFacts("cpu", "CPU", 0, "", "", 0))
    for name in ("sol-bf16-fp32", "sol-fp8-fp16-pure"):
        assert name in attention_sol.NAMES
    assert attention_sol.dense_identity("sol-bf16-fp32") == "bf16-fp32-dense"
    assert attention_sol.dense_identity("sol-fp8-fp16-pure") == "fp8-fp16-dense"
    assert {
        "kitchen-sol-producer-fast-shared-qkv",
        "kitchen-sol-producer-fp4-lowmem-shared-qkv",
    } <= attention_producer.SHARED_QKV


def test_dense_entrypoints_do_not_reach_sparse_policy_or_preprocessing() -> None:
    # Structural contract supplements the real GPU gate, which records actual CUDA kernels.
    for fn in (attention_precision.fp8_dense, attention_precision.bf16_dense):
        tree = ast.parse(inspect.getsource(fn))
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert not any(
            "sparse" in call or "prepare" in call or "attention_sol" in call for call in calls
        )
    assert "dense_forward_kernel" in inspect.getsource(attention_precision.fp8_dense)
    assert "SDPBackend.FLASH_ATTENTION" in inspect.getsource(attention_precision.bf16_dense)


def test_eight_step_policy_has_four_sparse_calls_and_dense_prefixes() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.author._attention_scope import AttentionLayout

    q = torch.zeros(1, 67, 2, 128, dtype=torch.bfloat16)
    calls = []

    def dense(*, query: Any, key: Any, value: Any, scale: Any) -> Any:
        calls.append(("dense", query.shape[1], key.shape[1]))
        return torch.ones_like(query)

    def sparse(query: Any, key: Any, value: Any, scale: Any, sink: Any) -> Any:
        calls.append(("sparse", query.shape[1], sink))
        return torch.full_like(query, 2)

    with attention_sol.observing() as counts:
        for step in range(8):
            result = attention_sol._execute(
                q,
                q,
                q,
                None,
                AttentionLayout(65, 7, step, dense_until_step=4),
                attention_sol.Site("dit", "blocks.2.attn"),
                sparse=sparse,
                dense=dense,
            )
            assert not torch.count_nonzero(result[:, 65:])
    assert counts == {"dense_step": 4, "sparse": 4, "dense_prefix": 4, "dense_path": 0}
    assert calls == [("dense", 65, 65)] * 4 + [("sparse", 65, 7), ("dense", 7, 65)] * 4


def test_dense_precision_refuses_cpu_without_silent_reference_fallback() -> None:
    torch = pytest.importorskip("torch")
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    for fn in (attention_precision.fp8_dense, attention_precision.bf16_dense):
        with pytest.raises(ValueError, match="SM120"):
            fn(q, q, q)
        with pytest.raises(ValueError, match="unmasked"):
            fn(q, q, q, is_causal=True)
