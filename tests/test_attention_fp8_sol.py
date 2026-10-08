"""CPU range/oracle contracts; actual CUDA output requires separate remote admission."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author._attention_scope import AttentionLayout
from cozy_runtime.internal import attention, attention_fp8_sol, attention_sol
from cozy_runtime.internal.encoding import DeviceFacts

torch = pytest.importorskip("torch")


def test_fp16_tile_ranges_are_finite_even_with_all_products_at_the_bound() -> None:
    maximum = torch.finfo(torch.float16).max
    assert 128 * 16**2 == 32768 < maximum
    assert 64 * 16**2 == 16384 < maximum
    # Emulate successive K32 partial accumulations at their positive/negative bound.
    for extent in (64, 128):
        for sign in (-1, 1):
            running = torch.tensor(0, dtype=torch.float16)
            for _ in range(extent // 32):
                running = (running.float() + sign * 32 * 16**2).half()
                assert torch.isfinite(running)
            assert float(running) == sign * extent * 16**2


@pytest.mark.parametrize("case", ["zeros", "extrema", "random"])
def test_cpu_oracle_all_exact_preserves_constant_values_and_bounds_error(case: str) -> None:
    from testdata.fp8_sol_cpu import quantize_cpu, reference

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(617)
    tensors = [torch.randn(1, 257, 2, 128, generator=generator).bfloat16() for _ in range(3)]
    if case in {"zeros", "extrema"}:
        for tensor in tensors:
            tensor.fill_(0 if case == "zeros" else 1)
    q, k, v = tensors
    quantized = [quantize_cpu(tensor) for tensor in tensors]
    codes, scales = zip(*quantized, strict=True)
    assert all(
        torch.isfinite(code.float()).all() and code.float().abs().max() <= 16 for code in codes
    )
    blocks = math.ceil(q.shape[1] / 64)
    kc = torch.stack([x.float().mean(1) for x in k.split(64, dim=1)], dim=1).bfloat16()
    vc = torch.stack([x.float().sum(1) for x in v.split(64, dim=1)], dim=1).bfloat16()
    threshold = torch.full((1, blocks, 2), -1e6)
    output = reference(q, codes[1], codes[2], codes[0], scales, kc, vc, threshold, 0)
    assert torch.isfinite(output).all()
    if case != "random":
        assert torch.equal(output, v)
    else:
        full = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
        ).transpose(1, 2)
        relative = (output.float() - full).norm() / full.norm()
        assert relative < 0.08, relative


def test_experimental_pins_are_explicit_head_local_and_share_sage_dense_reference() -> None:
    for name in ("sol-fp8-fp16", "kitchen-sol-sage"):
        candidate = attention.BY_NAME[name]
        assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
        assert candidate.local.head_local and name in attention_sol.NAMES
        assert attention_sol.dense_identity(name) == "sageattention"
        with pytest.raises(attention.AttentionRefusal):
            attention.pinned(name, DeviceFacts("cpu", "CPU", 0, "", "", 0))
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="SM120 experiment"):
        attention_fp8_sol.sparse(q, q, q, None, 0)


@pytest.mark.parametrize("dense_steps", [4, 10])
@pytest.mark.parametrize("prefix", [0, 65, 257])
def test_separate_prefix_callback_keeps_threshold_and_full_protection(
    dense_steps: int,
    prefix: int,
) -> None:
    """CPU policy proof, with distinct callbacks; no FP4/FP8 GPU arithmetic is simulated."""
    q = torch.zeros(1, 261, 2, 128, dtype=torch.bfloat16)
    calls = []

    def dense(*, query: Any, key: Any, **kwargs: Any) -> Any:
        calls.append(("dense", query.shape[1], key.shape[1]))
        return torch.ones_like(query)

    def protected(*, query: Any, key: Any, **kwargs: Any) -> Any:
        calls.append(("prefix", query.shape[1], key.shape[1]))
        return torch.full_like(query, 2)

    def sparse(query: Any, key: Any, value: Any, scale: Any, sink: int) -> Any:
        calls.append(("sparse", query.shape[1], sink))
        return torch.full_like(query, 3)

    site = attention_sol.Site("dit", "blocks.2.attn")
    for step in [dense_steps - 1, dense_steps]:
        calls.clear()
        with attention_sol.observing() as counts:
            result = attention_sol._execute(
                q,
                q,
                q,
                None,
                AttentionLayout(257, prefix, step, dense_steps),
                site,
                sparse=sparse,
                dense=dense,
                prefix_dense=protected,
            )
        assert not torch.count_nonzero(result[:, 257:])
        if step < dense_steps:
            assert calls == [("dense", 257, 257)]
            assert torch.equal(result[:, :257], torch.ones_like(q[:, :257]))
            assert counts == dict(sparse=0, dense_step=1, dense_path=0, dense_prefix=0)
        else:
            assert calls == [("sparse", 257, prefix)] + (
                [("prefix", prefix, 257)] if prefix else []
            )
            assert torch.all(result[:, :prefix] == 2)
            assert torch.all(result[:, prefix:257] == 3)
            assert counts == dict(
                sparse=1, dense_step=0, dense_path=0, dense_prefix=int(prefix > 0)
            )


@pytest.mark.parametrize("path,tokens", [("blocks.1.attn", 257), ("refiner.attn", 9)])
def test_protected_paths_and_refiners_use_dense_not_the_prefix_callback(
    path: str, tokens: int
) -> None:
    q = torch.zeros(1, tokens, 2, 128, dtype=torch.bfloat16)
    calls = []

    def dense(*, query: Any, **kwargs: Any) -> Any:
        calls.append(query.shape[1])
        return torch.ones_like(query)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dense model paths cannot use sparse/prefix callbacks")

    layout = AttentionLayout(257, 65, 12, 10, ("blocks.1", "refiner"))
    result = attention_sol._execute(
        q,
        q,
        q,
        None,
        layout,
        attention_sol.Site("dit", path),
        sparse=forbidden,
        dense=dense,
        prefix_dense=forbidden,
    )
    assert calls == [tokens] and torch.all(result == 1)


def test_fp4_hybrid_resolves_sage3_and_sage2_without_the_kitchen_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cozy_runtime.internal import attention_producer

    names = []

    def compiled(candidate: Any, *args: Any) -> dict[str, str]:
        names.append(candidate.name)
        return {"artifact": candidate.artifact}

    for book in ("_READY", "_BROKEN", "_STATES"):
        monkeypatch.setattr(attention, book, {})
    monkeypatch.setattr(attention, "_compiled", compiled)
    monkeypatch.setattr(attention, "_import", lambda _: SimpleNamespace())
    monkeypatch.setattr(attention, "_launch", lambda *args: None)
    name = "sol-fp8-fp16-fp4-dense"
    candidate = attention.BY_NAME[name]
    assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
    assert name not in attention_producer.NAMES
    assert candidate.local.head_local and name in attention_sol.NAMES
    assert attention._resolve(candidate, DeviceFacts("cuda", "unavailable", 120, "", "", 0))
    assert names == [name, "sageattention3-fp4-global", "sageattention"]
    assert attention_sol.dense_identity(name) == (
        "sageattention3-fp4-global; sparse-prefix=sageattention"
    )
