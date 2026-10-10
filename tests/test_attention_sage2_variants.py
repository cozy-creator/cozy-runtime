"""Explicit H100 precision recipes; CPU routing proof, not GPU quality/speed evidence."""

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope  # noqa: E402
from cozy_runtime.internal import (  # noqa: E402
    attention,
    attention_sol,
    attention_kitchen_sol,
    attention_ulysses,
    attention_upstream as upstream,
)
from cozy_runtime.internal import attention_producer as producer  # noqa: E402
from cozy_runtime.internal.encoding import DeviceFacts  # noqa: E402

BASE = "kitchen-sol-producer-fast-shared-qkv"
ARMS = (
    (
        "kitchen-sol-producer-sage2-shared-qkv",
        "sageattention-hopper-8bit-accurate",
        "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "per_thread",
        "fp32+fp32",
    ),
    (
        "kitchen-sol-producer-sage2-per-warp-shared-qkv",
        "sageattention-hopper-8bit-per-warp",
        "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "per_warp",
        "fp32+fp32",
    ),
    (
        "kitchen-sol-producer-sage2-fp16pv-shared-qkv",
        "sageattention-hopper-mixed8-16",
        "sageattn_qk_int8_pv_fp16_cuda",
        "per_thread",
        "fp32",
    ),
)


@pytest.mark.parametrize("name,dense,entry,granularity,accumulation", ARMS)
def test_explicit_hopper_recipe_has_shared_module_build_and_exact_precision(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    dense: str,
    entry: str,
    granularity: str,
    accumulation: str,
) -> None:
    recipe, selected = attention.BY_NAME[name], attention.BY_NAME[dense]
    assert recipe not in attention.CANDIDATES and selected not in attention.CANDIDATES
    assert recipe.sms == selected.sms == frozenset({90})
    assert selected.artifact == "sageattention-full" and selected.import_name == "sageattention"
    assert recipe.artifact == "comfy-kitchen" and not recipe.local.head_local
    assert name in producer.REUSE & producer.SHARED_QKV & attention_sol.NAMES
    assert attention_sol.dense_identity(name) == dense + "; sparse-prefix=kitchen-int8"
    for sm in (80, 89, 120, 121):
        for candidate in (recipe, selected):
            assert attention._device_admits(candidate, DeviceFacts("cuda", "test", sm, "", ""))
    with pytest.raises(attention.AttentionRefusal, match="context parallelism"):
        attention.admit(name, 4)

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    sentinel = object()

    def kernel(*args: Any, **kwargs: Any) -> Any:
        calls.append((entry, args, kwargs))
        return sentinel

    monkeypatch.setattr(upstream, "import_module", lambda _: SimpleNamespace(**{entry: kernel}))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    q = SimpleNamespace(device=torch.device("cuda", 0), shape=(1, 129, 56, 128))
    assert selected.local.forward(q, q, q, scale=0.125) is sentinel
    assert calls == [
        (
            entry,
            (q, q, q),
            {
                "tensor_layout": "NHD",
                "is_causal": False,
                "qk_quant_gran": granularity,
                "sm_scale": 0.125,
                "pv_accum_dtype": accumulation,
                "smooth_k": True,
            },
        )
    ]
    assert selected.local.impl(q) == f"{entry}({granularity}, {accumulation}, smooth_k)"


def test_all_variants_change_only_the_materialized_dense_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(attention_sol, "_attention", lambda *args, **kw: calls.append((args, kw)))
    q, k, v = object(), object(), object()
    producer.materialized_kitchen(q, k, v, scale=0.125)
    for name, *_ in ARMS:
        producer.LOCALS[name].forward(q, k, v, scale=0.125)
    original_args, original_kw = calls[0]
    common = dict(original_kw)
    assert common.pop("dense") is attention_kitchen_sol.dense_kitchen
    for arm, (args, options) in zip(ARMS, calls[1:], strict=True):
        assert args == original_args
        chosen = dict(options)
        callback = chosen.pop("dense")
        assert chosen == common
        dispatched: list[Any] = []
        monkeypatch.setattr(
            attention_ulysses, "_call", lambda *args, rows=dispatched: rows.append(args)
        )
        callback(query=q, key=k, value=v, scale=0.125)
        name, local, *tail = dispatched[0]
        assert name == arm[1] and local is attention.BY_NAME[name].local
        assert tail[:3] == [q, k, v]
        assert tail[3] == dict(
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=0.125, enable_gqa=False
        )


def test_sparse_prefix_chunks_and_previous_statistics_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    chunks_seen: list[Any] = []
    prior_mean, prior_scale, next_mean, next_scale, rope = (object() for _ in range(5))
    hidden = SimpleNamespace(shape=(1, 129, 5376), device=SimpleNamespace(type="cuda"))
    norm = SimpleNamespace(eps=1e-6, weight=torch.ones(128))
    attn = SimpleNamespace(heads=56, norm_q=norm, norm_k=norm, to_out=[lambda x: x, lambda x: x])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(producer, "rope_presentation", lambda *args: rope)

    def chunks(*args: Any, **kwargs: Any) -> Any:
        chunks_seen.append((args, kwargs["shared"]))
        yield "same chunk"

    def sparse(factory: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((list(factory()), args, kwargs))
        return torch.zeros(1, 129, 56, 128), next_mean, next_scale

    monkeypatch.setattr(producer, "qkv_chunks", chunks)
    monkeypatch.setattr(
        producer, "import_module", lambda _: SimpleNamespace(sol_attn_chunked=sparse)
    )
    key, geometry = ("dit", "transformer_blocks.2"), (129, 56, 17)
    token = attention_sol._SITE.set(attention_sol.Site(*key))
    observed = []
    try:
        for name in (BASE, *(arm[0] for arm in ARMS)):
            original = SimpleNamespace(
                _attention_backend="_cozy_" + name.replace("-", "_"), _parallel_config=None
            )
            with attention_sol.observing() as counts:
                state = producer._STATE.get()
                assert state is not None
                state.statistics[key] = (4, geometry, prior_mean, prior_scale)
                with attention_scope(AttentionLayout(129, 17, 5, 4)):
                    producer.Processor(original)(attn, hidden, rotary_emb=object())
                assert state.statistics[key] == (5, geometry, next_mean, next_scale)
                observed.append(dict(counts))
    finally:
        attention_sol._SITE.reset(token)
    assert calls == [calls[0]] * 4 and chunks_seen == [((attn, hidden, 129), True)] * 4
    assert calls[0][2]["kmean"] is prior_mean and calls[0][2]["vscale"] is prior_scale
    assert calls[0][2]["sink_q"] == [0, 1]
    assert observed == [dict(sparse=1, dense_step=0, dense_path=0, dense_prefix=1)] * 4
    assert producer.capabilities(BASE) == upstream.KITCHEN_SMS
    assert producer.capabilities("kitchen-sol-producer-fp4-lowmem-shared-qkv") == ((12, 0), (12, 1))


@pytest.mark.parametrize("name,dense,entry,granularity,accumulation", ARMS)
@pytest.mark.parametrize("dense_fails", [False, True])
def test_resolution_requires_the_exact_dense_companion_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    dense: str,
    entry: str,
    granularity: str,
    accumulation: str,
    dense_fails: bool,
) -> None:
    compiled, launched = [], []
    for table in ("_READY", "_BROKEN", "_STATES"):
        monkeypatch.setattr(attention, table, {})
    monkeypatch.setattr(attention, "_registry", lambda: None)
    monkeypatch.setattr(attention, "_member", lambda candidate: candidate.name)

    def compile_candidate(candidate: Any, device: Any, wait: bool) -> dict[str, str]:
        compiled.append((candidate.name, wait))
        return {"key": candidate.name}

    monkeypatch.setattr(attention, "_compiled", compile_candidate)
    monkeypatch.setattr(
        attention,
        "_import",
        lambda c: SimpleNamespace(
            sol_attn_is_available=lambda d: True,
            int8_attention_is_available=lambda d: True,
            sol_attn_chunked=lambda: None,
        ),
    )

    def launch(ready: Any, device: Any) -> None:
        launched.append(ready.name)
        if dense_fails and ready.name == dense:
            raise attention.AttentionRefusal("attention_probe_failed", "dense launch failed")

    monkeypatch.setattr(attention, "_launch", launch)
    device = DeviceFacts("cuda", "H100", 90, "", "")
    if dense_fails:
        with pytest.raises(attention.AttentionRefusal, match="dense launch failed"):
            attention._resolve(attention.BY_NAME[name], device)
        assert name not in launched
    else:
        ready = attention._resolve(attention.BY_NAME[name], device)
        assert ready.artifact and ready.artifact["dense_key"] == dense
        assert launched == [dense, name]
    assert compiled[:2] == [(name, False), (dense, False)]
