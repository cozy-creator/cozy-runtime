"""Explicit SM90 dense substitution; sparse producer and defaults remain identical."""

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope  # noqa: E402
from cozy_runtime.internal import attention, attention_sol  # noqa: E402
from cozy_runtime.internal import attention_producer as producer  # noqa: E402
from cozy_runtime.internal.encoding import DeviceFacts  # noqa: E402

NAME = "kitchen-sol-producer-sage2-shared-qkv"
BASE = "kitchen-sol-producer-fast-shared-qkv"


def test_new_recipe_is_explicit_sm90_only_and_preserves_existing_capabilities() -> None:
    candidate = attention.BY_NAME[NAME]
    assert candidate not in attention.CANDIDATES
    assert candidate.sms == frozenset({90})
    assert (candidate.min_sm, candidate.max_sm) == (90, 90)
    assert not candidate.local.head_local
    assert NAME in producer.REUSE & producer.SHARED_QKV & attention_sol.NAMES
    assert producer.capabilities(BASE) == producer.attention_upstream.KITCHEN_SMS
    assert producer.capabilities("kitchen-sol-producer-fp4-lowmem-shared-qkv") == ((12, 0), (12, 1))
    for sm in (80, 89, 120, 121):
        assert attention._device_admits(candidate, DeviceFacts("cuda", "test", sm, "", ""))
    assert not attention._device_admits(candidate, DeviceFacts("cuda", "H100", 90, "", ""))
    with pytest.raises(attention.AttentionRefusal, match="context parallelism"):
        attention.admit(NAME, 4)
    assert attention_sol.dense_identity(NAME) == "sageattention; sparse-prefix=kitchen-int8"


def test_dense_callback_uses_sage2_directly_without_sol_dense_fallback(monkeypatch: Any) -> None:
    captured = []
    monkeypatch.setattr(producer.attention_ulysses, "_call", lambda *args: captured.append(args))
    q, k, v = object(), object(), object()
    producer.dense_sage2(query=q, key=k, value=v, scale=128**-0.5)
    name, local, *tail = captured[0]
    assert name == "sageattention" and local is producer.attention_upstream.SAGE
    assert tail[:3] == [q, k, v]
    assert tail[3]["scale"] == 128**-0.5
    assert not tail[3]["is_causal"]


def test_materialized_policy_changes_only_dense_callback(monkeypatch: Any) -> None:
    captured = []
    monkeypatch.setattr(
        attention_sol, "_attention", lambda *args, **kw: captured.append((args, kw))
    )
    q, k, v = object(), object(), object()
    producer.materialized_kitchen(q, k, v, scale=0.125)
    producer.materialized_sage2(q, k, v, scale=0.125)
    left, right = captured
    assert left[0] == right[0]
    a, b = dict(left[1]), dict(right[1])
    assert a.pop("dense") is producer.attention_kitchen_sol.dense_kitchen
    assert b.pop("dense") is producer.dense_sage2
    assert a == b


def test_sparse_calls_share_exact_arguments_prefix_qkv_and_previous_statistics(
    monkeypatch: Any,
) -> None:
    calls = []
    prior_mean, prior_scale, next_mean, next_scale, rope = (object() for _ in range(5))
    hidden = SimpleNamespace(shape=(1, 129, 5376), device=SimpleNamespace(type="cuda"))
    norm = SimpleNamespace(eps=1e-6, weight=torch.ones(128))
    attn = SimpleNamespace(heads=56, norm_q=norm, norm_k=norm, to_out=[lambda x: x, lambda x: x])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(producer, "rope_presentation", lambda *args: rope)
    chunk_calls = []

    def chunks(*args: Any, **kwargs: Any) -> Any:
        chunk_calls.append((args, kwargs["shared"]))
        yield "identical chunk sentinel"

    def sparse(factory: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((list(factory()), args, kwargs))
        return torch.zeros(1, 129, 56, 128), next_mean, next_scale

    monkeypatch.setattr(producer, "qkv_chunks", chunks)
    monkeypatch.setattr(
        producer, "import_module", lambda _: SimpleNamespace(sol_attn_chunked=sparse)
    )
    key, geometry = ("dit", "transformer_blocks.2"), (129, 56, 17)
    token = attention_sol._SITE.set(attention_sol.Site(*key))
    observed_counts = []
    try:
        for name in (BASE, NAME):
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
                observed_counts.append(dict(counts))
    finally:
        attention_sol._SITE.reset(token)
    assert calls[0] == calls[1]
    assert chunk_calls == [((attn, hidden, 129), True)] * 2
    assert calls[0][2]["kmean"] is prior_mean and calls[0][2]["vscale"] is prior_scale
    assert calls[0][2]["sink_q"] == [0, 1]
    assert observed_counts == [dict(sparse=1, dense_step=0, dense_path=0, dense_prefix=1)] * 2


@pytest.mark.parametrize("dense_fails", [False, True])
def test_resolution_launches_sage2_companion_or_refuses(
    monkeypatch: Any, dense_fails: bool
) -> None:
    compiled, launched = [], []
    for table in ("_READY", "_BROKEN", "_STATES"):
        monkeypatch.setattr(attention, table, {})
    monkeypatch.setattr(attention, "_registry", lambda: None)
    monkeypatch.setattr(attention, "_member", lambda candidate: candidate.name)
    monkeypatch.setattr(
        attention,
        "_compiled",
        lambda c, d, wait: compiled.append((c.name, wait)) or {"key": c.name},
    )
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
        if dense_fails and ready.name == "sageattention":
            raise attention.AttentionRefusal("attention_probe_failed", "dense launch failed")

    monkeypatch.setattr(attention, "_launch", launch)
    device = DeviceFacts("cuda", "H100", 90, "", "")
    if dense_fails:
        with pytest.raises(attention.AttentionRefusal, match="dense launch failed"):
            attention._resolve(attention.BY_NAME[NAME], device)
        assert NAME not in launched
    else:
        ready = attention._resolve(attention.BY_NAME[NAME], device)
        assert ready.artifact["dense_key"] == "sageattention"
        assert launched == ["sageattention", NAME]
    assert compiled[:2] == [(NAME, False), ("sageattention", False)]
