"""Real encoded CPU modules and model LoRA hooks; GPU speed is a separate qualification."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from cozy_runtime.internal import attention_producer as producer  # noqa: E402
from cozy_runtime.internal.encoding.leaves import (  # noqa: E402
    PreQuantized,
    quantize_activation_rowwise,
)
from test_shared_qkv_activation import projections  # noqa: E402


def chunk_reference(attn: Any, hidden: Any) -> Any:
    return torch.cat(list(producer.qkv_chunks(attn, hidden, hidden.shape[1], shared=True)), 0)


def bits_equal(a: Any, b: Any) -> bool:
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


@pytest.mark.parametrize("state", ["inactive", "baked", "unbaked"])
def test_full_shared_projection_keeps_lora_and_ragged_partition_bits(state: str) -> None:
    attn, hooks = projections(state)
    attn.heads = 1
    hidden = torch.randn(1, producer.CHUNK + 65, 64, dtype=torch.bfloat16)
    before = hidden.clone()
    weights = [
        (site.data.clone(), site.scale.clone()) for site in (attn.to_q, attn.to_k, attn.to_v)
    ]
    with torch.inference_mode():
        full = chunk_reference(attn, hidden)
        for start, stop in ((0, 4161), (0, 2081), (2081, 4161)):
            shard = hidden[:, start:stop]
            counts: dict[str, int] = {}
            actual = producer._project_qkv(attn, shard, stop - start, counts)
            repeated = producer._project_qkv(attn, shard, stop - start)
            reference = chunk_reference(attn, shard)
            packed = torch.cat([part.flatten(2, 3) for part in actual], -1).squeeze(0)
            assert bits_equal(packed, reference) and bits_equal(packed, full[start:stop])
            assert all(bits_equal(a, b) for a, b in zip(actual, repeated, strict=True))
            assert counts == {"shared": 1}
            # Separate original module outputs, not views into an extra packed copy.
            assert len({part.untyped_storage().data_ptr() for part in actual}) == 3
    assert bits_equal(hidden, before)
    assert all(
        bits_equal(site.data, data) and bits_equal(site.scale, scale)
        for site, (data, scale) in zip((attn.to_q, attn.to_k, attn.to_v), weights, strict=True)
    )
    assert all(hook._cozy_baked == (state == "baked") for hook in hooks)


def test_shared_projection_quantizes_once_without_concatenating_outputs() -> None:
    attn, _ = projections("unbaked")
    attn.heads = 1
    hidden = torch.randn(1, producer.CHUNK + 65, 64, dtype=torch.bfloat16)
    with (
        torch.inference_mode(),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
    ):
        outputs = producer._project_qkv(attn, hidden, hidden.shape[1])
    assert all(part.shape == (1, hidden.shape[1], 1, 128) for part in outputs)
    operations = {event.key: event.count for event in profile.key_averages()}
    assert operations["aten::amax"] == 1
    assert operations.get("aten::cat", 0) == 0


@pytest.mark.parametrize("kind", ["pre", "post", "float", "prequantized"])
def test_unshared_semantics_keep_the_original_chunk_path(kind: str) -> None:
    attn, _ = projections("inactive")
    attn.heads = 1
    hidden = torch.randn(1, producer.CHUNK + 65, 64, dtype=torch.bfloat16)
    if kind == "pre":
        attn.to_q.register_forward_pre_hook(lambda module, args: (args[0] * 0.5,))
    elif kind == "post":
        attn.to_q.register_forward_hook(lambda module, args, output: output + 1)
    elif kind == "float":
        attn.to_q = torch.nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    else:
        payload, scales = quantize_activation_rowwise(torch, hidden)
        hidden = PreQuantized(payload, scales, tuple(hidden.shape))
    with torch.inference_mode():
        reference = chunk_reference(attn, hidden)
        counts: dict[str, int] = {}
        actual = producer._project_qkv(attn, hidden, hidden.shape[1], counts)
        packed = torch.cat([part.flatten(2, 3) for part in actual], -1).squeeze(0)
    assert bits_equal(packed, reference)
    reason = {
        "pre": "projection-pre-hook",
        "post": "unknown-projection-hook",
        "float": "non-rowwise-leaf",
        "prequantized": "already-quantized",
    }[kind]
    assert counts == {reason: 2}


def test_projection_keeps_original_forward_custody_and_refuses_partial_shards() -> None:
    attn, _ = projections("unbaked")
    attn.heads = 1
    hidden = torch.randn(1, 129, 64, dtype=torch.bfloat16)
    visits = []
    for name in ("to_q", "to_k", "to_v"):
        site = getattr(attn, name)
        forward = site.forward

        def wrapped(*args: Any, saved: Any = forward, name: str = name, **kwargs: Any) -> Any:
            visits.append(name)
            return saved(*args, **kwargs)

        site.forward = wrapped
    with torch.inference_mode():
        producer._project_qkv(attn, hidden, 129)
    assert visits == ["to_q", "to_k", "to_v"]
    with pytest.raises(ValueError, match="complete nonempty local token shard"):
        producer._project_qkv(attn, hidden, 128)
