"""CPU producer custody/policy contracts, not GPU performance or quality claims."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention  # noqa: E402

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope  # noqa: E402
from cozy_runtime.internal import attention, attention_sol  # noqa: E402
from cozy_runtime.internal import attention_producer as producer  # noqa: E402
from cozy_runtime.internal.encoding.leaves import PreQuantized  # noqa: E402
from cozy_runtime.models.minimax_h3.turbo import LoRAFactors, _LoRAHook  # noqa: E402


def module() -> Any:
    return MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128).to(dtype=torch.bfloat16)


def test_carrier_rows_are_views_of_original_payload_and_scales() -> None:
    payload = torch.arange(11 * 8).reshape(11, 8)
    scale = torch.arange(11).reshape(11, 1)
    source = PreQuantized(payload, scale, (1, 11, 8))
    part = producer.rows(source, 3, 8)
    assert part.shape == (1, 5, 8)
    assert part.payload.untyped_storage().data_ptr() == payload.untyped_storage().data_ptr()
    assert part.scale.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
    assert torch.equal(part.payload, payload[3:8]) and torch.equal(part.scale, scale[3:8])


@pytest.mark.parametrize("baked", [False, True])
def test_original_projection_modules_keep_fixed_chunk_turbo_hooks(baked: bool) -> None:
    count = producer.CHUNK + 65
    hidden = torch.ones(1, count, 8, dtype=torch.bfloat16)
    factors = LoRAFactors(8, 128, 2, 0.25)
    with torch.no_grad():
        factors.lora_down.fill_(0.125)
        factors.lora_up.fill_(0.125)
    owner = SimpleNamespace(
        _arming=SimpleNamespace(overlay=SimpleNamespace(get_submodule=lambda _: factors))
    )
    projections = [torch.nn.Linear(8, 128, bias=False, dtype=torch.bfloat16) for _ in range(3)]
    with torch.no_grad():
        for projection in projections:
            projection.weight.zero_()
            hook = _LoRAHook(owner, "projection")
            hook._cozy_baked = baked
            projection.register_forward_hook(hook)
        expected = torch.cat([projection(hidden) for projection in projections], dim=-1).squeeze(0)
        chunks = list(
            producer.qkv_chunks(
                SimpleNamespace(to_q=projections[0], to_k=projections[1], to_v=projections[2]),
                hidden,
                count,
            )
        )
    assert [x.shape[0] for x in chunks] == [producer.CHUNK, 65]
    assert torch.equal(torch.cat(chunks), expected)
    assert bool(torch.count_nonzero(expected)) is not baked


def test_rotary_conversion_preserves_split_half_signs_and_reuses_presentation() -> None:
    cos = torch.randn(1, 17, 1, 96)
    sin = torch.randn_like(cos)
    state = producer.RequestState()
    freqs = producer.rope_presentation((cos, sin), 17, state)
    assert freqs.shape == (17, 1, 48, 2, 2)
    assert producer.rope_presentation((cos, sin), 17, state) is freqs
    f = freqs[:, 0]
    x = torch.randn(17, 96)
    actual = torch.cat(
        (
            f[:, :, 0, 0] * x[:, :48] + f[:, :, 0, 1] * x[:, 48:],
            f[:, :, 1, 1] * x[:, 48:] + f[:, :, 1, 0] * x[:, :48],
        ),
        dim=-1,
    )
    expected = x * cos.reshape(17, 96) + torch.cat((-x[:, 48:], x[:, :48]), dim=-1) * sin.reshape(
        17, 96
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_request_scope_releases_all_statistics_after_success_or_failure() -> None:
    with producer.request_scope() as outer:
        outer.statistics[("dit", "block")] = (3, (1,), object(), object())
        with pytest.raises(RuntimeError), producer.request_scope() as inner:
            inner.statistics[("dit", "block")] = (7, (1,), object(), object())
            inner.rope = object(), object(), object()
            raise RuntimeError("failed request")
        assert producer._STATE.get() is outer
        assert not inner.statistics and inner.rope is None
    assert producer._STATE.get() is None and not outer.statistics


@pytest.mark.parametrize("dense_steps", [4, 10])
def test_step_and_protected_path_policy_is_unchanged(dense_steps: int) -> None:
    normal, protected = (
        attention_sol.Site("dit", "transformer_blocks.2.attn"),
        attention_sol.Site("dit", "transformer_blocks.1.attn"),
    )
    for step in range(dense_steps + 2):
        layout = AttentionLayout(
            100, 17, step, dense_steps, ("transformer_blocks.1", "token_refiner")
        )
        assert producer.sparse_step(layout, normal) == (step >= dense_steps)
        assert not producer.sparse_step(layout, protected)
        assert not producer.sparse_step(layout, attention_sol.Site("dit", "token_refiner.0"))


def test_real_h3_processor_can_restore_its_original_backend_without_reconstruction() -> None:
    attn = module()
    attn.processor._attention_backend = attention._member(attention.BY_NAME["sdpa"])
    hidden = torch.randn(1, 5, 256, dtype=torch.bfloat16)
    original = attn.processor
    phase = torch.randn(5, 48)
    rotary = tuple(torch.cat((fn(phase), fn(phase)), dim=-1) for fn in (torch.cos, torch.sin))
    expected = original(attn, hidden, rotary)
    producer.install(attn)
    assert isinstance(attn.processor, producer.Processor)
    actual = attn(hidden, rotary)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for backend, name in producer.BACKENDS.items():
        attn.processor._attention_backend = backend
        with pytest.raises(ValueError, match="invocation-scoped"):
            attn(hidden)
        assert attention.BY_NAME[name] not in attention.CANDIDATES
        with pytest.raises(attention.AttentionRefusal, match="context parallelism"):
            attention.admit(name, 4)
    attn.processor._attention_backend = attention._member(attention.BY_NAME["sdpa"])
    torch.testing.assert_close(attn(hidden, rotary), expected, rtol=0, atol=0)


def test_unknown_projection_hooks_and_sparse_cpu_execution_refuse() -> None:
    attn = module()
    handle = attn.to_q.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="unknown projection hook"):
        producer.install(attn)
    handle.remove()
    producer.install(attn)
    attn.processor._attention_backend = next(iter(producer.BACKENDS))
    attention_sol.install_site(attn, "dit", "transformer_blocks.2.attn")
    with (
        attention_sol.observing(),
        attention_scope(AttentionLayout(5, 1, 10, 10)),
        pytest.raises(ValueError, match="device capability"),
    ):
        attn(
            torch.ones(1, 5, 256, dtype=torch.bfloat16),
            (torch.ones(1, 5, 1, 96), torch.zeros(1, 5, 1, 96)),
        )


def test_later_site_refusal_restores_earlier_processor_and_metadata() -> None:
    first, second = module(), module()
    original = first.processor
    attention_sol.install_site(second, "other", "0")
    existing = second._cozy_sol_site
    candidate = attention.BY_NAME["kitchen-sol-producer-fp4-lowmem-shared-qkv"]
    entry = attention.Ready(candidate, attention._member(candidate))
    with pytest.raises(ValueError, match="assigned another Sol site"):
        attention._install_sites([(entry, "dit", "0", first), (entry, "dit", "1", second)])
    assert first.processor is original and not hasattr(first, "_cozy_sol_site")
    assert second._cozy_sol_site is existing


def test_statistics_reuse_requires_the_same_site_geometry_and_adjacent_step() -> None:
    state = producer.RequestState()
    key, geometry = ("dit", "block.2"), (129, 56, 17)
    mean, scale = object(), object()
    for step, shape, enabled, expected in [
        (5, geometry, True, (mean, scale)),
        (5, geometry, False, (None, None)),
        (4, geometry, True, (None, None)),
        (6, geometry, True, (None, None)),
        (5, (129, 56, 18), True, (None, None)),
    ]:
        state.statistics[key] = (4, geometry, mean, scale, None, None)
        assert producer.previous_statistics(state, key, step, shape, enabled) == expected
        assert (key in state.statistics) is enabled
    state.statistics[key] = (4, geometry, mean, scale, None, None)
    assert producer.previous_statistics(state, ("dit", "block.3"), 5, geometry, True) == (
        None,
        None,
    )
    assert key in state.statistics


def test_cleanup_drops_only_the_owned_upstream_rotary_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import weakref

    owned, foreign = torch.empty(2), torch.empty(2)
    packed = torch.empty(2)
    cache = {("ours",): (weakref.ref(owned), packed), ("theirs",): (weakref.ref(foreign), packed)}
    monkeypatch.setitem(
        sys.modules, "comfy_kitchen.backends.cuda", SimpleNamespace(_ROPE_FAB_CACHE=cache)
    )
    with producer.request_scope() as state:
        state.rope = owned, owned, owned
    assert ("ours",) not in cache and ("theirs",) in cache


@pytest.mark.parametrize("failure", ["kernel", "projection", "later_block"])
def test_same_step_replay_preserves_original_statistics_inputs(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A block's tensor/RNG rewind does not rewind the producer's Python dictionary."""
    key, geometry = ("dit", "transformer_blocks.2"), (129, 56, 17)
    previous_mean, previous_scale = object(), object()
    current_mean, current_scale = object(), object()
    observed: list[tuple[Any, Any]] = []
    projections = 0

    def kitchen_call(*args: Any, **kwargs: Any) -> Any:
        observed.append((kwargs["kmean"], kwargs["vscale"]))
        if failure == "kernel" and len(observed) == 1:
            raise RuntimeError("simulated allocation failure")
        return torch.zeros(1, 129, 56, 128), current_mean, current_scale

    def project(value: Any) -> Any:
        nonlocal projections
        projections += 1
        if failure == "projection" and projections == 1:
            raise RuntimeError("simulated allocation failure")
        return value

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(
        producer, "import_module", lambda _: SimpleNamespace(sol_attn_chunked=kitchen_call)
    )
    monkeypatch.setattr(producer, "rope_presentation", lambda *args: object())
    original = SimpleNamespace(
        _attention_backend="_cozy_kitchen_sol_producer_fast_shared_qkv", _parallel_config=None
    )
    processor = producer.Processor(original)
    norm = SimpleNamespace(eps=1e-6, weight=torch.ones(128))
    attn = SimpleNamespace(heads=56, norm_q=norm, norm_k=norm, to_out=[project, lambda x: x])
    hidden = SimpleNamespace(shape=(1, 129, 5376), device=SimpleNamespace(type="cuda"))
    token = producer.attention_sol._SITE.set(producer.attention_sol.Site(*key))
    try:
        with producer.request_scope() as state:
            state.statistics[key] = (4, geometry, previous_mean, previous_scale, None, None)
            layout = AttentionLayout(
                live_tokens=129, protected_prefix=17, step=5, dense_until_step=4
            )
            with attention_scope(layout):
                with pytest.raises(RuntimeError, match="simulated allocation failure"):
                    processor(attn, hidden, rotary_emb=object())
                    if failure == "later_block":
                        raise RuntimeError("simulated allocation failure")
                processor(attn, hidden, rotary_emb=object())
            assert observed == [(previous_mean, previous_scale)] * 2
            next_layout = AttentionLayout(
                live_tokens=129, protected_prefix=17, step=6, dense_until_step=4
            )
            with attention_scope(next_layout):
                processor(attn, hidden, rotary_emb=object())
            assert observed[-1] == (current_mean, current_scale)
    finally:
        producer.attention_sol._SITE.reset(token)


@pytest.mark.parametrize("raises", [False, True])
def test_request_cleanup_releases_current_and_replay_statistics(raises: bool) -> None:
    import weakref

    references = []
    try:
        with producer.request_scope() as state:
            tensors = [torch.zeros(1) for _ in range(4)]
            references = [weakref.ref(tensor) for tensor in tensors]
            state.statistics[("dit", "block.2")] = (5, (129, 56, 17), *tensors)
            del tensors
            if raises:
                raise RuntimeError("request failed")
    except RuntimeError:
        assert raises
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("steps", [8, 30])
def test_h3_loop_runs_one_model_forward_per_evaluation(reference: bool, steps: int) -> None:
    """The supported H3 loop has no separate conditional/unconditional CFG calls."""
    from diffusers.modular_pipelines.minimax_h3.denoise import (
        MiniMaxH3LoopDenoiser,
        MiniMaxH3Ref2VALoopDenoiser,
    )

    seen = []

    class Counter(torch.nn.Module):
        def forward(
            self,
            hidden_states: Any,
            audio_hidden_states: Any,
            encoder_hidden_states: Any,
            timestep: Any,
            timestep_indices: Any,
            attention_kwargs: Any,
            return_dict: bool,
            token_tags: Any,
        ) -> Any:
            seen.append(int(timestep.item()))
            assert hidden_states.shape[0] == audio_hidden_states.shape[0] == 1
            return hidden_states, audio_hidden_states

    model = Counter()
    components = SimpleNamespace(transformer=model, transformer_ref=model)
    state = SimpleNamespace(
        latents=torch.zeros(3, 2),
        audio_latents=torch.zeros(4, 2),
        prompt_embeds=torch.zeros(1, 5, 2),
        row_timestep_plan=[(torch.tensor(i), torch.zeros(12)) for i in range(steps)],
        denoiser_input_fields={"token_tags": torch.zeros(12)},
        attention_kwargs=None,
    )
    block = MiniMaxH3Ref2VALoopDenoiser() if reference else MiniMaxH3LoopDenoiser()
    for index in range(steps):
        block(components, state, i=index, t=torch.tensor(index))
    assert seen == list(range(steps))
