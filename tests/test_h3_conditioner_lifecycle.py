"""Actual Qwen forwards preserve H3 conditioning and release unused generation state."""

from contextlib import nullcontext
from typing import Any

import pytest


def _conditioner(torch: Any, device: str) -> Any:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "vocab_size": 128,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "out_hidden_size": 32,
            "num_heads": 4,
            "depth": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "num_position_embeddings": 256,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=3,
        video_token_id=6,
        vision_start_token_id=2,
        vision_end_token_id=4,
    )
    return Qwen3VLForConditionalGeneration(config).eval().to(device)


def _condition(torch: Any, model: Any, *, image: bool, release: bool) -> Any:
    from cozy_runtime.models.minimax_h3.conditioner import conditioning_scope

    device = model.device
    tokens = [1, 2, 3, 3, 3, 3, 4, 5] if image else [1, 5]
    inputs = {
        "input_ids": torch.tensor([tokens], device=device),
        "attention_mask": torch.ones((1, len(tokens)), device=device, dtype=torch.long),
        "mm_token_type_ids": torch.tensor(
            [[0, 0, 1, 1, 1, 1, 0, 0]] if image else [[0, 0]], device=device
        ),
        "use_cache": False,
        "output_hidden_states": True,
    }
    if image:
        inputs.update(
            pixel_values=torch.zeros((16, 12), device=device),
            image_grid_thw=torch.tensor([[1, 4, 4]], device=device),
        )
    with torch.inference_mode(), conditioning_scope(model) if release else nullcontext():
        result = model.model(**inputs)
        return tuple(state.cpu().clone() for state in result.hidden_states)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_qwen_image_conditioning_releases_position_cache_without_changing_hidden_states(
    device: str,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA allocation proof requires a local accelerator")
    from cozy_runtime.internal.worker.ledger import Ledger
    from cozy_runtime.models.minimax_h3.conditioner import conditioning_scope

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = _conditioner(torch, device)
        ledger = Ledger(worker_pid=0)

        def observe() -> dict[str, int]:
            if device == "cuda":
                torch.cuda.synchronize()
                ledger.observe_probe({"allocator_bytes": torch.cuda.memory_allocated()})
            return ledger.snapshot()

        # The actual first text-only forward pays cuBLAS's first-use workspace.
        opened = observe()
        _condition(torch, model, image=False, release=False)
        observe()
        assert ledger.close_attempt("text", opened)["closed"]
        assert model.model.rope_deltas is None

        # Unchanged upstream image conditioning retains its one int64 cache even
        # with use_cache=False; the normal second-attempt ledger detects it.
        opened = observe()
        expected = _condition(torch, model, image=True, release=False)
        assert type(model.model).__name__ == "Qwen3VLModel"
        deltas: Any = model.model.rope_deltas
        assert deltas.numel() == 1
        observe()
        if device == "cuda":
            assert ledger.device_allocated - opened["device_allocated"] == 512
            assert not ledger.close_attempt("unscoped-image", opened)["closed"]
        model.model.rope_deltas = None
        ledger.unreconciled = ""
        observe()

        for index in range(4):
            opened = observe()
            actual = _condition(torch, model, image=True, release=True)
            assert model.model.rope_deltas is None
            assert all(
                torch.equal(left, right) for left, right in zip(expected, actual, strict=True)
            )
            observe()
            assert ledger.close_attempt(f"scoped-image-{index}", opened)["closed"]

        with (
            pytest.raises(RuntimeError, match="conditioner failure"),
            conditioning_scope(model),
        ):
            _condition(torch, model, image=True, release=False)
            assert model.model.rope_deltas is not None
            raise RuntimeError("conditioner failure")
        assert model.model.rope_deltas is None
    finally:
        torch.set_num_threads(previous_threads)
