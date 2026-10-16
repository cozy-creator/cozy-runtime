"""`lm.generate` against Transformers' own `generate`: the same tokens, on Qwen3.5's hybrid
linear/full attention (with an image, mrope positions) and on a plain full-attention LM,
batched with left padding, with and without a presence penalty, across a window boundary. On
CUDA the decode runs as replayed graphs."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cozy_runtime import lm  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HYBRID = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def _qwen3_5() -> Any:
    config = transformers.AutoConfig.for_model(
        "qwen3_5",
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "vocab_size": 512,
            "layer_types": HYBRID,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
        },
        image_token_id=300,
        vision_start_token_id=301,
        vision_end_token_id=302,
    )
    model = transformers.AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    return model.to(DEVICE).eval()


def _qwen3() -> Any:
    config = transformers.AutoConfig.for_model(
        "qwen3",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    return model.to(DEVICE).eval()


def _reference(
    model: Any, inputs: dict[str, Any], new: int, stop: list[int], penalty: float
) -> Any:
    prompt = int(inputs["input_ids"].shape[1])

    def presence(input_ids: Any, scores: Any) -> Any:
        generated = input_ids[:, prompt:]
        if not generated.shape[1] or not penalty:
            return scores
        return scores - penalty * torch.zeros_like(scores).scatter_(1, generated, 1.0)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=new,
            do_sample=False,
            eos_token_id=stop,
            pad_token_id=0,
            logits_processor=transformers.LogitsProcessorList([presence]),
        )
    return out[:, prompt:]


def _same(model: Any, inputs: dict[str, Any], new: int, penalty: float = 0.0) -> None:
    stop = [7, 9]
    want = _reference(model, inputs, new, stop, penalty)
    got = lm.generate(model, inputs, max_new_tokens=new, stop=stop, pad=0, presence_penalty=penalty)
    assert got.shape == want.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize("make", [_qwen3_5, _qwen3], ids=["qwen3_5", "qwen3"])
@pytest.mark.parametrize("penalty", [0.0, 1.5])
def test_greedy_tokens_match_transformers(make: Any, penalty: float) -> None:
    torch.manual_seed(0)
    model = make()
    ids = torch.randint(10, 290, (2, 37), device=DEVICE)
    mask = torch.ones_like(ids)
    ids[1, :5], mask[1, :5] = 0, 0  # the second row is left-padded
    _same(model, {"input_ids": ids[:1], "attention_mask": mask[:1]}, 40, penalty)
    _same(model, {"input_ids": ids, "attention_mask": mask}, 40, penalty)


def test_image_prompt_across_a_window_boundary() -> None:
    torch.manual_seed(1)
    model = _qwen3_5()
    vision = model.config.vision_config
    merged = 16 // vision.spatial_merge_size**2
    image = [301, *[300] * merged, 302]
    text = torch.randint(10, 290, (1, lm.MIN_WINDOW - len(image) - 4), device=DEVICE)
    ids = torch.cat([text[:, :4], torch.tensor([image], device=DEVICE), text[:, 4:]], 1)
    pixels = torch.randn(16, vision.in_channels * vision.temporal_patch_size * vision.patch_size**2)
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": pixels.to(DEVICE),
        "image_grid_thw": torch.tensor([[1, 4, 4]], device=DEVICE),
        "mm_token_type_ids": (ids == 300).int(),
    }
    assert lm.window(ids.shape[1], ids.shape[1] + 30) < ids.shape[1] + 30
    _same(model, inputs, 30)


def test_windows_double_up_to_the_capacity() -> None:
    assert [lm.window(n, 5000) for n in (1, 256, 257, 1500, 4097)] == [256, 256, 512, 2048, 5000]


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA graphs need a GPU")
def test_decode_replays_graphs_on_cuda(capfd: pytest.CaptureFixture[str]) -> None:
    torch.manual_seed(2)
    model = _qwen3_5()
    ids = torch.randint(10, 290, (1, 300), device=DEVICE)
    _same(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, 64)
    assert "(graphs)" in capfd.readouterr().err


def _three_calls(model: Any, prompt: int, new: int, capturing: bool) -> list[bool]:
    """Three calls with a decoder layer that runs out of device memory once, inside a CUDA
    graph capture or outside one; each must give Transformers' tokens."""
    stop = [7, 9]
    calls = []
    for n in range(3):
        ids = torch.randint(10, 290, (1, prompt + n), device=DEVICE)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        calls.append((inputs, _reference(model, inputs, new, stop, 0.0)))
    armed = [True]
    layer = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")][1]

    def hook(*_: Any) -> None:
        inside = DEVICE == "cuda" and torch.cuda.is_current_stream_capturing()
        if armed[0] and inside == capturing:
            armed[0] = False
            raise torch.OutOfMemoryError("CUDA out of memory (injected)")

    layer.register_forward_pre_hook(hook)
    for inputs, want in calls:
        got = lm.generate(model, inputs, max_new_tokens=new, stop=stop, pad=0)
        assert torch.equal(got, want)
    return armed


def test_running_out_of_memory_decodes_through_transformers() -> None:
    from cozy_runtime.author._model import WARNINGS

    torch.manual_seed(4)
    WARNINGS.clear()
    assert not _three_calls(_qwen3_5(), 40, 24, capturing=False)[0]
    assert [message for message, _ in WARNINGS] == ["decode ran through Transformers' loop"]


@pytest.mark.skipif(DEVICE != "cuda", reason="CUDA graphs need a GPU")
def test_a_capture_out_of_memory_leaves_the_decode_serving(
    capfd: pytest.CaptureFixture[str],
) -> None:
    from cozy_runtime.author._model import WARNINGS

    torch.manual_seed(3)
    WARNINGS.clear()
    assert not _three_calls(_qwen3_5(), 100, 40, capturing=True)[0]
    assert [message for message, _ in WARNINGS] == ["a CUDA graph was not used"]
    lines = [line for line in capfd.readouterr().err.splitlines() if " tokens after " in line]
    assert len(lines) == 3 and "eager" in lines[0]
    assert all("(graphs)" in line for line in lines[1:])
