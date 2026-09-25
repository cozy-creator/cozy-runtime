"""Real tiny Qwen modules: staged execution agrees with the upstream pipeline."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")


def test_staged_qwen_image21_matches_upstream_and_preserves_scopes() -> None:
    import base64
    import zlib

    from diffusers import FlowMatchEulerDiscreteScheduler
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from cozy_runtime.author import Config
    from cozy_runtime.models.qwen_image21.model import (
        QwenImage21Model,
        build_processor,
        build_qwen_image21,
    )
    from cozy_runtime.models.qwen_image21.upstream.pipeline_qwenimage21 import QwenImage21Pipeline

    config = {
        "transformer": dict(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=16,
            mlp_ratio=2,
            axes_dims_rope=[4, 6, 6],
        ),
        "vae": dict(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=8,
            dim_mult=[1] * 5,
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True, True],
            latents_mean=[0.0] * 8,
            latents_std=[1.0] * 8,
        ),
        "text_encoder": dict(
            text_config=dict(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                rope_parameters=dict(rope_type="default", rope_theta=1e6, mrope_section=[1, 1, 2]),
            ),
            vision_config=dict(
                depth=1,
                hidden_size=16,
                intermediate_size=16,
                num_heads=2,
                out_hidden_size=16,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                num_position_embeddings=64,
                deepstack_visual_indexes=[0],
            ),
        ),
        "scheduler": dict(FlowMatchEulerDiscreteScheduler().config),
    }
    torch.manual_seed(53)
    graph = build_qwen_image21(Config(config))
    # Native serving fills values; this real tiny fixture initializes its own values.
    for root in graph.components.values():
        for p in root.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)
        root.eval()
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|image_pad|>": 2,
        "<|im_start|>": 3,
        "<|im_end|>": 4,
        "<|video_pad|>": 5,
        "<|vision_start|>": 6,
        "<|vision_end|>": 7,
    }
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=t,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=[
            "<|image_pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ],
    )
    chat = (
        '{% for message in messages %}{{"<|im_start|>" + message["role"] + "\\n"}}'
        '{% for item in message["content"] %}{{item["text"]}}{% endfor %}'
        '{{"<|im_end|>\\n"}}{% endfor %}'
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=1024, max_pixels=4096
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tok,
        chat_template=chat,
    )
    processor_bundle = {
        "tokenizer": base64.urlsafe_b64encode(
            zlib.compress(tok.backend_tokenizer.to_str().encode())
        ).decode(),
        "tokenizer_config": {"unk_token": "[UNK]", "pad_token": "[PAD]"},
        "image_processor": {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2},
        "video_processor": {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2},
        "chat_template": chat,
    }
    rebuilt = build_processor(processor_bundle)
    assert rebuilt.tokenizer.encode("blue jacket") == processor.tokenizer.encode("blue jacket")
    for invalid in (
        dict(processor_bundle, unexpected=True),
        dict(processor_bundle, tokenizer="invalid"),
        dict(
            processor_bundle,
            tokenizer=base64.urlsafe_b64encode(zlib.compress(b"x") + b"extra").decode(),
        ),
    ):
        with pytest.raises(ValueError):
            build_processor(invalid)
    model = QwenImage21Model.for_test(graph=graph, processor=processor)
    embeds, mask = model.encode("a person in a blue jacket")
    assert embeds.shape[-1] == 16
    latents = model.denoise(
        embeds,
        mask,
        width=32,
        height=32,
        steps=2,
        seed=44,
        on_step=lambda i: None,
        cancel=lambda: None,
    )
    actual = model.decode(latents, width=32, height=32)
    model.harness.assert_scopes("encode", "denoise", "decode")
    pipe = QwenImage21Pipeline(
        **graph.components,
        processor=processor,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config["scheduler"]),
    )
    expected = pipe(
        prompt="a person in a blue jacket",
        width=32,
        height=32,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(44),
        output_type="pt",
    ).images
    actual = (actual / 2 + 0.5).clamp(0, 1)
    assert actual.shape == (1, 4, 32, 32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert {x.dtype for x in graph.components["vae"].parameters()} == {torch.float32}
    assert {x.dtype for x in graph.components["transformer"].parameters()} == {torch.bfloat16}
