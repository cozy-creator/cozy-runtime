"""The shipped packages' 8-bit GEMMs (`internal/qualification_tables/gemm-shapes.json`): every
linear a lane encodes, from the real model classes on the meta device, with M from the packages'
serving token counts. Each lane's selection is its package recipe's (packages 2f4eaf2:
sdxl/sdxl/__init__.py INT8 and FP8, qwen-image-2/quantize.py, flux-3-klein/quantize.py,
character-describer INT8; H3's lanes encode both DiTs' blocks and the conditioner structurally).

    gemm-census.py <configs> > gemm-shapes.json

<configs> holds each component's upstream config.json, named as `config()` reads them:
sdxl-unet, sdxl-te1, sdxl-te2 (stabilityai/stable-diffusion-xl-base-1.0 unet, text_encoder,
text_encoder_2), qwen-transformer, qwen-te (Qwen/Qwen-Image-2.1), klein-transformer, klein-te
(black-forest-labs/flux-3-image-klein), describer (Qwen/Qwen3.5-9B), h3-transformer, h3-te
(MiniMaxAI/MiniMax-H3). It runs with Diffusers 7ecb1e1b (Klein's pin, which carries all three
DiT classes) and Transformers >= 5.15.
"""

import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from diffusers import (
    Flux3Transformer2DModel,
    MiniMaxH3Transformer3DModel,
    QwenImage21Transformer2DModel,
    UNet2DConditionModel,
)
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

CONFIGS = Path(sys.argv[1])


def config(name: str) -> dict:
    return {
        k: v
        for k, v in json.loads((CONFIGS / f"{name}.json").read_text()).items()
        if not k.startswith("_")
    }


def linears(model: torch.nn.Module, component: str) -> list[tuple[str, int, int]]:
    # Transformers 5 names CLIP-L's modules without the checkpoint's `text_model.` prefix.
    prefix = "text_model." if isinstance(model, CLIPTextModel) else ""
    return [
        (f"{component}.{prefix}{n}.weight", m.out_features, m.in_features)
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    ]


def vlm(name: str) -> torch.nn.Module:
    body = config(name)
    return AutoModelForImageTextToText.from_config(AutoConfig.for_model(**body))


with torch.device("meta"):
    MODELS = {
        "sdxl": {
            "unet": UNet2DConditionModel.from_config(config("sdxl-unet")),
            "text_encoder": CLIPTextModel(AutoConfig.for_model(**config("sdxl-te1"))),
            "text_encoder_2": CLIPTextModelWithProjection(
                AutoConfig.for_model(**config("sdxl-te2"))
            ),
        },
        "qwen-image": {
            "transformer": QwenImage21Transformer2DModel.from_config(config("qwen-transformer")),
            "text_encoder": vlm("qwen-te"),
        },
        "klein": {
            "transformer": Flux3Transformer2DModel.from_config(config("klein-transformer")),
            "text_encoder": vlm("klein-te"),
        },
        "describer": {"model": vlm("describer")},
        "h3": {
            "transformer": MiniMaxH3Transformer3DModel.from_config(config("h3-transformer")),
            "text_encoder": vlm("h3-te"),
        },
    }

_UNET_1280 = ("down_blocks.2", "mid_block", "up_blocks.0")
_LM = "model.model.language_model.layers.*"
#: Each lane's selection: (model, lane, encoding, include globs or None for structural, K multiple).
LANES = (
    (
        "sdxl",
        "int8",
        "int8",
        (
            *(
                f"unet.{level}.attentions.*.{key}"
                for level in _UNET_1280
                for key in (
                    "transformer_blocks.*.attn[12].to_[qkv].weight",
                    "transformer_blocks.*.attn[12].to_out.0.weight",
                    "transformer_blocks.*.ff.net.*weight",
                    "proj_[io]*.weight",
                )
            ),
            "unet.*_blocks.1.attentions.*.transformer_blocks.*.attn2.to_[kv].weight",
            "unet.*_blocks.1.attentions.*.transformer_blocks.*.ff.net.2.weight",
            "text_encoder.text_model.encoder.layers.*.self_attn.*_proj.weight",
            "text_encoder.text_model.encoder.layers.*.mlp.fc[12].weight",
            "text_encoder_2.text_model.encoder.layers.*.self_attn.*_proj.weight",
            "text_encoder_2.text_model.encoder.layers.*.mlp.fc[12].weight",
        ),
        256,
    ),
    ("sdxl", "fp8", "fp8", ("unet.*",), 32),
    (
        "qwen-image",
        "int8",
        "int8",
        (
            "transformer.transformer_blocks.*.attn.to_[qkv].weight",
            "transformer.transformer_blocks.*.attn.to_out.0.weight",
            "transformer.transformer_blocks.*.img_mlp.*.weight",
            "text_encoder.model.language_model.layers.*.self_attn.[qkvo]_proj.weight",
            "text_encoder.model.language_model.layers.*.mlp.*_proj.weight",
        ),
        256,
    ),
    (
        "klein",
        "int8",
        "int8",
        (
            "transformer.*blocks.*.attn.to_[qkv].weight",
            "transformer.*blocks.*.attn.to_out.0.weight",
            "transformer.*blocks.*.ff.linear_*.weight",
            "transformer.*_stream_modulations.*.linear.weight",
            "transformer.context_embedder.weight",
            "transformer.norm_out.linear.weight",
            "text_encoder.model.language_model.layers.[0-9].*_proj.weight",
            "text_encoder.model.language_model.layers.[12][0-9].*_proj.weight",
            "text_encoder.model.language_model.layers.3[01].*_proj.weight",
        ),
        256,
    ),
    (
        "describer",
        "int8",
        "int8",
        (
            f"{_LM}.linear_attn.in_proj_[qz]*.weight",
            f"{_LM}.linear_attn.out_proj.weight",
            f"{_LM}.self_attn.[qkvo]_proj.weight",
            f"{_LM}.mlp.*_proj.weight",
        ),
        256,
    ),
    (
        "h3",
        "int8",
        "int8",
        ("transformer.transformer_blocks.*", "text_encoder.model.language_model.*"),
        256,
    ),
    (
        "h3",
        "fp8",
        "fp8",
        ("transformer.transformer_blocks.*", "text_encoder.model.language_model.*"),
        32,
    ),
)


def tokens(model: str, name: str) -> tuple[int, ...]:
    """M per call at the packages' serving sizes (rows a linear sees)."""
    text = name.split(".")[0].startswith("text_encoder")
    if model == "sdxl":
        if text:
            return (154,)  # 77 tokens, prompt and negative
        if ".attn2.to_k" in name or ".attn2.to_v" in name:
            return (154,)
        if "time_emb" in name or "add_embedding" in name or "time_embedding" in name:
            return (2,)
        level = 1 if any(f"{b}_blocks.{1}" in name for b in ("down", "up")) else 2
        if "down_blocks.0" in name or "up_blocks.2" in name:
            level = 0
        side = {0: 128, 1: 64, 2: 32}[
            level
        ]  # 1 MP latent 128x128, CFG batch 2; 4 MP doubles the side
        return (2 * side * side, 2 * 4 * side * side)
    if model == "qwen-image":
        return (512,) if text else (4096, 16384)  # 1 MP and 4 MP image tokens (16x16 px each)
    if model == "klein":
        if text:
            return (512,)
        if "modulation" in name or "norm_out" in name:
            return (1,)
        if "context_embedder" in name:
            return (512,)
        return (4096 + 512,) if "single" in name else (4096,)
    if model == "describer":
        return (1, 2048)  # a decoded token; an image prompt's prefill
    if model == "h3":
        if text:
            return (512,)
        return (1,) if "norm" in name or "modulation" in name or "adaln" in name else (52 * 1024,)
    raise KeyError(model)


rows = []
for model, lane, fmt, include, multiple in LANES:
    found = Counter()
    for component, module in MODELS[model].items():
        for name, n, k in linears(module, component):
            if k % multiple or not any(fnmatch.fnmatchcase(name, rule) for rule in include):
                continue
            for m in tokens(model, name):
                found[m, n, k] += 1
    for (m, n, k), count in sorted(found.items()):
        rows.append(
            {"model": model, "lane": lane, "format": fmt, "m": m, "n": n, "k": k, "layers": count}
        )
sys.stdout.write(json.dumps(rows, separators=(",", ":")).replace("},{", "},\n{") + "\n")
