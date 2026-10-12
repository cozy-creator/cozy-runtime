"""Portable completed H3 latents and conditioning for learned spatial refinement.

The caller keeps the carrier with ordinary Outputs/FileAsset custody. No model weights,
pickle, raw prompt, CUDA handle or store path is included. Video is in Diffusers' normalized
24-channel latent space, so a learned upscaler must not normalize it a second time.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

import torch
from safetensors.torch import load, save

from cozy_runtime.author import InvalidRequest

SCHEMA = "cozy.minimax-h3.latents/1"
MAX_LATENT_BYTES = 2 << 30


@dataclass(frozen=True)
class ReferenceLayout:
    kind: str
    has_audio: bool = False


@dataclass(frozen=True)
class H3Latents:
    video: torch.Tensor
    audio: torch.Tensor
    waveform: torch.Tensor
    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    conditions: tuple[torch.Tensor, ...]
    audio_conditions: tuple[torch.Tensor, ...]
    references: tuple[ReferenceLayout, ...]
    anchors: tuple[str, ...]
    task: Literal["fl2va", "ref2va"]
    frames: int
    seed: int
    checkpoint: str
    sample_rate: int = 32000

    @property
    def height(self) -> int:
        return int(self.video.shape[-2]) * 16

    @property
    def width(self) -> int:
        return int(self.video.shape[-1]) * 16

    def validate(self) -> None:
        def refuse(message: str) -> None:
            raise InvalidRequest(message, code="h3_refinement")

        if self.task not in ("fl2va", "ref2va"):
            refuse("latent artifact must name the FL2VA or Ref2VA trunk")
        if type(self.frames) is not int or not 124 <= self.frames <= 362 or (self.frames - 5) % 17:
            refuse("latent artifact frames must follow H3's 17n+5 grid within 124..362")
        if self.video.ndim != 5 or tuple(self.video.shape[:2]) != (1, 24):
            refuse("video latents must have shape 1x24xTxHxW")
        if self.video.shape[2] != 5 * ((self.frames - 5) // 17) + 2:
            refuse("video latent time axis differs from its declared frame count")
        if min(self.video.shape[-2:]) < 2 or any(int(n) % 2 for n in self.video.shape[-2:]):
            refuse("video latent canvas must lie on H3's 32-pixel grid")
        if max(self.width, self.height) > 8192 or self.width * self.height > 16_777_216:
            refuse("latent canvas exceeds the 16-megapixel, 8192-edge limit")
        if self.audio.ndim != 3 or tuple(self.audio.shape[:2]) != (2, 32):
            refuse("audio latents must have shape 2x32xT")
        if (
            self.waveform.ndim != 3
            or tuple(self.waveform.shape[:2]) != (1, 2)
            or self.sample_rate != 32000
        ):
            refuse("retained audio must be 32 kHz stereo PCM with shape 1x2xS")
        if not 0 < self.waveform.shape[-1] <= (self.frames + 1) * 32000 // 24:
            refuse("retained audio does not fit the declared video duration")
        if self.prompt_embeds.ndim != 3 or self.prompt_embeds.shape[0] != 1:
            refuse("retained prompt embeddings must be one sequence")
        if (
            self.text_token_tags.ndim != 1
            or self.text_token_tags.numel() != self.prompt_embeds.shape[1]
        ):
            refuse("retained text tags differ from prompt embedding rows")
        if (
            len(self.conditions) > 12
            or len(self.audio_conditions) > 12
            or len(self.references) > 12
        ):
            refuse("retained reference count exceeds H3's limit")
        for condition in self.conditions:
            if condition.ndim != 5 or tuple(condition.shape[:2]) != (1, 24):
                refuse("visual conditioning must have shape 1x24xTxHxW")
        for condition in self.audio_conditions:
            if condition.ndim != 2 or condition.shape[-1] != 32:
                refuse("audio conditioning must be packed 32-channel rows")
        if any(r.kind not in ("image", "video", "audio") for r in self.references):
            refuse("retained reference modality is invalid")
        if self.task == "fl2va":
            if (
                self.references
                or self.audio_conditions
                or len(self.anchors) != len(self.conditions)
            ):
                refuse("FL2VA anchors differ from retained conditioning")
            if any(a not in ("first", "last") for a in self.anchors) or len(
                set(self.anchors)
            ) != len(self.anchors):
                refuse("FL2VA anchors must name distinct first/last frames")
            if any(tuple(c.shape[2:]) != (1, *self.video.shape[-2:]) for c in self.conditions):
                refuse("FL2VA conditioning must be upscaled with its target canvas")
        else:
            if self.anchors or sum(r.kind != "audio" for r in self.references) != len(
                self.conditions
            ):
                refuse("Ref2VA visual reference layout differs from conditioning")
            if sum(r.kind == "audio" or r.has_audio for r in self.references) != len(
                self.audio_conditions
            ):
                refuse("Ref2VA audio reference layout differs from conditioning")
        values = (
            self.video,
            self.audio,
            self.waveform,
            self.prompt_embeds,
            *self.conditions,
            *self.audio_conditions,
        )
        if any(not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in values):
            refuse("retained H3 tensors must be finite floating-point values")
        if self.text_token_tags.dtype not in (torch.int32, torch.int64):
            refuse("retained text tags must be integer values")
        if type(self.seed) is not int or not 0 <= self.seed < 1 << 64:
            refuse("retained seed is outside the uint64 range")
        if (
            not isinstance(self.checkpoint, str)
            or not self.checkpoint
            or len(self.checkpoint) > 512
        ):
            refuse("retained checkpoint identity is absent or too long")


def capture_latents(task: str, state: Any, *, checkpoint: str, seed: int) -> H3Latents:
    """Snapshot after denoise and audio decode; spatial refinement never rereads an MP4."""
    if state.get("continuation_context") is not None:
        raise InvalidRequest(
            "retain standalone H3 generations for refinement", code="h3_refinement"
        )

    def held(value: Any) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise InvalidRequest("completed H3 state lacks a required tensor", code="h3_refinement")
        return value.detach().cpu().contiguous().clone()

    trunk = task.removesuffix("_turbo")
    if trunk not in ("fl2va", "ref2va"):
        raise InvalidRequest("unknown H3 generation trunk", code="h3_refinement")
    value = H3Latents(
        held(state.latents),
        held(state.audio_latents),
        held(state.audio),
        held(state.prompt_embeds),
        held(state.text_token_tags),
        tuple(held(c) for c in (state.get("condition_latents") or ())),
        tuple(held(c) for c in (state.get("audio_condition_latents") or ())),
        tuple(
            ReferenceLayout(r.kind, bool(r.has_audio))
            for r in (state.get("normalized_references") or ())
        ),
        tuple(state.get("keyframe_anchors") or ()),
        cast(Literal["fl2va", "ref2va"], trunk),
        int(state.num_frames),
        seed,
        checkpoint,
        int(state.sampling_rate),
    )
    value.validate()
    return value


def encode_latents(value: H3Latents) -> bytes:
    value.validate()
    tensors = {
        name: getattr(value, name).detach().cpu().contiguous()
        for name in ("video", "audio", "waveform", "prompt_embeds", "text_token_tags")
    }
    tensors.update(
        {f"condition_{i}": t.detach().cpu().contiguous() for i, t in enumerate(value.conditions)}
    )
    tensors.update(
        {
            f"audio_condition_{i}": t.detach().cpu().contiguous()
            for i, t in enumerate(value.audio_conditions)
        }
    )
    metadata = {
        "schema": SCHEMA,
        "task": value.task,
        "frames": str(value.frames),
        "seed": str(value.seed),
        "checkpoint": value.checkpoint,
        "sample_rate": str(value.sample_rate),
        "conditions": str(len(value.conditions)),
        "audio_conditions": str(len(value.audio_conditions)),
        "references": json.dumps([[r.kind, r.has_audio] for r in value.references]),
        "anchors": json.dumps(value.anchors),
    }
    data: bytes = save(tensors, metadata=metadata)
    if len(data) > MAX_LATENT_BYTES:
        raise InvalidRequest("H3 latent carrier exceeds 2 GiB", code="h3_refinement")
    return data


def decode_latents(data: bytes) -> H3Latents:
    if not isinstance(data, bytes) or not 8 <= len(data) <= MAX_LATENT_BYTES:
        raise InvalidRequest("H3 latent carrier is absent or exceeds 2 GiB", code="h3_refinement")
    try:
        size = struct.unpack("<Q", data[:8])[0]
        if not 2 <= size <= min(65536, len(data) - 8):
            raise ValueError("invalid safetensors header length")
        metadata = json.loads(data[8 : 8 + size])["__metadata__"]
        if metadata["schema"] != SCHEMA:
            raise ValueError("unsupported H3 latent schema")
        count, audio_count = int(metadata["conditions"]), int(metadata["audio_conditions"])
        if not 0 <= count <= 12 or not 0 <= audio_count <= 12:
            raise ValueError("too many retained references")
        tensors = load(data)
        value = H3Latents(
            tensors["video"],
            tensors["audio"],
            tensors["waveform"],
            tensors["prompt_embeds"],
            tensors["text_token_tags"],
            tuple(tensors[f"condition_{i}"] for i in range(count)),
            tuple(tensors[f"audio_condition_{i}"] for i in range(audio_count)),
            tuple(ReferenceLayout(*r) for r in json.loads(metadata["references"])),
            tuple(json.loads(metadata["anchors"])),
            metadata["task"],
            int(metadata["frames"]),
            int(metadata["seed"]),
            metadata["checkpoint"],
            int(metadata["sample_rate"]),
        )
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise InvalidRequest(f"invalid H3 latent carrier: {exc}", code="h3_refinement") from exc
    value.validate()
    return value


def refinement_state(value: H3Latents, generator: Any) -> Any:
    """Fresh upstream state with retained conditioning; no caller tensor is mutated."""
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    value.validate()
    state = PipelineState()
    for name, tensor in (
        ("latents", value.video),
        ("audio_latents", value.audio),
        ("prompt_embeds", value.prompt_embeds),
        ("text_token_tags", value.text_token_tags),
    ):
        state.set(name, tensor.clone())
    for name, item in {
        "height": value.height,
        "width": value.width,
        "num_frames": value.frames,
        "num_inference_steps": 31,
        "generator": generator,
        "output_type": "pt",
        "condition_latents": [c.clone() for c in value.conditions],
        "audio_condition_latents": [c.clone() for c in value.audio_conditions],
        "normalized_references": list(value.references),
        "keyframe_anchors": value.anchors,
        # Workflow choice only: before-encode blocks are already represented by retained
        # embeddings and condition latents. No upstream image encoder sees these markers.
        "image": True if "first" in value.anchors else None,
        "last_image": True if "last" in value.anchors else None,
        "preserved_audio": value.waveform.clone(),
        "sampling_rate": value.sample_rate,
        "attention_kwargs": {},
    }.items():
        state.set(name, item)
    return state


def resize_video_chunk(chunk: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Aspect-preserving standard resize and letterbox, after native VAE decoding.

    The size is (width, height); this changes neither time nor audio. Work is bounded to
    the current decoded chunk, so it never allocates a second whole-video frame array.
    """
    vision = import_module("torchvision.transforms.functional")

    width, height = output_size
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) < 32
        or width % 2
        or height % 2
    ):
        raise InvalidRequest(
            "output canvas must have positive even dimensions", code="h3_refinement"
        )
    if max(width, height) > 8192 or width * height > 16_777_216:
        raise InvalidRequest("output canvas exceeds the 16-megapixel bound", code="h3_refinement")
    source_h, source_w = chunk.shape[-2:]
    ratio = min(width / source_w, height / source_h)
    scaled_h, scaled_w = min(height, round(source_h * ratio)), min(width, round(source_w * ratio))
    pixels = vision.resize(
        chunk, [scaled_h, scaled_w], interpolation=vision.InterpolationMode.BICUBIC, antialias=True
    )
    left, top = (width - scaled_w) // 2, (height - scaled_h) // 2
    return torch.nn.functional.pad(
        pixels, (left, width - scaled_w - left, top, height - scaled_h - top)
    )
