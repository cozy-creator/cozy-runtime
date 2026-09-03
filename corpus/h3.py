"""A real-torch H3 fixture: three AdaLN structures, two task partitions, one schedule.

`proofs/fixtures/tiny-models/tiny_h3.py` exercises the AUTHOR SURFACE and is deliberately
torch-free so the
descriptor stays derivable without a 3 GB wheel. This is its construction twin: the same
shape in real `nn.Module`s, so the cr-004 harness has something with genuine structural
variance to derive.

The three structures must never be conflated (§1.1.1):

* `full` carries the timestep embedder, every block AdaLN projection and the final
  modulation projection. Tables are computed live under the transformer's own lease.
* `curve` is the community pruned tensor schema: a shared interpolated curve plus narrow
  projections. It still EXECUTES projections and changes numeric semantics. It is a
  separately quality-approved lane, never shorthand for a baked table.
* `baked` carries exact modulation tables for a closed set of denoise plans and OMITS the
  projection weights. An uncovered plan cannot fall back inside this artifact.

They differ by CONSTRUCTION, not by a request mode: `baked` physically lacks weights that
`full` contains, so the tensor schema cannot change without selecting a different generation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from cozy_runtime.author import Config, Loader, Model, uses_components
from cozy_runtime.internal import canonical

Task = Literal["fl2va", "ref2va"]
Structure = Literal["full", "curve", "baked"]
StepPreset = Literal["draft", "standard", "quality"]

STEPS: Mapping[str, int] = {"draft": 8, "standard": 20, "quality": 40}


# --------------------------------------------------------------- the schedule builder


@dataclass(frozen=True, slots=True)
class TimestepPlan:
    """The exact denoise plan. PURE: same inputs, same bits, no model and no geometry.

    Canvas, frame count, reference count and packed row positions are deliberately NOT
    here (§1.1.1): AdaLN produces one modulation value per exact timestep and modality
    class, independent of how many rows consume it. That is what keeps one exact baked
    table reusable across every legal geometry without weakening the numeric match.
    """

    task: Task
    video: tuple[float, ...]
    audio: tuple[float, ...]
    condition_classes: tuple[str, ...]
    conditioning_noise: float
    tag_convention: str

    @property
    def digest(self) -> str:
        """Exact timestep BITS and order, through the ES6 number rule (cr-003's writer)."""
        return canonical.digest(
            {
                "task": self.task,
                "video": list(self.video),
                "audio": list(self.audio),
                "condition_classes": list(self.condition_classes),
                "conditioning_noise": self.conditioning_noise,
                "tag_convention": self.tag_convention,
            }
        )


def flow_match_sigmas(steps: int, *, shift: float, scheduler: str) -> tuple[float, ...]:
    """The scheduler's own timesteps, computed for real. Two implementations with the same
    step count produce different bits, which is the whole point of the digest."""
    base = [1.0 - i / steps for i in range(steps)]
    if scheduler == "flow_match":
        return tuple(shift * s / (1.0 + (shift - 1.0) * s) for s in base)
    if scheduler == "flow_match_exp":
        return tuple(math.expm1(shift * s) / math.expm1(shift) for s in base)
    raise ValueError(f"unknown scheduler implementation {scheduler!r}")


def derive_plan(
    *,
    preset: StepPreset,
    task: Task,
    has_visual: bool,
    has_audio: bool,
    video: Mapping[str, Any],
    audio: Mapping[str, Any],
) -> TimestepPlan:
    """The package's PURE schedule/modulation-plan builder — what conformance executes
    per typed step preset and condition-presence cell (cr-004)."""
    steps = STEPS[preset]
    classes = ["denoise"]
    if has_visual:
        classes.append("visual_cond")
    if has_audio:
        classes.append("audio_cond")
    return TimestepPlan(
        task=task,
        video=flow_match_sigmas(steps, shift=float(video["shift"]), scheduler=str(video["impl"])),
        audio=flow_match_sigmas(steps, shift=float(audio["shift"]), scheduler=str(audio["impl"])),
        condition_classes=tuple(classes),
        conditioning_noise=float(video.get("conditioning_noise", 0.0)),
        tag_convention=str(video.get("tags", "modality-v1")),
    )


def coverage_map(config: Config) -> dict[str, str]:
    """The canonical `(preset, task, has_visual, has_audio)` -> exact plan digest map.

    Published beside the artifact's advertised covered digests: Tensorhub joins request,
    task and artifact on it BEFORE renting a GPU or downloading the variant, and the worker
    re-derives and verifies before any large allocation.
    """
    video = config.section("video_scheduler")
    audio = config.section("audio_scheduler")
    video_cfg = {
        "shift": video.as_float("shift", 3.0),
        "impl": video.as_str("impl", "flow_match"),
        "conditioning_noise": video.as_float("conditioning_noise", 0.0),
        "tags": video.as_str("tags", "modality-v1"),
    }
    audio_cfg = {"shift": audio.as_float("shift", 5.0), "impl": audio.as_str("impl", "flow_match")}
    out: dict[str, str] = {}
    for preset in ("draft", "standard", "quality"):
        for task in ("fl2va", "ref2va"):
            for visual in (False, True):
                for sound in (False, True):
                    cell = f"{preset}/{task}/visual={int(visual)}/audio={int(sound)}"
                    out[cell] = derive_plan(
                        preset=preset,  # type: ignore[arg-type]
                        task=task,  # type: ignore[arg-type]
                        has_visual=visual,
                        has_audio=sound,
                        video=video_cfg,
                        audio=audio_cfg,
                    ).digest
    return out


# ------------------------------------------------------------------- the construction


def _block(width: int, structure: Structure, index: int) -> nn.Module:
    block = nn.Module()
    block.attn_q = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)  # type: ignore[assignment]
    block.attn_k = nn.Linear(width, width, bias=False, dtype=torch.bfloat16)  # type: ignore[assignment]
    if structure == "full":
        # every block carries its OWN wide AdaLN projection — the ~13B the model card
        # calls removable from inference-only loading once the outputs are cached
        block.adaln_proj = nn.Linear(width, width * 6, dtype=torch.bfloat16)  # type: ignore[assignment]
    elif structure == "curve":
        # the community pruned tensor schema: a NARROW projection over a shared curve
        block.adaln_narrow = nn.Linear(width // 4, width * 6, bias=False, dtype=torch.bfloat16)  # type: ignore[assignment]
    else:
        # baked: the exact table IS checkpoint data, and the projection does not exist
        block.register_buffer("modulation_table", torch.zeros(8, width * 6, dtype=torch.bfloat16))
    del index
    return block


class H3Transformer(nn.Module):
    def __init__(self, width: int, blocks: int, structure: Structure) -> None:
        super().__init__()
        self.structure = structure
        self.blocks = nn.ModuleList([_block(width, structure, i) for i in range(blocks)])
        if structure == "full":
            self.time_embed = nn.Linear(256, width, dtype=torch.bfloat16)
            self.final_modulation = nn.Linear(width, width * 2, dtype=torch.bfloat16)
        elif structure == "curve":
            self.time_embed = nn.Linear(256, width // 4, dtype=torch.bfloat16)
            self.curve = nn.Linear(width // 4, width // 4, bias=False, dtype=torch.bfloat16)
            self.final_modulation = nn.Linear(width // 4, width * 2, dtype=torch.bfloat16)


class H3Pipeline:
    """The component-staged object the factory returns whole (§1.1: no partial binding)."""

    def __init__(self, config: Config, *, task: Task | None, both: bool = False) -> None:
        width = config.as_int("width", 32)
        depth = config.as_int("blocks", 2)
        structure: Structure = config.as_str("structure", "full")  # type: ignore[assignment]
        self.structure = structure
        components: dict[str, object] = {
            "text_encoder": _encoder(width),
            "visual_vae": _vae(width),
            "audio_vae": _vae(width // 2),
        }
        roles: Sequence[Task] = (
            ("fl2va", "ref2va") if both else ((task,) if task is not None else ())
        )
        for role in roles:
            components[f"{role}_transformer"] = H3Transformer(width, depth, structure)
        self.components = components
        self.coverage = coverage_map(config)


def _encoder(width: int) -> nn.Module:
    module = nn.Module()
    module.embed = nn.Embedding(64, width, dtype=torch.bfloat16)  # type: ignore[assignment]
    module.head = nn.Linear(width, 64, bias=False, dtype=torch.bfloat16)  # type: ignore[assignment]
    module.head.weight = module.embed.weight  # a real tie, both names are destinations
    return module


def _vae(width: int) -> nn.Module:
    module = nn.Module()
    module.conv_in = nn.Conv1d(width, width, 3, dtype=torch.bfloat16)  # type: ignore[assignment]
    # NaN-prone VAE head pinned to fp32 — the dtype-diet rule must carry the deviation
    module.out = nn.Linear(width, 3, dtype=torch.float32)  # type: ignore[assignment]
    return module


def build_h3(config: Config, *, task: Task) -> H3Pipeline:
    """The ROLE-SCOPED family factory: the graph contains exactly what the class declares."""
    return H3Pipeline(config, task=task)


def build_h3_both(config: Config) -> H3Pipeline:
    """The planted violation: one class constructing BOTH transformers could never pass
    exact binding against a narrow artifact."""
    return H3Pipeline(config, task=None, both=True)


class _H3Base(Model[H3Pipeline]):
    pipe: H3Pipeline

    def _load(self, loader: Loader, *, task: Task) -> None:
        self.pipe = loader.construct(H3Pipeline, factory=lambda config: build_h3(config, task=task))
        if self.pipe.structure != "baked":
            # exactly one modulation source exists per bound tensor schema (§1.1.1); a baked
            # construction never touches this cache
            self.tables = loader.cache("adaln_modulation_tables", max_entries=8)

    @uses_components("text_encoder")
    def encode_presentation(self, tokens: int) -> int:
        return tokens

    @uses_components("visual_vae")
    def encode_visual_conditions(self, count: int) -> int:
        return count

    @uses_components("audio_vae")
    def encode_audio_conditions(self, count: int) -> int:
        return count

    @uses_components("visual_vae")
    def decode_video(self, latents: int) -> int:
        return latents

    @uses_components("audio_vae")
    def decode_audio(self, latents: int) -> int:
        return latents


class Fl2VAModel(_H3Base, task="fl2va"):
    def load(self, loader: Loader) -> None:
        self._load(loader, task="fl2va")

    @uses_components("fl2va_transformer")
    def denoise(self, steps: int) -> int:
        return steps


class Ref2VAModel(_H3Base, task="ref2va"):
    def load(self, loader: Loader) -> None:
        self._load(loader, task="ref2va")

    @uses_components("ref2va_transformer")
    def denoise(self, steps: int) -> int:
        return steps


COMPONENT_USE: Mapping[str, tuple[str, ...]] = {
    "encode_presentation": ("text_encoder",),
    "encode_visual_conditions": ("visual_vae",),
    "encode_audio_conditions": ("audio_vae",),
    "decode_video": ("visual_vae",),
    "decode_audio": ("audio_vae",),
}
