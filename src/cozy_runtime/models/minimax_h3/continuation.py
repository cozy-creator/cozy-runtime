"""Bounded, completed-segment H3 AV context and exact delivered-frame clocks.

Reference: AIMixer/ComfyUI_MiniMaxH3_Director@4b4fe6e1a7e68b17f4a36c8b3e942667defab61f
and yolain/ComfyUI-Easy-Media@ed20432dcb744895c94c8b5b0eed90f36234a55b.
This is model-local data/math, not a filesystem or mid-denoise checkpoint service.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import numpy as np
import torch
from safetensors.torch import load, save

from cozy_runtime.author import InvalidRequest

FPS = 24
CONTEXT_FRAMES = (22, 39, 56)
MAX_CONTEXT_FRAMES = 56
MAX_CONTEXT_BYTES = 64 << 20
MAX_CONTEXT_PIXELS = 1344 * 768
CONTEXT_SCHEMA = "cozy.minimax-h3.av-context/2"


def _refuse(message: str) -> NoReturn:
    raise InvalidRequest(message, code="h3_continuation", fields=["context"])


def audio_sample_range(start: int, end: int, sample_rate: int) -> tuple[int, int]:
    """Half-open audio interval on the absolute frame clock; never round each clip."""
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        _refuse("context frame interval is invalid")
    if sample_rate != 32000:
        _refuse("H3 continuation requires its native 32000 Hz audio clock")
    return start * sample_rate // FPS, end * sample_rate // FPS


@dataclass(frozen=True)
class ContinuationPlan:
    delivered_frames: int
    prefix_frames: int = 0

    def __post_init__(self) -> None:
        if type(self.delivered_frames) is not int or not 120 <= self.delivered_frames <= 360:
            _refuse("continuation must deliver between 120 and 360 new frames (5-15 seconds)")
        if type(self.prefix_frames) is not int or self.prefix_frames not in (0, *CONTEXT_FRAMES):
            _refuse("continuation context must contain 22, 39 or 56 frames")
        if self.sample_frames > 362:
            _refuse(
                f"{self.delivered_frames} new frames plus {self.prefix_frames} context frames "
                f"requires {self.sample_frames} sampled frames, above the qualified 362-frame "
                "envelope; shorten the segment"
            )

    @property
    def sample_frames(self) -> int:
        frames = self.delivered_frames + self.prefix_frames
        return frames + (5 - frames) % 17

    @property
    def delivery_start(self) -> int:
        return self.prefix_frames

    @property
    def delivery_end(self) -> int:
        return self.prefix_frames + self.delivered_frames


def plan_continuation(delivered_frames: int, *, context_frames: int = 0) -> ContinuationPlan:
    return ContinuationPlan(delivered_frames, context_frames)


@dataclass(frozen=True)
class DeliveredAVWindow:
    frames: Any  # uint8 CPU ndarray, [frames,height,width,RGB]
    audio: Any  # float32 CPU tensor, [stereo,samples]
    sample_rate: int
    source_start_frame: int
    source_end_frame: int
    provenance: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.frames, np.ndarray) or self.frames.dtype != np.uint8:
            _refuse("context video must be uint8 RGB pixels")
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            _refuse("context video must have shape [frames,height,width,3]")
        n, height, width, _ = self.frames.shape
        if n not in CONTEXT_FRAMES or min(height, width) < 32 or height % 32 or width % 32:
            _refuse("context frame count or 32-pixel canvas stride is invalid")
        if height * width > MAX_CONTEXT_PIXELS:
            _refuse("context canvas exceeds the qualified H3 canvas area")
        if type(self.source_start_frame) is not int or type(self.source_end_frame) is not int:
            _refuse("context source frame clock must use integer frames")
        if self.source_end_frame - self.source_start_frame != n or self.source_end_frame > 362:
            _refuse("context source frame interval does not match its video")
        start, end = audio_sample_range(
            self.source_start_frame, self.source_end_frame, self.sample_rate
        )
        if (
            not isinstance(self.audio, torch.Tensor)
            or self.audio.device.type != "cpu"
            or self.audio.dtype != torch.float32
            or self.audio.ndim != 2
            or tuple(self.audio.shape) != (2, end - start)
            or not bool(torch.isfinite(self.audio).all())
        ):
            _refuse("context audio must be finite float32 stereo at its exact frame interval")
        if (
            not isinstance(self.provenance, str)
            or not self.provenance
            or len(self.provenance.encode("utf-8")) > 16384
        ):
            _refuse("context requires bounded renderer provenance")

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def select(self, frames: int) -> DeliveredAVWindow:
        self.validate()
        if type(frames) is not int or frames not in CONTEXT_FRAMES or frames > self.frame_count:
            _refuse("requested continuation window is absent from the completed tail")
        start_frame = self.source_end_frame - frames
        old_start, _ = audio_sample_range(
            self.source_start_frame, self.source_end_frame, self.sample_rate
        )
        start, end = audio_sample_range(start_frame, self.source_end_frame, self.sample_rate)
        return DeliveredAVWindow(
            self.frames[-frames:].copy(),
            self.audio[:, start - old_start : end - old_start].clone(),
            self.sample_rate,
            start_frame,
            self.source_end_frame,
            self.provenance,
        )


@dataclass(frozen=True)
class AVContext:
    """Native normalized AV windows, independently encoded from delivered RGB/audio.

    A shorter causal VAE window is not a suffix of a longer window's latents, and the audio
    grid's phase is window-local, so every window a successor may select is its own
    encoding. A producer encodes only the windows its caller names.
    """

    windows: dict[int, tuple[torch.Tensor, torch.Tensor]]
    height: int
    width: int
    source_end_frame: int
    provenance: str
    frame_count: int = MAX_CONTEXT_FRAMES
    sample_rate: int = 32000

    def __post_init__(self) -> None:
        self.validate()

    @property
    def source_start_frame(self) -> int:
        return self.source_end_frame - self.frame_count

    @property
    def video_latents(self) -> torch.Tensor:
        return self.windows[self.frame_count][0]

    @property
    def audio_latents(self) -> torch.Tensor:
        return self.windows[self.frame_count][1]

    def validate(self) -> None:
        if (
            type(self.frame_count) is not int
            or not isinstance(self.windows, dict)
            or self.frame_count not in self.windows
            or not set(self.windows) <= set(CONTEXT_FRAMES)
            or any(type(n) is not int for n in self.windows)
        ):
            _refuse("context must contain its selected independently encoded AV window")
        if (
            type(self.height) is not int
            or type(self.width) is not int
            or min(self.height, self.width) < 32
            or self.height % 32
            or self.width % 32
            or self.height * self.width > MAX_CONTEXT_PIXELS
        ):
            _refuse("context canvas is outside the qualified H3 geometry")
        if type(self.source_end_frame) is not int or not 56 <= self.source_end_frame <= 362:
            _refuse("context source frame clock is invalid")
        audio_sample_range(self.source_end_frame - 56, self.source_end_frame, self.sample_rate)
        if (
            not isinstance(self.provenance, str)
            or not self.provenance
            or len(self.provenance.encode("utf-8")) > 16384
        ):
            _refuse("context requires bounded renderer provenance")
        for n, pair in self.windows.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                _refuse("context AV window is invalid")
            video, audio = pair
            shapes = (
                (1, 24, 5 * ((n - 5) // 17) + 2, self.height // 16, self.width // 16),
                (2 * ((n * 5 + 2) // 3), 32),
            )
            for tensor, shape in zip((video, audio), shapes, strict=True):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or tuple(tensor.shape) != shape
                    or not bool(torch.isfinite(tensor).all())
                ):
                    _refuse("context must contain finite native float32 AV conditioning tensors")

    def select(self, frames: int) -> AVContext:
        self.validate()
        if type(frames) is not int or frames not in self.windows:
            _refuse("requested continuation window is absent from the completed tail")
        return AVContext(
            self.windows,
            self.height,
            self.width,
            self.source_end_frame,
            self.provenance,
            frames,
            self.sample_rate,
        )


@dataclass(frozen=True)
class ContextReferenceLayout:
    """Private packed-layout metadata, appended only after public reference encoding."""

    kind: str = "video"
    has_audio: bool = True


def encode_context(context: AVContext) -> bytes:
    context.validate()
    tensors = {
        f"{kind}_{n}": tensor.contiguous()
        for n, pair in context.windows.items()
        for kind, tensor in zip(("video", "audio"), pair, strict=True)
    }
    data = save(
        tensors,
        metadata={
            "schema": CONTEXT_SCHEMA,
            "sample_rate": str(context.sample_rate),
            "height": str(context.height),
            "width": str(context.width),
            "frame_count": str(context.frame_count),
            "source_end_frame": str(context.source_end_frame),
            "provenance": context.provenance,
        },
    )
    if len(data) > MAX_CONTEXT_BYTES:
        _refuse("encoded context exceeds its 64 MiB bound")
    return cast(bytes, data)


def decode_context(data: bytes) -> AVContext:
    if not isinstance(data, bytes) or not 8 <= len(data) <= MAX_CONTEXT_BYTES:
        _refuse("context carrier is absent or exceeds its 64 MiB bound")
    length = struct.unpack("<Q", data[:8])[0]
    if not 2 <= length <= min(65536, len(data) - 8):
        _refuse("context header is invalid or exceeds its bound")
    try:
        header = json.loads(data[8 : 8 + length])
        metadata = header.get("__metadata__", {})
        present = [n for n in CONTEXT_FRAMES if f"video_{n}" in header]
        if (
            not present
            or not {
                "schema",
                "sample_rate",
                "height",
                "width",
                "frame_count",
                "source_end_frame",
                "provenance",
            }
            <= set(metadata)
            or metadata["schema"] != CONTEXT_SCHEMA
        ):
            _refuse("context schema or fields do not match")
        # Bounded, uncompressed safetensors validates offsets before allocation.
        tensors = load(data)
        return AVContext(
            {n: (tensors[f"video_{n}"], tensors[f"audio_{n}"]) for n in present},
            int(metadata["height"]),
            int(metadata["width"]),
            int(metadata["source_end_frame"]),
            metadata["provenance"],
            int(metadata["frame_count"]),
            int(metadata["sample_rate"]),
        )
    except InvalidRequest:
        raise
    except Exception as exc:
        raise InvalidRequest(
            "invalid completed H3 AV context", code="h3_continuation", fields=["context"]
        ) from exc


def align_context_rows(state: Any, context: AVContext, patch_size: tuple[int, int, int]) -> None:
    """Place the hidden AV condition at the target head, retaining all public references.

    It is encoded as the final internal video/audio condition after text encoding,
    so it consumes neither a Picture/Video label nor a Qwen vision-token block.
    The native scheduler already keeps these condition rows fixed on every step.
    """
    _, patch_h, patch_w = patch_size
    latent = state.condition_latents[-1]
    frames, height, width = (int(x) for x in latent.shape[2:])
    expected_latents = 5 * ((context.frame_count - 5) // 17) + 2
    if frames != expected_latents:
        _refuse("context VAE temporal grid differs from its frame clock")
    video_count = frames * (height // patch_h) * (width // patch_w)
    audio_count = int(state.audio_condition_latents[-1].shape[0])
    if (
        audio_count != 2 * ((context.frame_count * 5 + 2) // 3)
        or video_count > state.num_condition_video_rows
    ):
        _refuse("context packed geometry is inconsistent")
    video_rows = state.video_indices[
        state.num_condition_video_rows - video_count : state.num_condition_video_rows
    ]
    audio_rows = state.audio_indices[
        state.num_condition_audio_rows - audio_count : state.num_condition_audio_rows
    ]
    target_start = state.video_indices[state.num_condition_video_rows]
    origin = state.position_ids[target_start, 0].clone()
    state.position_ids[video_rows, 0] += origin - state.position_ids[video_rows[0], 0].clone()
    # Audio has one rotary tick per 1/40 second. Its latent grid rounds up,
    # just as native target audio does; the final tick may contain encoder padding.
    ticks = audio_count // 2
    end_tick = (context.frame_count * 5 + 2) // 3
    desired_start = origin + end_tick - ticks
    state.position_ids[audio_rows, 0] += (
        desired_start - state.position_ids[audio_rows[0], 0].clone()
    )


class CompletedTail:
    """The delivery window of a streamed decode. The consumer lands the delivered frames as
    RGB8 for its output; the context tail is read back from that buffer, not copied twice."""

    def __init__(self, plan: ContinuationPlan) -> None:
        self.plan = plan
        self.seen = 0
        self.delivered = 0
        self.complete = False

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.complete or chunk.ndim != 4 or chunk.shape[1] != 3:
            _refuse("continuation decode stream is invalid")
        start = max(0, self.plan.delivery_start - self.seen)
        end = min(int(chunk.shape[0]), self.plan.delivery_end - self.seen)
        self.seen += int(chunk.shape[0])
        selected = chunk[start : max(start, end)]
        self.delivered += int(selected.shape[0])
        return selected

    def finish(self) -> None:
        if self.seen != self.plan.sample_frames or self.delivered != self.plan.delivered_frames:
            _refuse("completed continuation decode differs from its sampled/delivered frame plan")
        self.complete = True

    def export(
        self, frames: torch.Tensor, audio: torch.Tensor, sample_rate: int, provenance: str
    ) -> DeliveredAVWindow:
        """`frames` are every delivered frame as landed: uint8 CPU `[delivered, H, W, 3]`."""
        if (
            not self.complete
            or not isinstance(frames, torch.Tensor)
            or frames.dtype != torch.uint8
            or frames.device.type != "cpu"
            or len(frames) != self.plan.delivered_frames
        ):
            _refuse("continuation is not a completed decoded segment")
        end_frame = self.plan.delivery_end
        start_frame = end_frame - MAX_CONTEXT_FRAMES
        start, end = audio_sample_range(start_frame, end_frame, sample_rate)
        if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] < end:
            _refuse("completed audio does not cover the delivered continuation window")
        return DeliveredAVWindow(
            frames[-MAX_CONTEXT_FRAMES:].numpy().copy(),
            audio[:, start:end].float().cpu().clone(),
            sample_rate,
            start_frame,
            end_frame,
            provenance,
        )
