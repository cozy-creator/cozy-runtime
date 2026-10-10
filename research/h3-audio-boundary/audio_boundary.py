"""Research entrypoint: ordinary H3 Turbo denoise, audio decode, and one decode replay.

This is not a video benchmark. The caller supplies the unchanged captured H3 package;
its normal model bindings, scheduler, attention selection, and denoise method remain
authoritative. Only audio boundary copies and output assets are added.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Annotated, Any

import h3
import msgspec
import torch

from cozy_runtime.author import AssetBound, Context, FileAsset, Outputs, Telemetry
from cozy_runtime.models.minimax_h3.model import H3TurboBase, H3TurboLoRA

app = h3.app


class AudioBoundaryOutput(msgspec.Struct):
    final_audio_latent: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/octet-stream",))
    ]
    original_audio: Annotated[
        FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
    ]
    repeat_audio: Annotated[
        FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
    ]
    diagnostics: Annotated[
        FileAsset, AssetBound(max_bytes=64 << 10, media_types=("application/json",))
    ]


def snapshot(value: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    """Own canonical CPU bytes without changing dtype, values, or the source tensor."""
    host = value.detach().to(device="cpu", copy=True).contiguous()
    payload = host.view(torch.uint8).numpy().tobytes()
    return payload, {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "byte_order": sys.byteorder,
        "layout": "C contiguous",
        "bytes": len(payload),
        "nonfinite_elements": int((~torch.isfinite(host)).sum().item()),
        "source_stride": list(value.stride()),
        "source_device": str(value.device),
    }


def audio_finish(
    model: H3TurboBase,
    state: Any,
    schedule: Any,
    *,
    ctx: Context,
    out: Outputs,
    tel: Telemetry,
    checks: h3.NumericalChecks,
    seed: int | None,
) -> AudioBoundaryOutput:
    ctx.raise_if_cancelled()
    latent_bytes, latent = snapshot(state.audio_latents)
    decoded = []
    payloads = []
    custody = []
    for label in ("original", "repeat"):
        ctx.raise_if_cancelled()
        with tel.stage("decode_audio_" + label):
            started = time.perf_counter()
            audio, sample_rate = model.decode_audio("fl2va_turbo", state, checks=checks)
            if audio.ndim != 3 or int(audio.shape[0]) != 1:
                raise h3.OutputError("H3 audio boundary requires batch-one waveform")
            if sample_rate != model.pipe.sample_rate:
                raise h3.OutputError("H3 audio boundary sample rate changed")
            # This is the same canonical waveform landing as the full H3 finish.
            waveform = audio[0].to(torch.float32).contiguous().cpu()
            elapsed = time.perf_counter() - started
        # waveform is already CPU; this creates no additional GPU transfer.
        payload, facts = snapshot(waveform)
        payloads.append(payload)
        decoded.append(
            {
                "label": label,
                "sample_rate": sample_rate,
                "decode_and_cpu_landing_seconds": elapsed,
                **facts,
            }
        )
        _, after = snapshot(state.audio_latents)
        custody.append({"after": label, "unchanged": after["sha256"] == latent["sha256"], **after})
    report = {
        "format": "h3-audio-boundary/1",
        "seed": seed,
        "process": {
            "pid": os.getpid(),
            "latent_device": str(state.audio_latents.device),
            "device_uuid": str(
                getattr(torch.cuda.get_device_properties(state.audio_latents.device), "uuid", "")
            )
            if state.audio_latents.is_cuda
            else None,
        },
        "final_audio_latent": latent,
        "latent_stage": "After denoise.after_denoise; before audio mean/std and VAE decode",
        "latent_custody": custody,
        "decoded_audio": decoded,
        "decode_repeat_bytes_equal": payloads[0] == payloads[1],
        "schedule": {
            "timestep_plan_digest": schedule.timestep_plan_digest,
            "video_sigma_digest": schedule.video_sigma_digest,
            "audio_sigma_digest": schedule.audio_sigma_digest,
            "video_timestep_digest": schedule.video_timestep_digest,
            "audio_timestep_digest": schedule.audio_timestep_digest,
            "sigma_grid_points": schedule.sigma_grid_points,
            "transformer_evaluations": schedule.transformer_evaluations,
        },
        "scope": "Full ordinary eight-step H3 Turbo denoise; audio-only finish and one replay. "
        "No video decode or encode, no generation speed claim.",
    }
    return AudioBoundaryOutput(
        final_audio_latent=out.save_bytes(latent_bytes, media_type="application/octet-stream"),
        original_audio=out.save_bytes(payloads[0], media_type="application/octet-stream"),
        repeat_audio=out.save_bytes(payloads[1], media_type="application/octet-stream"),
        diagnostics=out.save_bytes(
            json.dumps(report, sort_keys=True, allow_nan=False).encode(),
            media_type="application/json",
        ),
    )


@app.entrypoint(
    defaults={"base_model": h3._DEFAULT_MODEL_LADDER, "turbo_lora": h3._DEFAULT_TURBO_LORA_LADDER},
    components={"base_model": h3._FL2VA, "turbo_lora": ("fl2va_turbo",)},
)
def audio_boundary(
    ctx: Context,
    payload: h3.ClipInput,
    assets: h3.KeyframeAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> AudioBoundaryOutput:
    """Same text-only full Turbo denoise; expose the final latent and audio decode boundary."""
    ctx.raise_if_cancelled()
    first_index, last_index = h3._keyframe_roles(assets)
    if first_index is not None or last_index is not None:
        raise h3.InvalidRequest("The audio boundary diagnostic accepts text-only requests")
    prompt = h3.encoder_prompt("fl2va_turbo", payload.prompt)
    view = base_model.for_request(ctx, seed=payload.seed)
    checks = h3.NumericalChecks(tel, base_model.pipe.resident)
    h3.log_prompt(tel, "h3 encoder prompt", prompt)
    with tel.stage("prepare", overall_range=(0.00, 0.03)):
        state = base_model.pipe.start_fl2va(
            prompt=prompt,
            first_frame=None,
            last_frame=None,
            generator=base_model.pipe.generator(view.generator),
            steps=h3.TURBO_STEPS,
            frames=h3.frames_for(payload.duration_s),
            task="fl2va_turbo",
        )
    with tel.stage("condition_text", overall_range=(0.03, 0.08)):
        base_model.condition_text("fl2va_turbo", state, checks=checks)
    with tel.stage("denoise", overall_range=(0.15, 0.85)):
        schedule = base_model.sample_fl2va_turbo(
            state,
            turbo_lora=turbo_lora,
            on_step=tel.step_callback(h3.TURBO_STEPS, stage="denoise", overall_range=(0.15, 0.85)),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    return audio_finish(
        base_model, state, schedule, ctx=ctx, out=out, tel=tel, checks=checks, seed=payload.seed
    )
