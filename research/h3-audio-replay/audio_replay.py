"""Replay a verified normalized audio latent through the unchanged public H3 decoder."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Annotated, Any, Literal

import msgspec
import torch
from diffusers.modular_pipelines.modular_pipeline import PipelineState

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    InvalidRequest,
    Outputs,
    Telemetry,
)
from cozy_runtime.models.minimax_h3.model import H3TurboBase
from cozy_runtime.models.minimax_h3.official import NumericalChecks

app = App()


class ReplayInput(msgspec.Struct):
    latent: Annotated[
        FileAsset, AssetBound(max_bytes=4 << 20, media_types=("application/octet-stream",))
    ]
    sha256: str
    shape: tuple[int, int, int]
    dtype: Literal["torch.float32"] = "torch.float32"
    byte_order: Literal["little", "big"] = "little"


Waveform = Annotated[
    FileAsset, AssetBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
]


class ReplayOutput(msgspec.Struct):
    baseline_0: Waveform
    baseline_1: Waveform
    baseline_2: Waveform
    deterministic_0: Waveform
    deterministic_1: Waveform
    deterministic_2: Waveform
    diagnostics: Annotated[
        FileAsset, AssetBound(max_bytes=64 << 10, media_types=("application/json",))
    ]


def restore(payload: ReplayInput, device: torch.device) -> torch.Tensor:
    """Validate the existing artifact before allocating its bounded tensor on the device."""
    if payload.dtype != "torch.float32" or payload.byte_order != sys.byteorder:
        raise InvalidRequest("Replay requires native-byte-order float32 latent bytes")
    if payload.shape[:2] != (2, 32) or not 1 <= payload.shape[2] <= 2048:
        raise InvalidRequest("Replay requires a bounded stereo [2,32,T] audio latent")
    raw = payload.latent.read_bytes()
    if len(raw) != 2 * 32 * payload.shape[2] * 4:
        raise InvalidRequest("Audio latent byte length does not match its shape")
    if hashlib.sha256(raw).hexdigest() != payload.sha256:
        raise InvalidRequest("Audio latent bytes do not match the supplied digest")
    host = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(payload.shape)
    if not bool(torch.isfinite(host).all()):
        raise InvalidRequest("Audio latent contains nonfinite values")
    return host.to(device=device, copy=True)


def snapshot(value: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    host = value.detach().to(device="cpu", copy=True).contiguous()
    raw = host.view(torch.uint8).numpy().tobytes()
    return raw, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "byte_order": sys.byteorder,
        "layout": "C contiguous",
        "bytes": len(raw),
        "nonfinite_elements": int((~torch.isfinite(host)).sum().item()),
    }


def numerical_context() -> dict[str, Any]:
    """Read the serving context without changing any numerical setting."""
    return {
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_benchmark_limit": torch.backends.cudnn.benchmark_limit,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision,
        "cudnn_depthwise_kernel": torch.backends.cudnn.depthwise_kernel,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
        "cuda_autocast_dtype": str(torch.get_autocast_dtype("cuda")),
    }


def deterministic_decode_scope():
    """Change only deterministic/benchmark, preserving every other flags() argument."""
    cudnn = torch.backends.cudnn
    return cudnn.flags(
        enabled=cudnn.enabled,
        benchmark=False,
        benchmark_limit=cudnn.benchmark_limit,
        deterministic=True,
        allow_tf32=cudnn.allow_tf32,
        fp32_precision=cudnn.fp32_precision,
        depthwise_kernel=cudnn.depthwise_kernel,
    )


def memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


@app.entrypoint(
    defaults={
        "model": [{"gpu": "H100", "gpus": 1, "lane": "paul/minimax-h3@1.0.0-rc.3/fp8-pruned"}]
    },
    components={"model": ("audio_vae",)},
)
def audio_replay(
    ctx: Context,
    payload: ReplayInput,
    model: H3TurboBase,
    out: Outputs,
    tel: Telemetry,
) -> ReplayOutput:
    """Three baseline and three scoped deterministic decodes, without denoising or video."""
    ctx.raise_if_cancelled()
    latent = restore(payload, torch.device(str(ctx.device)))
    state = PipelineState({"audio_latents": latent, "image": None, "last_image": None})
    checks = NumericalChecks(tel, model.pipe.resident)
    before = numerical_context()
    _, latent_facts = snapshot(latent)
    outputs = {}
    rows = []
    custody = []

    def decode_three(case: str) -> None:
        for index in range(3):
            label = f"{case}_{index}"
            ctx.raise_if_cancelled()
            before_memory = memory(latent.device)
            with tel.stage("decode_audio_" + label):
                started = time.perf_counter()
                audio, rate = model.decode_audio("fl2va_turbo", state, checks=checks)
                waveform = audio[0].to(torch.float32).contiguous().cpu()
                elapsed = time.perf_counter() - started
            after_memory = memory(latent.device)
            raw, facts = snapshot(waveform)
            outputs[label] = raw
            rows.append(
                {
                    "label": label,
                    "case": case,
                    "index": index,
                    "sample_rate": rate,
                    "decode_and_cpu_landing_seconds": elapsed,
                    "memory_before": before_memory,
                    "memory_after": after_memory,
                    "numerical_context": numerical_context(),
                    **facts,
                }
            )
            _, after = snapshot(state.audio_latents)
            custody.append(
                {"after": label, "unchanged": after["sha256"] == payload.sha256, **after}
            )

    decode_three("baseline")
    before_scope = numerical_context()
    with deterministic_decode_scope():
        inside_scope = numerical_context()
        decode_three("deterministic")
    restored = numerical_context()
    report = {
        "format": "h3-audio-replay/1",
        "input_latent": latent_facts,
        "latent_custody": custody,
        "decoded_audio": rows,
        "within_case_bytes_equal": {
            name: len({outputs[f"{name}_{i}"] for i in range(3)}) == 1
            for name in ("baseline", "deterministic")
        },
        "numerical_context_before": before,
        "numerical_context_before_scope": before_scope,
        "numerical_context_inside_scope": inside_scope,
        "numerical_context_restored": restored,
        "context_restored_exactly": restored == before_scope,
        "process": {
            "pid": os.getpid(),
            "device": str(latent.device),
            "device_uuid": str(getattr(torch.cuda.get_device_properties(latent.device), "uuid", ""))
            if latent.is_cuda
            else None,
        },
        "scope": "Existing H3TurboBase.decode_audio, including official mean/std, called three "
        "times unchanged and three with scoped cuDNN deterministic=True/benchmark=False. "
        "Only audio_vae selected; no conditioning, denoise, video decode or precision changes.",
        "timing_limit": "The first call of each case may include cold algorithm/setup work; "
        "Retain every sample and compare warmed calls separately. This is decoder time only. "
        "Memory peaks are owned/reset by Runtime component entry.",
    }
    return ReplayOutput(
        **{
            name: out.save_bytes(raw, media_type="application/octet-stream")
            for name, raw in outputs.items()
        },
        diagnostics=out.save_bytes(
            json.dumps(report, sort_keys=True, allow_nan=False).encode(),
            media_type="application/json",
        ),
    )
