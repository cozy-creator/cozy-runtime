"""Owned upstream replay and two isolated controls; no process-wide patching."""

from __future__ import annotations

import hashlib
from pathlib import Path

API_SHA256 = "9aec83f8d7165e3b563ba4fc91f6ca5e09b07ef3445b6db5d1105912e35dfb90"
VARIANTS = ("upstream", "token_last", "fixed_means", "fixed_all")


def verify_api(api):
    path = Path(api.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != API_SHA256:
        raise RuntimeError("Sage3 preprocessing API differs from the reviewed source")
    return {"path": str(path), "sha256": actual}


def pad_128(value):
    import torch.nn.functional as functional

    pad = (-value.shape[2]) % 128
    return (
        functional.pad(value, (0, 0, 0, pad), value=0).contiguous() if pad else value.contiguous()
    )


def upstream_preprocess(q, k, v, *, token_last=False):
    """Exact global API operation order with exposed means and correction rounding."""
    import torch

    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    k = k.clone(memory_format=torch.contiguous_format)
    if token_last:
        km = k.transpose(-1, -2).contiguous().mean(dim=-1, keepdim=True).transpose(-1, -2)
    else:
        km = k.mean(dim=-2, keepdim=True)
    k -= km
    q, k, v = (pad_128(x) for x in (q, k, v))
    if token_last:
        qm = q.transpose(-1, -2).contiguous().mean(dim=-1, keepdim=True).transpose(-1, -2)
    else:
        qm = q.mean(dim=-2, keepdim=True)
    q = q - qm
    delta_bf16 = torch.matmul(qm, k.transpose(-2, -1))
    return {
        "q": q,
        "k": k,
        "v": v,
        "qm": qm.contiguous(),
        "km": km.contiguous(),
        "delta_bf16": delta_bf16,
        "delta": delta_bf16.to(torch.float32).contiguous(),
    }


def preprocess(values, variant):
    if variant in ("upstream", "token_last"):
        return upstream_preprocess(*values, token_last=variant == "token_last")
    if variant not in VARIANTS:
        raise ValueError("Unknown preprocessing variant")
    import torch

    from .kernels import fixed_preprocess

    stages = fixed_preprocess(*values)
    if variant == "fixed_means":
        # Keep the same fixed mean/centering candidate but isolate the correction.
        delta = torch.matmul(stages["qm"], stages["k"].transpose(-2, -1))
        stages["delta"] = delta.to(torch.float32).contiguous()
        stages["delta_bf16"] = delta
    else:
        stages["delta_bf16"] = stages["delta"].to(torch.bfloat16)
    return stages


def quantize(api, stages):
    import torch

    quantized = {}
    batch, heads, tokens, dim = stages["q"].shape
    if (batch, dim) != (1, 128) or tokens % 128:
        raise RuntimeError("Unexpected padded Sage3 quantizer geometry")
    for name, function in (
        ("q", api.scale_and_quant_fp4),
        ("k", api.scale_and_quant_fp4_permute),
        ("v", api.scale_and_quant_fp4_transpose),
    ):
        payload, scales = function(stages[name])
        payload_shape = (
            (batch, heads, dim, tokens // 2) if name == "v" else (batch, heads, tokens, dim // 2)
        )
        scale_shape = (
            (batch, heads, dim, tokens // 16) if name == "v" else (batch, heads, tokens, dim // 16)
        )
        if (
            tuple(payload.shape) != payload_shape
            or tuple(scales.shape) != scale_shape
            or payload.dtype != torch.uint8
            or scales.dtype != torch.float8_e4m3fn
        ):
            raise RuntimeError("Pinned quantizer changed head-axis layout or representation")
        quantized[name + "_payload"] = payload
        quantized[name + "_scales"] = scales
    return quantized


def finish(api, stages, quantized, tokens):
    pairs = [
        tuple(quantized[name + suffix] for suffix in ("_payload", "_scales"))
        for name in ("q", "k", "v")
    ]
    output = api.blockscaled_fp4_attn(*pairs, stages["delta"], tokens, False, False, True)[0]
    return output[:, :, :tokens, :].transpose(1, 2).contiguous()


def complete(api, values, variant):
    stages = preprocess(values, variant)
    quantized = quantize(api, stages)
    return finish(api, stages, quantized, values[0].shape[1])


def fixed_complete(api, values):
    """Lean candidate attention: release centered buffers after each public quantizer."""
    from .kernels import fixed_preprocess

    tokens = values[0].shape[1]
    prepared = fixed_preprocess(*values)
    del prepared["qm"], prepared["km"]
    qlist = api.scale_and_quant_fp4(prepared.pop("q"))
    klist = api.scale_and_quant_fp4_permute(prepared.pop("k"))
    vlist = api.scale_and_quant_fp4_transpose(prepared.pop("v"))
    output = api.blockscaled_fp4_attn(
        qlist, klist, vlist, prepared["delta"], tokens, False, False, True
    )[0]
    return output[:, :, :tokens, :].transpose(1, 2).contiguous()
