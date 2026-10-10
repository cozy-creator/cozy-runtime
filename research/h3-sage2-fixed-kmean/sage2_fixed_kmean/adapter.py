"""Exact per-thread SM90 upstream stages; only the optional K mean is replaced."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

CORE_SHA = "e475182fc55d6a683499e1eb4d8886fc06e050ed938d8563b090773eede2a99e"


def core():
    module = importlib.import_module("sageattention.core")
    path = Path(module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORE_SHA:
        raise RuntimeError("Pinned upstream SM90 call contract changed")
    return module


def stages(module, q, k, v, *, fixed=False):
    """B1/NHD/D128, BF16, equal Q/KV, noncausal, no LSE; the actual H3 fixture."""
    import torch

    if (
        q.shape != k.shape
        or q.shape != v.shape
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
    ):
        raise ValueError("The replay supports equal B1/NHD/D128 only")
    if any(
        x.device != q.device or x.dtype != torch.bfloat16 or not x.is_contiguous()
        for x in (q, k, v)
    ):
        raise ValueError("Fresh contiguous BF16 operands on one device are required")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("This first fixed-mean comparison admits SM90 only")
    if not module.SM90_ENABLED:
        raise RuntimeError("Actual upstream SM90 extension is absent")
    torch.cuda.set_device(v.device)
    if fixed:
        from .kernels import mean

        km = mean(k)
    else:
        km = k.mean(dim=1, keepdim=True)
    qi, qs, ki, ks = module.per_thread_int8_triton(
        q, k, km, tensor_layout="NHD", BLKQ=64, WARPQ=16, BLKK=128, WARPK=128
    )
    o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
    kv_len = k.size(1)
    pad = 128 - (kv_len % 128) if kv_len % 128 else 0
    if pad > 0:
        v = torch.cat(
            [v, torch.zeros(v.size(0), pad, v.size(2), v.size(3), dtype=v.dtype, device=v.device)],
            dim=1,
        )
    vf, vs, _ = module.per_channel_fp8(v, tensor_layout="NHD", smooth_v=False)
    module.sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        qi, ki, vf, o, qs, ks, vs, 0, 0, 3, 128**-0.5, 0
    )
    return {
        "k_mean": km,
        "q_codes": qi,
        "q_scales": qs,
        "k_codes": ki,
        "k_scales": ks,
        "v_codes": vf,
        "v_scales": vs,
        "output": o,
    }


def forward(module, q, k, v, *, fixed=False):
    # The instrumentation record is dropped before returning the output. It adds no
    # arithmetic or copies to the replay; timing includes these small Python objects.
    return stages(module, q, k, v, fixed=fixed)["output"]


def canonical_record(torch, record, tokens):
    """Head-major bytes; exclude only scale slots with no live token owner."""
    result = {}
    for name in ("k_mean", "q_codes", "k_codes", "output"):
        result[name] = record[name][0].transpose(0, 1).contiguous()
    for name, block, lanes in [("q_scales", 16, 8), ("k_scales", 128, 4)]:
        value = record[name][0]
        idx = torch.arange(value.shape[-1], device=value.device)
        first = idx // lanes * block + idx % lanes * (2 if lanes == 4 else 1)
        result[name + "_live"] = value.index_select(-1, idx[first < tokens]).contiguous()
    # Upstream NHD per-channel FP8 returns [B,D,H,Npad], scales [B,H,D].
    vf = record["v_codes"]
    vs = record["v_scales"]
    if vf.shape[:3] != (1, 128, record["output"].shape[2]) or vs.shape != (
        1,
        record["output"].shape[2],
        128,
    ):
        raise RuntimeError("Unexpected upstream V representation/head axis")
    result["v_codes"] = vf[0].transpose(0, 1).contiguous()
    result["v_scales"] = vs[0].contiguous()
    return result
