"""Separately replay upstream's exposed Q/K quantizer calls; never patch attention."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any

SOURCE_HASHES = {
    "core.py": "e475182fc55d6a683499e1eb4d8886fc06e050ed938d8563b090773eede2a99e",
    "quant.py": "86bf415d4d24793a591c2a056e60f41b4bc579519ac7a875e484a65816940bf7",
    "triton/quant_per_thread.py": (
        "e8aa4dfc16780326b90a9235439a0cb77875ff75d7c5376a1aac6adc82f4d643"
    ),
}


def parameters(mode: str) -> tuple[str, dict[str, int | str]]:
    """Exact core.py call arguments for NHD, D128 and the three requested modes."""
    if mode == "per-warp":
        return "per_warp_int8_cuda", {"tensor_layout": "NHD", "BLKQ": 64, "WARPQ": 16, "BLKK": 128}
    if mode == "accurate":
        return "per_thread_int8_triton", {
            "tensor_layout": "NHD",
            "BLKQ": 64,
            "WARPQ": 16,
            "BLKK": 128,
            "WARPK": 128,
        }
    if mode == "fp16pv":
        return "per_thread_int8_triton", {
            "tensor_layout": "NHD",
            "BLKQ": 128,
            "WARPQ": 32,
            "BLKK": 64,
            "WARPK": 64,
        }
    raise ValueError(f"No Sage preprocessing mode: {mode}")


def load() -> tuple[Any, dict[str, str]]:
    core = importlib.import_module("sageattention.core")
    root = Path(inspect.getfile(core)).parent
    actual = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_HASHES
    }
    if actual != SOURCE_HASHES:
        raise RuntimeError("Upstream preprocessing differs from the reviewed pinned source")
    return core, actual


def active_scale_indices(torch: Any, scale: Any, tokens: int, block: int, lanes: int) -> Any:
    """Exclude quantizer slots whose first token is already outside the live sequence.

    Each per-thread Q scale serves residue t within a WARPQ group (8 lanes);
    K serves residues 2t and 2t+1 within WARPK (4 lanes). Per-warp/block scales
    use one contiguous token group. Unused padded scale slots are not operands
    of any live query/key and their unspecified bits are not an identity claim.
    """
    index = torch.arange(scale.shape[-1], device=scale.device)
    first = (index // lanes) * block + index % lanes * (2 if lanes == 4 else 1)
    return index[first < tokens]


def run(torch: Any, core: Any, mode: str, q: Any, k: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if q.shape != k.shape or q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("Preprocessing inspection is the equal-head B1/D128 NHD fixture only")
    name, options = parameters(mode)
    # This is core.py's exact smooth_k expression, with the same BF16 inputs.
    mean = k.mean(dim=1, keepdim=True)
    qi, qs, ki, ks = getattr(core, name)(q, k, mean, **options)
    thread = name == "per_thread_int8_triton"
    q_indices = active_scale_indices(
        torch, qs, q.shape[1], int(options["WARPQ"]), 8 if thread else 1
    )
    k_indices = active_scale_indices(
        torch,
        ks,
        k.shape[1],
        int(options["WARPK"] if thread else options["BLKK"]),
        4 if thread else 1,
    )
    output = {
        "k_mean": mean[0].transpose(0, 1).contiguous(),
        "q_codes": qi[0].transpose(0, 1).contiguous(),
        "k_codes": ki[0].transpose(0, 1).contiguous(),
        "q_scales_live": qs[0].index_select(-1, q_indices).contiguous(),
        "k_scales_live": ks[0].index_select(-1, k_indices).contiguous(),
    }
    evidence = {
        "helper": name,
        "arguments": options,
        "q_scale_shape": list(qs.shape),
        "k_scale_shape": list(ks.shape),
        "q_unused_scale_slots_per_head": qs.shape[-1] - q_indices.numel(),
        "k_unused_scale_slots_per_head": ks.shape[-1] - k_indices.numel(),
        "scope": (
            "Separate execution of the exact K mean and exposed upstream Q/K quantizer; "
            "not an in-call capture. V/P conversion and attention accumulation are not inspected."
        ),
    }
    return output, evidence
