"""Attention selection against REAL diffusers modules on a REAL card (cr-124).

Run as a child process by `tests/test_attention_selection.py`, which skips it where torch,
diffusers 0.40.0 or a CUDA device is missing. Nothing here is faked: the modules are
diffusers' own `Attention` at real head dimensions and dtypes, the pin is read back off the
processors diffusers itself would dispatch through, and one pinned site runs one real forward.

Prints one JSON line.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from diffusers.models.attention_processor import Attention

from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts


def _device() -> DeviceFacts:
    major, minor = torch.cuda.get_device_capability()
    return DeviceFacts(
        kind="cuda",
        name=torch.cuda.get_device_name(0),
        sm=major * 10 + minor,
        driver=str(torch.version.cuda),
        configuration="",
        index=0,
    )


def _site(dim: int, heads: int, dtype: torch.dtype) -> Attention:
    return Attention(query_dim=dim, heads=heads, dim_head=dim // heads).to("cuda", dtype)


def _backends(roots: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for component, _module, processor in attention._sites(roots):
        member = getattr(processor, "_attention_backend", None)
        out.setdefault(component, []).append(getattr(member, "value", str(member)))
    return out


def _refusal(run: Any) -> dict[str, str]:
    try:
        run()
    except attention.AttentionRefusal as exc:
        return {"code": exc.code, "detail": str(exc)[:300]}
    return {"code": "", "detail": "no refusal"}


def main() -> None:
    device = _device()
    out: dict[str, Any] = {
        "sm": device.sm,
        "offered": [r.name for r in attention.available(device)],
    }
    out["image_kernels"] = attention.image_kernels()

    # A real DiT-shaped site (head dim 128) and a real wide site (head dim 256) in one
    # construction, each read back off the processor diffusers would dispatch through.
    dit = _site(1024, 8, torch.bfloat16)
    wide = _site(1024, 4, torch.float16)
    roots = {"dit": dit, "vae": wide}
    out["head_dims"] = {"dit": attention._head_dim(dit), "vae": attention._head_dim(wide)}
    out["dtypes"] = {"dit": attention._dtype(dit), "vae": attention._dtype(wide)}

    # AUTO. On a box without the baked snapshot this is the floor, and the floor takes both.
    applied = attention.select(device, roots)
    out["auto"] = applied.document()
    out["auto_backends"] = _backends(roots)

    # PER-SITE ADMISSION. cuDNN serves head dim 128 and not 256, so one site takes it and
    # the other keeps the floor — the same construction, two kernels, no per-call fallback.
    ready = [
        attention._resolve(attention.BY_NAME["cudnn"]),
        attention._resolve(attention.BY_NAME["sdpa"]),
    ]
    mixed = attention.install(ready, roots)
    out["mixed"] = mixed.document()
    out["mixed_backends"] = _backends(roots)

    # A PIN DOES NOT FALL BACK. cuDNN alone cannot serve the 256-dim site, and that is a
    # typed refusal rather than a quiet demotion to the floor.
    out["pin_unserved"] = _refusal(lambda: attention.install(ready[:1], roots, "cudnn"))
    out["pin_served"] = attention.install(ready[:1], {"dit": dit}, "cudnn").document()
    out["pin_backends"] = _backends({"dit": dit})

    out["unknown"] = _refusal(lambda: attention.pinned("no-such-kernel", device))
    out["too_old"] = _refusal(lambda: attention.pinned("flash-attn3", _device_at(80)))
    out["absent"] = _refusal(lambda: attention.pinned("sageattention", _device_at(90)))

    # ONE REAL FORWARD through a pinned site, so the pin is a thing that computes and not a
    # field that was set.
    attention.install([attention._resolve(attention.BY_NAME["cudnn"])], {"dit": dit})
    hidden = torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        result = dit(hidden)
    out["forward"] = {"shape": list(result.shape), "dtype": str(result.dtype)}

    print(json.dumps(out))


def _device_at(sm: int) -> DeviceFacts:
    return DeviceFacts(kind="cuda", name="card", sm=sm, driver="580", configuration="", index=0)


if __name__ == "__main__":
    main()
