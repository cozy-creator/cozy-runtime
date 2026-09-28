"""Attention selection against REAL diffusers modules on a REAL card (cr-124, cr-127).

Run as a child process by `tests/test_attention_selection.py`, which skips it where torch,
diffusers 0.40.0 or a CUDA device is missing. Nothing here is faked: the modules are
diffusers' own `WanAttention` at real head dimensions and dtypes, the pin is read back off the
processors diffusers itself would dispatch through, and one pinned site runs one real forward.

Prints one JSON line.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

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


def _site(dim: int, heads: int, dtype: torch.dtype) -> WanAttention:
    return WanAttention(
        dim=dim, heads=heads, dim_head=dim // heads, processor=WanAttnProcessor()
    ).to("cuda", dtype)


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
    out["image_kernels"] = attention.image_kernels(device)

    # A real DiT-shaped site (head dim 128) and a real wide site (head dim 256) in one
    # construction, each read back off the processor diffusers would dispatch through.
    dit = _site(1024, 8, torch.bfloat16)
    wide = _site(1024, 4, torch.float16)
    roots = {"dit": dit, "vae": wide}
    out["head_dims"] = {"dit": attention._head_dim(dit), "vae": attention._head_dim(wide)}
    out["dtypes"] = {"dit": attention._dtype(dit), "vae": attention._dtype(wide)}

    # AUTO. On a machine that has compiled nothing this is the floor, and it takes both.
    applied = attention.select(device, roots)
    out["auto"] = applied.document()
    out["auto_backends"] = _backends(roots)
    # AUTO on a placement sharded at degree 4 (cr-138): every ranked kernel is head-local, so
    # the Ulysses wrapper shards the same choice.
    out["auto_degree4"] = attention.select(device, roots, degree=4).document()

    # PER-SITE ADMISSION. cuDNN serves head dim 128 and not 256, so one site takes it and
    # the other keeps the floor — the same construction, two kernels, no per-call fallback.
    ready = [
        attention._resolve(attention.BY_NAME["cudnn"], device),
        attention._resolve(attention.BY_NAME["sdpa"], device),
    ]
    mixed = attention.install(ready, roots)
    out["mixed"] = mixed.document()
    out["mixed_backends"] = _backends(roots)

    # A PIN DOES NOT FALL BACK. cuDNN alone cannot serve the 256-dim site, and that is a
    # typed refusal rather than a quiet demotion to the floor.
    out["pin_unserved"] = _refusal(lambda: attention.install(ready[:1], roots, "cudnn"))
    out["pin_served"] = attention.install(ready[:1], {"dit": dit}, "cudnn").document()
    out["pin_backends"] = _backends({"dit": dit})

    # A GROUP PREPARE has one record, merged from its models' own (cr-135): two real
    # constructions pinned to one kernel, each read back off its own processors, and the
    # merge `_prepare_many` runs — which used to drop the pin — renders PINNED.
    second = _site(1024, 8, torch.bfloat16)
    out["group_pinned"] = attention.merge(
        {
            "fl2va": attention.install(ready[:1], {"dit": dit}, "cudnn").document(),
            "ref2va": attention.install(ready[:1], {"dit": second}, "cudnn").document(),
        }
    )
    out["group_auto"] = attention.merge(
        {
            "fl2va": attention.select(device, {"dit": dit}).document(),
            "ref2va": attention.select(device, {"dit": second}).document(),
        }
    )

    out["unknown"] = _refusal(lambda: attention.pinned("no-such-kernel", device))
    out["too_old"] = _refusal(lambda: attention.pinned("flash-attn3", _device_at(80)))
    out["absent"] = _refusal(lambda: attention.pinned("sageattention", _device_at(90)))
    # h3a-022. ABOVE the ceiling, through the real registry: a card outside a kernel's declared
    # range is refused by name, and the kernel is never reached — which is the only outcome
    # that helps when reaching it exits the process from C++ rather than raising.
    out["too_new"] = _refusal(lambda: attention.pinned("sageattention-fp16pv", _device_at(120)))

    # ONE REAL FORWARD through a pinned site, so the pin is a thing that computes and not a
    # field that was set.
    attention.install([attention._resolve(attention.BY_NAME["cudnn"], device)], {"dit": dit})
    hidden = torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        result = dit(hidden)
    out["forward"] = {"shape": list(result.shape), "dtype": str(result.dtype)}

    out["fp8"] = _fp8(device, dit)

    print(json.dumps(out))


def _fp8(device: DeviceFacts, dit: WanAttention) -> dict[str, Any]:
    """cr-136: the fp8 path through the real registry: refused by name where the card cannot
    run it or this machine has not compiled it, otherwise pinned and run."""
    try:
        ready = attention.pinned("flash-attn3-fp8", device)
    except attention.AttentionRefusal as exc:
        return {"code": exc.code, "detail": str(exc)[:300]}
    applied = attention.install(ready, {"dit": dit}, "flash-attn3-fp8")
    hidden = torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        result = dit(hidden)
    return {
        "code": "",
        "backend": _backends({"dit": dit})["dit"][0],
        "line": applied.line(),
        "forward": {"shape": list(result.shape), "dtype": str(result.dtype)},
        "sharded": attention.install(ready, {"dit": dit}, "flash-attn3-fp8", degree=4).totals(),
    }


def _device_at(sm: int) -> DeviceFacts:
    return DeviceFacts(kind="cuda", name="card", sm=sm, driver="580", configuration="", index=0)


if __name__ == "__main__":
    main()
