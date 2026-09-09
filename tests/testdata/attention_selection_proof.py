"""Attention selection against REAL diffusers modules on a REAL card (cr-124, cr-127).

Run as a child process by `tests/test_attention_selection.py`, which skips it where torch,
diffusers 0.40.0 or a CUDA device is missing. Nothing here is faked: the modules are
diffusers' own `Attention` at real head dimensions and dtypes, the pin is read back off the
processors diffusers itself would dispatch through, and one pinned site runs one real forward.

The cr-127 self-check arms plant the h3a-013 defect the only faithful way: by replacing what
diffusers' OWN backend registry dispatches to, so the runtime reaches a kernel that returns its
uninitialized output buffer exactly as the shipped `sageattention` sm90 kernel did — right shape,
right dtype, no exception. Nothing in `cozy_runtime` is stubbed.

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


def _dead(kind: str) -> Any:
    """A backend that does what the shipped sm90 kernel did: return without computing.

    `torch.empty_like` IS the defect — upstream's `register_fake` meta implementation allocated
    the output and returned, so the caller got its uninitialized buffer at the right shape with
    no error raised anywhere.
    """

    def backend(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        if kind == "empty":
            return torch.empty_like(query)
        if kind == "nan":
            return torch.full_like(query, float("nan"))
        # FINITE AND PLAUSIBLE and still not attention: V straight through. This is the arm
        # neither a finiteness test nor an implausible-rate test can catch, and it is the
        # shape h3a-013 actually saw — a right-shaped tensor of the right magnitude that is
        # simply not the answer.
        return value.clone()

    return backend


def _selfcheck(out: dict[str, Any], device: DeviceFacts) -> None:
    """Prove the self-check on this card: it passes real kernels and refuses dead ones."""
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    # GREEN. Every kernel this device and image offer agrees with exact fp32 attention inside
    # its declared budget, measured here rather than asserted.
    measured: dict[str, float] = {}
    for name in ("sdpa", "cudnn"):
        candidate = attention.BY_NAME[name]
        if candidate.min_sm and device.sm < candidate.min_sm:
            continue
        entry = attention._resolve(candidate, device)
        measured[name] = _rel_l2(entry, device)
    out["selfcheck_rel_l2"] = measured
    out["selfcheck_budgets"] = {
        name: attention.BY_NAME[name].rel_l2_max for name in sorted(attention.BY_NAME)
    }

    # RED. Same registry, same dispatch, a backend that computes nothing.
    for kind in ("empty", "nan", "wrong"):
        for name in ("sdpa", "cudnn"):
            candidate = attention.BY_NAME[name]
            if candidate.min_sm and device.sm < candidate.min_sm:
                continue
            member = attention._member(candidate.backend)
            real = _AttentionBackendRegistry._backends[member]
            _AttentionBackendRegistry._backends[member] = _dead(kind)
            try:
                out[f"dead_{kind}_{name}_pin"] = _refusal(
                    lambda pinned=name: attention.pinned(pinned, device)
                )
                if name == "sdpa":
                    # The FLOOR is dead, so auto-selection must offer NOTHING rather than
                    # silently serve a kernel that returns uninitialized memory.
                    out[f"dead_{kind}_auto"] = [r.name for r in attention.available(device)]
                    out[f"dead_{kind}_image_kernels"] = attention.image_kernels(device)
            finally:
                _AttentionBackendRegistry._backends[member] = real

    # And the green arm again AFTER the plants are removed, so a passing red arm cannot be a
    # registry left broken.
    out["selfcheck_restored"] = [r.name for r in attention.available(device)]


def _rel_l2(entry: Any, device: DeviceFacts) -> float:
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    batch, sequence, heads, dim = attention.PROBE
    seed = torch.Generator(device="cuda").manual_seed(0x5A6E)
    q, k, v = (
        (
            torch.randn(
                (batch, sequence, heads, dim), generator=seed, device="cuda", dtype=torch.float32
            )
            * 0.5
        ).to(torch.bfloat16)
        for _ in range(3)
    )
    with torch.no_grad():
        got = dispatch_attention_fn(q, k, v, backend=entry.member).float()
    exact = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v))
    ).transpose(1, 2)
    return float((got - exact).norm() / exact.norm())


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

    # AUTO. On a box without the baked snapshot this is the floor, and the floor takes both.
    applied = attention.select(device, roots)
    out["auto"] = applied.document()
    out["auto_backends"] = _backends(roots)

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
    out["too_new"] = _refusal(lambda: attention.pinned("sageattention", _device_at(120)))

    # ONE REAL FORWARD through a pinned site, so the pin is a thing that computes and not a
    # field that was set.
    attention.install([attention._resolve(attention.BY_NAME["cudnn"], device)], {"dit": dit})
    hidden = torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        result = dit(hidden)
    out["forward"] = {"shape": list(result.shape), "dtype": str(result.dtype)}

    _selfcheck(out, device)

    print(json.dumps(out))


def _device_at(sm: int) -> DeviceFacts:
    return DeviceFacts(kind="cuda", name="card", sm=sm, driver="580", configuration="", index=0)


if __name__ == "__main__":
    main()
