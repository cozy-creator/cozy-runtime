"""Each ported Inductor decision, measured on its own at the production shape.

Interleaved A/B (see ab.py). Every arm's numerics are checked against the eager arm
before it is timed: a faster kernel that changes the output is not a win.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ab
import ports
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _geglu_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    """Inductor's `triton_poi_fused_gelu_mul_split_view`, hand-written: chunk, gelu and
    multiply in one pass instead of eager's two kernels over the 2*dim projection."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row.to(tl.int64) * (2 * n_cols)
    a = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + base + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    gelu = 0.5 * b * (1.0 + libdevice.erf(b * 0.7071067811865476))
    tl.store(out_ptr + row.to(tl.int64) * n_cols + cols, (a * gelu).to(tl.float16), mask=mask)


S, C, H, D = 4096, 2048, 16, 128
DT = torch.bfloat16


def err(a: torch.Tensor, b: torch.Tensor) -> str:
    diff = (a.float() - b.float()).abs()
    scale = b.float().abs().mean().clamp_min(1e-12)
    return f"max|Δ| {diff.max().item():.3e}  mean|Δ|/mean|ref| {(diff.mean() / scale).item():.3e}"


# ---------------------------------------------------------------------------- M1: AdaLN-Zero


def m1_adaln(reps: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    resid = torch.randn(1, S, C, device="cuda", dtype=DT, generator=g)
    branch = torch.randn(1, S, C, device="cuda", dtype=DT, generator=g)
    mod_prev = torch.randn(1, 3 * C, device="cuda", dtype=DT, generator=g)
    mod_next = torch.randn(1, 3 * C, device="cuda", dtype=DT, generator=g)
    norm = torch.nn.LayerNorm(C, elementwise_affine=False, eps=1e-6).cuda()

    def eager() -> torch.Tensor:
        _, _, gate = mod_prev.chunk(3, dim=-1)
        shift, scale, _ = mod_next.chunk(3, dim=-1)
        h = torch.addcmul(resid, gate.unsqueeze(1), branch)
        return norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def fused() -> torch.Tensor:
        return ports.gate_add_adaln(resid.clone(), branch, mod_prev, mod_next)

    reference = eager()
    print(f"  numerics: {err(fused(), reference)}")
    results = ab.interleaved([("eager 4-kernel", eager), ("triton fused", fused)], reps=reps)
    ab.report(
        f"M1  AdaLN-Zero residual+LayerNorm+modulate, (1,{S},{C}) bf16", results, "eager 4-kernel"
    )


# ------------------------------------------------------------------- M2: repeat_interleave(1)


def m2_repeat_interleave(reps: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    proj = torch.randn(1, S, H * D, device="cuda", dtype=DT, generator=g)
    q = proj.unflatten(2, (H, -1)).transpose(1, 2)
    k = torch.empty_like(q).copy_(q)
    v = torch.empty_like(q).copy_(q)

    def with_ri() -> torch.Tensor:
        kk = k.repeat_interleave(1, dim=3)
        vv = v.repeat_interleave(1, dim=3)
        return torch.nn.functional.scaled_dot_product_attention(q, kk, vv)

    def without_ri() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    print(f"  numerics: {err(without_ri(), with_ri())}")
    print(
        f"  k stride with ri {tuple(k.repeat_interleave(1, dim=3).stride())} vs {tuple(k.stride())}"
    )
    results = ab.interleaved(
        [("with repeat_interleave", with_ri), ("without", without_ri)], reps=reps
    )
    ab.report(f"M2  self-attention, ratio-1 repeat_interleave, (1,{H},{S},{D}) bf16", results)


# ------------------------------------------------------------------- M3: AdaLN LoRA GEMM batch


def m3_adaln_gemms(reps: int, layers: int = 28) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    count = 3 * layers
    rank = 256
    t = torch.randn(1, C, device="cuda", dtype=DT, generator=g)
    temb = torch.randn(1, 3 * C, device="cuda", dtype=DT, generator=g)
    w1 = [torch.randn(rank, C, device="cuda", dtype=DT, generator=g) for _ in range(count)]
    w2 = [torch.randn(3 * C, rank, device="cuda", dtype=DT, generator=g) for _ in range(count)]
    w1c = torch.cat(w1, dim=0).contiguous()
    w2s = torch.stack(w2, dim=0).contiguous()

    def per_norm() -> torch.Tensor:
        act = torch.nn.functional.silu(t)
        outs = [
            torch.nn.functional.linear(torch.nn.functional.linear(act, a), b) + temb
            for a, b in zip(w1, w2, strict=True)
        ]
        return torch.cat(outs, dim=0)

    def batched() -> torch.Tensor:
        act = torch.nn.functional.silu(t)
        low = (act @ w1c.t()).view(count, 1, rank)
        return torch.bmm(low, w2s.transpose(1, 2)).squeeze(1) + temb

    print(f"  numerics: {err(batched(), per_norm())}")
    results = ab.interleaved(
        [(f"{2 * count} small GEMMs", per_norm), ("2 batched GEMMs", batched)], reps=reps
    )
    ab.report(f"M3  AdaLN LoRA modulation for {layers} blocks (batch 1)", results)


# ----------------------------------------------------------------------- M4: SDXL GEGLU fusion


def m4_geglu(reps: int) -> None:
    import triton

    rows, dim = 2 * 4096, 1280
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(rows, 2 * dim, device="cuda", dtype=torch.float16, generator=g)

    def eager() -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * torch.nn.functional.gelu(b)

    def fused() -> torch.Tensor:
        out = torch.empty(rows, dim, device="cuda", dtype=torch.float16)
        _geglu_kernel[(rows,)](x, out, dim, BLOCK=triton.next_power_of_2(dim), num_warps=8)
        return out

    print(f"  numerics: {err(fused(), eager())}")
    results = ab.interleaved([("eager chunk+gelu+mul", eager), ("triton fused", fused)], reps=reps)
    ab.report(f"M4  GEGLU, ({rows},{2 * dim}) fp16", results)


# ---------------------------------------------- M5: which SDPA backend the dispatcher picks


def m5_sdpa_backend(reps: int) -> None:
    """The default SDPA dispatch is not the fastest backend at Anima's attention shape."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    g = torch.Generator(device="cuda").manual_seed(0)
    proj = torch.randn(1, S, H * D, device="cuda", dtype=DT, generator=g)
    q = proj.unflatten(2, (H, -1)).transpose(1, 2)
    k = torch.empty_like(q).copy_(q)
    v = torch.empty_like(q).copy_(q)

    def run(backend: object) -> object:
        def call() -> torch.Tensor:
            if backend is None:
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
            with sdpa_kernel(backend):  # type: ignore[arg-type]
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)

        return call

    default = run(None)
    flash = run(SDPBackend.FLASH_ATTENTION)
    cudnn = run(SDPBackend.CUDNN_ATTENTION)
    efficient = run(SDPBackend.EFFICIENT_ATTENTION)
    print(f"  flash vs default numerics: {err(flash(), default())}")
    results = ab.interleaved(
        [
            ("default dispatch", default),
            ("flash", flash),
            ("cudnn", cudnn),
            ("mem-efficient", efficient),
        ],
        reps=reps,
    )
    ab.report(f"M5  SDPA backend, (1,{H},{S},{D}) bf16 BSHD-strided", results, "default dispatch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    print(ab.device_note())
    chosen = set(args.only.split(",")) if args.only else None
    for name, fn in (
        ("m1", m1_adaln),
        ("m2", m2_repeat_interleave),
        ("m3", m3_adaln_gemms),
        ("m4", m4_geglu),
        ("m5", m5_sdpa_backend),
    ):
        if chosen and name not in chosen:
            continue
        print(f"\n--- {name}")
        fn(args.reps)


if __name__ == "__main__":
    main()
