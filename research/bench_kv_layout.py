"""Inductor's asymmetric attention layout, tested against ours.

Inductor writes Q in the projection's own `(B, S, H, D)`-physical layout (stride
`(S*H*D, D, H*D, 1)`, zero copies) but materializes K and V CONTIGUOUS in `(B, H, S, D)`,
and flash attention's output then comes back Q-shaped so the `flatten(2, 3)` into the
out-projection is a free view.

The shipping plan writes both Q and K through `torch.empty_like`, i.e. both in
`(B, S, H, D)`-physical, and then diffusers' ratio-1 `repeat_interleave` copies K and V to
contiguous `(B, H, S, D)` anyway. The kernel already takes output strides as arguments, so
"write K the way Inductor wants it" costs nothing but an allocation choice. This measures
the three combinations at Anima's exact self-attention shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ab

from cozy_runtime._kernels import _C

B, H, S, D = 1, 16, 4096, 128
EPS = 1e-5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=21)
    args = ap.parse_args()
    print(ab.device_note())
    print(f"self-attention shape (B,H,S,D)=({B},{H},{S},{D}) bf16")

    g = torch.Generator(device="cuda").manual_seed(0)

    def projection() -> Any:
        flat = torch.randn(B, S, H * D, device="cuda", dtype=torch.bfloat16, generator=g)
        return flat.unflatten(2, (H, -1)).transpose(1, 2)

    q_in, k_in, value = projection(), projection(), projection()
    cos = torch.randn(S, D // 2, device="cuda", generator=g)
    sin = torch.randn(S, D // 2, device="cuda", generator=g)
    freqs = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, S, D // 2, 2, 2)
    q_scale = torch.randn(D, device="cuda", dtype=torch.bfloat16, generator=g)
    k_scale = torch.randn(D, device="cuda", dtype=torch.bfloat16, generator=g)

    def kernel(q_out: Any, k_out: Any) -> None:
        _C.rms_rope_split_half(
            q_in,
            k_in,
            freqs,
            q_scale,
            k_scale,
            q_out,
            k_out,
            EPS,
            torch.cuda.current_stream(q_in.device).cuda_stream,
            0,
        )

    def like() -> Any:
        return torch.empty_like(q_in)

    def contiguous() -> Any:
        return torch.empty(B, H, S, D, device="cuda", dtype=torch.bfloat16)

    def plan() -> Any:
        """What ships today: both outputs `empty_like`, then ratio-1 repeat_interleave."""
        q_out, k_out = like(), like()
        kernel(q_out, k_out)
        k = k_out.repeat_interleave(1, dim=3)
        v = value.repeat_interleave(1, dim=3)
        return torch.nn.functional.scaled_dot_product_attention(q_out, k, v)

    def no_repeat() -> Any:
        q_out, k_out = like(), like()
        kernel(q_out, k_out)
        return torch.nn.functional.scaled_dot_product_attention(q_out, k_out, value)

    def inductor_layout() -> Any:
        """Q in the projection's layout, K written contiguous, V copied contiguous once."""
        q_out, k_out = like(), contiguous()
        kernel(q_out, k_out)
        return torch.nn.functional.scaled_dot_product_attention(q_out, k_out, value.contiguous())

    def inductor_layout_strided_v() -> Any:
        q_out, k_out = like(), contiguous()
        kernel(q_out, k_out)
        return torch.nn.functional.scaled_dot_product_attention(q_out, k_out, value)

    reference = plan()
    for name, fn in (
        ("no_repeat", no_repeat),
        ("inductor_layout", inductor_layout),
        ("inductor_layout_strided_v", inductor_layout_strided_v),
    ):
        out = fn()
        diff = (out.float() - reference.float()).abs()
        print(f"  {name:26s} max|Δ| {diff.max().item():.3e}   out stride {tuple(out.stride())}")
    del reference

    results = ab.interleaved(
        [
            ("plan (empty_like + repeat_interleave)", plan),
            ("no repeat_interleave", no_repeat),
            ("inductor layout (K,V contiguous)", inductor_layout),
            ("inductor K contiguous, V strided", inductor_layout_strided_v),
        ],
        reps=args.reps,
    )
    ab.report("fused QK-norm/RoPE + flash attention, one self-attention", results)


if __name__ == "__main__":
    main()
