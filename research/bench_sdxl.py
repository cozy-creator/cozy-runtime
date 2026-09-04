"""SDXL UNet A/B: the two layout/pointwise decisions Inductor made that eager does not.

channels_last   Inductor ran the ENTIRE conv stack in NHWC (every conv activation buffer
                is `(N, 1, W*C, C)`-strided and every conv weight is transposed into
                NHWC by a generated kernel before the cuDNN call). Eager runs NCHW.
rescale div     `AttnProcessor2_0` ends with `hidden_states / attn.rescale_output_factor`
                where the factor is 1.0 — a full elementwise pass per attention that
                Inductor folded into a neighbouring fusion for free.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ab
import sdxl_model


class NoRescaleProcessor:
    """AttnProcessor2_0 without the `/ rescale_output_factor` pass (the factor is 1.0)."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if attn.rescale_output_factor != 1.0 or attn.spatial_norm is not None:
            raise ValueError("this port covers the plain SDXL attention only")
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        batch = hidden_states.shape[0]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, hidden_states.shape[1], batch
            )
            attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if attn.norm_cross and encoder_hidden_states is not None:
            context = attn.norm_encoder_hidden_states(context)
        query = attn.to_q(hidden_states)
        key = attn.to_k(context)
        value = attn.to_v(context)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


def install_no_rescale(model: Any) -> int:
    from diffusers.models.attention_processor import Attention

    n = 0
    for module in model.modules():
        if isinstance(module, Attention):
            module.set_processor(NoRescaleProcessor())
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--latent-hw", type=int, default=128)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    print(ab.device_note())
    print(f"SDXL UNet2DConditionModel, fp16, batch {args.batch}, latent {args.latent_hw}^2, warm")

    model = sdxl_model.build()
    kwargs = sdxl_model.inputs(latent_hw=args.latent_hw, batch=args.batch)
    from diffusers.models.attention_processor import Attention

    attentions = sum(1 for m in model.modules() if isinstance(m, Attention))
    print(f"  {attentions} Attention modules -> {attentions} rescale divisions per forward")

    def call() -> None:
        with torch.no_grad():
            model(**kwargs)

    with torch.no_grad():
        reference = model(**kwargs)[0].float().clone()

    arms: list[tuple[str, Any, Any]] = [("nchw", call, None)]

    def to_nhwc() -> None:
        model.to(memory_format=torch.channels_last)
        kwargs["sample"] = kwargs["sample"].contiguous(memory_format=torch.channels_last)

    def to_nchw() -> None:
        model.to(memory_format=torch.contiguous_format)
        kwargs["sample"] = kwargs["sample"].contiguous(memory_format=torch.contiguous_format)

    arms[0] = ("nchw", call, to_nchw)
    arms.append(("nhwc (channels_last)", call, to_nhwc))

    to_nhwc()
    with torch.no_grad():
        out = model(**kwargs)[0].float()
    print(
        f"  nhwc  max|Δ| {(out - reference).abs().max().item():.3e}  "
        f"mean|Δ|/mean|ref| {((out - reference).abs().mean() / reference.abs().mean()).item():.3e}"
    )
    del out

    n = install_no_rescale(model)
    with torch.no_grad():
        out = model(**kwargs)[0].float()
    print(f"  nhwc+no-rescale ({n} attns)  max|Δ| {(out - reference).abs().max().item():.3e}")
    del out
    arms.append(("nhwc + no rescale div", call, None))

    if args.compile:
        start = time.perf_counter()
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled(**kwargs)
        torch.cuda.synchronize()
        print(f"  cold compile {time.perf_counter() - start:.1f} s")

        def ccall() -> None:
            with torch.no_grad():
                compiled(**kwargs)

        arms.append(("compiled (nhwc)", ccall, None))

    torch.cuda.empty_cache()
    results = ab.interleaved(arms, reps=args.reps, inner=args.inner)
    ab.report(f"SDXL UNet forward (batch {args.batch}, {args.latent_hw}^2 latent)", results, "nchw")
    print(f"  peak CUDA memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
