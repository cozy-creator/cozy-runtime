"""Whole-transformer A/B on ONE weight set: eager, the shipping fused plan, each port.

Every arm is a Python-callable plan installed on the same module tree (parameters and
storage are invariants), so the card holds one model, not one per arm. Installing an arm
happens outside the timed region. Every arm's output is compared against `eager_stock`.
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
import anima_model
import ports


class Arms:
    """Install/restore plans on one shared model."""

    def __init__(self, layers: int, latent_hw: int) -> None:
        self.model = anima_model.build(num_layers=layers)
        self.kwargs = anima_model.inputs(latent_hw=latent_hw)
        self.strict = layers == 28
        self.pristine = self._snapshot()

    def _snapshot(self) -> dict[int, Any]:
        state: dict[int, Any] = {}
        for module in self.model.modules():
            entry: dict[str, Any] = {}
            if "forward" in module.__dict__:
                entry["forward"] = module.__dict__["forward"]
            else:
                entry["forward"] = None
            processor = getattr(module, "processor", None)
            if processor is not None:
                entry["processor"] = processor
            state[id(module)] = entry
        return state

    def restore(self) -> None:
        for module in self.model.modules():
            entry = self.pristine.get(id(module))
            if entry is None:
                continue
            if entry["forward"] is None:
                module.__dict__.pop("forward", None)
            else:
                module.__dict__["forward"] = entry["forward"]
            if (
                "processor" in entry
                and getattr(module, "processor", None) is not entry["processor"]
            ):
                module.set_processor(entry["processor"])
        self.model.__dict__.pop("_repeated_blocks", None)

    def install(self, name: str) -> None:
        self.restore()
        model = self.model
        if name == "eager_stock":
            return
        if name == "stock_no_ri":
            ports.install_no_repeat_interleave(model)
            return
        if name == "stock_fused_adaln":
            ports.install_fused_adaln(model, batched=False, triton_norm=True)
            return
        if name == "stock_batched_adaln":
            ports.install_fused_adaln(model, batched=True, triton_norm=False)
            return
        if name == "stock_all_ports":
            ports.install_no_repeat_interleave(model)
            ports.install_fused_adaln(model, batched=True, triton_norm=True)
            return
        if name.startswith("plan") or name == "eager_plan":
            anima_model.apply_fused_plan(model, via="raw", strict=self.strict)
            if name in ("plan_no_ri", "plan_all_ports"):
                _drop_repeat_interleave(model)
            if name == "plan_fused_adaln":
                ports.install_fused_adaln(model, batched=False, triton_norm=True)
            if name == "plan_batched_adaln":
                ports.install_fused_adaln(model, batched=True, triton_norm=False)
            if name == "plan_all_ports":
                ports.install_fused_adaln(model, batched=True, triton_norm=True)
            return
        if name == "eager_plan_op":
            anima_model.apply_fused_plan(model, via="op", strict=self.strict)
            return
        raise SystemExit(f"unknown arm {name}")

    def call(self) -> None:
        with torch.no_grad():
            self.model(**self.kwargs)

    def output(self) -> torch.Tensor:
        with torch.no_grad():
            return self.model(**self.kwargs)[0]


def _drop_repeat_interleave(model: Any) -> None:
    """Delete the ratio-1 repeat_interleave from the shipping fused self-attention."""
    from cozy_runtime.internal.anima_optimization import _FusedSelfAttention

    class NoRepeat(_FusedSelfAttention):
        def __call__(  # type: ignore[override]
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            image_rotary_emb=None,
        ):
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
            query, key = self.anima_kernels.rms_rope_split_half(
                query,
                key,
                self.rope.get(image_rotary_emb),
                attn.norm_q.weight,
                attn.norm_k.weight,
                attn.norm_q.eps,
            )
            output = dispatch_attention_fn(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            return attn.to_out[1](attn.to_out[0](output.flatten(2, 3).type_as(query)))

    for module in model.modules():
        proc = getattr(module, "processor", None)
        if type(proc) is _FusedSelfAttention:
            module.set_processor(NoRepeat(proc.torch, proc.anima_kernels, proc.rope))
    ports.install_no_repeat_interleave(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--latent-hw", type=int, default=128)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--compile", default="", help="also time: stock | plan_op | repeated_blocks")
    ap.add_argument("arms", nargs="*", default=["eager_stock"])
    args = ap.parse_args()

    print(ab.device_note())
    tokens = (args.latent_hw // 2) ** 2
    print(
        f"anima CosmosTransformer3DModel, num_layers={args.layers}, latent {args.latent_hw}^2 "
        f"-> {tokens} tokens, bf16, batch 1, warm"
    )

    arms = Arms(args.layers, args.latent_hw)
    reference: torch.Tensor | None = None
    timed: list[tuple[str, Any, Any]] = []

    for name in args.arms:
        arms.install(name)
        out = arms.output()
        if reference is None:
            reference = out.float().clone()
            print(f"  {name:22s} reference")
        else:
            diff = (out.float() - reference).abs()
            rel = (diff.mean() / reference.abs().mean()).item()
            print(f"  {name:22s} max|Δ| {diff.max().item():.3e}  mean|Δ|/mean|ref| {rel:.3e}")
        del out
        timed.append((name, arms.call, _installer(arms, name)))

    if args.compile:
        arms.install("eager_plan_op" if args.compile == "plan_op" else "eager_stock")
        model = arms.model
        start = time.perf_counter()
        if args.compile == "repeated_blocks":
            model._repeated_blocks = ["CosmosTransformerBlock"]
            model.compile_repeated_blocks(fullgraph=True)
            compiled: Any = model
        else:
            compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            first = compiled(**arms.kwargs)[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        diff = (first.float() - reference).abs()  # type: ignore[operator]
        rel = (diff.mean() / reference.abs().mean()).item()  # type: ignore[union-attr]
        print(
            f"  compile_{args.compile:14s} max|Δ| {diff.max().item():.3e}  "
            f"mean|Δ|/mean|ref| {rel:.3e}   COLD COMPILE {seconds:.1f} s"
        )
        del first

        def compiled_call() -> None:
            with torch.no_grad():
                compiled(**arms.kwargs)

        timed.append((f"compile_{args.compile}", compiled_call, None))

    torch.cuda.empty_cache()
    results = ab.interleaved(timed, reps=args.reps, inner=args.inner)
    ab.report(
        f"anima transformer forward ({args.layers} blocks, {tokens} tokens)", results, args.arms[0]
    )
    print(f"  peak CUDA memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def _installer(arms: Arms, name: str) -> Any:
    def setup() -> None:
        arms.install(name)

    return setup


if __name__ == "__main__":
    main()
