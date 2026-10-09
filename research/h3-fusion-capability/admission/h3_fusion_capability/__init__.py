"""Real fused block/encoded GEMM/LoRA admission; run through ordinary Cozy CLI only."""

from __future__ import annotations

import json
from importlib.metadata import version
from types import SimpleNamespace

import msgspec

from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Result(msgspec.Struct):
    runtime: str
    diffusers: str
    gpu: str
    passed: bool
    evidence_json: str


def compile_architecture(capability: tuple[int, int]) -> int:
    major, minor = capability
    return major * 10 + minor


def construct_case(
    mode: str, device: object, *, hidden_size: int = 5376, ffn_dim: int = 14336
) -> SimpleNamespace:
    """The same real constructor path is used by the GPU gate and CPU admission check."""
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

    from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise
    from cozy_runtime.models.minimax_h3.turbo import LoRAFactors, _LoRAHook

    if mode not in ("bf16", "encoded", "encoded_inactive", "encoded_baked", "encoded_unbaked"):
        raise ValueError(f"Unknown fusion gate case {mode}")
    torch.manual_seed(7132)
    with torch.device(device):
        model = (
            MiniMaxH3Transformer3DModel(
                num_layers=1,
                num_refiner_layers=0,
                hidden_size=hidden_size,
                ffn_dim=ffn_dim,
                num_attention_heads=2,
                attention_head_dim=128,
                text_dim=128,
                time_embed_hidden_dim=128,
                time_embed_dim=128,
            )
            .bfloat16()
            .eval()
        )
    block = model.transformer_blocks[0]
    # Stable, finite modulation keeps this a sensitive arithmetic agreement test.
    for parameter in model.parameters():
        parameter.normal_(std=0.02)
    slots = [
        (block.attn, "to_q"),
        (block.attn, "to_k"),
        (block.attn, "to_v"),
        (block.ff.net[0], "proj"),
        (block.ff.net, "2"),
    ]
    hooks = []
    owners = []
    if mode != "bf16":
        for parent, name in slots:
            linear = getattr(parent, name)
            if mode.startswith("encoded_"):
                factors = LoRAFactors(linear.in_features, linear.out_features, 4, 0.25).to(device)
                factors.lora_down.normal_(std=0.01)
                factors.lora_up.normal_(std=0.01)
                owner = SimpleNamespace(
                    _arming=None
                    if mode == "encoded_inactive"
                    else SimpleNamespace(
                        overlay=SimpleNamespace(get_submodule=lambda _, f=factors: f)
                    )
                )
                owners.append(owner)
                if mode == "encoded_baked":
                    linear.weight.add_((factors.lora_up @ factors.lora_down) * factors.scale)
            data, scale = quantize_activation_rowwise(torch, linear.weight)
            leaf = RowwiseNativeLeaf(encoding="fp8-rowwise/1").leaf(
                torch, {"data": data, "scale": scale}, linear, torch.bfloat16
            )
            if mode.startswith("encoded_"):
                hook = _LoRAHook(owner, "projection")
                hook._cozy_baked = mode == "encoded_baked"
                leaf.register_forward_hook(hook)
                hooks.append(hook)
            setattr(parent, name, leaf)
    hidden = torch.randn(1, 129, hidden_size, device=device, dtype=torch.bfloat16)
    temb = torch.randn(2, 128, device=device, dtype=torch.bfloat16)
    indices = torch.arange(129, device=device) % 6
    rotary = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16).to(device)(
        torch.randint(0, 19, (129, 3), device=device)
    )
    holder = SimpleNamespace(pipe=SimpleNamespace(components={"fl2va_dit": model}))
    return SimpleNamespace(
        model=model,
        block=block,
        slots=slots,
        hooks=hooks,
        owners=owners,
        hidden=hidden,
        temb=temb,
        indices=indices,
        rotary=rotary,
        holder=holder,
    )


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context) -> Result:
    import torch

    from cozy_runtime.internal import fusion, fusion_install

    if ctx.device.type != "cuda":
        raise ValueError("Fusion admission requires the allocated rented GPU")
    device = torch.device(str(ctx.device))
    fusion_install._require_capabilities()
    fusion.expect(compile_architecture(torch.cuda.get_device_capability(device)))
    cases = []
    with torch.inference_mode():
        for mode in ("bf16", "encoded", "encoded_inactive", "encoded_baked", "encoded_unbaked"):
            case = construct_case(mode, device)
            model, block, slots, hooks = case.model, case.block, case.slots, case.hooks
            hidden, temb, indices, rotary = case.hidden, case.temb, case.indices, case.rotary
            expected = block(hidden, temb, indices, rotary)
            holder = case.holder
            before = fusion_install._tensor_census(model)
            applied = fusion_install.apply_fusion_plan(
                holder, torch=torch, device=device, required=True
            )
            actual = block(hidden, temb, indices, rotary)
            repeated = block(hidden, temb, indices, rotary)
            torch.cuda.synchronize(device)
            delta = (actual.float() - expected.float()).abs()
            max_error = delta.max().item()
            relative_l2 = (delta.norm() / expected.float().norm()).item()
            range_error = max_error / max(expected.float().abs().max().item(), 1e-12)
            finite = bool(torch.isfinite(actual).all())
            same_hooks = all(
                list(getattr(parent, name)._forward_hooks.values()) == [hook]
                for (parent, name), hook in zip(slots, hooks, strict=False)
            )
            expected_sites = 3 if mode == "encoded" else 0
            passed = (
                applied.applied
                and applied.fp8_sites == expected_sites
                and finite
                and range_error <= 0.015625
                and relative_l2 <= 0.01
                and torch.equal(actual, repeated)
                and before == fusion_install._tensor_census(model)
                and same_hooks
            )
            row = {
                "mode": mode,
                "applied": applied.document(),
                "block_class": type(block).__name__,
                "finite": finite,
                "max_abs_error": max_error,
                "relative_l2": relative_l2,
                "max_error_over_output_range": range_error,
                "repeat_exact": torch.equal(actual, repeated),
                "hooks_preserved": same_hooks,
                "tensor_custody_preserved": before == fusion_install._tensor_census(model),
                "passed": passed,
            }
            cases.append(row)
            if not passed:
                raise ValueError(json.dumps(row, sort_keys=True))
            del case, model, block, holder, slots, expected, actual, repeated, hidden, hooks
    return Result(
        version("cozy-runtime"),
        version("diffusers"),
        torch.cuda.get_device_name(device),
        True,
        json.dumps(cases, sort_keys=True),
    )
