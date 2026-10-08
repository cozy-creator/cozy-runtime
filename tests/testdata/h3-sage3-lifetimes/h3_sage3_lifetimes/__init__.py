"""Actual-kernel agreement gate, through ordinary CLI; not a full-model speed claim."""

from __future__ import annotations

import json
from importlib.metadata import version

import msgspec

from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Result(msgspec.Struct):
    runtime: str
    gpu: str
    passed: bool
    evidence_json: str


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context) -> Result:
    if ctx.device.type != "cuda":
        raise ValueError("Sage3 lifetime admission requires an allocated SM120 GPU")
    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    from cozy_runtime.author import AttentionLayout, attention_scope
    from cozy_runtime.internal import attention, attention_producer, attention_sol
    from cozy_runtime.internal import attention_upstream as upstream
    from cozy_runtime.internal.encoding import DeviceFacts
    from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise

    index, device = ctx.device.index or 0, torch.device(str(ctx.device))
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise ValueError("Sage3 lifetime admission requires SM120")
    facts = DeviceFacts("cuda", torch.cuda.get_device_name(index), 120, "", "", index)
    attention.pinned("sageattention3-fp4-global-lowmem", facts)
    generator = torch.Generator(device=device).manual_seed(71831)
    direct = []
    with torch.inference_mode():
        for label, ql, kl, dtype, separate_stream in [
            ("aligned", 128, 128, torch.bfloat16, False),
            ("ragged", 4097, 4097, torch.bfloat16, False),
            ("unequal_fp16", 129, 257, torch.float16, False),
            ("current_stream", 257, 257, torch.bfloat16, True),
        ]:
            q = torch.randn((1, ql, 2, 128), device=device, dtype=dtype, generator=generator)
            kv = torch.randn((1, kl, 2, 2, 128), device=device, dtype=dtype, generator=generator)
            k, v, saved = kv[:, :, :, 0], kv[:, :, :, 1], kv.clone()
            saved_q = q.clone()
            # Warm both implementations before comparing Torch-tracked live peaks.
            warmed = upstream.sage3_fp4_global(q, k, v)
            warmed = upstream.sage3_fp4_global_lowmem(q, k, v)
            del warmed
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            expected = upstream.sage3_fp4_global(q, k, v)
            torch.cuda.synchronize(device)
            reference_peak = torch.cuda.max_memory_allocated(device) - allocated
            stream = (
                torch.cuda.Stream(device=device)
                if separate_stream
                else torch.cuda.current_stream(device)
            )
            stream.wait_stream(torch.cuda.current_stream(device))
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            with torch.cuda.stream(stream):
                got = upstream.sage3_fp4_global_lowmem(q, k, v)
            torch.cuda.synchronize(device)
            candidate_peak = torch.cuda.max_memory_allocated(device) - allocated
            with torch.cuda.stream(stream):
                repeated = upstream.sage3_fp4_global_lowmem(q, k, v)
            torch.cuda.synchronize(device)
            agrees = torch.equal(got, expected) and torch.equal(got, repeated)
            finite = bool(torch.isfinite(got).all())
            unchanged = torch.equal(kv, saved) and torch.equal(q, saved_q)
            if not agrees or not finite or not unchanged or not got.is_contiguous():
                raise ValueError(f"Sage3 lifetime contract failed: {label}")
            direct.append(
                {
                    "case": label,
                    "shape": list(got.shape),
                    "stride": list(got.stride()),
                    "dtype": str(dtype),
                    "exact_reference": agrees,
                    "inputs_unchanged": unchanged,
                    "finite": finite,
                    "nondefault_stream": separate_stream,
                    "torch_peak_increment_bytes": {
                        "reference": reference_peak,
                        "candidate": candidate_peak,
                    },
                }
            )

        # Real H3 instance dispatch, including the shared-rowwise projection path.
        tokens = 4097
        torch.manual_seed(1933)
        attn = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128).bfloat16().to(device)
        for name in ("to_q", "to_k", "to_v"):
            original = getattr(attn, name)
            data, scale = quantize_activation_rowwise(torch, original.weight)
            setattr(
                attn,
                name,
                RowwiseNativeLeaf().leaf(
                    torch, {"data": data, "scale": scale}, original, torch.bfloat16
                ),
            )
        hidden = torch.randn(
            (1, tokens, 256), device=device, dtype=torch.bfloat16, generator=generator
        )
        phase = torch.randn((tokens, 48), device=device, generator=generator)
        rotary = tuple(torch.cat((f(phase), f(phase)), dim=-1) for f in (torch.cos, torch.sin))
        mixed = []
        for dense_steps in (4, 10):
            for suffix in ("", "-shared-qkv"):
                control = "kitchen-sol-producer-fp4" + suffix
                candidate = "kitchen-sol-producer-fp4-lowmem" + suffix
                values, counters = {}, {}
                for name in (control, candidate):
                    ready = attention.pinned(name, facts)
                    attention.install(ready, {"dit": attn}, "dit=" + name, device=facts)
                    outputs = []
                    with attention_sol.observing() as counts:
                        for step in (0, dense_steps, dense_steps + 1):
                            with attention_scope(AttentionLayout(tokens, 65, step, dense_steps)):
                                outputs.append(attn(hidden, rotary).clone())
                    torch.cuda.synchronize(device)
                    if attention_producer._STATE.get() is not None:
                        raise ValueError("Producer invocation state escaped")
                    values[name], counters[name] = outputs, dict(counts)
                agrees = all(
                    torch.equal(a, b)
                    for a, b in zip(values[control], values[candidate], strict=True)
                )
                if not agrees or counters[control] != counters[candidate]:
                    raise ValueError(f"Mixed policy/output changed: {candidate}")
                if not all(bool(torch.isfinite(x).all()) for x in values[candidate]):
                    raise ValueError("Mixed producer returned nonfinite output")
                mixed.append(
                    {
                        "selector": candidate,
                        "dense_steps": dense_steps,
                        "exact_reference": agrees,
                        "counts": counters[candidate],
                    }
                )
    return Result(
        version("cozy-runtime"),
        torch.cuda.get_device_name(index),
        True,
        json.dumps(
            {"direct": direct, "mixed": mixed, "kitchen": version("comfy-kitchen")}, allow_nan=False
        ),
    )
