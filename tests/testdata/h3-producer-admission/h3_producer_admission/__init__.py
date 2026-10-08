"""Small actual-kernel gate through ordinary Cozy CLI; never a model speed claim."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Row(msgspec.Struct):
    selector: str
    projection_calls: list[int]
    elapsed_ms: list[float]
    finite: bool
    repeat_identical: bool
    retained_scope_empty: bool


class Result(msgspec.Struct):
    gpu: str
    kitchen: str
    tokens: int
    rows: list[Row]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context) -> Result:
    if ctx.device.type != "cuda":
        raise ValueError("Producer admission requires an allocated SM120 GPU")
    import time
    from importlib.metadata import version

    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

    from cozy_runtime.author import AttentionLayout, attention_scope
    from cozy_runtime.internal import attention, attention_producer, attention_sol
    from cozy_runtime.internal.encoding import DeviceFacts

    index = ctx.device.index or 0
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise ValueError("Producer admission requires SM120")
    device = torch.device(str(ctx.device))
    facts = DeviceFacts("cuda", torch.cuda.get_device_name(index), 120, "", "", index)
    tokens = 4097  # one4K producer chunk plus a non-aligned live tail
    generator = torch.Generator().manual_seed(8193)
    torch.manual_seed(91)
    attn = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128).bfloat16().to(device)
    hidden = torch.randn(1, tokens, 256, generator=generator).bfloat16().to(device)
    phase = torch.randn(1, tokens, 1, 48, generator=generator).to(device)
    cos, sin = (torch.cat((f(phase), f(phase)), dim=-1) for f in (torch.cos, torch.sin))
    calls = [0]

    def count(*args: object) -> None:
        calls[0] += 1

    handles = [p.register_forward_pre_hook(count) for p in (attn.to_q, attn.to_k, attn.to_v)]
    results = []
    try:
        for name in sorted(attention_producer.NAMES):
            ready = attention.pinned(name, facts)
            attention.install(ready, {"dit": attn}, "dit=" + name, device=facts)
            runs, counts, elapsed = [], [], []
            for repeat in range(2):
                outputs = []
                with torch.no_grad(), attention_sol.observing():
                    for step in (0, 4, 5):
                        calls[0] = 0
                        torch.cuda.synchronize(device)
                        began = time.perf_counter()
                        with attention_scope(AttentionLayout(tokens, 65, step, 4)):
                            output = attn(hidden, (cos, sin))
                        torch.cuda.synchronize(device)
                        if not torch.isfinite(output).all() or output.float().square().mean() <= 0:
                            raise ValueError("Producer returned nonfinite or degenerate output")
                        outputs.append(output.clone())
                        if repeat == 0:
                            elapsed.append((time.perf_counter() - began) * 1000)
                            counts.append(calls[0])
                if attention_producer._STATE.get() is not None:
                    raise ValueError("Producer state escaped its invocation")
                runs.append(outputs)
            identical = all(torch.equal(a, b) for a, b in zip(*runs, strict=True))
            expected = [3, 12, 6 if name in attention_producer.REUSE else 12]
            if counts != expected or not identical:
                raise ValueError(
                    f"Producer replay contract failed: {name}: {counts} vs{expected}, repeat={identical}"
                )
            results.append(Row(name, counts, elapsed, True, identical, True))
    finally:
        for handle in handles:
            handle.remove()
    return Result(torch.cuda.get_device_name(index), version("comfy-kitchen"), tokens, results)
