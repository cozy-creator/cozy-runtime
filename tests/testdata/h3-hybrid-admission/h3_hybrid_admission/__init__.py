"""Actual-kernel policy gate through ordinary Cozy CLI; never a model speed/quality claim."""

from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Context

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Row(msgspec.Struct):
    case: str
    tokens: int
    prefix: int
    counts: dict[str, int]
    implementations: str
    finite: bool
    repeat_identical: bool
    exact_composition: bool
    inputs_unchanged: bool
    relative_l2_to_sage2: float


class Result(msgspec.Struct):
    runtime: str
    gpu: str
    selector: str
    rows: list[Row]


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context) -> Result:
    if ctx.device.type != "cuda":
        raise ValueError("Hybrid admission requires an allocated SM120 GPU")
    from importlib.metadata import version

    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    from cozy_runtime.author import AttentionLayout, attention_scope
    from cozy_runtime.internal import (
        attention,
        attention_fp8_sol,
        attention_sol,
        attention_ulysses,
        attention_upstream,
    )
    from cozy_runtime.internal.encoding import DeviceFacts

    index = ctx.device.index or 0
    if torch.cuda.get_device_capability(index) != (12, 0):
        raise ValueError("Hybrid admission requires SM120")
    device = torch.device(str(ctx.device))
    name = "sol-fp8-fp16-fp4-dense"
    facts = DeviceFacts("cuda", torch.cuda.get_device_name(index), 120, "", "", index)
    ready = attention.pinned(name, facts)[0]
    rows = []
    for case, tokens, prefix, step, path in [
        ("last_dense", 4097, 65, 3, "blocks.2.attn"),
        ("first_sparse_partial_group_tail", 4097, 65, 4, "blocks.2.attn"),
        ("full_protected_document", 257, 257, 4, "blocks.2.attn"),
        ("protected_model_path", 257, 65, 4, "blocks.1.attn"),
        ("short_refiner", 129, 65, 4, "refiner.attn"),
    ]:
        generator = torch.Generator().manual_seed(8193 + tokens)
        tensors = [
            torch.randn(1, tokens, 2, 128, generator=generator).bfloat16().to(device)
            for _ in range(3)
        ]
        before = [x.clone() for x in tensors]
        live = 4097 if case == "short_refiner" else tokens
        layout = AttentionLayout(live, prefix, step, 4, ("blocks.1", "refiner"))
        site = attention_sol._SITE.set(attention_sol.Site("dit", path))
        try:
            with (
                torch.inference_mode(),
                attention_scope(layout),
                attention_sol.observing() as counts,attention_ulysses.observing() as seen
            ):
                got = dispatch_attention_fn(*tensors, backend=ready.member)
            with torch.inference_mode(), attention_scope(layout), attention_sol.observing():
                repeat = dispatch_attention_fn(*tensors, backend=ready.member)
        finally:
            attention_sol._SITE.reset(site)
        q, k, v = tensors
        with torch.inference_mode():
            reference = attention_upstream.SAGE.forward(q, k, v, scale=128**-0.5)
            is_sparse = case in {"first_sparse_partial_group_tail", "full_protected_document"}
            if is_sparse:
                expected = attention_fp8_sol.sparse(q, k, v, 128**-0.5, prefix)
                expected[:, :prefix] = attention_upstream.SAGE.forward(
                    q[:, :prefix],
                    k,
                    v,
                    scale=128**-0.5,
                )
                wanted = dict(sparse=1, dense_step=0, dense_path=0, dense_prefix=1)
            else:
                expected = attention_upstream.SAGE3_GLOBAL.forward(q, k, v, scale=128**-0.5)
                wanted = dict(
                    sparse=0, dense_step=int(step < 4), dense_path=int(step >= 4), dense_prefix=0
                )
        finite = bool(torch.isfinite(got).all()) and bool(got.float().square().mean() > 0)
        identical = bool(torch.equal(got, repeat))
        composition = bool(torch.equal(got, expected))
        unchanged = all(torch.equal(x, y) for x, y in zip(tensors, before, strict=True))
        if not (finite and identical and composition and unchanged and counts == wanted):
            raise ValueError(
                f"Hybrid gate failed {case}: finite={finite}, repeat={identical}, "
                f"composition={composition}, inputs={unchanged}, counts={counts}"
            )
        relative = float(
            (got.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-8)
        )
        rows.append(
            Row(
                case,
                tokens,
                prefix,
                counts,
                attention.implementations(seen),
                finite,
                identical,
                composition,
                unchanged,
                relative,
            )
        )
    return Result(version("cozy-runtime"), torch.cuda.get_device_name(index), name, rows)
