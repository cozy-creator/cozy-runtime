"""Whether the ggml path orders itself on torch's default stream (`bench.ggml_batch_shapes`, xs-213, cozy-runtime#2309).

batch-shapes (the varena suite's case, #2291) found it: Qwen3.5-9B's MLP linears in q8_0 and q4_k, the same rows in decode
batches and prefill chunks through the leaf (`ggml_linear.linear`) and as one launch (`ggml_linear.matmul`), each held
against its group's largest. Run as it first ran, every call on the default stream and read at once with no sync, it
was red: 0 of N rows identical, NaNs, differences of 5-15 (44407876384, 44407881866). Here it runs that way (`raced`)
beside the same cases settled (each call on its own stream, synced and copied before a read; the largest called twice as
the control). Where the path is ordered, every raced row equals its settled row."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from types import ModuleType

import torch

from cozy_runtime.internal import ggml_linear, kernel_ready

#: (site, out, in): the 9B's MLP up and down projections.
SITES = (("up", 12288, 4096), ("down", 4096, 12288))
SCHEMES = ("ggml.q8_0", "ggml.q4_k")
GROUPS = (("decode", (1, 2, 4, 8, 13, 16)), ("prefill", (640, 1500, 4096)))

type Row = dict[str, int | float | str]


def _rows(ops: ModuleType, raced: bool) -> dict[str, Row]:
    own = torch.cuda.Stream()

    def run(call: Callable[[torch.Tensor], torch.Tensor], r: torch.Tensor) -> torch.Tensor:
        if raced:
            return call(r)
        with torch.cuda.stream(own):
            out = call(r)
        own.synchronize()
        held = out.clone()
        torch.cuda.synchronize()
        return held

    def row(got: torch.Tensor, against: torch.Tensor) -> Row:
        diff = float((got - against).abs().max())
        return {"rows_identical": int((got == against).all(dim=1).sum()), "max_abs_diff": diff if diff == diff else "nan"}

    made = torch.Generator(device="cpu").manual_seed(213)
    rows: dict[str, Row] = {}
    with torch.cuda.stream(torch.cuda.default_stream()):
        for scheme in SCHEMES:
            for site, n, k in SITES:
                weight = torch.randn((n, k), generator=made).mul_(0.02)
                blocks = ggml_linear.encode(torch, scheme, weight).to("cuda")
                x = torch.randn((4096, k), generator=made).to("cuda", torch.bfloat16)
                paths: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
                    "leaf": partial(ggml_linear.linear, torch, blocks=blocks, scheme=scheme, n=n, bias=None, out_dtype=torch.float32),
                    "launch": partial(ggml_linear.matmul, ops, torch, blocks=blocks, scheme=scheme, n=n),
                }  # fmt: skip
                for path, call in paths.items():
                    for group, sizes in GROUPS:
                        most = max(sizes)
                        against = run(call, x[:most])
                        if not raced:
                            rows[f"{scheme} {site} {path} {group} {most} repeat"] = row(run(call, x[:most]), against)
                        for m in sizes[:-1]:
                            rows[f"{scheme} {site} {path} {group} {m}"] = row(run(call, x[:m]), against[:m])
                del paths, blocks, x
        torch.cuda.synchronize()
    return rows


def measure() -> dict[str, object]:
    if why := ggml_linear.carried(ggml_linear.capability(torch, 0)):
        return {"served": False, "why": why}
    # This runs inside a request: a kernel the request opens lands, and turns ready only at the next quiet boundary
    # (`kernel_ready.ready`), so polling `native` here sees None for good (44536379975 waited 900 s). A request that names
    # its kernel waits for it on its own thread instead, and the leaf then serves it for every call that follows.
    try:
        ops = kernel_ready.wait(f"{ggml_linear.OP} cuda:0", partial(ggml_linear._verified, torch, 0))
    except Exception as exc:  # refused on this card: told, and the session judges it
        return {"served": False, "why": f"{type(exc).__name__}: {exc}"[:300]}
    raced, settled = _rows(ops, raced=True), _rows(ops, raced=False)
    differs = [key for key, value in raced.items() if settled.get(key) != value]
    return {
        "served": True,
        "rows": len(raced),
        "raced_equals_settled": not differs,
        "repeat_identical": all(v["max_abs_diff"] == 0.0 for key, v in settled.items() if key.endswith(" repeat")),
        "differs": differs[:8],
        "raced": raced,
        "settled": settled,
    }
