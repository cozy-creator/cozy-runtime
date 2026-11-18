"""Whether a row's result depends on how many rows share its launch (the varena suite's
`batch-shapes` case). xs-213's identical-token bar holds a capped arm against an uncapped one, and
the two hand a linear different row counts: fewer rows decoding at once, a smaller prefill chunk.
llama.cpp's MMQ splits K by stream-k per its config (`mmq.cuh`), so the same row in batches of
different sizes may not sum alike; MMVQ serves one row and is another kernel altogether.

Qwen3.5-9B's MLP linears in the q8_0 lane's type and q4_k (Q4_K_M's), on blocks TensorFS's
ggml-ref/1 writes: the same rows in a batch of 1 to 16 (decode) and in prefills of three sizes,
through the leaf's own path (`ggml_linear.linear`, which takes rows a piece at a time) and as one
launch (`ggml_linear.matmul`). Every call runs on an explicit stream and is synchronised and copied
before anything reads it, and each shape is called twice alike first (`repeat`): a call that
differs from itself is not a shape's doing. It reports; it judges nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import torch

from cozy_runtime.internal import ggml_linear

type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]

#: (site, out, in): the MLP's up and down projections.
SITES = (("up", 12288, 4096), ("down", 4096, 12288))
SCHEMES = ("ggml.q8_0", "ggml.q4_k")
#: Rows decoding at once, each held against the same rows in the largest.
DECODE = (1, 2, 4, 8, 13, 16)
#: Prefill chunks' rows, each held against the same rows in the largest.
PREFILL = (640, 1500, 4096)


def measure() -> dict[str, JSON]:
    deadline = time.monotonic() + 120
    while (ops := ggml_linear.native(torch, 0)) is None:
        if why := ggml_linear.why(0) or ggml_linear.carried(ggml_linear.capability(torch, 0)):
            return {"served": False, "why": why}
        if time.monotonic() > deadline:
            return {"served": False, "why": "the card's ggml kernels were not verified in 120 s"}
        time.sleep(0.1)
    made = torch.Generator(device="cpu").manual_seed(213)
    rows: list[JSON] = []
    stream = torch.cuda.Stream()

    def settled(call: Callable[[torch.Tensor], torch.Tensor], r: torch.Tensor) -> torch.Tensor:
        with torch.cuda.stream(stream):
            out = call(r)
        stream.synchronize()
        held = out.clone()
        torch.cuda.synchronize()
        return held

    for scheme in SCHEMES:
        for site, n, k in SITES:
            weight = torch.randn((n, k), generator=made).mul_(0.02)
            blocks = ggml_linear.encode(torch, scheme, weight).to("cuda")
            x = torch.randn((max(PREFILL), k), generator=made).to("cuda", torch.bfloat16)
            paths: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
                "leaf": lambda r: ggml_linear.linear(torch, r, blocks, scheme, n, None, torch.float32),
                "launch": lambda r: ggml_linear.matmul(ops, torch, r, blocks, scheme, n),
            }
            for path, call in paths.items():
                for group, sizes in (("decode", DECODE), ("prefill", PREFILL)):
                    most = max(sizes)
                    against = settled(call, x[:most])
                    again = settled(call, x[:most])
                    rows.append({
                        "type": scheme, "site": site, "path": path, "group": group, "rows": most,
                        "against": most, "repeat": True,
                        "kernel": ggml_linear.kernel(torch, 0, scheme, k, n, most, blocks),
                        "kernel_against": ggml_linear.kernel(torch, 0, scheme, k, n, most, blocks),
                        "rows_identical": int((again == against).all(dim=1).sum()),
                        "max_abs_diff": float((again - against).abs().max()),
                    })  # fmt: skip
                    for m in sizes:
                        if m == most:
                            continue
                        got = settled(call, x[:m])
                        same = (got == against[:m]).all(dim=1)
                        rows.append({
                            "type": scheme, "site": site, "path": path, "group": group,
                            "rows": m, "against": most, "repeat": False,
                            "leaf_piece": max(1, ggml_linear.WORKING // (6 * k + 4 * n)),
                            "kernel": ggml_linear.kernel(torch, 0, scheme, k, n, m, blocks),
                            "kernel_against": ggml_linear.kernel(torch, 0, scheme, k, n, most, blocks),
                            "rows_identical": int(same.sum()),
                            "max_abs_diff": float((got - against[:m]).abs().max()),
                        })  # fmt: skip
            del blocks, x
    torch.cuda.synchronize()
    return {
        "served": True,
        "serving": ggml_linear.serving(0),
        "leaf_failed": ggml_linear.why(0),  # the leaf decodes and multiplies by torch after a failure
        "shapes": rows,
    }
