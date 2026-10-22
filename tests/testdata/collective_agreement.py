"""Two or more gloo ranks agree the Ulysses cut level once per pass (varena-slim §1).

Each rank loads a tiny sharded attention model through varena's host pages, with the execution
layer installed, and runs four passes in one request. In the first, every rank's exchange runs
out of memory before any rank enters its collectives: none runs it again (its peers would wait
on it), each fails typed, and each tells Rust, which keeps a finer level for the next pass's
agreement. From the second pass every rank crosses the heads in the same smaller groups, each
pass equal to degree 1 byte for byte. The ranks' rows are gathered once per shape. One JSON
report per rank.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import tensorfs
import torch
import torch.distributed as dist
from diffusers.models._modeling_parallel import ContextParallelConfig, ParallelConfig
from diffusers.models.attention_dispatch import dispatch_attention_fn
from tensorfs.varena import GpuId
from tensorfs.varena.torch import Declared, Memory, Weights
from torch import nn
from torch.multiprocessing.spawn import spawn

from cozy_runtime.author._errors import classify
from cozy_runtime.internal import attention_ulysses, execution
from cozy_runtime.internal.execution import agree
from cozy_runtime.internal.parallel import ulysses
from cozy_runtime.internal.parallel.cp import CpComms, _GroupMesh

DIM, HEADS, ROWS = 64, 8, 37


def scores(query: Any, key: Any, value: Any, **options: Any) -> Any:
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    weights = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(-1)
    return (weights @ v).transpose(1, 2)


class Sharded(nn.Module):
    """Q/K/V projections and attention over this rank's rows, Ulysses when configured."""

    def __init__(self) -> None:
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(DIM, DIM) for _ in range(3))
        self.member = attention_ulysses.member(
            "_cozy_test_agreement",
            attention_ulysses.Local(scores, lambda *_: "scores", head_local=True, rows=8),
        )

    def forward(self, x: Any, parallel: Any = None) -> Any:
        b, s, _ = x.shape
        q, k, v = (p(x).view(b, s, HEADS, DIM // HEADS) for p in (self.q, self.k, self.v))
        return dispatch_attention_fn(q, k, v, backend=self.member, parallel_config=parallel)


def rank_main(rank: int, degree: int, root: Path) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "cpu:gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=degree
    )
    try:
        device = torch.device("cpu")
        config = ParallelConfig(
            context_parallel_config=ContextParallelConfig(
                ulysses_degree=degree, ulysses_anything=True
            )
        )
        mesh = _GroupMesh(CpComms(dist.group.WORLD, rank, device), degree, rank)
        config.setup(rank, degree, device, mesh=cast(Any, mesh))
        attention_ulysses.install(execution)
        exchanged: list[int] = []
        armed = [False]
        cross = ulysses._cross

        def crossing(engines: Any, query: Any, *rest: Any) -> Any:
            if armed[0]:  # its buffers do not fit, before any collective of this exchange
                armed[0] = False
                raise torch.OutOfMemoryError("out of memory. Tried to allocate 36.00 KiB")
            exchanged.append(int(query.shape[2]))
            return cross(engines, query, *rest)

        ulysses._cross = crossing
        gathers: list[int] = []
        sizes = agree.sizes

        def counted(gather: Callable[[], list[int]]) -> list[int]:
            def counting() -> list[int]:
                gathers.append(1)
                return gather()

            return sizes(counting)

        agree.sizes = counted
        torch.manual_seed(0)
        plain = Sharded().to(torch.bfloat16).eval()
        writer = tensorfs.Writer(cache_dir=root / f"store-{rank}")
        component = writer.component("attn", cls=("cozy-test", "0", "Sharded"), config={})
        for key, tensor in plain.state_dict().items():
            held = tensor.detach().contiguous()
            data = held.view(torch.uint8).numpy().tobytes()
            component.put(key, data, dtype="BF16", shape=list(held.shape))
        ck = writer.commit()
        mem = Memory.open(GpuId(f"cozy-agreement-{os.getpid()}-{rank}", "cpu", 1 << 30))
        with torch.device("meta"):
            model = Sharded().to(torch.bfloat16).eval()
        vars(model)["_varena"] = Declared(Weights(ck, "attn"))
        mem.load(model, "gpu")
        execution.install(model)
        model._parallel_config = config  # what `agree` reads at each pass start
        seed = torch.Generator().manual_seed(0x51D3)
        x = torch.randn(1, ROWS, DIM, generator=seed).to(torch.bfloat16)
        mine = x.tensor_split(degree, dim=1)[rank].contiguous()
        passes: list[dict[str, Any]] = []
        with torch.no_grad():
            whole = plain(x).tensor_split(degree, dim=1)[rank]
            with mem.request("agreement") as r, execution.serving(r):
                for n in range(4):
                    armed[0] = n == 0
                    exchanged.clear()
                    gathers.clear()
                    row: dict[str, Any] = {}
                    try:
                        got = model(mine, config)
                        row["byte_equal"] = bool(torch.equal(got, whole))
                    except Exception as exc:
                        row["failed"] = classify(exc).code
                    row |= {"exchanged_heads": list(exchanged), "gathered": len(gathers)}
                    passes.append(row)
        (root / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "passes": passes}))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()
    start: Any = spawn
    start(rank_main, args=(args.degree, args.root), nprocs=args.degree, join=True)


if __name__ == "__main__":
    main()
