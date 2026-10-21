"""Two or more gloo ranks agree the Ulysses cut level once per pass (varena-slim §1).

Each rank loads a tiny sharded attention model through varena's host pages, with the catch
installed, and runs five passes in one request. In the first (unmeasured: under the per-op
layer), one rank's exchange runs out of memory allocating its buffers: Rust answers it locally
(never a cut mid-pass) and keeps the finer level, so from the second pass every rank crosses the
heads in the same smaller groups. Measured, the second and third run their collective block
bare. In the fourth every rank runs out in its bare exchange: no rank runs it again, each fails
typed, and the fifth (the same shapes) runs under the layer. Every pass that completes equals
degree 1, byte for byte. The ranks' rows are gathered once, in the first pass: every later
pass has the same shapes. One JSON report per rank.
"""

from __future__ import annotations

import argparse
import json
import os
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
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode

from cozy_runtime.author._errors import classify
from cozy_runtime.internal import attention_ulysses, execution
from cozy_runtime.internal.execution import catch
from cozy_runtime.internal.parallel.cp import CpComms, _GroupMesh

DIM, HEADS, ROWS = 64, 8, 37
#: whether each forward of `Sharded` ran under the per-op layer
LAYERED: list[bool] = []


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
        LAYERED.append(isinstance(_get_current_dispatch_mode(), catch.Layer))
        b, s, _ = x.shape
        q, k, v = (p(x).view(b, s, HEADS, DIM // HEADS) for p in (self.q, self.k, self.v))
        return dispatch_attention_fn(q, k, v, backend=self.member, parallel_config=parallel)


class ExchangeShort(TorchDispatchMode):  # type: ignore[misc]
    """The first buffer an exchange allocates runs out, once, while `armed`."""

    def __init__(self) -> None:
        super().__init__()
        self.armed, self.refused = False, 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        if self.armed and func is torch.ops.aten.new_empty.default:
            self.armed, self.refused = False, self.refused + 1
            nbytes = out.numel() * out.element_size()
            del out
            catch._refused(0, nbytes, 0, 0)
            raise torch.OutOfMemoryError(f"out of memory. Tried to allocate {nbytes} B")
        return out


def rank_main(rank: int, degree: int, root: Path, short: int) -> None:
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
        counted = attention_ulysses._exchange

        def counting(query: Any, *rest: Any) -> Any:
            exchanged.append(int(query.shape[2]))
            return counted(query, *rest)

        attention_ulysses._exchange = counting
        gathers: list[int] = []
        gather = attention_ulysses._gather

        def gathering(rows: int, *rest: Any) -> list[int]:
            gathers.append(rows)
            return gather(rows, *rest)

        attention_ulysses._gather = gathering
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
        injector = ExchangeShort()
        passes: list[dict[str, Any]] = []
        with torch.no_grad():
            whole = plain(x).tensor_split(degree, dim=1)[rank]
            with injector, mem.request("agreement") as r, execution.serving(r):
                for n in range(5):
                    injector.armed = (rank == short and n == 0) or n == 3
                    exchanged.clear()
                    gathers.clear()
                    row: dict[str, Any] = {}
                    try:
                        got = model(mine, config)
                        row["byte_equal"] = bool(torch.equal(got, whole))
                    except Exception as exc:
                        row["failed"] = classify(exc).code
                    row |= {"exchanged_heads": list(exchanged), "layered": LAYERED[-1]}
                    row["gathered"] = len(gathers)
                    passes.append(row)
        report = {"rank": rank, "refused": injector.refused, "passes": passes}
        (root / f"rank-{rank}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--short", type=int, default=1, help="the rank whose exchange runs out")
    args = parser.parse_args()
    start: Any = spawn
    start(rank_main, args=(args.degree, args.root, args.short), nprocs=args.degree, join=True)


if __name__ == "__main__":
    main()
