"""Real two-process CP hook flow using SDPA on CPU; no pretend Kitchen kernel."""

import argparse
import datetime
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cp_producer_lifecycle.model import Component, read_report, report_tensor

from cozy_runtime.author import Model, uses_components
from cozy_runtime.internal import attention
from cozy_runtime.internal.parallel.cp import CpComms, gated_call, install_context_parallel


class CPUComponent(Component):
    def forward(self, step: int, seed: int, first: bool):
        hidden = torch.randn(
            (1, 130, 128), generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16
        )
        positions = torch.arange(130).repeat(3, 1).t()
        rotary = self.rope(positions)
        local = self.transformer_blocks[0](hidden_states=hidden, rotary_emb=rotary)
        full = self.proj_out(local)
        assert full.shape == hidden.shape and torch.isfinite(full).all()
        return report_tensor(
            {
                "step": step,
                "local_tokens": local.shape[1],
                "rope_tokens": rotary[0].shape[0],
                "full_tokens": full.shape[1],
            }
        )


class Holder(Model[object]):
    dit: Any

    @uses_components("dit")
    def call(self):
        return self.dit


def worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group(
        "cpu:gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    handle = dist.new_group(ranks=[0, 1], backend="cpu:gloo")
    root = CPUComponent()
    root.transformer_blocks[0].attn.processor._attention_backend = attention._member(
        attention.BY_NAME["sdpa"]
    )
    holder = Holder.for_test(dit=root)
    assert install_context_parallel(
        holder, degree=2, comms=CpComms(handle, rank, torch.device("cpu"))
    ) == ("dit",)
    rows = []
    for step in (4, 5, 5, 6):
        with torch.inference_mode(), gated_call():
            rows.append(read_report(root(step, 581204, step in (4, 6))))
    assert all(row["local_tokens"] == row["rope_tokens"] == 65 for row in rows)
    Path(folder, f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "passed": True, "rows": rows}, indent=2) + "\n"
    )
    dist.destroy_process_group(handle)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp, "join")), str(args.out)), nprocs=2, join=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "scope": (
                    "actual tiny-H3 CP hook/carrier with CPU SDPA at even130; "
                    "Gloo cannot gather uneven outputs; GPU fixture stays4225"
                ),
                "ranks": [
                    json.loads((args.out / f"rank-{rank}.json").read_text()) for rank in range(2)
                ],
            },
            indent=2,
        )
        + "\n"
    )
