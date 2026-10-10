"""Actual two-process Diffusers CP hook ordering, without CUDA attention."""

import argparse
import datetime
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

from cozy_runtime.author import Model, uses_components
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import attention
from cozy_runtime.internal import attention_producer as producer
from cozy_runtime.internal.parallel.cp import CpComms, install_context_parallel


class Holder(Model[object]):
    dit: Any

    @uses_components("dit")
    def call(self):
        return self.dit


def component():
    root = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=128,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=0,
        ffn_dim=32,
        in_channels=2,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=4,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=8,
        rope_freq_dim=16,
    )
    for block in root.transformer_blocks:
        block.attn.to(dtype=torch.bfloat16)
    return root


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
    roots = [component(), component()]
    n = 129
    positions = torch.arange(n).repeat(3, 1).t()
    full = [root.rope(positions) for root in roots]
    for root in roots:
        for block in root.transformer_blocks:
            block.attn.processor._attention_backend = attention._member(attention.BY_NAME["sdpa"])
        model = Holder.for_test(dit=root)
        assert install_context_parallel(
            model, degree=2, comms=CpComms(handle, rank, torch.device("cpu"))
        ) == ("dit",)
    layout = AttentionLayout(n, 17, 4, 4)
    with producer.request_scope() as empty, attention_scope(layout):
        roots[0].rope(positions)
        assert not empty.parallel_rotary
    fixed = attention._member(attention.BY_NAME["kitchen-sol-producer-fp4-fixed-means-shared-qkv"])
    for root in roots:
        root.transformer_blocks[0].attn.processor._attention_backend = fixed
    with producer.request_scope() as state, attention_scope(layout):
        keys = []
        for i, root in enumerate(roots):
            local = root.rope(positions)
            key = root.transformer_blocks[0].attn._cozy_producer_rotary_key
            keys.append(key)
            held = state.parallel_rotary[key]
            for got, reference, part in zip(held, full[i], local, strict=True):
                assert torch.equal(got, reference) and got.shape == (n, 96)
                assert torch.equal(part, reference.tensor_split(2, dim=0)[rank])
                assert got.untyped_storage().data_ptr() == part.untyped_storage().data_ptr()
        assert keys[0] is not keys[1] and len(state.parallel_rotary) == 2
        original = state.parallel_rotary[keys[0]]
        with producer.request_scope() as inner:
            roots[0].rope(positions)
            assert len(inner.parallel_rotary) == 1
        assert not inner.parallel_rotary and state.parallel_rotary[keys[0]] is original
    assert not state.parallel_rotary and producer._STATE.get() is None
    with producer.request_scope() as next_state, attention_scope(layout):
        assert not next_state.parallel_rotary
        roots[1].rope(positions)
        assert keys[0] not in next_state.parallel_rotary and keys[1] in next_state.parallel_rotary
    Path(folder, f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "passed": True,
                "full_rope_shape": [129, 96],
                "local_tokens": 65 if rank == 0 else 64,
                "no_clone_same_storage": True,
                "component_isolation": True,
                "nested_and_next_request_reset": True,
                "default_backend_no_retention": True,
            }
        )
    )
    dist.destroy_process_group(handle)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp, "join")), str(args.out)), nprocs=2, join=True)
    rows = [json.loads((args.out / f"rank-{r}.json").read_text()) for r in range(2)]
    (args.out / "summary.json").write_text(
        json.dumps({"passed": True, "ranks": rows}, indent=2) + "\n"
    )
