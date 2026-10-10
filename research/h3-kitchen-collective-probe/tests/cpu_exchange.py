import argparse
import datetime
import hashlib
import inspect
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from kitchen_collective_probe.model import Pipeline, ProbeModel
from kitchen_collective_probe.probe import DEGREE, HEADS, token_bounds

from cozy_runtime.author._loader import census
from cozy_runtime.internal import attention_ulysses
from cozy_runtime.internal.parallel.cp import CpComms, install_context_parallel


def worker(rank, rendezvous, folder):
    torch.set_num_threads(1)
    dist.init_process_group(
        "cpu:gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=DEGREE,
        timeout=datetime.timedelta(seconds=60),
    )
    handle = dist.new_group(
        ranks=list(range(DEGREE)), backend="cpu:gloo", timeout=datetime.timedelta(seconds=60)
    )
    pipe = Pipeline()
    model = ProbeModel.for_test(pipe=pipe)
    assert install_context_parallel(
        model, degree=DEGREE, comms=CpComms(handle, rank, torch.device("cpu"))
    ) == ("pipe.dit",)
    assert attention_ulysses.group(pipe.dit._parallel_config) == handle
    assert len(census(pipe).destinations) == 10 and census(pipe).components == ("dit",)
    results = []
    for n in (129, 132):
        gen = torch.Generator().manual_seed(581204)
        packed = torch.randn(1, n, 168, 128, dtype=torch.bfloat16, generator=gen)
        full = packed.chunk(3, dim=2)
        start, stop = token_bounds(rank, n)
        local = tuple(x[:, start:stop] for x in full)
        observation = {}

        def identity(q, k, v, *, observation=observation, full=full):
            observation["shape"] = list(q.shape)
            observation["strides"] = [list(x.stride()) for x in (q, k, v)]
            observation["head_range"] = [(HEADS // DEGREE) * rank, (HEADS // DEGREE) * (rank + 1)]
            for actual, expected in zip((q, k, v), full, strict=True):
                assert torch.equal(
                    actual.view(torch.uint8),
                    expected[:, :, (HEADS // DEGREE) * rank : (HEADS // DEGREE) * (rank + 1)]
                    .contiguous()
                    .view(torch.uint8),
                )
            return q.clone()

        out = attention_ulysses.exchange(*local, handle, identity)
        assert torch.equal(out.view(torch.uint8), local[0].contiguous().view(torch.uint8))
        again = attention_ulysses.exchange(*local, handle, identity)
        assert torch.equal(out.view(torch.uint8), again.view(torch.uint8))
        results.append(
            {
                "tokens": n,
                "local_tokens": stop - start,
                "rank": rank,
                "callback": observation,
                "qkv_and_inverse_bytes_equal": True,
                "repeat_bytes_equal": True,
            }
        )
    Path(folder, f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "results": results,
                "actual_runtime_group_bound": True,
                "component_census_tensors": len(census(pipe).destinations),
            }
        )
    )
    dist.destroy_process_group(handle)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    folder = parser.parse_args().out
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ulysses-gloo-") as temp:
        mp.spawn(
            worker, args=(str(Path(temp, "rendezvous")), str(folder)), nprocs=DEGREE, join=True
        )
    rows = [json.loads((folder / f"rank-{r}.json").read_text()) for r in range(DEGREE)]
    result = {
        "passed": True,
        "ranks": rows,
        "no_local_gpu": True,
        "runtime_exchange_sha256": hashlib.sha256(
            Path(inspect.getfile(attention_ulysses)).read_bytes()
        ).hexdigest(),
        "full_shape_token_counts": [token_bounds(r)[1] - token_bounds(r)[0] for r in range(DEGREE)],
        "limits": "Two-process Gloo permutation passed; CUDA/full-shape attention unqualified",
    }
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
