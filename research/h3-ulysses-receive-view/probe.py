"""CPU/Gloo proof of the real Ulysses send/unpack seam, including view custody."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import weakref
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def worker(rank: int, world: int, store: str, expect_view: bool, directory: str) -> None:
    import torch
    import torch.distributed as dist

    from cozy_runtime.internal.parallel import turn, ulysses

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=world,
        timeout=timedelta(seconds=30),
    )
    records = []
    try:
        sizes = [3, 5]
        total, heads, dim = sum(sizes), 8, 4
        matrix = [[rows] * world for rows in sizes]
        first = sum(sizes[:rank])
        # One-head edges and a two-head middle exercise the pipelined tail geometry.
        groups = ulysses._groups(heads, 4, world)
        assert groups == [(0, 1), (2, 2), (6, 1)]
        for batch in (1, 2):
            for dtype in (torch.float32, torch.bfloat16):
                for narrow in (False, True):
                    torch.manual_seed(298)
                    full = [torch.randn(batch, total, heads, dim, dtype=dtype) for _ in range(3)]
                    sources = tuple(x[:, first:first + sizes[rank]] for x in full)
                    assert all(not x.is_contiguous() for x in sources) if batch > 1 else True
                    packer = ulysses._Packer(sources[0], narrow)

                    def send(start: int, local: int):
                        return ulysses._send(
                            dist.group.WORLD, None, packer, sources, start, total,
                            local, matrix, rank, narrow, None,
                        )

                    arriving = send(*groups[0])
                    retained = []
                    for index, (start, local) in enumerate(groups):
                        current = arriving
                        arriving = send(*groups[index + 1]) if index + 1 < len(groups) else None
                        receive_ref = weakref.ref(current.values)
                        receive_pointer = current.values.untyped_storage().data_ptr()
                        actual = ulysses._unpack(packer, current)
                        eligible = batch == 1 and not narrow
                        aliases = all(x.untyped_storage().data_ptr() == receive_pointer for x in actual)
                        if expect_view:
                            assert aliases == eligible, (rank, batch, dtype, narrow, "receive alias", aliases)
                            assert (current.operands is None) == eligible
                        else:
                            assert not aliases
                            assert current.operands is not None
                        begin = start + rank * local
                        for got, source in zip(actual, full, strict=True):
                            want = source[:, :, begin:begin + local]
                            if narrow:
                                want = turn.law_back(*turn.law(want))
                            assert got.shape == want.shape
                            assert got.dtype == want.dtype
                            assert got.stride(-1) == 1
                            assert torch.equal(got.view(torch.uint8), want.contiguous().view(torch.uint8))
                        del current
                        gc.collect()
                        if expect_view and eligible:
                            assert receive_ref() is not None, "view lost receive storage with _Arriving"
                        # Later receives/unpacks must not overwrite earlier views.
                        retained.extend((x, x.clone()) for x in actual)
                        records.append(dict(
                            rank=rank, batch=batch, dtype=str(dtype), narrow=narrow,
                            start=start, local_heads=local, rows=total, aliases=aliases,
                            strides=[list(x.stride()) for x in actual],
                            avoided_bytes=3 * batch * total * local * dim * sources[0].element_size()
                            if aliases else 0,
                        ))
                    for got, original in retained:
                        assert torch.equal(got.view(torch.uint8), original.view(torch.uint8))
                    packer.settle()

        # An unsuccessful transport wait must fail before producing an operand view.
        like = torch.empty(1, 3, 8, 4)
        packer = ulysses._Packer(like, False)
        values = torch.empty(3, 8, 1, 2, 4)

        def fail() -> None:
            raise RuntimeError("injected transport failure")

        group = ulysses._Arriving(fail, values, None, None if expect_view else values.clone(), None)
        try:
            ulysses._unpack(packer, group)
        except RuntimeError as exc:
            assert str(exc) == "injected transport failure"
        else:
            raise AssertionError("failed wait exposed operands")
        finally:
            packer.settle()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main() -> None:
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-view", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "build" / str(os.getpid())
    directory.mkdir(parents=True)
    mp.spawn(worker, args=(2, str(directory / "store"), args.expect_view, str(directory)), nprocs=2)
    rows = [r for rank in range(2) for r in json.loads((directory / f"rank-{rank}.json").read_text())]
    result = dict(
        scope="CPU Gloo send/unpack storage and values only; no CUDA/Varena/performance qualification",
        expected_receive_views=args.expect_view, cases=len(rows), records=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS {len(rows)} actual Gloo send/unpack cases; views={args.expect_view}")


if __name__ == "__main__":
    main()
