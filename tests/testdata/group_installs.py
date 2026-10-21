"""A CPU follower for group-install tests: one construction whose execution plan waits for a
kernel set (`Plan.when_ready`) and whose attention site starts on a kernel rank 0 does not
serve. Under `--root`: the set is built for GPU n once `built-<n>` exists, its build failed
once `failed-<n>` does, and its install writes `installed-<n>`, or raises once `refuse-<n>`
exists. The machine's kernel store and a card's attention kernels need CUDA; these files
stand in for the store, everything else is the executor's own path."""

import argparse
import functools
import socket
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models.attention_processor import Attention

from cozy_runtime.author import Model
from cozy_runtime.author.kernels import KernelsUnavailable, Plan, TritonKernels
from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor, serve
from cozy_runtime.internal.seam import Channel

KEY = "construction#glued"
GLUE = TritonKernels("test-group-glue", [])
FACTS = DeviceFacts("cpu", "fixture", 0, "", "", 0)


def glued(root: Path, rank: int) -> Model[object]:
    """A model whose execution plan installs GLUE when Runtime says every GPU has it."""

    def install(launch: object) -> None:
        if (root / f"refuse-{rank}").exists():
            raise RuntimeError(f"GPU {rank} cannot install")
        (root / f"installed-{rank}").write_text("glue")

    class Glued(Model[object]):
        def execution_plan(self, plan: Plan) -> str:
            assert plan.when_ready is not None
            plan.when_ready(GLUE, install)
            return "glue"

    return Glued.for_test()


def built(root: Path, rank: int) -> object:
    """`triton_kernels.ready` for GPU `rank`, read from the marker files."""

    def ready(kernels: TritonKernels, *, device: object) -> object:
        if (root / f"failed-{rank}").exists():
            raise KernelsUnavailable("kernels_failed", f"{kernels.name}: its build failed")
        return (lambda *args: None) if (root / f"built-{rank}").exists() else None

    return ready


def never(kernels: TritonKernels, *, device: object) -> object:
    raise AssertionError("a construction serves while its kernels compile: nothing waits")


def construction(worker: Executor, root: Path) -> Executor:
    """The construction both ranks hold: its deferred install, and one attention site."""
    child = worker._construction_child("construction")
    child._group_model_key = KEY
    child._execution_plan(glued(root, worker.rank), torch, never, built(root, worker.rank))
    site = Attention(query_dim=32, heads=2, dim_head=16)
    child.backend = SimpleNamespace(roots={"dit": site})
    child.device_facts = FACTS
    child._reselect = functools.partial(
        attention.select, FACTS, child.backend.roots, degree=worker.world
    )
    child.attention = child._reselect()
    return child


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--rank-fd", type=int, required=True)
    parser.add_argument("--leader")
    args = parser.parse_args()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    worker = Executor(channel, args.root, rank=args.rank, world=args.world)
    worker.torch, worker.device_kind, worker.ready = torch, "cpu", True
    held = construction(worker, args.root)
    # This GPU starts on a kernel rank 0 does not serve: what it serves after the agreement
    # is rank 0's, or the group refuses.
    processor = held.backend.roots["dit"].processor
    processor._attention_backend = attention._member(attention.known()["flash-attn3"])
    worker.constructions["construction"] = held
    raise SystemExit(serve(worker, channel))
