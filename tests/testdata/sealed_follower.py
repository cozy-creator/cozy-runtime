"""A real group follower that loads one weight set as an executor's prepare does: every
request to the machine (header, assets, sealed layouts) goes through rank 0."""

from __future__ import annotations

import argparse
import hashlib
import socket
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights import Weights
from test_weight_plane import HIDDEN, load, run


def digest(output: Any) -> str:
    return hashlib.sha256(output.numpy().tobytes()).hexdigest()


def adopt(executor: Any, root: Path, manifest: str, construction: str) -> dict[str, Any]:
    checkpoint = Checkpoint(
        manifest,
        sourced=lambda name: executor._model_source(manifest, name),
    )
    weights = Weights(torch, torch.device("cpu"), "cpu")
    model = load(weights, checkpoint, construction, tiers=executor._tiers(False, True))
    x = torch.rand(4, HIDDEN, generator=torch.Generator().manual_seed(3)).half()
    coarse = run(model, x, passes=1)[0]
    sealed = model.residency.components["stack"].sealed
    # Later, on its own: this rank moves to the finer layout and asks the machine for it.
    refined = model.residency._refine(model.residency.components["stack"].key)
    return {
        "ok": True,
        "store": checkpoint.store is not None,
        "assets": len(checkpoint.model_assets()),
        "sealed": [sealed, model.residency.components["stack"].sealed],
        "refined": refined,
        "outputs": [digest(coarse), digest(run(model, x, passes=1)[0])],
        "refused": executor._ask_tier("stack", "layout") is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world", type=int)
    parser.add_argument("--rank-fd", type=int)
    parser.add_argument("--leader")
    args = parser.parse_args()
    torch.set_num_threads(1)
    _capture_seal()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    executor: Any = Executor(channel, args.root, rank=args.rank, world=args.world)
    while (command := channel.recv()) is not None:
        if command["cmd"] == "shutdown":
            return
        reply = adopt(executor, args.root, command["manifest"], command["construction"])
        channel.send({**reply, "reply": "adopt"})


if __name__ == "__main__":
    main()
