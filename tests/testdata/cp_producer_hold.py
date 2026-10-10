"""Real follower process for the existing Attention begin/end command lifetime."""

import argparse
import socket
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

from cozy_runtime.internal import attention
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor_commands import decode
from cozy_runtime.internal.seam import Channel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-fd", type=int, required=True)
    parser.add_argument("--leader", type=int)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    _capture_seal()
    channel = Channel(socket.socket(fileno=args.rank_fd))
    follower = Executor(channel, args.root, rank=args.rank, world=args.world)
    target = Executor(
        channel, args.root, rank=args.rank, world=args.world, host=follower, construction="proof"
    )
    root = torch.nn.Module()
    root.attn = MiniMaxH3Attention(hidden_size=256, heads=2, dim_head=128)
    root.attn.processor._attention_backend = attention._member(
        attention.BY_NAME["kitchen-sol-producer-fp4-fixed-means-shared-qkv"]
    )
    target.backend = SimpleNamespace(roots={"dit": root})
    target.attention_defaults = attention.snapshot(target.backend.roots)
    follower.constructions["proof"] = target
    previous = None
    while (command := channel.recv()) is not None:
        kind = command["cmd"]
        if kind == "shutdown":
            return
        if kind == "hello":
            reply = follower.hello()
        elif kind == "attention":
            reply = follower.select_attention(decode(command))
        elif kind == "state-seed":
            assert follower._producer_hold is not None
            previous = follower._producer_hold[1]
            previous.statistics[("dit", "site")] = (4, (129,), object(), object())
            reply = {"ok": True, "rank": args.rank}
        elif kind == "state-read":
            reply = {
                "ok": True,
                "rank": args.rank,
                "live": follower._producer_hold is not None,
                "cleared": previous is not None and not previous.statistics,
            }
        else:
            raise AssertionError(kind)
        channel.send({**reply, "reply": kind})


if __name__ == "__main__":
    main()
