"""Real follower process for the existing Attention begin/end command lifetime."""

import argparse
import socket
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention

from cozy_runtime.author import AttentionProducer, register_attention
from cozy_runtime.internal import attention
from cozy_runtime.internal.executor import Executor, _capture_seal
from cozy_runtime.internal.executor_commands import Attention, decode
from cozy_runtime.internal.seam import Channel


class State:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _never(call: object, *args: object, **kwargs: object) -> object:
    raise AssertionError("no attention call reaches a producer here")


def main() -> None:
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
    name = "test-cp-producer"
    register_attention(
        AttentionProducer(
            name,
            "kitchen-sol-fp4-fixed-means",
            _never,
            lambda module: None,
            state=State,
            context_parallel=True,
        )
    )
    root.attn.processor._attention_backend = attention._member(attention.known()[name])
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
            attention_command = decode(command)
            assert isinstance(attention_command, Attention)
            reply = follower.select_attention(attention_command)
        elif kind == "state-seed":
            assert follower._producer_hold is not None
            previous = State()
            follower._producer_hold[1][name] = previous
            reply = {"ok": True, "rank": args.rank}
        elif kind == "state-read":
            reply = {
                "ok": True,
                "rank": args.rank,
                "live": follower._producer_hold is not None,
                "closed": previous is not None and previous.closed,
            }
        else:
            raise AssertionError(kind)
        channel.send({**reply, "reply": kind})


if __name__ == "__main__":
    main()
