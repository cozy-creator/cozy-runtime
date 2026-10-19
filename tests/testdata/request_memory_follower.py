"""CPU executor for request-memory transport tests; no simulated memory implementation."""

import argparse
import socket
from pathlib import Path

import torch
from tensorfs.varena.torch import Memory

from cozy_runtime.internal.executor import Executor, serve
from cozy_runtime.internal.seam import Channel

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--world", type=int, required=True)
parser.add_argument("--rank-fd", type=int, required=True)
parser.add_argument("--leader")
args = parser.parse_args()
channel = Channel(socket.socket(fileno=args.rank_fd))
worker = Executor(channel, args.root, rank=args.rank, world=args.world)
worker.memory = Memory.open(torch.device("cpu"))
worker.torch, worker.device_kind, worker.ready = torch, "cpu", True
construction = worker._construction_child("construction")
construction.ready = True
worker.constructions["construction"] = construction
raise SystemExit(serve(worker, channel))
