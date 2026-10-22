"""A CPU follower executor for the request-memory tests: a real memory over host pages."""

import argparse
import os
import socket
from pathlib import Path

from tensorfs.varena import GpuId
from tensorfs.varena.torch import Memory

from cozy_runtime.internal import execution
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
worker.memory = Memory.open(GpuId(f"cozy-follower-{os.getpid()}", "cpu", 1 << 30))
worker.execution, worker.store = execution, args.root / "store"
worker.store.mkdir(parents=True, exist_ok=True)
worker.device_kind, worker.ready = "cpu", True
construction = worker._construction_child("construction")
construction.ready = True
worker.constructions["construction"] = construction
raise SystemExit(serve(worker, channel))
