"""Another executor's idle resident: a model loaded on the shared ledger, run once so its weights
are home, then left idle until stdin closes (`test_stream_room`)."""

import sys
from pathlib import Path

import tensorfs
import torch
from tensorfs.varena.torch import LedgerGrant, Memory

fd, events, nbytes, store = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])


class Resident(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(nbytes // 4), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight[:1]


writer = tensorfs.Writer(cache_dir=store)
component = writer.component("resident", cls=("torch", "", "Resident"), config={})
component.put("weight", bytes(nbytes), dtype="F32", shape=[nbytes // 4])
checkpoint = writer.commit()
memory = Memory.open(torch.device("cuda", 0), LedgerGrant(0, fd, events))
with torch.device("meta"):
    module = Resident()
model = memory.load(module, ck=checkpoint)
with torch.no_grad(), memory.request("warm"):
    module(torch.zeros(1, device="cuda"))
torch.cuda.synchronize()
print(f"resident {model.nbytes}", flush=True)
sys.stdin.read()
