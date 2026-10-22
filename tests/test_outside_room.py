"""Torch allocates before a request's first pass even when a load filled the device (v6 13633/13640:
H3's 2 MiB of input ids, before its text encoder's pass, found nothing). Standalone, with no
TensorD to key a measured outside room: opening the request lends idle kept pages, and the
allocator callback claims one for the allocation. Real varena on a real GPU, its own process."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tensorfs = pytest.importorskip("tensorfs")
from tensorfs.varena.torch import Declared, Memory, Weights  # noqa: E402

pytestmark = pytest.mark.real_gpu
#: the device cap every real-GPU varena test in a process shares (the first opener sets it)
CAP = 1 << 30
#: 64 MiB blocks: 24 of them are half again the cap, so a GPU-level load fills it
WIDTH, DEPTH = 4096, 24


class Stack(torch.nn.Module):
    _no_split_modules = ("Linear",)

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(DEPTH)
        )

    def forward(self, x: Any) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x


def test_a_request_allocates_before_its_first_pass_after_a_load_filled_the_device(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the allocator callback is CUDA's")
    memory = Memory.open(0, limit=CAP)
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    component = writer.component("stack", cls=("torch", "", "Module"), config={})
    for key in Stack().state_dict():
        component.put(key, bytes(WIDTH * WIDTH * 4), dtype="F32", shape=[WIDTH, WIDTH])
    checkpoint = writer.commit()
    with torch.device("meta"):
        module = Stack()
    vars(module)["_varena"] = Declared(Weights(checkpoint, "stack"))
    model = memory.load(module, "gpu")
    try:
        print(f"torch may hold {memory.native.torch_limit()} bytes after the load")
        with memory.request("before-its-first-pass"):
            ids = torch.ones(1 << 18, dtype=torch.long, device="cuda")  # 2 MiB, as H3's
            assert int(ids.sum()) == 1 << 18
    finally:
        model.close()
