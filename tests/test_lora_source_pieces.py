"""A LoRA source's factors (H3's turbo overlay) are another loaded model's tensors, read by the
bake outside that model's forward. Each is bound as its own piece while it is copied: a few MB a
site, not the source whole (cr-203 U9 (c): the overlay's 620 MB mapped per read kept H3 off 8 GB).
Real TensorFS checkpoint, real varena Memory on host pages, the Runtime's own Backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import tensorfs
import torch
from tensorfs.varena import GpuId
from tensorfs.varena.torch import Memory
from torch import nn

from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.weights import Backend

DIM, RANK, SITES, SCALE = 32, 4, 6, 0.5


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.lin(x))


class Net(nn.Module):
    _no_split_modules: ClassVar[list[str]] = ["Block"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Factors(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_down = nn.Parameter(torch.empty(RANK, DIM))
        self.lora_up = nn.Parameter(torch.empty(DIM, RANK))


class Site(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = Factors()


class Overlay(nn.Module):
    """Per-site factors under the consumer's own paths, one piece each."""

    def __init__(self, pieces: bool) -> None:
        super().__init__()
        if pieces:
            self._no_split_modules = ["Factors"]
        self.sites = [Site() for _ in range(SITES)]
        self.blocks = nn.ModuleList(self.sites)


class Source:
    """A consumer hook that names another model's factors (`weights.LoRASource`)."""

    def __init__(self, overlay: Overlay, index: int) -> None:
        self.overlay, self.index = overlay, index
        self._cozy_baked = False

    def __call__(self, module: nn.Module, args: tuple[object, ...], out: torch.Tensor) -> None:
        assert self._cozy_baked, "a baked weight carries the delta; the hook adds nothing"

    def lora_factors(self) -> tuple[object, torch.Tensor, torch.Tensor, float]:
        factors = self.overlay.sites[self.index].lin
        return factors, factors.lora_down, factors.lora_up, SCALE


def _store(tmp_path: Path, seed: int) -> tuple[tensorfs.Checkpoint, dict[str, torch.Tensor]]:
    """Weights and factors from `seed`: the same seed is the same bytes (one fingerprint)."""
    torch.manual_seed(seed)
    values = {f"blocks.{i}.lin.weight": torch.randn(DIM, DIM) for i in range(2)}
    factors = {
        f"blocks.{i}.lin.lora_{role}": torch.randn(*shape)
        for i in range(SITES)
        for role, shape in (("down", (RANK, DIM)), ("up", (DIM, RANK)))
    }
    writer = tensorfs.Writer(cache_dir=tmp_path / "store")
    for name, tensors in (("dit", values), ("overlay", factors)):
        component = writer.component(name, cls=("torch", "", "Module"), config={})
        for key, value in tensors.items():
            component.put(key, value.numpy().tobytes(), dtype="F32", shape=list(value.shape))
        del component
    native = writer.commit()
    return native, {**values, **factors}


def _backend(native: tensorfs.Checkpoint, table: str) -> tuple[Backend, Memory, list[Any]]:
    checkpoint = Checkpoint(native)
    rows = [*checkpoint.rows("overlay"), *checkpoint.rows("dit")]
    memory = Memory.open(GpuId(f"cozy-lora-pieces-{os.getpid()}-{table}", "cpu", 1 << 30))
    backend = Backend.for_script(
        {"dit": checkpoint, "overlay": checkpoint},
        rows,
        torch=torch,
        memory=memory,
        device=torch.device("cpu"),
        construction="lora-pieces",
        release="lora-pieces",
    )
    return backend, memory, rows


@pytest.mark.parametrize("pieces", [True, False], ids=["pieces", "no-blocks"])
def test_a_bake_reads_each_sources_factors_in_its_piece(tmp_path: Path, pieces: bool) -> None:
    native, values = _store(tmp_path, 5 + pieces)  # each case its own bytes: each bakes fresh
    backend, memory, rows = _backend(native, str(pieces))
    with torch.device("meta"):
        overlay, net = Overlay(pieces), Net()
    for i, block in enumerate(net.blocks):
        cast(Block, block).lin.register_forward_hook(Source(overlay, i))
    loaded = [
        backend._register(name, root, [r.key for r in rows if r.component == name])
        for name, root in (("overlay", overlay), ("dit", net))
    ]
    try:
        x = torch.randn(3, DIM)
        with torch.no_grad(), memory.request("lora-pieces") as r:
            y = net(x)
        want = x
        for i in range(2):
            down, up = values[f"blocks.{i}.lin.lora_down"], values[f"blocks.{i}.lin.lora_up"]
            want = want @ (values[f"blocks.{i}.lin.weight"] + SCALE * up @ down).T
        torch.testing.assert_close(y, want)
        assert backend.bakes["dit"].baked == 2
        # the binds the bake made: the overlay's, once the DiT's first block bound
        assert r.ended is not None
        trace = [(ref.set.rsplit("/", 1)[1], ref.block) for ref in r.ended.trace]
        read = [block for name, block in trace[trace.index(("dit", 0)) :] if name == "overlay"]
        regions = {path: block.region for path, block in loaded[0].blocks.items()}
        if pieces:  # each read binds its own site's factors: 2 of the 6 pieces
            assert len(regions) == SITES and loaded[0].common is None
            assert read == [regions["blocks.0.lin"], regions["blocks.1.lin"]], (read, regions)
        else:  # a source that declares no pieces is read whole (its common), and still works
            common = loaded[0].common
            assert not regions and common is not None and read == [common.region] * 2, read
    finally:
        for model in reversed(loaded):
            model.close()
        backend.close()
        native.close()
