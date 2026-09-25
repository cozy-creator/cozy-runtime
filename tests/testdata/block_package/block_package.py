"""One square block whose size is its checkpoint config: sized device residency tests."""

from __future__ import annotations

from typing import Any, cast

import msgspec
import torch

from cozy_runtime.author import App, Config, Loader, Model, uses_components

app = App()


class BlockPipe:
    def __init__(self, config: Config) -> None:
        size = int(cast(dict[str, Any], config.mapping())["block"]["size"])
        self.components = {"block": torch.nn.Linear(size, size, bias=False)}


class Block(Model[BlockPipe]):
    pipe: BlockPipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(BlockPipe, factory=BlockPipe)

    @uses_components("block")
    def apply(self) -> float:
        block = self.pipe.components["block"]
        ones = torch.ones(1, block.weight.shape[1], device=block.weight.device)
        with torch.no_grad():
            return float(block(ones).sum())


class Input(msgspec.Struct):
    value: int = 1


class Output(msgspec.Struct):
    value: int
    checksum: float


@app.entrypoint
def generate(payload: Input, block: Block) -> Output:
    return Output(payload.value, block.apply())
