"""A sharded diffusers component, reviewed in ``testdata/native_source/sharded-registry.json``.

The registry binds names, shapes and dtypes only, so bodies are generated per run.
"""

from __future__ import annotations

import json
import random
import struct

REPOSITORY = "example/sharded"
REVISION = "b" * 40
PROFILE = "fixture/sharded/1"
INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
SHARDS = 8
TENSORS_PER_SHARD = 2
ROWS = COLUMNS = 4096  # float16: 32 MiB per tensor, 64 MiB per shard


def shard_name(index: int) -> str:
    return f"transformer/diffusion_pytorch_model-{index + 1:05d}-of-{SHARDS:05d}.safetensors"


def keys(shard: int) -> list[str]:
    return [f"blocks.{shard * TENSORS_PER_SHARD + i}.weight" for i in range(TENSORS_PER_SHARD)]


def header(shard: int) -> bytes:
    size = ROWS * COLUMNS * 2
    fields = {
        key: {"dtype": "F16", "shape": [ROWS, COLUMNS], "data_offsets": [i * size, (i + 1) * size]}
        for i, key in enumerate(keys(shard))
    }
    raw = json.dumps(fields, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def body_length(shard: int) -> int:
    return len(header(shard)) + TENSORS_PER_SHARD * ROWS * COLUMNS * 2


def index() -> bytes:
    weight_map = {
        key: shard_name(shard).rsplit("/", 1)[1] for shard in range(SHARDS) for key in keys(shard)
    }
    return json.dumps({"metadata": {}, "weight_map": weight_map}, sort_keys=True).encode()


def files(seed: int = 7) -> dict[str, bytes]:
    """Every member's exact bytes; tensor values are arbitrary but reproducible."""
    generator = random.Random(seed)
    out = {INDEX: index()}
    for shard in range(SHARDS):
        payload = generator.randbytes(TENSORS_PER_SHARD * ROWS * COLUMNS * 2)
        out[shard_name(shard)] = header(shard) + payload
    return out


def order() -> list[list[str]]:
    return [["transformer", key] for shard in range(SHARDS) for key in keys(shard)]
