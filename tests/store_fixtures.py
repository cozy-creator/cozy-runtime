"""A Runtime test's own TensorFS store: D10 checkpoints written into it, and an attempt's
`Context` bound to it as the executor binds one (`weights_store.Outputs`), the checkpoint
items it shows recorded."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime import weights_store
from cozy_runtime.author import Context

DTYPES = {"float16": "F16", "float32": "F32", "uint8": "U8", "int64": "I64"}


def home(path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`path` is the store of this process and of every process it starts."""
    monkeypatch.setenv("TENSORFS_HOME", str(path))
    tensorfs.configure(cache_dir=path)
    return path


def checkpoint(
    components: Mapping[str, Mapping[str, Any]],
    *,
    classes: Mapping[str, str] = {},
    configs: Mapping[str, Mapping[str, object]] = {},
    config_files: Mapping[str, str] = {},
    assets: Mapping[str, Mapping[str, bytes]] = {},
    pipeline: tuple[str, Mapping[str, object]] | None = None,
) -> str:
    """`components`' numpy arrays as one checkpoint in the store (each component's in insertion
    order), with their classes, configs and assets: its header ref."""
    writer = tensorfs.Writer()
    if pipeline is not None:
        writer.pipeline(*pipeline)
    for name in dict.fromkeys([*components, *assets]):
        component = writer.component(
            name,
            cls=classes.get(name, ""),
            config=dict(configs.get(name, {})),
            config_file=config_files.get(name, "config.json"),
        )
        for key, value in components.get(name, {}).items():
            component.put(key, value, dtype=DTYPES[str(value.dtype)], shape=list(value.shape))
        for asset, data in assets.get(name, {}).items():
            component.asset(asset, data)
    return writer.commit().id


def context(
    request_id: str = "test",
    outputs: Mapping[str, int] = {"model": 64 << 20},
    *,
    shown: list[tuple[str, str, str, str]] | None = None,
    code: str = "",
) -> Context:
    """An attempt's `Context` reading its `Weights` sources from the store and writing
    `outputs` into it; each checkpoint item it shows is appended to `shown`."""
    items = shown if shown is not None else []
    bound = weights_store.Outputs(
        request_id=request_id,
        actor="tester",
        code=code or request_id,
        bounds=outputs,
        show=lambda *item: items.append(item),
    )
    return Context(
        request_id,
        math.inf,
        _tensorfs_source=lambda weights: weights_store.open_checkpoint(
            weights.checkpoint_ref, None
        ),
        _tensorfs_output=bound.open,
        _tensorfs_adopt=bound.adopt,
    )


def adopted(shown: list[tuple[str, str, str, str]], slot: str = "model") -> tensorfs.Checkpoint:
    """The checkpoint the attempt last showed for `slot`."""
    _, repo_id, revision, _ = [item for item in shown if item[0] == slot][-1]
    return tensorfs.open(repo_id, revision=revision, local_files_only=True)
