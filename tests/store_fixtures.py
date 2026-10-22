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
from cozy_runtime.internal import executor

DTYPES = {"float16": "F16", "float32": "F32", "uint8": "U8", "uint16": "BF16", "int64": "I64"}

#: One shown checkpoint item: output, repo, revision, variant.
type Shown = list[tuple[str, str, str, str, str]]


def home(path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The store at `path`, which an executor this test drives opens as its machine's."""
    monkeypatch.setenv("TENSORFS_HOME", str(path))
    # These in-process executors receive the same root as a freshly sealed child.
    monkeypatch.setitem(executor._SEALED, "TENSORFS_HOME", str(path))
    return path


def checkpoint(
    store: Path,
    components: Mapping[str, Mapping[str, Any]],
    *,
    classes: Mapping[str, tuple[str, str, str]] = {},
    configs: Mapping[str, Mapping[str, object]] = {},
    assets: Mapping[str, Mapping[str, bytes]] = {},
    pipeline: tuple[tuple[str, str, str], Mapping[str, object]] | None = None,
) -> str:
    """`components`' numpy arrays (a uint16 one is bf16) as one checkpoint in `store`, each
    component's in insertion order, with their classes, configs and assets: its header ref."""
    writer = tensorfs.Writer(cache_dir=store)
    if pipeline is not None:
        writer.pipeline(*pipeline)
    for name in dict.fromkeys([*components, *assets]):
        component = writer.component(
            name,
            cls=classes.get(name, ("", "", "")),
            config=dict(configs.get(name, {})),
        )
        for key, value in components.get(name, {}).items():
            component.put(
                key, memoryview(value), dtype=DTYPES[str(value.dtype)], shape=list(value.shape)
            )
        for asset, data in assets.get(name, {}).items():
            component.asset(asset, data)
    ref = writer.commit().id
    assert ref is not None
    return ref


def context(
    store: Path,
    request_id: str = "test",
    outputs: Mapping[str, int] = {"model": 64 << 20},
    *,
    shown: Shown | None = None,
    destination: str = "",
) -> Context:
    """An attempt's `Context` reading its `Weights` sources from `store` and writing `outputs`
    into it, uploaded to `destination` (`org/name@tag`) when it names one; each checkpoint item
    it shows is appended to `shown`."""
    items = shown if shown is not None else []
    bound = weights_store.Outputs(
        store,
        request_id=request_id,
        actor="tester",
        code=request_id,
        bounds=outputs,
        show=lambda *item: items.append(item),
        destination=destination,
    )
    return Context(
        request_id,
        math.inf,
        _tensorfs_source=lambda weights: tensorfs.open(
            weights.checkpoint_ref, cache_dir=store, local_files_only=True
        ),
        _tensorfs_output=bound.open,
        _tensorfs_adopt=bound.adopt,
    )


def adopted(store: Path, shown: Shown, slot: str = "model") -> tensorfs.Checkpoint:
    """The checkpoint the attempt last showed for `slot`."""
    _, repo_id, revision, _, _ = [item for item in shown if item[0] == slot][-1]
    assert repo_id == weights_store.DERIVED
    return tensorfs.open(revision, cache_dir=store, local_files_only=True)
