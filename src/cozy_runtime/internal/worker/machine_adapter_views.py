"""CPU-only immutable adapter views, using the ordinary native writer and local roots."""

from __future__ import annotations

import hashlib
import threading
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author import ObjectRef
from cozy_runtime.internal import fill, lora_contract, storage_admission
from cozy_runtime.internal.canonical import Json

from . import machine_model_defaults, source_steps
from .source_calls import diagnosis
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker


class Adapter(msgspec.Struct, frozen=True):
    selected: machine_model_defaults.Selected
    component: str
    source_component: str = "adapter"
    scale: str = "1"


_LOCKS: WeakValueDictionary[str, threading.Lock] = WeakValueDictionary()
_LOCK = threading.Lock()


def compose(
    worker: Worker,
    base: Mapping[str, Json],
    adapters: Sequence[Adapter],
    *,
    check: Callable[[], None],
    cancellation: machine_model_defaults.Cancellation | None = None,
    native_base: bool = False,
) -> dict[str, Json]:
    """Keep the original base reference and return one ordinary composed component view.

    Native ModelArtifact inputs must already have an authorized hold owned by the caller.
    Catalog/source choices use the normal materialization path before any source is read.
    This function neither requests a GPU lane nor imports Torch.
    """
    if not adapters:
        return dict(base)
    assert worker.workspace is not None
    store = fill.store(worker.workspace.store_root)
    selected = msgspec.convert(base, machine_model_defaults.Selected)
    closure = [msgspec.to_builtins(row.selected) for row in adapters]
    if not native_base:
        closure.insert(0, dict(base))
    check()
    machine_model_defaults.materialize(worker, closure, cancellation=cancellation)
    identity = (
        "sha256:"
        + hashlib.sha256(
            canonical_json.encode(
                [
                    lora_contract.FORMAT,
                    msgspec.to_builtins(selected.manifest),
                    msgspec.to_builtins(adapters),
                ]
            )
        ).hexdigest()
    )
    local = "adapters-" + identity[7:47]
    with _LOCK:
        lock = _LOCKS.setdefault(local, threading.Lock())
    with lock:
        check()
        try:
            held = store.resolve_local(local)
            manifest = ObjectRef(held["manifest_digest"], held["manifest_length"])
            store.verify_checkpoint_source("local/" + local, manifest.digest, manifest.length)
        except fill.tensorfs_module().errors.Refusal:
            manifest = None
        if manifest is None:
            launch = source_steps.Launch(
                store=str(worker.workspace.store_root),
                parent_pid=os.getpid(),
                step=source_steps.PrepareAdapterView(
                    identity=identity,
                    base=ObjectRef(selected.manifest.digest, selected.manifest.length),
                    adapters=tuple(
                        source_steps.AdapterViewInput(
                            ObjectRef(row.selected.manifest.digest, row.selected.manifest.length),
                            row.component,
                            row.source_component,
                            row.scale,
                        )
                        for row in adapters
                    ),
                ),
            )
            with storage_admission.admit(
                storage_admission.native_write(
                    worker.workspace.store_root,
                    source_steps.MAX_BYTES,
                )
            ):
                status, output, stderr = source_steps.run(
                    "cozy_runtime.internal.adapter_view_child",
                    launch,
                    observe=lambda _sample: check(),
                )
                if status or not output:
                    raise WorkspaceRefusal(
                        "adapter preparation failed: " + diagnosis(status, stderr, "")
                    )
                answer = source_steps.answer(output, source_steps.AdapterViewReady)
                if isinstance(answer, source_steps.Refused):
                    raise WorkspaceRefusal("adapter preparation refused: " + answer.code)
                check()
                manifest = answer.manifest
                try:
                    current: bytes | None = store.repo_get("local", local)
                except fill.tensorfs_module().errors.Refusal:
                    current = None
                store.replace_local(current, local, identity, manifest.digest, manifest.length)
                store.derived_dispose(identity)
        header = store.manifest(manifest.digest)["header"]
        if header is None:
            raise WorkspaceRefusal("prepared adapter view has no header")
        graph = lora_contract.read(
            fill.tensorfs_module().parse_header(header)["configs"][lora_contract.GRAPH_CONFIG]
        )
        result = dict(base)
        result["composed"] = {
            "model": "local/" + local,
            "manifest": manifest.digest,
            "manifest_length": manifest.length,
        }
        result["adapters"] = [
            {
                "model": row.selected.repository,
                "manifest": row.selected.manifest.digest,
                "manifest_length": row.selected.manifest.length,
                "component": reference.component,
                "source_component": reference.source_component,
                "scale": row.scale,
                "release": row.selected.release,
                "lane": row.selected.lane,
            }
            for row, reference in zip(adapters, graph.adapters, strict=True)
        ]
        return result
