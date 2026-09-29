"""Runtime preparation requests verified disk custody from TensorFS, never the Host."""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_model_defaults, machine_model_resolve
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker

_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MANIFEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def models(
    delegation: bytes, native: Iterable[pb.NativeModelBinding] = ()
) -> list[pb.DownloadModelRef | pb.DownloadAdapterRef]:
    """An exact base/adapter closure. Native base checkpoints are already retained."""
    selected = documents.parse(delegation, pb.DownloadDelegation) if delegation else None
    result: dict[tuple[str, str], pb.DownloadModelRef | pb.DownloadAdapterRef] = {}
    for row in () if selected is None else selected.models:
        result.setdefault((row.model, row.manifest), row)
        for adapter in row.adapters:
            result.setdefault((adapter.model, adapter.manifest), adapter)
    for binding in native:
        for adapter in binding.adapters:
            result.setdefault((adapter.model, adapter.manifest), adapter)
    for model in result.values():
        if not _MODEL.fullmatch(model.model) or not _MANIFEST.fullmatch(model.manifest):
            raise WorkspaceRefusal("model preparation requires an exact repository checkpoint")
    return list(result.values())


def ensure(
    worker: Worker,
    delegation: bytes,
    *,
    hub: str = "",
    native: Iterable[pb.NativeModelBinding] = (),
) -> None:
    """Ensure each exact selected checkpoint, retaining the entire preparation's closure.

    TensorFS owns shared flights, disk admission, transfer retry and integrity. A cache hit
    needs no Hub. A miss uses only the machine's scoped authority for the selected Hub.
    """
    selected = models(delegation, native)
    if not selected:
        return
    if worker.workspace is None:
        raise WorkspaceRefusal("model preparation needs the machine's TensorFS workspace")
    facade = fill.tensorfs_module()
    store = fill.store(worker.workspace.store_root)
    keep = sorted({row.manifest for row in selected})
    authority = machine_model_resolve.registration(worker, hub)
    origin = authority.origin if authority is not None else hub

    def progress(event: dict[str, object]) -> None:
        if worker.stop.is_set():
            raise WorkspaceRefusal("machine stopped during model preparation")
        worker.note(
            "model_preparation",
            f"{event.get('bytes_done', 0)}/{event.get('bytes_total', 0)} bytes verified",
        )

    def land(row: pb.DownloadModelRef | pb.DownloadAdapterRef) -> None:
        if worker.stop.is_set():
            raise WorkspaceRefusal("machine stopped during model preparation")
        try:
            raw = store.manifest(row.manifest)["manifest"]
            store.verify_checkpoint_source(row.model, row.manifest, len(raw))
            return
        except facade.errors.Refusal:
            pass
        if not origin:
            raise WorkspaceRefusal("model_access_absent: this checkpoint is not held locally")
        if "ensure/1" not in fill.capabilities():
            raise WorkspaceRefusal("model_materialization_unsupported: TensorFS needs ensure/1")
        try:
            facade.ensure(
                store,
                row.model + "@" + row.manifest,
                hub=origin,
                lane=row.lane,
                credential=machine_model_defaults.credential(worker, origin),
                ca_file=machine_model_defaults.hub_ca(worker, origin),
                allowed_hosts=machine_model_defaults.storage_hosts(worker, origin),
                allow_local=machine_model_defaults.local_development(worker),
                keep=keep,
                progress=progress,
            )
            raw = store.manifest(row.manifest)["manifest"]
            store.verify_checkpoint_source(row.model, row.manifest, len(raw))
        except facade.errors.Refusal as exc:
            raise WorkspaceRefusal(f"model_materialization_refused: {exc.code}") from exc

    with ThreadPoolExecutor(
        max_workers=min(8, len(selected)), thread_name_prefix="model-ensure"
    ) as pool:
        pending = [pool.submit(land, row) for row in selected]
        for future in pending:
            future.result()
