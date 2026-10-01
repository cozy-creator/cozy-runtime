"""Runtime preparation requests verified disk custody from TensorFS, never the Host."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING

from cozy_runtime.internal import fill
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_model_defaults, machine_model_resolve
from .workspace import WorkspaceRefusal

if TYPE_CHECKING:
    from .session import Worker

_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MANIFEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _counter(value: object) -> int | None:
    return value if type(value) is int and 0 <= value < 1 << 64 else None


def _document_length(value: object) -> int | None:
    return len(value) if isinstance(value, bytes) else None


def sample(event: object) -> tuple[int, int] | None:
    if not isinstance(event, Mapping):
        return None
    done, total = _counter(event.get("bytes_done")), _counter(event.get("bytes_total"))
    if done is None or total is None or (total and done > total):
        return None
    return done, total


def forward(progress: Callable[[int, int], None], event: object) -> None:
    """The native callback validates optional data before calling any observer."""
    measured = sample(event)
    if measured is not None:
        progress(*measured)


def snapshot(worker: Worker) -> list[pb.PrepareModelProgress]:
    """Copy optional observations without waiting for preparation or transfer locks."""
    with worker.model_transfer_lock:
        return [
            pb.PrepareModelProgress.FromString(row.SerializeToString())
            for row in worker.model_preparation_progress.values()
        ]


def observe(
    worker: Worker,
    model: pb.DownloadModelRef,
    available: object,
    total: object,
    *,
    result: Mapping[str, object] | None = None,
    begin: bool = False,
) -> pb.PrepareModelProgress | None:
    """Retain native per-checkpoint observations; this owns no flight or cancellation."""
    done, whole = _counter(available), _counter(total)
    if done is None or whole is None or (whole and done > whole):
        return None
    key = (model.model, model.manifest)
    with worker.model_transfer_lock:
        if begin or key not in worker.model_preparation_progress:
            worker.model_preparation_progress[key] = pb.PrepareModelProgress(
                model=pb.DownloadModelRef(model=model.model, manifest=model.manifest)
            )
        row = worker.model_preparation_progress.pop(key)
        if whole == 0 and row.total_bytes and done > row.total_bytes:
            worker.model_preparation_progress[key] = row
            return None
        row.total_bytes = max(row.total_bytes, whole)
        row.transferred_bytes = max(row.transferred_bytes, done)
        if result is not None:
            for field, counter in (
                ("origin_bytes", "bytes_fetched"),
                ("cached_bytes", "bytes_cached"),
                ("cache_written_bytes", "cache_written_bytes"),
            ):
                value = _counter(result.get(counter))
                if value is not None:
                    setattr(row, field, value)
        worker.model_preparation_progress[key] = row
        # Telemetry retains recent rows, with active transfers refreshed on each native
        # sample. This is the existing model-slot display bound, never an admission cap.
        while len(worker.model_preparation_progress) > weights_limits.MAX_MODEL_SLOT_PATHS:
            old = next(
                (
                    identity
                    for identity in worker.model_preparation_progress
                    if identity not in worker.model_preparation_observers
                ),
                None,
            )
            if old is None:
                break
            del worker.model_preparation_progress[old]
        return row


@contextmanager
def observing(worker: Worker, model: pb.DownloadModelRef) -> Iterator[None]:
    key = (model.model, model.manifest)
    with worker.model_transfer_lock:
        if key not in worker.model_preparation_observers:
            worker.model_preparation_progress[key] = pb.PrepareModelProgress(
                model=pb.DownloadModelRef(model=model.model, manifest=model.manifest)
            )
        worker.model_preparation_observers[key] = worker.model_preparation_observers.get(key, 0) + 1
    try:
        yield
    finally:
        with worker.model_transfer_lock:
            remaining = worker.model_preparation_observers[key] - 1
            if remaining:
                worker.model_preparation_observers[key] = remaining
            else:
                del worker.model_preparation_observers[key]
                row = worker.model_preparation_progress.get(key)
                if row is not None and (
                    not row.total_bytes or row.transferred_bytes < row.total_bytes
                ):
                    # A canceled/failed flight has no live owner. Keep completed availability,
                    # but do not let a historical partial row impersonate a new download.
                    del worker.model_preparation_progress[key]


def completed(worker: Worker, model: pb.DownloadModelRef, result: object) -> None:
    """A missing/malformed optional counter never changes the native operation's answer."""
    if isinstance(result, Mapping):
        total = result.get("bytes_total")
        observe(worker, model, total, total, result=result)


def held(worker: Worker, model: pb.DownloadModelRef, store: fill.Store) -> None:
    """Read a verified native closure's availability, without transferring payloads."""
    with worker.model_transfer_lock:
        known = worker.model_preparation_progress.get((model.model, model.manifest))
        total = known.total_bytes if known is not None else 0
    if not total:
        try:
            observed = {}
            for row in store.walk_cozytensors(model.manifest):
                identity, size = row.get("id"), _counter(row.get("length"))
                if not isinstance(identity, str) or size is None:
                    return
                observed[identity] = size
            length = _document_length(store.manifest(model.manifest)["manifest"])
            if length is None:
                return
            observed[model.manifest] = length
            total = sum(observed.values())
        except (fill.tensorfs_module().errors.Refusal, OSError, ValueError, KeyError, TypeError):
            # Custody was already verified by the caller. An optional size read cannot
            # turn that successful preparation into a second transfer or a refusal.
            return
    observe(worker, model, total, total)


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

    def land(row: pb.DownloadModelRef | pb.DownloadAdapterRef) -> None:
        model = pb.DownloadModelRef(
            model=row.model, manifest=row.manifest, release=row.release, lane=row.lane
        )
        if isinstance(row, pb.DownloadModelRef):
            model.CopyFrom(row)
        try:
            raw = store.manifest(row.manifest)["manifest"]
            store.verify_checkpoint_source(row.model, row.manifest, len(raw))
            held(worker, model, store)
            return
        except facade.errors.Refusal:
            pass
        if not origin:
            raise WorkspaceRefusal("model_access_absent: this checkpoint is not held locally")

        def note(text: str) -> None:
            worker.note("model_preparation", f"{row.model}: {text}")

        def progress(event: Mapping[str, object]) -> None:
            asking(event)
            measured = sample(event)
            if measured is not None:
                available, total = measured
                observe(worker, model, available, total)
                worker.note("model_preparation", f"{available}/{total} bytes verified")

        try:
            with observing(worker, model), machine_model_defaults.hub_waiting(note) as asking:
                result = facade.ensure(
                    store,
                    row.model + "@" + row.manifest,
                    hub=origin,
                    lane=row.lane,
                    credential=machine_model_defaults.credential(worker, origin),
                    ca_file=machine_model_defaults.hub_ca(worker, origin),
                    allowed_hosts=machine_model_defaults.storage_hosts(worker, origin),
                    allow_local=machine_model_defaults.local_development(worker),
                    keep=keep,
                    cancellation=worker.transfers,
                    progress=progress,
                )
                raw = store.manifest(row.manifest)["manifest"]
                store.verify_checkpoint_source(row.model, row.manifest, len(raw))
                completed(worker, model, result)
        except facade.errors.Refusal as exc:
            if worker.transfers.cancelled:
                raise WorkspaceRefusal("machine stopped during model preparation") from exc
            raise WorkspaceRefusal(f"model_materialization_refused: {exc}") from exc

    with ThreadPoolExecutor(
        max_workers=min(8, len(selected)), thread_name_prefix="model-ensure"
    ) as pool:
        pending = [pool.submit(land, row) for row in selected]
        for future in pending:
            future.result()
