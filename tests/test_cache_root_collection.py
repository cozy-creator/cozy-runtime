"""Placement activation and pressure collection share the actual worker fences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, RLock
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal.worker import store_gc
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def worker(tmp_path: Path, *, owns: bool = True) -> Any:
    return SimpleNamespace(
        options=SimpleNamespace(tensorfs_root=tmp_path, owns_tensorfs_store=owns),
        control_lock=RLock(),
        preparation_lock=Lock(),
        placement=None,
        pending=None,
        hosted={},
        pending_hosted=[],
        prepared_revisions={},
        local_operations={},
        accepted=SimpleNamespace(desired_state_body=b"", accepted_desired_state_revision=1),
        held_manifests=SimpleNamespace(rebuild=lambda _: None),
    )


def manifest(name: str) -> bytes:
    return documents.digest_of(name.encode())


def placement(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        document=pb.Placement(models=[pb.Model(manifest=pb.Ref(digest=manifest(name)))])
    )


def test_collection_protects_active_intent_without_treating_prepared_metadata_as_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = worker(tmp_path)
    w.placement = placement("current")
    w.pending = placement("pending")
    w.hosted = {"hosted": SimpleNamespace(placement=placement("hosted"))}
    w.pending_hosted = [placement("incoming").document]
    w.prepared_revisions = {"revision": placement("revision")}
    w.local_operations = {"operation": placement("operation")}
    calls = []

    def collect(root: Path, **kwargs: object) -> pb.CollectStoreGarbageResult:
        calls.append((root, kwargs))
        return pb.CollectStoreGarbageResult(store_busy=True)

    monkeypatch.setattr(store_gc, "collect", collect)
    assert store_gc.collect_worker(w).store_busy
    assert calls == [
        (
            tmp_path,
            {
                "keep_manifests": sorted(
                    documents.spell(manifest(name))
                    for name in ("current", "hosted", "incoming", "pending")
                )
            },
        )
    ]


def test_foreground_preparation_wins_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = worker(tmp_path)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("busy preparation reached native collection")

    monkeypatch.setattr(store_gc, "collect", unexpected)
    with w.preparation_lock:
        assert store_gc.collect_worker(w).store_busy


def test_activation_cannot_enter_between_census_and_native_reclamation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = worker(tmp_path)
    entered, release, activated = Event(), Event(), Event()

    def collect(root: Path, **kwargs: object) -> pb.CollectStoreGarbageResult:
        entered.set()
        assert release.wait(5)
        return pb.CollectStoreGarbageResult(store_busy=True)

    def activate() -> None:
        with w.control_lock:
            activated.set()
            w.placement = placement("replacement")

    monkeypatch.setattr(store_gc, "collect", collect)
    with ThreadPoolExecutor(max_workers=2) as pool:
        collecting = pool.submit(store_gc.collect_worker, w)
        assert entered.wait(5)
        activation = pool.submit(activate)
        try:
            assert not activated.wait(0.05)
        finally:
            release.set()
        collecting.result(timeout=5)
        activation.result(timeout=5)
    assert activated.is_set()


def test_shared_store_never_authorizes_cache_root_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = worker(tmp_path, owns=False)
    calls = []

    def collect(root: Path, **kwargs: object) -> pb.CollectStoreGarbageResult:
        calls.append(kwargs)
        return pb.CollectStoreGarbageResult()

    monkeypatch.setattr(store_gc, "collect", collect)
    store_gc.collect_worker(w)
    assert calls == [{}]


def test_managed_worker_uses_native_cached_collection_and_refreshes_residency(
    tmp_path: Path,
) -> None:
    native = tensorfs.Store.init(tmp_path / "store")
    w = worker(Path(native.root))
    observed = []
    w.held_manifests = SimpleNamespace(rebuild=lambda store: observed.append(store.root))
    result = store_gc.collect_worker(w)
    assert not result.store_busy and result.reclaimed_bytes == 0
    assert observed == [native.root]


def test_fresh_runtime_preserves_cache_roots_before_owner_replays_desired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = worker(tmp_path)
    w.accepted.accepted_desired_state_revision = 0
    calls = []

    def collect(root: Path, **kwargs: object) -> pb.CollectStoreGarbageResult:
        calls.append(kwargs)
        return pb.CollectStoreGarbageResult()

    monkeypatch.setattr(store_gc, "collect", collect)
    store_gc.collect_worker(w)
    assert calls == [{}]
