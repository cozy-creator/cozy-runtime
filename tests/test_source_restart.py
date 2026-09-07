"""Native source-operation lifetime across a private worker replacement.

The opt-in fixture is exported by TensorFS's real coordinator, with prepared
non-inline checkpoint heads and a second open source operation. No SQL is forged.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.refusal import LaunchRefusal
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb


def _fixture(root: Path) -> tuple[Any, dict[str, Any]]:
    import tensorfs

    path = os.environ.get("COZY_TEST_SOURCE_RESTART_FIXTURE", "")
    if not path or not hasattr(tensorfs.Store, "model_source_operations"):
        pytest.skip("requires native source restart fixture and future TensorFS0.3.11")
    source = Path(path)
    document: dict[str, Any] = json.loads(source.read_text())
    shutil.copytree(source.parent / "store", root / "store")
    return tensorfs.Store.open(str(root / "store")), document


def _worker(root: Path, *, owns: bool, required: bool = False) -> Worker:
    return Worker(
        RuntimeConfig(cozy_home=root / "home", credentials=Credentials()),
        WorkerOptions(
            root=root / "worker",
            tensorfs_root=root / "store",
            no_warm=True,
            worker_boot_id="source-restart-boot",
            ownership_required=required,
            owns_tensorfs_store=owns,
        ),
        InMemoryControlHost(),
    )


def _heads(store: Any, fixture: dict[str, Any]) -> list[Any]:
    return [
        store.checkpoint_page(
            row["head_id"],
            row["head_length"],
            operation_id=fixture["operation"],
            slot=row["slot"],
            plan_digest=row["plan_digest"],
        )
        for row in fixture["heads"]
    ]


def test_restart_and_normal_recycle_preserve_sources_until_explicit_owner_release(
    tmp_path: Path,
) -> None:
    store, fixture = _fixture(tmp_path)
    expected = sorted([fixture["operation"], fixture["open_operation"]])
    before = _heads(store, fixture)
    worker = _worker(tmp_path, owns=True)
    assert sorted(worker.model_source_operations) == expected
    worker.request_restart()
    refused = worker.prepare_model_source(pb.PrepareModelSourceRequest())
    assert refused.safe_code == "model_source_preparer_unavailable"
    worker.shutdown()
    assert store.model_source_operations() == expected
    assert _heads(store, fixture) == before
    replacement = _worker(tmp_path, owns=True, required=True)
    assert sorted(replacement.model_source_operations) == expected
    replacement.shutdown()  # no source prepare was replayed in this process
    assert store.model_source_operations() == expected
    assert _heads(store, fixture) == before
    owner = _worker(tmp_path, owns=True, required=True)
    owner.release_model_source(fixture["open_operation"])
    assert store.model_source_operations() == [fixture["operation"]]
    assert _heads(store, fixture) == before
    owner.release_model_source(fixture["operation"])
    owner.shutdown()
    assert store.model_source_operations() == []


def test_shared_local_store_does_not_release_other_source_operations(tmp_path: Path) -> None:
    store, fixture = _fixture(tmp_path)
    before = store.model_source_operations()
    pages = _heads(store, fixture)
    worker = _worker(tmp_path, owns=False)
    worker.shutdown()
    assert store.model_source_operations() == before
    assert _heads(store, fixture) == pages


def test_restart_intent_never_enters_worker_queues_or_stops_transport(tmp_path: Path) -> None:
    worker = _worker(tmp_path, owns=False)
    host = worker.host
    assert isinstance(host, InMemoryControlHost)
    queues: list[Any] = [worker.outbound, host.inbound]
    queues.extend(queue for lane in worker.lanes for queue in (lane.work, lane.post))
    try:
        with ThreadPoolExecutor(max_workers=1) as executor, ExitStack() as held:
            for queue in queues:
                held.enter_context(queue.mutex)
            # A signal may interrupt the thread while any queue mutex is held.
            # Intent must return without trying to re-enter those real locks.
            executor.submit(worker.request_restart).result(timeout=1)
            assert worker.restart_requested and worker.stop.is_set()
            assert not host.closed.is_set()
    finally:
        worker.shutdown()


@pytest.mark.parametrize("restart", [False, True])
def test_queued_preparation_observes_stop_after_acquiring_lock(
    tmp_path: Path, restart: bool
) -> None:
    store, fixture = _fixture(tmp_path)
    operations = store.model_source_operations()
    pages = _heads(store, fixture)
    worker = _worker(tmp_path, owns=True)
    lock = worker.preparation_lock
    entered = threading.Event()

    class ObservedLock:
        """Observe the real lock attempt; never replace native work or its answer."""

        def __enter__(self) -> None:
            entered.set()
            lock.acquire()

        def __exit__(self, *args: Any) -> None:
            lock.release()

    observed: Any = ObservedLock()
    worker.preparation_lock = observed
    with ThreadPoolExecutor(max_workers=1) as executor:
        lock.acquire()
        try:
            pending = executor.submit(
                worker.prepare_model_source,
                pb.PrepareModelSourceRequest(operation_id="queued-after-stop"),
            )
            assert entered.wait(5), "prepare handler did not reach the held lock"
            assert not pending.done()
            if restart:
                worker.request_restart()
            else:
                worker.request_stop()
        finally:
            lock.release()
        worker.shutdown()
        result = pending.result(timeout=5)
    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code == "model_source_preparer_unavailable"
    assert store.model_source_operations() == operations
    assert _heads(store, fixture) == pages
    replacement = _worker(tmp_path, owns=True, required=True)
    replacement.shutdown()
    assert store.model_source_operations() == operations


def test_refused_restart_history_keeps_source_operations(tmp_path: Path) -> None:
    store, fixture = _fixture(tmp_path)
    before = store.model_source_operations()
    pages = _heads(store, fixture)
    with pytest.raises(LaunchRefusal, match="worker_ownership_absent"):
        _worker(tmp_path, owns=True, required=True)
    assert store.model_source_operations() == before
    worker = _worker(tmp_path, owns=True)
    worker.request_restart()
    worker.shutdown()
    (tmp_path / "worker/ownership.json").write_bytes(b"{}")
    with pytest.raises(LaunchRefusal, match="worker_ownership_invalid"):
        _worker(tmp_path, owns=True, required=True)
    assert store.model_source_operations() == before
    assert _heads(store, fixture) == pages
