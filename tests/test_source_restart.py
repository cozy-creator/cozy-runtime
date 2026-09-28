"""A restart request from a signal handler never blocks on the worker's own queues."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions


def test_restart_intent_never_enters_worker_queues_or_stops_transport(tmp_path: Path) -> None:
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(
            root=tmp_path / "worker",
            tensorfs_root=tmp_path / "store",
            worker_boot_id="source-restart-boot",
            owns_tensorfs_store=False,
        ),
        InMemoryControlHost(),
    )
    host = worker.host
    assert isinstance(host, InMemoryControlHost)
    queues: list[Any] = [worker.outbound, host.inbound]
    try:
        with ThreadPoolExecutor(max_workers=1) as executor, ExitStack() as held:
            for queue in queues:
                held.enter_context(queue.mutex)
            # A signal may interrupt the thread while any queue mutex is held.
            # Intent must return without trying to re-enter those real locks; the bound
            # only turns a deadlock into a failure.
            executor.submit(worker.request_restart).result(timeout=60)
            assert worker.restart_requested and worker.stop.is_set()
            assert not host.closed.is_set()
    finally:
        worker.shutdown()
