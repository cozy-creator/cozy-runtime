"""Native source-operation lifetime across a private worker replacement.

The opt-in fixture is exported by TensorFS's real coordinator, with prepared
non-inline checkpoint heads and a second open source operation. No SQL is forged.
"""

from __future__ import annotations

import json
import os
import shutil
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
        store.source_checkpoint_page(
            row["head_id"],
            row["head_length"],
            operation_id=fixture["operation"],
            slot=row["slot"],
            plan_digest=row["plan_digest"],
        )
        for row in fixture["heads"]
    ]


def test_restart_preserves_native_sources_and_normal_stop_releases_them(tmp_path: Path) -> None:
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
    assert store.model_source_operations() == []


def test_shared_local_store_does_not_release_other_source_operations(tmp_path: Path) -> None:
    store, fixture = _fixture(tmp_path)
    before = store.model_source_operations()
    pages = _heads(store, fixture)
    worker = _worker(tmp_path, owns=False)
    worker.shutdown()
    assert store.model_source_operations() == before
    assert _heads(store, fixture) == pages


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
