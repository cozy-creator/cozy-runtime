"""The reader borrows real disk admission without blocking cancellation on its mutex."""

from __future__ import annotations

import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import Cancelled
from cozy_runtime.internal import storage_admission as storage
from test_model_reader import model, plain, reader
from test_storage_admission import operation


@pytest.mark.parametrize("phase", ["open", "read"])
@pytest.mark.parametrize("allowed", [True, False])
def test_native_reader_metadata_admission_has_no_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, allowed: bool
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    ref = model(store, "admission", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, _, _, _ = reader(tmp_path, store, {"source": ref})
    broker.store_root = tmp_path / "store"
    view = service.open(models["source"]) if phase == "read" else None
    host, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    released = threading.Event()
    failures: list[BaseException] = []
    seen: list[dict[str, Any]] = []

    def controller() -> None:
        try:
            rows, conn = operation(host)
            seen.extend(rows)
            with conn:
                conn.sendall(
                    b'{"ok":true}\n' if allowed else b'{"ok":false,"facts":{"available":0}}\n'
                )
                assert conn.recv(1) == b""
            released.set()
        except BaseException as exc:
            failures.append(exc)

    with host, client:
        monkeypatch.setattr(storage, "_channel", client)
        server = threading.Thread(target=controller, daemon=True)
        server.start()

        def execute() -> None:
            if phase == "open":
                service.open(models["source"]).close()
            else:
                assert view is not None
                target = bytearray(16)
                view.read_part_into("model", "weight", "value", 0, target)
                assert target == np.arange(4, dtype=np.float32).tobytes()
                view.close()

        if allowed:
            execute()
        else:
            with pytest.raises(tensorfs.errors.Refusal) as refused:
                execute()
            assert refused.value.code == "insufficient_storage"
        assert released.wait(3)
        server.join(3)
        assert not server.is_alive() and not failures
    broker.close()
    assert not broker.sources
    assert seen and all(row["path"] == str(tmp_path / "store") for row in seen)
    assert not list((tmp_path / "spool").glob("model-reader-*-output"))


def test_cancel_can_close_source_while_metadata_admission_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    ref = model(store, "cancel", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, record, _, _ = reader(tmp_path, store, {"source": ref})
    view = service.open(models["source"])
    host, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    asked, allow = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def controller() -> None:
        _, conn = operation(host)
        with conn:
            asked.set()
            assert allow.wait(3)
            conn.sendall(b'{"ok":true}\n')
            assert conn.recv(1) == b""

    def read() -> None:
        try:
            view.read_part_into("model", "weight", "value", 0, bytearray(4))
        except BaseException as exc:
            errors.append(exc)

    with host, client:
        monkeypatch.setattr(storage, "_channel", client)
        server = threading.Thread(target=controller, daemon=True)
        worker = threading.Thread(target=read, daemon=True)
        server.start()
        worker.start()
        assert asked.wait(3)
        record.canceling = "requested"
        closed = threading.Event()

        def close() -> None:
            broker.close_attempt(record)
            closed.set()

        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        try:
            assert closed.wait(10), "read broker held its mutex while waiting for disk admission"
            assert not broker.sources
        finally:
            allow.set()
        worker.join(3)
        server.join(3)
        closer.join(3)
        assert not worker.is_alive() and not server.is_alive() and not closer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], Cancelled)
    assert not list((tmp_path / "spool").glob("model-reader-*-output"))


def test_payload_reads_on_an_existing_native_lease_require_no_disk_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    reference = model(store, "resident-read", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, _, _, _ = reader(tmp_path, store, {"source": reference})
    with service.open(models["source"]) as view:

        def refuse(*args: Any) -> Any:
            raise AssertionError("resident read requested a disk write grant")

        monkeypatch.setattr(storage, "acquire", refuse)
        output = bytearray(64)
        view.read_part_into("model", "weight", "value", 0, output)
        assert output == np.arange(16, dtype=np.float32).tobytes()
    broker.close()
    assert list((tmp_path / "spool").iterdir()) == []
