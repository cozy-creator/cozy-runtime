"""The reader borrows real disk admission without blocking cancellation on its mutex."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs
from tensorfs.derived import serve_source

from cozy_runtime.author import Cancelled
from cozy_runtime.internal import storage_admission as storage
from test_model_reader import model, plain, reader
import weights_channel


@pytest.mark.parametrize("warm", [True, False])
def test_native_reader_metadata_admission_has_no_spool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, warm: bool
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    ref = model(store, "admission", {"weight": plain(np.arange(16, dtype=np.float32))})
    sources, broker, _, _, handles, _ = reader(tmp_path, store, {"source": ref})
    view = sources["source"]
    with handles:
        if warm:
            assert view.components() == ("model",)

        def legacy_admission(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("native source requested a legacy disk write grant")

        # Source reads use TensorFS's own exclusion. They allocate no spool file and
        # must not ask the retired Python write-admission channel for a grant.
        monkeypatch.setattr(storage, "acquire", legacy_admission)
        assert view.components() == ("model",)
        target = bytearray(16)
        view.read_into("model", "weight", "value", 0, target)
        assert target == np.arange(4, dtype=np.float32).tobytes()
    broker.close()
    assert not broker.sources
    assert not list((tmp_path / "spool").glob("model-reader-*-output"))


def test_cancel_can_close_source_while_native_metadata_check_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    ref = model(store, "cancel", {"weight": plain(np.arange(16, dtype=np.float32))})
    sources, broker, record, _, _, _ = reader(tmp_path, store, {"source": ref})
    view = sources["source"]
    asked, allow = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def serve(*args: Any, **kwargs: Any) -> None:
        check = kwargs["check_current"]

        def waiting_check() -> None:
            asked.set()
            assert allow.wait(10)
            check()

        kwargs["check_current"] = waiting_check
        serve_source(*args, **kwargs)

    monkeypatch.setattr(weights_channel, "serve_source", serve)

    def read() -> None:
        try:
            view.read_into("model", "weight", "value", 0, bytearray(4))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    assert asked.wait(10)
    record.canceling = "requested"
    closed = threading.Event()

    def close() -> None:
        broker.close_attempt(record)
        closed.set()

    closer = threading.Thread(target=close, daemon=True)
    closer.start()
    try:
        assert closed.wait(10), "read broker held its mutex while checking native metadata"
        assert not broker.sources
    finally:
        allow.set()
    worker.join(10)
    closer.join(10)
    assert not worker.is_alive() and not closer.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], Cancelled)
    assert not list((tmp_path / "spool").glob("model-reader-*-output"))


def test_payload_reads_on_an_existing_native_lease_require_no_disk_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    reference = model(store, "resident-read", {"weight": plain(np.arange(16, dtype=np.float32))})
    sources, broker, _, _, handles, _ = reader(tmp_path, store, {"source": reference})
    with handles:
        view = sources["source"]
        view.components()

        def refuse(*args: Any) -> Any:
            raise AssertionError("resident read requested a disk write grant")

        monkeypatch.setattr(storage, "acquire", refuse)
        output = bytearray(64)
        view.read_into("model", "weight", "value", 0, output)
        assert output == np.arange(16, dtype=np.float32).tobytes()
    broker.close()
    assert list((tmp_path / "spool").iterdir()) == []
