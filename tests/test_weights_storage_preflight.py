"""Native writer openings check remaining additions before author computation."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import MAX_WEIGHTS_SOURCE_READ
from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.weights_writer import StoreProxy, WriterBroker
from test_storage_admission import operation


def _rig(
    root: Path, lengths: dict[str, int], maximum: int
) -> tuple[Any, Any, Any, Callable[..., Any]]:
    store = tensorfs.Store.init(root / "store")
    spool = root / "spool"
    spool.mkdir()
    targets = {
        "model": {
            "drop": [],
            "add": {
                key: {
                    "logical_dtype": "u8",
                    "shape": [length],
                    "encoding": dict(tensorfs.seed_digests())["plain/1"],
                    "parts": {"value": {"dtype": "u8", "shape": [length]}},
                }
                for key, length in lengths.items()
            },
        }
    }
    order = [("model", key) for key in lengths]
    fingerprint = "sha256:" + "71" * 32
    transaction = "sha256:" + "72" * 32
    declaration = store.derived_declaration(
        {}, targets, {}, order, maximum, work_fingerprint=fingerprint
    )
    attempt = SimpleNamespace(
        request_id="storage-preflight",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        weights_work_fingerprint=fingerprint,
        spec={"inputs": [], "outputs": [{"output_id": "model", "max_bytes": maximum}]},
    )
    broker = WriterBroker(lambda: store, store_root=Path(store.root))
    proxy = StoreProxy(store, spool, lambda _kind, frame: broker.handle(attempt, frame))

    def opened(output: str = "model", selected: str = transaction) -> Any:
        broker.authorize(
            attempt, selected, attempt.attempt, hashlib.sha256(declaration).digest(), output
        )
        return proxy.begin_derived(
            selected,
            attempt.attempt,
            {},
            targets,
            {},
            order,
            maximum,
            work_fingerprint=fingerprint,
        )

    return store, broker, attempt, opened


def _host(channel: socket.socket, accept: bool, rows: list[list[dict[str, Any]]]) -> None:
    # The first hold admits only opening metadata. The second is the whole pending
    # writer preflight, whose release is essential before later per-part writes.
    for index in range(2):
        writes, connection = operation(channel)
        rows.append(writes)
        with connection:
            connection.sendall(
                json.dumps(
                    {"ok": index == 0 or accept, "facts": {"available": 4_883_300_352}}
                ).encode()
                + b"\n"
            )
            if index == 0 or accept:
                assert connection.recv(1) == b""


def test_model_sized_additions_refuse_before_any_payload_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, broker, attempt, opened = _rig(tmp_path, {"weight": 5 << 30}, 12 << 30)
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    rows: list[list[dict[str, Any]]] = []
    with host, worker:
        monkeypatch.setattr(storage, "_channel", worker)
        thread = threading.Thread(target=_host, args=(host, False, rows), daemon=True)
        thread.start()
        try:
            with pytest.raises(CapabilityError) as refused:
                opened()
            assert refused.value.code == "insufficient_storage"
            assert "available=4883300352" in str(refused.value)
            assert not broker.writers and not broker.pending
            assert not list(attempt.spool.glob("*-input"))
            assert store.derived_lookup("sha256:" + "72" * 32)["state"] != "committed"
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[1][0]["bytes"] == (5 << 30) + 4 * tensorfs.manifest_max_bytes()


def test_preflight_uses_geometry_instead_of_the_loose_output_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, broker, attempt, opened = _rig(tmp_path, {"weight": 4096}, 1 << 40)
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    rows: list[list[dict[str, Any]]] = []
    with host, worker:
        monkeypatch.setattr(storage, "_channel", worker)
        thread = threading.Thread(target=_host, args=(host, True, rows), daemon=True)
        thread.start()
        try:
            opened()
            assert broker.writers
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[1][0]["bytes"] == 4096 + 4 * tensorfs.manifest_max_bytes()
    assert rows[1][1]["bytes"] == 4096 + MAX_WEIGHTS_SOURCE_READ


def test_resume_excludes_native_completed_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, broker, attempt, opened = _rig(tmp_path, {"large": 4096, "small": 64}, 1 << 40)
    writer = opened()
    writer.add_part("model", "large", "value", io.BytesIO(bytes(4096)))
    broker.close_attempt(attempt)
    attempt.attempt = 2
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    rows: list[list[dict[str, Any]]] = []
    with host, worker:
        monkeypatch.setattr(storage, "_channel", worker)
        thread = threading.Thread(target=_host, args=(host, True, rows), daemon=True)
        thread.start()
        try:
            opened()
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[1][0]["bytes"] == 64 + 4 * tensorfs.manifest_max_bytes()
    assert rows[1][1]["bytes"] == 64 + MAX_WEIGHTS_SOURCE_READ


def test_open_outputs_share_one_remaining_payload_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, broker, attempt, opened = _rig(tmp_path, {"weight": 4096}, 1 << 40)
    opened()
    attempt.spec["outputs"].append({"output_id": "second", "max_bytes": 1 << 40})
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    rows: list[list[dict[str, Any]]] = []
    with host, worker:
        monkeypatch.setattr(storage, "_channel", worker)
        thread = threading.Thread(target=_host, args=(host, True, rows), daemon=True)
        thread.start()
        try:
            opened("second", "sha256:" + "73" * 32)
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[1][0]["bytes"] == 8192 + 4 * tensorfs.manifest_max_bytes()
    assert rows[1][1]["bytes"] == 4096 + MAX_WEIGHTS_SOURCE_READ
