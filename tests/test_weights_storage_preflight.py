"""Native writer openings check remaining additions before author computation."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Source, Target, Tensor

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.weights_writer import WriterBroker
from cozy_runtime.internal.worker.grants import MODEL_PREFIX
from test_storage_admission import operation
from weights_channel import open_output


def _rig(
    root: Path, lengths: dict[str, int], maximum: int
) -> tuple[Any, Any, Any, Callable[..., Any]]:
    store = tensorfs.Store.init(root / "store")
    spool = root / "spool"
    spool.mkdir()
    order = [("model", key) for key in lengths]
    fingerprint = "sha256:" + "71" * 32
    transaction = "sha256:" + "72" * 32
    attempt = SimpleNamespace(
        request_id="storage-preflight",
        attempt=1,
        state="running",
        canceling="",
        spool=spool,
        weights_work_fingerprint=fingerprint,
        spec={"inputs": [], "outputs": [{"output_id": "model", "max_bytes": maximum}]},
    )
    recorded: list[Mapping[str, object]] = []
    broker = WriterBroker(
        lambda: store,
        store_root=Path(store.root),
        checkpoint=lambda _a, _t, _b, facts: recorded.append(facts),
        receipt=lambda _a, _t, _b, facts: recorded.append(facts),
    )
    definition = Derivation(
        {},
        {
            "model": Target(
                add={
                    key: Tensor(
                        "u8",
                        (length,),
                        dict(tensorfs.seed_digests())["plain/1"],
                        {"value": Part("u8", (length,))},
                    )
                    for key, length in lengths.items()
                }
            )
        },
        {},
        order,
    )

    def opened(output: str = "model", selected: str = transaction) -> Any:
        return open_output(broker, attempt, selected, definition, output=output)

    return store, broker, attempt, opened


def _host(channel: socket.socket, accept: bool, rows: list[list[dict[str, Any]]]) -> None:
    # The output door first admits remaining bytes, then accepts the native handle
    # through its metadata-only receipt query. Neither call reserves a role spool.
    for _index in range(2 if accept else 1):
        try:
            if not channel.recv(1, socket.MSG_PEEK):
                return
        except OSError:
            return
        writes, connection = operation(channel)
        rows.append(writes)
        with connection:
            connection.sendall(
                json.dumps({"ok": accept, "facts": {"available": 4_883_300_352}}).encode() + b"\n"
            )
            if accept:
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
            assert not broker.writers
            assert not list(attempt.spool.glob("*-input"))
            assert store.derived_lookup("sha256:" + "72" * 32)["state"] != "committed"
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[0][0]["bytes"] == (5 << 30) + 4 * tensorfs.manifest_max_bytes()
    assert len(rows[0]) == 1


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
            writer = opened()
            assert broker.writers
            assert writer.receipt is None
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[0][0]["bytes"] == 4096 + 4 * tensorfs.manifest_max_bytes()
    assert len(rows[0]) == 1


def test_inherited_checkpoint_needs_no_new_payload_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, broker, attempt, opened = _rig(tmp_path, {"weight": 4096}, 4096)
    writer = opened()
    writer.add_part("model", "weight", "value", io.BytesIO(bytes(4096)))
    manifest = writer.commit()["manifest"]
    source = "sha256:" + manifest["sha256"]
    sources = {"source": (source, manifest["length"])}
    targets = {"model": {"source": "source", "source_component": "model", "drop": [], "add": {}}}
    order = [("model", "weight")]
    declaration = store.derived_declaration(
        sources, targets, {}, order, 0, work_fingerprint=attempt.weights_work_fingerprint
    )
    transaction = "sha256:" + "74" * 32
    broker.authorize(attempt, transaction, 1, hashlib.sha256(declaration).digest(), "inherited")
    attempt.spec["inputs"] = [
        {"input_id": MODEL_PREFIX + "source", "digest": source, "length": manifest["length"]}
    ]
    attempt.spec["outputs"].append({"output_id": "inherited", "max_bytes": 0})
    definition = Derivation(
        {"source": Source(source, manifest["length"])},
        {"model": Target(source="source", source_component="model")},
        {},
        order,
    )
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    rows: list[list[dict[str, Any]]] = []
    with host, worker:
        monkeypatch.setattr(storage, "_channel", worker)
        thread = threading.Thread(target=_host, args=(host, True, rows), daemon=True)
        thread.start()
        try:
            inherited = open_output(broker, attempt, transaction, definition, output="inherited")
            assert inherited.receipt is None
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[0][0]["bytes"] == 4 * tensorfs.manifest_max_bytes()
    assert len(rows[0]) == 1


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
            resumed = opened()
            assert resumed.receipt is None
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[0][0]["bytes"] == 64 + 4 * tensorfs.manifest_max_bytes()
    assert len(rows[0]) == 1


def test_open_outputs_share_one_remaining_payload_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, broker, attempt, opened = _rig(tmp_path, {"weight": 4096}, 1 << 40)
    first = opened()
    attempt.spec["outputs"].append({"output_id": "second", "max_bytes": 1 << 40})
    host, worker = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    rows: list[list[dict[str, Any]]] = []
    with host, worker:
        monkeypatch.setattr(storage, "_channel", worker)
        thread = threading.Thread(target=_host, args=(host, True, rows), daemon=True)
        thread.start()
        try:
            second = opened("second", "sha256:" + "73" * 32)
            assert first.receipt is None and second.receipt is None
        finally:
            broker.close_attempt(attempt)
        thread.join(3)
        assert not thread.is_alive()
    assert rows[0][0]["bytes"] == 8192 + 4 * tensorfs.manifest_max_bytes()
    assert len(rows[0]) == 1
