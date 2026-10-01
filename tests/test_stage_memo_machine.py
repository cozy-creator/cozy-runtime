"""The machine tier of memoized Model methods (tracker #301): the real executor engine
against the real Worker tier over a real TensorFS store, in-process. The durable exchange is
answered directly; its seam is the existing executor channel."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author._executor_requests import (
    MemoEntry,
    StageMemoLookup,
    StageMemoStore,
)
from cozy_runtime.internal import local_storage_admission, stage_memo, storage_admission
from cozy_runtime.internal.config import MemoConfig
from cozy_runtime.internal.executor_commands import MemoSettings
from cozy_runtime.internal.worker.stage_memo import SPACE, MachineMemo
from test_stage_memo import Encoder, _model, torch

tensorfs = pytest.importorskip("tensorfs")


class _Worker:
    """The executor's durable exchange, answered by the real Worker tier in-process."""

    def __init__(self, memo: Any, spool: Path, attempt: str = "req#1") -> None:
        self.memo, self.spool, self.attempt = memo, spool, attempt
        self.requests: list[str] = []

    def __call__(self, request: Any, into: type[Any]) -> Any:
        self.requests.append(type(request).__name__)
        if isinstance(request, StageMemoLookup):
            return self.memo.lookup(self.attempt, self.spool, request)
        assert isinstance(request, StageMemoStore)
        return self.memo.store(self.attempt, self.spool, request)


class _Recorded(MachineMemo):
    notes: list[tuple[str, str]]


def _machine(tmp_path: Path, **config: int) -> _Recorded:
    if not hasattr(tensorfs.Store, "put_keyed_root"):
        pytest.skip("this TensorFS has no keyed roots; the machine tier stays off")
    store = tmp_path / "store"
    tensorfs.Store.ensure(str(store))
    notes: list[tuple[str, str]] = []
    memo = _Recorded(
        store, tmp_path / "staging", MemoConfig(**config), lambda *note: notes.append(note)
    )
    memo.notes = notes
    return memo


def _committed(memo: MachineMemo) -> None:
    while any(row.staged is not None for row in memo.rows.values()):
        time.sleep(0.01)


def _engine(machine: MachineMemo, spool: Path, attempt: str = "req#1") -> stage_memo.Engine:
    spool.mkdir(exist_ok=True)
    engine = stage_memo.Engine()
    engine.configure(machine.settings())
    engine.open(_Worker(machine, spool, attempt), spool)
    return engine


def test_a_new_process_hits_the_machine_tier(tmp_path: Path) -> None:
    machine, spool = _machine(tmp_path), tmp_path / "spool"
    producer = _engine(machine, spool)
    first = _model(engine=producer).encode("shared prompt")
    _committed(machine)
    consumer = _engine(machine, spool, "req#2")
    second = _model(engine=consumer).encode("shared prompt")
    assert Encoder.runs == 0, "the second executor process read the stored entry"
    assert torch.equal(first.prompt, second.prompt) and torch.equal(first.pooled, second.pooled)
    assert consumer.calls[0].outcome == "hit:machine"
    assert consumer.calls[0].sha256 == producer.calls[0].sha256
    assert not list(spool.iterdir()), "spool copies are consumed"


def test_entries_outlive_a_worker_restart_and_expire_after_their_ttl(tmp_path: Path) -> None:
    machine = _machine(tmp_path, ttl_ms=60_000)
    _model(engine=_engine(machine, tmp_path / "spool")).encode("kept")
    _committed(machine)
    restarted = _machine(tmp_path, ttl_ms=60_000)
    assert list(restarted.rows) == list(machine.rows)
    (row,) = restarted.rows.values()
    assert restarted.expire(now_ms=row.created_ms + 59_999) == 0
    assert restarted.expire(now_ms=row.created_ms + 60_000) == 1
    assert not restarted.rows and not restarted.native.keyed_roots(SPACE)
    assert not (tmp_path / "store/roots/keyed" / SPACE).exists() or not any(
        (tmp_path / "store/roots/keyed" / SPACE).iterdir()
    ), "nothing is left behind"
    assert tensorfs.gc(str(tmp_path / "store"))["reclaimed_bytes"] == 0, "the drop collected"


def test_beyond_its_cap_the_least_recently_used_go_first(tmp_path: Path) -> None:
    machine = _machine(tmp_path)
    engine = _engine(machine, tmp_path / "spool")
    model = _model(engine=engine)
    for text in ("one", "two", "thr"):
        model.encode(text)
    _committed(machine)
    oldest, second, _ = machine.rows
    engine.configure(MemoSettings(process_bytes=0, entry_bytes=machine.config.entry_bytes))
    model.encode("one")  # a machine hit: "one" becomes the most recently used
    assert Encoder.runs == 3 and next(iter(machine.rows)) == second
    machine.config = MemoConfig(store_bytes=sum(r.bytes for r in machine.rows.values()) - 1)
    assert machine.trim() == 1
    assert second not in machine.rows and oldest in machine.rows


def test_low_disk_evicts_and_very_low_disk_skips_writes(tmp_path: Path) -> None:
    machine = _machine(tmp_path, entry_bytes=1 << 50)
    _model(engine=_engine(machine, tmp_path / "spool")).encode("pressure")
    _committed(machine)
    # Low disk is the storage-pressure policy's own watermark on this real filesystem.
    pressure = local_storage_admission.pressure_target(tmp_path)
    assert machine.relieve() == (1 if pressure else 0)
    # Very low disk: storage admission refuses a write larger than the free space; the
    # entry is not written and the request carries on.
    (tmp_path / "store/.cozy-workspace").mkdir()
    held = storage_admission.register_workspace(tmp_path / "store", lambda target: 0)
    try:
        free = shutil.disk_usage(tmp_path).free
        payload = tmp_path / "spool/entry"
        payload.write_bytes(b"x")
        answer = machine.store(
            "req#9",
            tmp_path / "spool",
            StageMemoStore(key="b" * 64, stage="s", numerics="n", local=str(payload), length=free),
        )
    finally:
        held.close()
    assert answer.ok and not answer.stored and answer.reason == "disk_low"


def test_a_corrupt_entry_is_a_miss_and_is_released(tmp_path: Path) -> None:
    machine = _machine(tmp_path)
    _model(engine=_engine(machine, tmp_path / "spool")).encode("flip me")
    _committed(machine)
    (key,) = machine.rows
    (blob,) = (path for path in (tmp_path / "store/blobs").rglob("*") if path.is_file())
    blob.chmod(0o600)
    data = bytearray(blob.read_bytes())
    data[-1] ^= 0xFF
    blob.write_bytes(bytes(data))
    fresh = _engine(machine, tmp_path / "spool", "req#2")
    _model(engine=fresh).encode("flip me")
    assert Encoder.runs == 1 and fresh.calls[0].outcome == "miss"
    assert [kind for kind, _ in machine.notes] == ["memo.integrity"]
    assert key in machine.rows, "the recomputed entry replaced the released one"


def test_a_waiting_lookup_takes_over_when_the_producer_ends(tmp_path: Path) -> None:
    machine = _machine(tmp_path)
    lookup = StageMemoLookup(key="a" * 64, stage="s", numerics="n")
    assert machine.lookup("p#1", tmp_path, lookup).local == ""
    answers: list[MemoEntry] = []
    waiter = threading.Thread(
        target=lambda: answers.append(machine.lookup("w#1", tmp_path, lookup))
    )
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive(), "the second lookup waits for the producer, not a clock"
    machine.release("p#1")
    waiter.join(10)
    assert answers and answers[0].local == "" and machine.claims["a" * 64].attempt == "w#1"


def test_an_entry_is_taken_only_from_the_attempts_own_spool(tmp_path: Path) -> None:
    machine = _machine(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    secret = tmp_path / "secret"
    secret.write_bytes(b"x" * 8)
    (spool / "link").symlink_to(secret)
    for local in (secret, spool / "link"):
        request = StageMemoStore(key="c" * 64, stage="s", numerics="n", local=str(local), length=8)
        assert machine.store("req#1", spool, request).reason == "local"
    assert not machine.rows
