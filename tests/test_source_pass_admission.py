from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.internal import fill
from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.worker import model_source_prepare
from cozy_runtime.protocol import worker_pb2 as pb


@pytest.mark.parametrize("mode", ["preferred", "minimum", "pending", "replay", "full"])
def test_source_pass_uses_native_bounds_instead_of_whole_carriers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    profiles = [("left", "fixture/left"), ("right", "fixture/right")]
    request = pb.PrepareModelSourceRequest(
        operation_id="source-pass", source_selection_digest=b"a" * 32
    )
    for slot, profile in profiles:
        request.profiles.add(slot=slot, profile=profile)
    roster = [("weights", "sha256:" + "b" * 64, 200 << 30, tmp_path / "weights", b"header")]
    monkeypatch.setattr(model_source_prepare, "_request", lambda _: (profiles, roster, [], []))
    monkeypatch.setattr(storage, "_channel", object())
    admitted, reservations, calls = [], [], []
    minimum, preferred = 128 << 10, 64 << 20

    @contextmanager
    def admit(*writes: storage.Write) -> Iterator[None]:
        reservation = sum(row.bytes for row in writes)
        reservations.append(reservation)
        if mode == "full" or (mode == "minimum" and reservation > 1 << 20):
            raise storage.StorageRefusal("test disk capacity")
        admitted.append(True)
        try:
            yield
        finally:
            admitted.pop()

    def advance(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert admitted, "native preparation started without admission"
        calls.append(kwargs["write_budget_bytes"])
        replay = mode == "replay"
        return {
            "replayed": replay,
            "complete": replay,
            "sources": [
                {
                    "slot": slot,
                    "profile": profile,
                    "manifest_digest": "sha256:" + "c" * 64,
                    "manifest_length": 1,
                }
                for slot, profile in profiles
            ]
            if replay
            else [],
            "checkpoints": [],
            "spent": [],
            "required_write_bytes": minimum
            if len(calls) == 1 and mode not in {"pending", "replay"}
            else 0,
            "write_interval_bytes": preferred,
        }

    def ensure(_: Path) -> SimpleNamespace:
        assert admitted, "Store creation started before metadata admission"
        return SimpleNamespace(prepare_model_source=advance)

    facade = SimpleNamespace(manifest_max_bytes=lambda: 128)
    monkeypatch.setattr(fill, "tensorfs_module", lambda: facade)
    monkeypatch.setattr(fill, "ensure_store", ensure)
    monkeypatch.setattr(storage, "admit", admit)
    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "store")
    assert not admitted
    if mode == "full":
        assert not calls
        assert result.safe_code == "model_source_insufficient_storage"
        assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    elif mode == "preferred":
        assert calls == [0, preferred]
        assert max(reservations) < preferred + 4096
    elif mode == "minimum":
        assert calls == [0, minimum]
        assert len(reservations) == 3
    else:
        assert calls == [0]
        assert len(reservations) == 1
        if mode == "replay":
            assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED
