from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal import storage_admission as storage
from cozy_runtime.internal.weights_writer import WriterBroker
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from test_workspace_byte_outputs import producing
from weights_channel import open_output


def test_native_import_refusal_preserves_input_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    original = path.read_bytes()

    def refuse(*writes: storage.Write) -> NoReturn:
        raise storage.StorageRefusal("no native import capacity")

    with monkeypatch.context() as scoped:
        scoped.setattr(storage, "admit", refuse)
        with pytest.raises(storage.StorageRefusal):
            outputs.commit(
                workspace,
                "owner",
                "producer",
                1,
                spec,
                "report",
                manifest,
                [("payload", path)],
                200000,
            )
    assert path.read_bytes() == original
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    path.unlink()
    # Exact completed native custody replays without the now-absent spool input.
    assert (
        outputs.commit(
            workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
        )
        == source
    )


def test_parent_writer_checks_admission_before_creating_native_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    spool = tmp_path / "spool"
    spool.mkdir()
    encoding = next(d for a, d in tensorfs.seed_digests() if a == "plain/1")
    targets = {
        "model": {
            "drop": [],
            "add": {
                "w": {
                    "logical_dtype": "bf16",
                    "shape": [2],
                    "encoding": encoding,
                    "parts": {"value": {"dtype": "bf16", "shape": [2]}},
                }
            },
        }
    }
    order, maximum = [("model", "w")], 4
    fingerprint, transaction = "sha256:" + "71" * 32, "sha256:" + "72" * 32
    declaration = store.derived_declaration(
        {}, targets, {}, order, maximum, work_fingerprint=fingerprint
    )
    attempt = SimpleNamespace(
        request_id="r",
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
        store_root=tmp_path / "store",
        checkpoint=lambda _a, _t, _b, facts: recorded.append(facts),
        receipt=lambda _a, _t, _b, facts: recorded.append(facts),
    )
    broker.authorize(attempt, transaction, 1, hashlib.sha256(declaration).digest(), "model")
    definition = Derivation(
        {},
        {
            "model": Target(
                add={
                    "w": Tensor(
                        "bf16",
                        (2,),
                        dict(tensorfs.seed_digests())["plain/1"],
                        {"value": Part("bf16", (2,))},
                    )
                }
            )
        },
        {},
        order,
    )

    def refuse(*writes: storage.Write) -> NoReturn:
        raise storage.StorageRefusal("native capacity exhausted", code="insufficient_storage")

    with monkeypatch.context() as scoped:
        scoped.setattr(storage, "acquire", refuse)
        with pytest.raises(CapabilityError) as denied:
            open_output(broker, attempt, transaction, definition)
        assert denied.value.code == "insufficient_storage"
        assert store.derived_lookup(transaction)["state"] == "absent"
    writer = open_output(broker, attempt, transaction, definition)
    writer.add_part("model", "w", "value", io.BytesIO(b"\0" * 4))
    assert writer.commit()["manifest"]["length"] > 0
    broker.close_attempt(attempt)
