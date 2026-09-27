"""Interrupted derived work resumes after a Runtime upgrade or reinstall of the same release."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.weights_sink import work_fingerprint

NUMERICAL = bytes(range(32))


def _spec(installation: str, *, payload: str = "sha256:" + "aa" * 32) -> dict[str, object]:
    return {
        "payload_digest": payload,
        "inputs": [{"input_id": "source", "digest": "sha256:" + "bb" * 32}],
        "outputs": [{"output_id": "model", "max_bytes": 4096}],
        "job": {"installation_id": installation, "job_descriptor_id": "sha256:" + "33" * 32},
    }


BEFORE = work_fingerprint(_spec("install-before-upgrade"), NUMERICAL)
AFTER = work_fingerprint(_spec("install-after-upgrade"), NUMERICAL)


def test_output_determining_facts_still_separate_work() -> None:
    assert BEFORE == AFTER
    assert work_fingerprint(_spec("x", payload="sha256:" + "cc" * 32), NUMERICAL) != BEFORE
    assert work_fingerprint(_spec("x"), bytes(32)) != BEFORE
    assert (
        work_fingerprint(_spec("x"), NUMERICAL, operation_identity="sha256:" + "d" * 64) != BEFORE
    )


def test_interrupted_derived_work_resumes_after_a_new_installation(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    plain = next(value for alias, value in tensorfs.seed_digests() if alias == "plain/1")
    tensor = {
        "logical_dtype": "f32",
        "shape": [512],
        "encoding": plain,
        "parts": {"value": {"dtype": "f32", "shape": [512]}},
    }
    targets = {"model": {"add": {"a": tensor, "b": tensor}, "drop": []}}
    order = [("model", "a"), ("model", "b")]
    transaction = "sha256:" + "44" * 32
    writer = store.begin_derived(
        transaction, 1, {}, targets, {}, order, 4096, work_fingerprint=BEFORE
    )
    writer.add_part("model", "a", "value", io.BytesIO(b"\x31" * 2048))
    facts = writer.checkpoint("before-upgrade", "model")
    writer.fence()  # the worker stopped here; Runtime is upgraded and the release reinstalled
    head_length = facts["head_length"]
    assert isinstance(head_length, int)
    head = (str(facts["head"]), head_length)
    resumed = store.begin_derived(
        transaction, 2, {}, targets, {}, order, 4096, work_fingerprint=AFTER, checkpoint=head
    )
    assert resumed.completed_parts() == [("model", "a", "value")]
    resumed.add_part("model", "b", "value", io.BytesIO(b"\x32" * 2048))
    assert resumed.commit()["manifest"]
    with pytest.raises(tensorfs.errors.Refusal):
        store.begin_derived(
            "sha256:" + "55" * 32, 1, {}, targets, {}, order, 4096, work_fingerprint="sha256:"
        )
