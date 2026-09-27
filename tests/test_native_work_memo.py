"""A real native writer rejects old parts after a same-version helper algorithm edit."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import struct
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import tensorfs

from cozy_runtime.author._calls import _export
from cozy_runtime.internal.hostfacts import HostFacts
from cozy_runtime.internal.memo_implementation import describe
from cozy_runtime.internal.numerical_environment import fingerprint
from cozy_runtime.internal.weights_sink import work_fingerprint


def algorithm(path: Path, delta: int, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path.write_text(f"""from cozy_runtime.author import Context, invocable
def helper(value: int) -> int:
    return value + {delta}
@invocable(memoize=True, memo_version="unchanged-manual-recipe/1", memo_dependencies=(helper,))
async def calculate(ctx: Context, *, value: int) -> int:
    return helper(value)
""")
    spec = importlib.util.spec_from_file_location("native_math_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def implementation(fn: Callable[..., object]) -> Callable[..., Coroutine[Any, Any, Any]]:
    exported = _export(fn)
    assert exported is not None
    return cast(Callable[..., Coroutine[Any, Any, Any]], exported.implementation)


def test_sdk_only_scope_reuses_but_changed_helper_rejects_partial_and_complete_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = algorithm(tmp_path / "before.py", 1, monkeypatch)
    original = implementation(before.calculate)
    identity = describe(original)["operation_identity"]
    original_value = asyncio.run(original(None, value=2))
    assert original_value == 3
    policy = fingerprint(HostFacts(backend="cpu"), threads=1, inherited={})
    spec = {
        "job": {"installation_id": "sdk-one", "job_descriptor_id": "caller-one"},
        "payload_digest": "sha256:" + "a" * 64,
    }
    work = work_fingerprint(spec, policy, operation_identity=identity)
    changed_sdk = {**spec, "job": {"installation_id": "sdk-two", "job_descriptor_id": "caller-two"}}
    assert work_fingerprint(changed_sdk, policy, operation_identity=identity) == work

    store = tensorfs.Store.init(tmp_path / "store")
    plain = dict(tensorfs.seed_digests())["plain/1"]
    tensor = {
        "logical_dtype": "f32",
        "shape": [512],
        "encoding": plain,
        "parts": {"value": {"dtype": "f32", "shape": [512]}},
    }
    targets = {"model": {"add": {"a": tensor, "b": tensor}, "drop": []}}
    order = [("model", "a"), ("model", "b")]
    args: tuple[Any, ...] = ({}, targets, {}, order, 4096)
    declaration = store.derived_declaration(*args, work_fingerprint=work)
    first, caller, changed = ("sha256:" + byte * 64 for byte in ("1", "2", "3"))
    writer = store.begin_derived(first, 1, *args, work_fingerprint=work)
    old_bytes = struct.pack("<f", original_value) * 512
    writer.add_part("model", "a", "value", io.BytesIO(old_bytes))
    checkpoint = writer.checkpoint("first", "model")
    head, head_length = cast(str, checkpoint["head"]), cast(int, checkpoint["head_length"])
    writer.fence()
    adopted = store.adopt_derived_checkpoint(
        caller,
        first,
        declaration,
        head,
        head_length,
        operation_id="caller",
        slot="model",
    )
    resumed = store.begin_derived(
        caller,
        1,
        *args,
        work_fingerprint=work,
        checkpoint=(adopted["head"], adopted["head_length"]),
    )
    assert resumed.completed_parts() == [("model", "a", "value")]
    resumed.add_part("model", "b", "value", io.BytesIO(old_bytes))
    complete = resumed.commit()

    after = algorithm(tmp_path / "after.py", 2, monkeypatch)
    corrected = implementation(after.calculate)
    changed_identity = describe(corrected)["operation_identity"]
    assert changed_identity != identity
    changed_work = work_fingerprint(spec, policy, operation_identity=changed_identity)
    changed_declaration = store.derived_declaration(*args, work_fingerprint=changed_work)
    assert changed_declaration != declaration
    with pytest.raises(Exception) as refused:
        store.adopt_derived_checkpoint(
            changed,
            first,
            changed_declaration,
            head,
            head_length,
            operation_id="changed",
            slot="model",
        )
    assert getattr(refused.value, "code", None) == "TRANSACTION_CONFLICT"
    with pytest.raises(Exception) as refused_complete:
        store.begin_derived(caller, 2, *args, work_fingerprint=changed_work)
    assert getattr(refused_complete.value, "code", None) == "TRANSACTION_CLOSED"

    fresh = store.begin_derived(changed, 1, *args, work_fingerprint=changed_work)
    assert fresh.completed_parts() == []
    new_value = asyncio.run(corrected(None, value=2))
    assert new_value == 4
    new_bytes = struct.pack("<f", new_value) * 512
    for key in ("a", "b"):
        fresh.add_part("model", key, "value", io.BytesIO(new_bytes))
    replacement = fresh.commit()
    assert replacement["manifest"] != complete["manifest"]
    for receipt, expected in ((complete, old_bytes), (replacement, new_bytes)):
        manifest = "sha256:" + receipt["manifest"]["sha256"]
        header = store.manifest(manifest)["header"]
        assert header is not None
        with store.acquire_cozytensors(manifest) as lease:
            for key in ("a", "b"):
                actual = bytearray(2048)
                lease.read_part_into(header, "model", key, "value", 0, actual)
                assert actual == expected
