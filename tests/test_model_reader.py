"""Real native inputs, scoped broker reads, and banked format reconstruction."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import Cancelled, CapabilityError, WeightsReader
from cozy_runtime.author._executor_requests import (
    Answer,
    Reply,
    Request,
    WriterSource,
    decode,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import Attempt
from cozy_runtime.derive.quantization import QuantizationSource
from cozy_runtime.internal.encoding.formats import (
    SPEC_MXFP8,
    SPEC_PLAIN,
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
)
from cozy_runtime.internal.model_values import read_values_into
from cozy_runtime.internal.weights_writer import ExecutionStorage, WriterBroker
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.grants import MODEL_PREFIX
from durable_seam import seam
from weights_channel import broker_lane


def model(
    store: Any, label: str, tensors: dict[str, Any], source: tuple[str, int] | None = None
) -> tuple[str, int]:
    targets = {
        "model": {
            "drop": [],
            "add": {
                name: {
                    "logical_dtype": row["logical"]["dtype"],
                    "shape": row["logical"]["shape"],
                    "encoding": row["encoding"],
                    "parts": {
                        role: {"dtype": part["dtype"], "shape": part["shape"]}
                        for role, part in row["roles"].items()
                    },
                }
                for name, row in tensors.items()
            },
        }
    }
    sources = {}
    order = [("model", name) for name in tensors]
    if source is not None:
        sources = {"original": source}
        targets["model"].update(source="original", source_component="model", drop=list(tensors))
        header = tensorfs.parse_header(store.manifest(source[0])["header"])
        order = [("model", name) for name in header["components"]["model"]]
    transaction = "sha256:" + hashlib.sha256(label.encode()).hexdigest()
    size = sum(len(part["bytes"]) for row in tensors.values() for part in row["roles"].values())
    writer = store.begin_derived(
        transaction, 1, sources, targets, {}, order, size, work_fingerprint=transaction
    )
    for name, row in tensors.items():
        for role, part in row["roles"].items():
            writer.add_part("model", name, role, io.BytesIO(part["bytes"]))
    manifest = writer.commit()["manifest"]
    return "sha256:" + manifest["sha256"], manifest["length"]


def plain(values: np.ndarray, dtype: str = "f32") -> dict[str, Any]:
    return {
        "logical": {"dtype": dtype, "shape": list(values.shape)},
        "encoding": SPEC_PLAIN,
        "roles": {
            "value": {"dtype": dtype, "shape": list(values.shape), "bytes": values.tobytes()}
        },
    }


def vector(name: str, encoding: str) -> tuple[dict[str, Any], np.ndarray]:
    path = Path(__file__).parents[1] / "src/cozy_runtime/internal/encoding/vectors" / name
    case = next(
        row for row in json.loads(path.read_bytes())["cases"] if row["logical"]["dtype"] == "f32"
    )
    expected = np.frombuffer(base64.b64decode(case["expect_logical_bits"]), dtype="<f4").reshape(
        case["logical"]["shape"]
    )
    row = {
        "logical": case["logical"],
        "encoding": encoding,
        "roles": {
            name: {**part, "bytes": base64.b64decode(part["bits"])}
            for name, part in case["roles"].items()
        },
    }
    return row, expected


def reader(
    tmp_path: Path, store: Any, references: dict[str, tuple[str, int]]
) -> tuple[
    WeightsReader,
    dict[str, QuantizationSource],
    WriterBroker,
    AttemptRecord,
    Attempt,
    list[Request],
]:
    spool = tmp_path / "spool"
    spool.mkdir(exist_ok=True)
    record = AttemptRecord(
        "reader-proof",
        1,
        b"\x31" * 32,
        {
            "inputs": [
                {"input_id": MODEL_PREFIX + name, "digest": ref[0], "length": ref[1]}
                for name, ref in references.items()
            ]
        },
        state="running",
        spool=spool,
        kind="job",
    )
    broker = WriterBroker(lambda: store, store_root=Path(store.root))
    frames: list[Request] = []
    handle = broker_lane(broker, record)

    def recorded(request: Request) -> Reply:
        frames.append(request)
        return handle(request)

    client = ExecutionStorage(spool, seam(recorded), {})
    models = {name: _derive_model(QuantizationSource, ref[0]) for name, ref in references.items()}
    attempt = Attempt(record.request_id, spool)
    service = WeightsReader(
        attempt,
        models,
        lambda source: client.source(source.checkpoint_ref),
        lambda: bool(record.canceling),
        read_values_into,
    )
    return service, models, broker, record, attempt, frames


@pytest.mark.parametrize("changed_role", ["data", "scale"])
def test_inherited_identity_and_changed_scale_use_native_read_custody(
    tmp_path: Path, changed_role: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    encoded, expected = vector("fp8-rowwise.json", SPEC_ROWWISE)
    original = model(
        store, "original", {"inherited": plain(np.arange(9, dtype=np.float32)), "changed": encoded}
    )
    changed = {**encoded, "roles": dict(encoded["roles"])}
    scale = changed["roles"]["scale"]
    changed_expected = expected.copy()
    if changed_role == "scale":
        changed["roles"]["scale"] = {
            **scale,
            "bytes": (np.frombuffer(scale["bytes"], dtype="<f4") * 2).tobytes(),
        }
        changed_expected *= 2
    else:
        data = changed["roles"]["data"]
        payload = bytearray(data["bytes"])
        payload[0] = 0x38  # exact E4M3 1.0, replacing the vector's first zero
        changed["roles"]["data"] = {**data, "bytes": bytes(payload)}
        changed_expected.flat[0] = np.frombuffer(scale["bytes"], dtype="<f4")[0]
    candidate = model(store, "candidate", {"changed": changed}, original)
    service, models, broker, _, attempt, frames = reader(
        tmp_path, store, {"reference": original, "candidate": candidate}
    )
    with service.open(models["reference"]) as ref, service.open(models["candidate"]) as cand:
        assert ref.components() == ("model",)
        assert ref.keys("model") == ("inherited", "changed")
        assert ref.identity("model", "inherited") == cand.identity("model", "inherited")
        assert all(isinstance(frame, WriterSource) for frame in frames)
        assert ref.identity("model", "changed") != cand.identity("model", "changed")
        for role, facts in encoded["roles"].items():
            original_part = bytearray(len(facts["bytes"]))
            changed_part = bytearray(len(changed["roles"][role]["bytes"]))
            ref.read_part_into("model", "changed", role, 0, original_part)
            cand.read_part_into("model", "changed", role, 0, changed_part)
            assert original_part == facts["bytes"]
            assert changed_part == changed["roles"][role]["bytes"]
        if importlib.util.find_spec("torch") is not None:
            left = np.empty(expected.shape, dtype=np.float32)
            right = np.empty_like(left)
            ref.read_values_into("model", "changed", 0, left)
            cand.read_values_into("model", "changed", 0, right)
            np.testing.assert_array_equal(left.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(right, changed_expected)
    broker.close()
    assert not broker.sources
    with pytest.raises(CapabilityError):
        ref.identity("model", "inherited")
    with pytest.raises(CapabilityError):
        service.open(_derive_model(QuantizationSource, original[0]))
    attempt.closed = True
    with pytest.raises(CapabilityError):
        service.open(models["reference"])


@pytest.mark.parametrize(
    ("filename", "encoding"),
    [
        ("fp8-rowwise.json", SPEC_ROWWISE),
        ("fp8-rowwise-keepdim.json", SPEC_ROWWISE_KEEPDIM),
        ("mxfp8.json", SPEC_MXFP8),
    ],
)
def test_logical_ranges_match_banked_format_vectors(
    tmp_path: Path, filename: str, encoding: str
) -> None:
    pytest.importorskip("torch", reason="encoded reconstruction uses the optional Torch decoder")
    store = tensorfs.Store.init(tmp_path / "store")
    row, expected = vector(filename, encoding)
    reference = model(store, filename, {"weight": row})
    service, models, _, _, _, _ = reader(tmp_path, store, {"source": reference})
    with service.open(models["source"]) as view:
        for offset, count in [(0, expected.size), (3, 71), (31, 35), (expected.size - 3, 3)]:
            target = np.empty(count, dtype=np.float32)
            view.read_values_into("model", "weight", offset, target)
            np.testing.assert_array_equal(
                target.view(np.uint32),
                expected.reshape(-1)[offset : offset + count].view(np.uint32),
            )
        with pytest.raises(CapabilityError):
            view.read_values_into("model", "weight", expected.size, np.empty(1, dtype=np.float32))
        with pytest.raises(CapabilityError):
            view.read_values_into("model", "weight", 0, np.empty(1, dtype=np.float64))


def test_plain_bf16_fp16_and_scalar_reads(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    values = np.array([0, -1.5, 2.25, 8], dtype=np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype("<u2")
    reference = model(
        store,
        "plain",
        {
            "bf16": plain(bf16, "bf16"),
            "fp16": plain(values.astype("<f2"), "f16"),
            "scalar": plain(np.array(3.5, dtype=np.float32)),
        },
    )
    service, models, _, _, _, _ = reader(tmp_path, store, {"source": reference})
    with service.open(models["source"]) as view:
        for key in ("bf16", "fp16"):
            output = np.empty(4, dtype=np.float32)
            view.read_values_into("model", key, 0, output)
            np.testing.assert_array_equal(output, values)
        scalar = np.empty((), dtype=np.float32)
        view.read_values_into("model", "scalar", 0, scalar)
        assert scalar.item() == 3.5


def test_foreign_stale_canceled_and_released_views_refuse(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    reference = model(store, "scope", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, record, _, _ = reader(tmp_path, store, {"source": reference})
    view = service.open(models["source"])
    assert len(broker.sources) == 1
    with pytest.raises(CapabilityError) as retired:
        decode({"kind": "weights_writer", "operation": "check", "view_id": "foreign"})
    assert retired.value.code == "durable_request_malformed"
    assert len(broker.sources) == 1
    ungranted = broker.handle(record, WriterSource(manifest="sha256:" + "ff" * 32))
    assert isinstance(ungranted, Answer) and not ungranted.ok
    record.spec["inputs"][0]["length"] += 1
    with pytest.raises(tensorfs.errors.Refusal):
        service.open(models["source"])
    record.spec["inputs"][0]["length"] -= 1
    record.canceling = "requested"
    broker.close_attempt(record)
    assert not broker.sources
    with pytest.raises(Cancelled):
        view.identity("model", "weight")
    record.canceling = ""
    view = service.open(models["source"])
    broker.close_attempt(record)
    with pytest.raises((EOFError, OSError, tensorfs.errors.Refusal)):
        view.read_part_into("model", "weight", "value", 0, bytearray(4))


def test_ragged_mxfp8_and_partial_final_block(tmp_path: Path) -> None:
    pytest.importorskip("torch", reason="encoded reconstruction uses the optional Torch decoder")
    store = tensorfs.Store.init(tmp_path / "store")
    row = {
        "logical": {"dtype": "f32", "shape": [2, 35]},
        "encoding": SPEC_MXFP8,
        "roles": {
            "data": {"dtype": "f8_e4m3fn", "shape": [2, 35], "bytes": bytes([0x38]) * 70},
            "scale": {"dtype": "u8", "shape": [2, 2], "bytes": bytes([127, 128, 126, 129])},
        },
    }
    expected = np.array([1] * 32 + [2] * 3 + [0.5] * 32 + [4] * 3, dtype=np.float32)
    ref = model(store, "ragged", {"weight": row})
    service, models, _, _, _, _ = reader(tmp_path, store, {"source": ref})
    with service.open(models["source"]) as view:
        for offset, count in [(0, 70), (31, 8), (67, 3)]:
            target = np.empty(count, dtype=np.float32)
            view.read_values_into("model", "weight", offset, target)
            np.testing.assert_array_equal(target, expected[offset : offset + count])


def test_missing_or_corrupt_native_header_cannot_open_a_view(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    ref = model(store, "corrupt", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, _, _, _ = reader(tmp_path, store, {"source": ref})
    header = store.manifest(ref[0])["header"]
    assert header is not None
    files = [
        path
        for path in (tmp_path / "store").rglob("*")
        if path.is_file() and path.stat().st_size == len(header) and path.read_bytes() == header
    ]
    assert len(files) == 1
    path = files[0]
    path.chmod(0o600)
    path.write_bytes(header[:-1] + bytes([header[-1] ^ 1]))
    with pytest.raises(tensorfs.errors.Refusal):
        service.open(models["source"])
    broker.close()
    assert not broker.sources
    assert not path.exists(), "native verification must quarantine the corrupt header"
    with pytest.raises(tensorfs.errors.Refusal):
        service.open(models["source"])
    broker.close()
    assert not broker.sources


def test_source_descriptor_is_bound_to_the_accepted_digest(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    reference = model(store, "digest-bound", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, record, _, _ = reader(tmp_path, store, {"source": reference})
    view = service.open(models["source"])
    view.identity("model", "weight")
    record.digest = b"\x32" * 32
    try:
        with pytest.raises(tensorfs.errors.Refusal) as refused:
            view.read_part_into("model", "weight", "value", 0, bytearray(4))
        assert refused.value.code == "weights_writer_closed"
    finally:
        view.close()
        broker.close()


def test_source_handle_limit_refuses_before_exhausting_worker_descriptors(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    reference = model(store, "bounded-handles", {"weight": plain(np.arange(16, dtype=np.float32))})
    service, models, broker, _, _, _ = reader(tmp_path, store, {"source": reference})
    views = [service.open(models["source"]) for _ in range(16)]
    try:
        with pytest.raises(CapabilityError) as refused:
            service.open(models["source"])
        assert refused.value.code == "weights_reader_bounds"
        assert sum(len(sources) for sources in broker.sources.values()) == 16
    finally:
        for view in views:
            view.close()
        broker.close()
    assert not broker.sources
