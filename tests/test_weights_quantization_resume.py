"""Actual author quantization, native role admission, interruption and restart."""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import (
    App,
    Context,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
    WeightsSink,
)
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import invoke_with_fakes
from cozy_runtime.derive import QuantizeResult, plan, quantize
from cozy_runtime.derive.quantization import ArtifactQuantizationRequest, QuantizationSource
from cozy_runtime.internal.weights_sink import WeightsTransactionHost

app = App()
BOUND = 32768


@app.job(name="native-quantization", weights=(WeightsOutput("model", max_new_bytes=BOUND),))
def native_quantization(
    ctx: Context,
    source: QuantizationSource,
    payload: ArtifactQuantizationRequest,
    weights: WeightsSink,
    tel: Telemetry,
) -> QuantizeResult:
    return quantize(
        source,
        plan(("model",), "fp8-rowwise/1", max_relative_frobenius=payload.max_relative_frobenius),
        sink=weights,
        ctx=ctx,
        tel=tel,
    )


def _source(root: Path):
    store = tensorfs.Store.init(root)
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    values = (np.random.default_rng(812).standard_normal((128, 64)) * 0.05).astype("<f2")
    source_id = "sha256:" + "61" * 32
    writer = store.begin_derived(
        source_id,
        1,
        {},
        {
            "model": {
                "drop": [],
                "add": {
                    "linear.weight": {
                        "logical_dtype": "f16",
                        "shape": [128, 64],
                        "encoding": plain,
                        "parts": {"value": {"dtype": "f16", "shape": [128, 64]}},
                    },
                },
            }
        },
        {},
        [("model", "linear.weight")],
        values.nbytes,
        work_fingerprint="sha256:" + "62" * 32,
    )
    writer.add_part("model", "linear.weight", "value", io.BytesIO(values.tobytes()))
    source = writer.commit()["manifest"]
    return store, source_id, "sha256:" + source["sha256"], source["length"]


class InterruptedAfterCheckpoint(RuntimeError):
    pass


class ObservedNativeTransaction:
    """Inject a short native scale stream or interrupt after native checkpoint acceptance."""

    def __init__(self, native: Any, mode: str, reads: list[int]) -> None:
        self.native, self.mode, self.reads = native, mode, reads

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def add_part(self, component: str, key: str, role: str, data: bytes) -> None:
        if self.mode == "short-scale" and role == "scale":
            data = data[: len(data) // 2]
        self.native.add_part(component, key, role, data)

    def source_read_into(self, *args: Any) -> None:
        assert self.mode != "reuse", "complete native role group reread its source"
        self.reads.append(len(args[-1]))
        self.native.source_read_into(*args)

    def checkpoint(self) -> None:
        self.native.checkpoint()
        if self.mode == "checkpoint":
            raise InterruptedAfterCheckpoint()


def _run(
    store: Any,
    source: str,
    length: int,
    epoch: int,
    mode: str,
    root: Path,
    *,
    threshold: float = 0.25,
):
    checkpoints: list[Any] = []
    host = WeightsTransactionHost(
        store=store,
        owner_scope="native-quantization-proof",
        request_id="quantize",
        invocation_spec_digest="sha256:" + "63" * 32,
        work_fingerprint="sha256:" + "64" * 32,
        writer_session_id=epoch,
        allowed_sources={source: length},
        output_bounds={"model": BOUND},
        record_checkpoint=checkpoints.append,
    )
    reads: list[int] = []
    captured: list[Any] = []

    def open_native(request: Any):
        current = host.open(request)
        captured.append(current)
        return ObservedNativeTransaction(current, mode, reads)

    def invoke():
        return invoke_with_fakes(
            app.get("native-quantization"),
            {"max_relative_frobenius": threshold},
            request_id="quantize",
            spool=root / f"attempt-{epoch}",
            models={"source": _derive_model(QuantizationSource, source)},
            weights=open_native,
            weights_source_structure=host.structure,
        ).result

    return invoke, reads, captured, checkpoints


def _gc(store: Any) -> None:
    subprocess.run(
        [str(Path(sys.executable).with_name("tfs")), "gc", store.root],
        check=True,
        capture_output=True,
    )


def test_native_partial_group_recomputes_then_complete_group_reuses_without_reads(
    tmp_path: Path,
) -> None:
    store, source_id, source, length = _source(tmp_path / "resumed")
    invoke, reads, captured, _ = _run(store, source, length, 1, "short-scale", tmp_path)
    with pytest.raises(CapabilityError):
        invoke()
    assert sum(reads) == 16384
    transaction = captured[0].transaction_id

    invoke, reads, captured, checkpoints = _run(store, source, length, 2, "checkpoint", tmp_path)
    with pytest.raises(CapabilityError) as interrupted:
        invoke()
    assert isinstance(interrupted.value.__cause__, InterruptedAfterCheckpoint)
    assert sum(reads) == 16384  # one accepted role is not a complete quality-checked group
    accepted_head = captured[0].checkpoint_head
    assert accepted_head is not None
    store.derived_dispose(source_id)
    _gc(store)  # the unfinished derived writer owns its input custody after origin release

    invoke, reads, captured, _ = _run(store, source, length, 3, "reuse", tmp_path)
    result = invoke()
    assert isinstance(result, QuantizeResult)
    assert result.reused_keys == 1 and result.worst_relative_frobenius is None
    assert result.source_bytes_read_this_run == result.new_bytes_written_this_run == 0
    assert reads == []
    recovered = store.derived_lookup(transaction)["receipt"]
    _gc(store)
    # Local commit preserves the outstanding checkpoint metadata while its uploader
    # may still be banking the Link; the final receipt alone is not remote custody.
    page = store.checkpoint_page(
        *accepted_head,
        operation_id="quantize",
        slot="model",
        plan_digest=checkpoints[-1].plan_digest,
    )
    assert page["index"] == checkpoints[-1].index
    with store.acquire_cozytensors("sha256:" + recovered["manifest"]["sha256"]) as lease:
        assert lease.bytes > 0

    clean, _, clean_source, clean_length = _source(tmp_path / "clean")
    invoke, reads, captured, _ = _run(
        clean, clean_source, clean_length, 1, "clean", tmp_path / "clean-run"
    )
    expected = invoke()
    assert expected.reused_keys == 0 and expected.worst_relative_frobenius is not None
    assert sum(reads) == 16384
    assert clean.derived_lookup(captured[0].transaction_id)["receipt"] == recovered


def test_native_quality_refusal_accepts_no_role(tmp_path: Path) -> None:
    store, _, source, length = _source(tmp_path / "store")
    invoke, reads, captured, _ = _run(store, source, length, 1, "quality", tmp_path, threshold=0.0)
    with pytest.raises(UnsupportedInput) as error:
        invoke()
    assert error.value.code == "quantization_tripwire"
    assert sum(reads) == 16384
    row = captured[0]
    again = row.host
    again.writer_session_id = 2
    resumed = again.open(row.request)
    assert resumed.completed_parts == frozenset()
    assert resumed.completed_configs == frozenset()
    resumed.fence()
