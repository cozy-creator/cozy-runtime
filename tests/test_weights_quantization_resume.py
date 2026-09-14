"""Actual author quantization, native role admission, interruption and restart."""

from __future__ import annotations

import io
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs
from tensorfs.derived import Derivation, OutputCapability

from cozy_runtime.author import (
    App,
    Context,
    ModelArtifact,
    ObjectRef,
    Telemetry,
    UnsupportedInput,
    WeightsOutput,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive import QuantizeResult, plan, quantize
from cozy_runtime.derive.quantization import ArtifactQuantizationRequest, QuantizationSource
from native_weights import NativeExecution

app = App()
BOUND = 32768


@app.job(name="native-quantization", weights=(WeightsOutput("model", max_new_bytes=BOUND),))
def native_quantization(
    ctx: Context,
    source: QuantizationSource,
    payload: ArtifactQuantizationRequest,
    tel: Telemetry,
) -> QuantizeResult:
    return quantize(
        source,
        plan(("model",), "fp8-rowwise/1", max_relative_frobenius=payload.max_relative_frobenius),
        ctx=ctx,
        tel=tel,
    )


def _source(root: Path) -> tuple[tensorfs.Store, str, str, int]:
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

    def __init__(
        self, native: Any, mode: str, reads: list[int], transaction: str, definition: Derivation
    ) -> None:
        self.native, self.mode, self.reads = native, mode, reads
        self.transaction_id, self.definition = transaction, definition
        self.checkpoint_head: tuple[str, int] | None = None

    def __enter__(self) -> ObservedNativeTransaction:
        return self

    def __exit__(self, *_: Any) -> None:
        self.native.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    def add_part(self, component: str, key: str, role: str, data: Any) -> None:
        if self.mode == "short-scale" and role == "scale":
            data = data[: len(data) // 2]
        self.native.add_part(component, key, role, data)

    def source_read_into(self, *args: Any) -> None:
        assert self.mode != "reuse", "complete native role group reread its source"
        self.reads.append(len(args[-1]))
        self.native.source_read_into(*args)

    def checkpoint(self) -> None:
        facts = self.native.checkpoint()
        self.checkpoint_head = (facts["head"], facts["head_length"])
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
) -> tuple[Callable[[], QuantizeResult], list[int], list[Any], list[Any]]:
    checkpoints: list[Any] = []
    reads: list[int] = []
    captured: list[Any] = []
    artifact = ModelArtifact("source", "model", ObjectRef(source, length), "sha256:" + "11" * 32)

    def invoke() -> QuantizeResult:
        with NativeExecution(
            store,
            root,
            "quantize",
            {"source": artifact},
            {"model": BOUND},
            epoch=epoch,
            after_checkpoint=checkpoints.append,
        ) as host:

            def open_native(slot: str, definition: Derivation) -> Any:
                current = host.client.open_output(slot, definition)
                wrapped = ObservedNativeTransaction(
                    current, mode, reads, next(iter(host.client.opened)), definition
                )
                captured.append(wrapped)
                return wrapped

            ctx = host.context()
            ctx._tensorfs_output = lambda slot: OutputCapability(
                lambda definition: open_native(slot, definition)
            )
            result = quantize(
                _derive_model(QuantizationSource, source),
                plan(("model",), "fp8-rowwise/1", max_relative_frobenius=threshold),
                ctx=ctx,
                tel=fake_telemetry(fake_attempt("quantize")),
            )
            assert isinstance(result, QuantizeResult)
            return result

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
    with pytest.raises(tensorfs.errors.Refusal):
        invoke()
    assert sum(reads) == 16384
    transaction = captured[0].transaction_id

    invoke, reads, captured, checkpoints = _run(store, source, length, 2, "checkpoint", tmp_path)
    with pytest.raises(InterruptedAfterCheckpoint):
        invoke()
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
        plan_digest=checkpoints[-1]["plan_digest"],
    )
    assert page["index"] == checkpoints[-1]["index"]
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
    artifact = ModelArtifact("source", "model", ObjectRef(source, length), "sha256:" + "11" * 32)
    with (
        NativeExecution(
            store, tmp_path, "quantize", {"source": artifact}, {"model": BOUND}, epoch=2
        ) as owner,
        owner.client.open_output("model", row.definition) as resumed,
    ):
        assert resumed.completed_parts() == []
        assert resumed.completed_configs() == []
