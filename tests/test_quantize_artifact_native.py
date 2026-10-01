"""Stable managed quantization results and real native group recovery (job-022)."""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    Loader,
    Model,
    ModelArtifact,
    Telemetry,
    WeightsOutput,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._errors import Outcome
from cozy_runtime.author._loader import TensorSpec
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import Attempt
from cozy_runtime.derive import plan, quantize_artifact
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.microscale import encode_fp8_rowwise, encode_mxfp8
from cozy_runtime.derive.quantization import prepare_source_quantization, quantization_additions
from cozy_runtime.internal import plane
from cozy_runtime.internal.encoding import TORCH_DTYPES
from cozy_runtime.internal.fill import Checkpoint
from cozy_runtime.internal.weights import PlaneBackend, Weights
from native_weights import NativeExecution

BOUND = 1 << 20


class Source(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("quantization must not construct inference")


@invocable(memoize=True)
async def quantize(
    ctx: Context,
    *,
    source: Source,
    component: str,
    encoding: str,
    tel: Telemetry,
    threshold: float | None = None,
) -> ModelArtifact:
    return quantize_artifact(
        source,
        plan((component,), encoding, max_relative_frobenius=threshold),
        ctx=ctx,
        tel=tel,
    )


def _host(
    store: tensorfs.Store,
    request: str,
    source: ModelArtifact | None = None,
    *,
    epoch: int = 1,
    checkpoint: Callable[[Any], None] | None = None,
) -> NativeExecution:
    return NativeExecution(
        store,
        Path(store.root).parent / "executions",
        request,
        {} if source is None else {"source": source},
        {"model": BOUND},
        epoch=epoch,
        after_checkpoint=checkpoint,
    )


def _source(
    store: tensorfs.Store,
    component: str,
    dtype: str = "bf16",
    *,
    keys: tuple[str, ...] = ("z.weight", "a.weight", "bias"),
) -> ModelArtifact:
    plain = dict(tensorfs.seed_digests())["plain/1"]
    tensors = {key: Tensor(dtype, (8, 64), plain, {"value": Part(dtype, (8, 64))}) for key in keys}
    with (
        _host(store, "source") as host,
        host.client.open_output(
            "model",
            Derivation(
                {}, {component: Target(add=tensors)}, {}, tuple((component, key) for key in keys)
            ),
        ) as writer,
    ):
        values = np.random.default_rng(22).normal(0, 0.05, (8, 64)).astype(np.float32)
        for index, key in enumerate(keys):
            writer.add_part(
                component, key, "value", io.BytesIO(st.from_f32(values + index, dtype.upper()))
            )
        return host.client.adopt_model(writer.commit())


@pytest.mark.parametrize("encoding,payload_bytes", [("fp8-rowwise/1", 544), ("mxfp8/1", 528)])
def test_two_durable_quantization_groups_skip_source_reads_and_payload_writes(
    tmp_path: Path, encoding: str, payload_bytes: int
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "unet", keys=("z.weight", "a.weight", "b.weight", "bias"))
    cancelled = [False]
    checkpoints: list[Any] = []

    def stop_after_two(value: Any) -> None:
        checkpoints.append(value)
        cancelled[0] = len(checkpoints) == 2

    partial, outcome, _ = _run(
        tmp_path / "partial",
        _host(store, "two-groups", source, checkpoint=stop_after_two),
        source,
        "unet",
        encoding,
        cancelled=cancelled,
    )
    assert partial is None and outcome.terminal == "canceled" and len(checkpoints) == 2
    resumed, outcome, record = _run(
        tmp_path / "resumed",
        _host(store, "two-groups", source, epoch=2),
        source,
        "unet",
        encoding,
    )
    assert outcome.terminal == "succeeded" and resumed is not None
    measured = next(
        row["fields"] for row in record.ring.rows() if row.get("name") == "quantization measurement"
    )
    assert isinstance(measured, dict)
    assert measured["reused_keys"] == 2
    assert measured["source_bytes_read_this_run"] == 1024
    assert measured["new_bytes_written_this_run"] == payload_bytes
    clean, outcome, _ = _run(
        tmp_path / "clean",
        _host(store, "clean-three-groups", source),
        source,
        "unet",
        encoding,
    )
    assert outcome.terminal == "succeeded" and clean is not None
    assert clean.manifest == resumed.manifest


def _run(
    root: Path,
    host: NativeExecution,
    source: ModelArtifact,
    component: str,
    encoding: str,
    *,
    cancelled: list[bool] | None = None,
    threshold: float | None = None,
) -> tuple[ModelArtifact | None, Outcome, Attempt]:
    app = App()
    app.job(quantize, weights=(WeightsOutput("model", BOUND),))
    describe(app)
    with host:
        result, outcome, record = attempt(
            app.get("quantize"),
            {
                "source": msgspec.to_builtins(source),
                "component": component,
                "encoding": encoding,
                "threshold": threshold,
            },
            Invocation(
                host.attempt.request_id,
                root,
                math.inf,  # no stopwatch: a test attempt ends on its own outcome
                models={"source": _derive_model(Source, source.manifest.digest)},
                tensorfs_source=host.client.source,
                tensorfs_output=host.client.open_output,
                tensorfs_adopt=host.client.adopt_model,
                cancel=lambda: bool(cancelled and cancelled[0]),
            ),
        )
    artifact = result.result if result is not None else None
    assert artifact is None or isinstance(artifact, ModelArtifact)
    return artifact, outcome, record


@pytest.mark.parametrize("component", ["unet", "transformer"])
@pytest.mark.parametrize("encoding", ["fp8-rowwise/1", "mxfp8/1"])
def test_managed_quantization_recovers_groups_and_returns_stable_artifact(
    tmp_path: Path,
    component: str,
    encoding: str,
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, component)
    uninterrupted, outcome, _ = _run(
        tmp_path / "uninterrupted",
        _host(store, "uninterrupted", source),
        source,
        component,
        encoding,
    )
    assert outcome.terminal == "succeeded", outcome
    assert isinstance(uninterrupted, ModelArtifact)
    cancelled = [False]
    checkpoints: list[Any] = []

    def checkpoint(value: Any) -> None:
        checkpoints.append(value)
        cancelled[0] = True

    partial, outcome, _ = _run(
        tmp_path / "interrupted",
        _host(store, "resume", source, checkpoint=checkpoint),
        source,
        component,
        encoding,
        cancelled=cancelled,
    )
    assert partial is None and outcome.terminal == "canceled", outcome
    assert len(checkpoints) == 1
    # A replacement host/process with a higher native epoch resumes the exact declaration.
    resumed, outcome, record = _run(
        tmp_path / "resumed",
        _host(store, "resume", source, epoch=2),
        source,
        component,
        encoding,
    )
    assert outcome.terminal == "succeeded", outcome
    assert isinstance(resumed, ModelArtifact)
    assert resumed.manifest == uninterrupted.manifest
    measurement = [
        row for row in record.ring.rows() if row.get("name") == "quantization measurement"
    ]
    assert measurement, record.ring.rows()
    # One finished data+scale group was reused, so it was neither read nor encoded again.
    fields = measurement[0]["fields"]
    assert isinstance(fields, dict)
    assert fields["reused_keys"] == 1
    assert fields["source_bytes_read_this_run"] == 1024
    assert fields["worst_relative_frobenius"] is None
    assert "reused_keys" not in msgspec.to_builtins(resumed)
    replay, outcome, _ = _run(
        tmp_path / "replay",
        _host(store, "resume", source, epoch=3),
        source,
        component,
        encoding,
    )
    assert outcome.terminal == "succeeded", outcome
    assert replay == resumed


@pytest.mark.parametrize("encoding", ["fp8-rowwise/1", "mxfp8/1"])
def test_half_written_group_recomputes_and_inherits_unchanged_parts(
    tmp_path: Path,
    encoding: str,
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "unet")
    host = _host(store, "partial-group", source)
    with host.client.source(source.manifest.digest) as capability:
        structure = capability.inspect()
    selected = prepare_source_quantization(structure, components=("unet",))
    additions = quantization_additions(encoding, selected, "unet", target_logical_dtype="bf16")
    writer = host.client.open_output(
        "model",
        Derivation(
            {"source": structure.source},
            {
                "unet": Target(
                    source="source",
                    source_component="unet",
                    drop=tuple(sorted(additions)),
                    add=additions,
                )
            },
            {},
            tuple(
                (component, key) for component, rows in structure.components.items() for key in rows
            ),
        ),
    )
    first = selected.tensors[0]
    raw = bytearray(1024)
    writer.source_read_into("source", "unet", first.key, "value", 0, memoryview(raw))
    values = st.to_f32(raw, "BF16", list(first.shape))
    encoded = (encode_fp8_rowwise if encoding == "fp8-rowwise/1" else encode_mxfp8)(
        first.key, values
    )
    writer.add_part("unet", first.key, "data", io.BytesIO(encoded.payload))
    # Persist only the data role. A restarted kernel must not regard half a tensor as done.
    writer.checkpoint()
    writer.close()
    host.__exit__()
    resumed, outcome, record = _run(
        tmp_path / "resumed",
        _host(store, "partial-group", source, epoch=2),
        source,
        "unet",
        encoding,
    )
    assert outcome.terminal == "succeeded", outcome
    assert isinstance(resumed, ModelArtifact)
    fields = next(
        row["fields"] for row in record.ring.rows() if row["name"] == "quantization measurement"
    )
    assert isinstance(fields, dict)
    assert fields["reused_keys"] == 0
    assert fields["source_bytes_read_this_run"] == 2048
    complete, outcome, _ = _run(
        tmp_path / "complete",
        _host(store, "complete", source),
        source,
        "unet",
        encoding,
    )
    assert outcome.terminal == "succeeded", outcome
    assert isinstance(complete, ModelArtifact)
    assert complete.manifest == resumed.manifest
    original_bytes = store.manifest(source.manifest.digest)["header"]
    result_bytes = store.manifest(resumed.manifest.digest)["header"]
    assert original_bytes is not None and result_bytes is not None
    original = tensorfs.parse_header(original_bytes)
    result = tensorfs.parse_header(result_bytes)
    assert result["components"]["unet"]["bias"] == original["components"]["unet"]["bias"]


@pytest.mark.parametrize("source_dtype", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("encoding", ["fp8-rowwise/1", "mxfp8/1"])
def test_quantized_encoding_keeps_the_source_precision_contract(
    tmp_path: Path, source_dtype: str, encoding: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "unet", source_dtype)
    result, outcome, _ = _run(
        tmp_path / "quantize", _host(store, "quantize", source), source, "unet", encoding
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    raw_header = store.manifest(result.manifest.digest)["header"]
    assert raw_header is not None
    header = tensorfs.parse_header(raw_header)
    expected = "bf16" if source_dtype == "f32" else source_dtype
    assert {row["logical"]["logical_dtype"] for row in header["components"]["unet"].values()} == {
        expected
    }


def test_fp16_quantized_native_header_enters_fp16_fill(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not plane.available():
        pytest.skip("a TensorFS with the weight plane")
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "unet", "f16")
    result, outcome, _ = _run(
        tmp_path / "quantize", _host(store, "quantize", source), source, "unet", "fp8-rowwise/1"
    )
    assert outcome.terminal == "succeeded" and result is not None
    checkpoint = Checkpoint(tmp_path / "store", result.manifest.digest)
    rows = checkpoint.rows("unet")
    backend = PlaneBackend.for_script(
        {"unet": checkpoint},
        rows,
        # The host tier alone: the fill's validation needs no card.
        weights=Weights(torch, torch.device("cpu"), "cpu"),
        construction="fp16-contract",
        release="fp16-contract-proof",
        encoded_leaves="refuse",
    )
    try:
        backend.expect({"unet": [row.key for row in rows]})
        destinations = {row.key: torch.empty(row.shape, dtype=torch.float16) for row in rows}
        for row in rows:
            backend.fill(
                row.key, TensorSpec(row.shape, "float16", nbytes=row.nbytes), destinations[row.key]
            )
        assert set(backend.enqueued) == set(destinations)
        # Read through a real native lease and run the selected decoder into those exact
        # admitted FP16 buffers; no fabricated header, provider or destination stands in.
        with store.acquire_cozytensors(result.manifest.digest) as lease:
            header = checkpoint.header_bytes
            for row in rows:
                parts = {}
                for part in row.encoded.parts:
                    raw = bytearray(part.nbytes)
                    lease.read_part_into(header, row.component, row.name, part.role, 0, raw)
                    parts[part.role] = torch.frombuffer(
                        raw, dtype=getattr(torch, TORCH_DTYPES[part.dtype])
                    ).reshape(part.shape)
                provider = backend.selected[row.key].provider
                if provider.route == "verbatim":
                    destinations[row.key].copy_(parts["value"])
                else:
                    provider.decode(torch, parts, destinations[row.key])
        assert all(
            value.dtype == torch.float16 and bool(torch.isfinite(value).all())
            for value in destinations.values()
        )
    finally:
        backend.close()
