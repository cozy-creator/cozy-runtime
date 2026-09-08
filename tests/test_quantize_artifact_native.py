"""Stable managed quantization results and real native group recovery (job-022)."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import msgspec
import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    Loader,
    Model,
    ModelArtifact,
    Telemetry,
    WeightsOutput,
    WeightsPart,
    WeightsSink,
    WeightsTarget,
    WeightsTensor,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._errors import Outcome
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import Attempt
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.derive import plan, quantize_artifact
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.microscale import encode_fp8_rowwise, encode_mxfp8
from cozy_runtime.derive.quantization import prepare_source_quantization, quantization_additions
from cozy_runtime.internal.weights_sink import WeightsCheckpoint, WeightsTransactionHost

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
    weights: WeightsSink,
    tel: Telemetry,
    threshold: float | None = None,
) -> ModelArtifact:
    return quantize_artifact(
        source,
        plan((component,), encoding, max_relative_frobenius=threshold),
        sink=weights,
        ctx=ctx,
        tel=tel,
    )


def _host(
    store: tensorfs.Store,
    request: str,
    source: ModelArtifact | None = None,
    *,
    epoch: int = 1,
    checkpoint: Callable[[WeightsCheckpoint], None] | None = None,
) -> WeightsTransactionHost:
    return WeightsTransactionHost(
        store=store,
        owner_scope="quantization",
        request_id=request,
        invocation_spec_digest="sha256:" + hashlib.sha256(request.encode()).hexdigest(),
        work_fingerprint="sha256:" + "52" * 32,
        writer_session_id=epoch,
        allowed_sources={} if source is None else {source.manifest.digest: source.manifest.length},
        output_bounds={"model": BOUND},
        record_checkpoint=checkpoint,
    )


def _source(store: tensorfs.Store, component: str) -> ModelArtifact:
    plain = dict(tensorfs.seed_digests())["plain/1"]
    keys = ("z.weight", "a.weight", "bias")
    tensors = {
        key: WeightsTensor("bf16", (8, 64), plain, {"value": WeightsPart("bf16", (8, 64))})
        for key in keys
    }
    host = _host(store, "source")
    writer = host.open(
        WeightsCommit(
            output_slot="model",
            sources={},
            targets={component: WeightsTarget(add=tensors)},
            configs={},
            order=tuple((component, key) for key in keys),
            max_new_bytes=BOUND,
        )
    )
    values = np.random.default_rng(22).normal(0, 0.05, (8, 64)).astype(np.float32)
    for index, key in enumerate(keys):
        writer.add_part(component, key, "value", st.from_f32(values + index, "BF16"))
    return writer.commit().artifact


def _run(
    root: Path,
    host: WeightsTransactionHost,
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
    result, outcome, record = attempt(
        app.get("quantize"),
        {
            "source": msgspec.to_builtins(source),
            "component": component,
            "encoding": encoding,
            "threshold": threshold,
        },
        Invocation(
            host.request_id,
            root,
            time.monotonic() + 30,
            models={"source": _derive_model(Source, source.manifest.digest)},
            weights=host.open,
            weights_source_structure=host.structure,
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
    checkpoints: list[WeightsCheckpoint] = []

    def checkpoint(value: WeightsCheckpoint) -> None:
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
    structure = host.structure(source.manifest.digest)
    selected = prepare_source_quantization(structure, components=("unet",))
    additions = quantization_additions(encoding, selected, "unet", target_logical_dtype="bf16")
    writer = host.open(
        WeightsCommit(
            output_slot="model",
            sources={"source": source.manifest.digest},
            targets={
                "unet": WeightsTarget(
                    source="source",
                    source_component="unet",
                    drop=tuple(sorted(additions)),
                    add=additions,
                )
            },
            configs={},
            order=tuple((t.component, t.key) for t in structure.tensors),
            max_new_bytes=BOUND,
        )
    )
    first = selected.tensors[0]
    raw = bytearray(1024)
    writer.source_read_into("source", "unet", first.key, "value", 0, memoryview(raw))
    values = st.to_f32(raw, "BF16", list(first.shape))
    encoded = (encode_fp8_rowwise if encoding == "fp8-rowwise/1" else encode_mxfp8)(
        first.key, values
    )
    writer.add_part("unet", first.key, "data", encoded.payload)
    # Persist only the data role. A restarted kernel must not regard half a tensor as done.
    writer.checkpoint()
    writer.fence()
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
