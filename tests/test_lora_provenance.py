"""Prepared serving identity changes while the original native input remains authorized."""

from __future__ import annotations

import struct
import time
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.author import (
    App,
    Artifact,
    Config,
    Context,
    Invocation,
    ModelRegistry,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.internal import lora_composition
from cozy_runtime.internal.fill import Checkpoint, tensor_schema_of
from native_weights import NativeExecution
from test_lora_composition import compose, fixture

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from test_lora_peft import TinyModel


class Identity(msgspec.Struct):
    checkpoint: str
    factor: float


@invocable
async def inspect_adapted(ctx: Context, *, model: TinyModel) -> Identity:
    raw = bytearray(4)
    with ctx.tensorfs_source(model) as source:
        source.read_part_into("transformer", "proj.lora_A.adapter_0.weight", "value", 0, raw)
    return Identity(model.checkpoint_ref, struct.unpack("<f", raw)[0])


def test_effective_checkpoint_and_source_keep_exact_admitted_base(tmp_path: Path) -> None:
    store, base, first, _second = fixture(tmp_path)
    positive, _ = compose(store, tmp_path, "positive", base, [(first, 0.5)])
    negative, _ = compose(store, tmp_path, "negative", base, [(first, -0.5)])
    registry = ModelRegistry(release="provenance-proof", substrate=lambda: torch.device("meta"))

    def construct(name: str, digest: str) -> TinyModel:
        checkpoint = Checkpoint.from_store(store, digest)
        bound = lora_composition.bind(
            Artifact(
                base.manifest.digest, tensor_schema_of(checkpoint.rows("transformer")), Config({})
            ),
            TinyModel,
            checkpoint.header["configs"][lora_composition.GRAPH_CONFIG],
            prepared_snapshot=digest,
        )
        return registry.acquire(name + ".models.model", TinyModel, bound, mode="derive")

    model = construct("positive", positive.manifest.digest)
    other = construct("negative", negative.manifest.digest)
    assert construct("same", positive.manifest.digest) is model
    assert other is not model
    assert model.checkpoint_ref == positive.manifest.digest
    assert other.checkpoint_ref == negative.manifest.digest
    assert model._cozy_selection_ref == other._cozy_selection_ref == base.manifest.digest
    app = App()
    app.job(inspect_adapted)
    describe(app)
    try:
        # Real native source custody contains the admitted base and its effective view.
        # The other view is physically present in CAS but is outside this input closure.
        with NativeExecution(
            store, tmp_path, "inspect", {"base": base, "composed": positive}, {}
        ) as execution:
            invocation = Invocation(
                "inspect",
                tmp_path / "spool",
                time.monotonic() + 30,
                models={"model": model},
                tensorfs_source=execution.client.source,
            )
            result, outcome, _ = attempt(
                app.get("inspect_adapted"), {"model": msgspec.to_builtins(base)}, invocation
            )
            assert outcome.terminal == "succeeded", outcome
            assert result is not None and result.result == Identity(positive.manifest.digest, 1.0)
            changed = msgspec.to_builtins(negative)
            _, refused, _ = attempt(app.get("inspect_adapted"), {"model": changed}, invocation)
            assert refused.code == "model_artifact_binding"
            with pytest.raises(Exception, match="granted|admitted|bound"):
                execution.client.source(other.checkpoint_ref)
    finally:
        registry.unload_all()
