"""Public quantization helpers over real worker-owned native source/output channels."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorfs
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from cozy_runtime.author import ModelArtifact, UnsupportedInput
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_telemetry
from cozy_runtime.derive import QuantizePlan, QuantizeResult, plan, quantize
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.derive.quantization import (
    ArtifactQuantizationRequest,
    QuantizationSource,
    bf16_targets,
    inherited_configs,
    prepare_bf16,
    prepare_source_quantization,
    quantization_additions,
    quantize_component_into,
    write_bf16,
)
from cozy_runtime.internal.worker.workspace import Workspace
from native_weights import NativeExecution

BOUND = 1 << 20
Row = tuple[str, tuple[int, ...], bytes]


def mint(root: Path, rows: Mapping[str, Mapping[str, Row]]) -> tuple[Any, ModelArtifact]:
    store = tensorfs.Store.init(root / "store")
    plain = dict(tensorfs.seed_digests())["plain/1"]
    targets = {
        component: Target(
            add={
                key: Tensor(dtype, shape, plain, {"value": Part(dtype, shape)})
                for key, (dtype, shape, _) in tensors.items()
            }
        )
        for component, tensors in rows.items()
    }
    with (
        NativeExecution(store, root, "source", {}, {"model": BOUND}) as owner,
        owner.client.open_output(
            "model",
            Derivation(
                {},
                targets,
                {"model": Config("add")},
                [(component, key) for component, tensors in rows.items() for key in tensors],
            ),
        ) as writer,
    ):
        for component, tensors in rows.items():
            for key, (_, _, data) in tensors.items():
                writer.add_part(component, key, "value", io.BytesIO(data))
        writer.add_config("model", io.BytesIO(b"{}"))
        return store, owner.client.adopt_model(writer.commit())


def sdxl_rows(*, poison: bool = False) -> dict[str, dict[str, Row]]:
    rng = np.random.default_rng(76)

    def row(dtype: str, shape: tuple[int, ...]) -> Row:
        value = rng.standard_normal(shape).astype(np.float32) * 0.05
        return dtype, shape, st.from_f32(value, dtype.upper())

    q = row("f16", (8, 64))
    if poison:
        bad = np.frombuffer(q[2], dtype="<f2").copy()
        bad[3] = np.nan
        q = q[0], q[1], bad.tobytes()
    return {
        "unet": {
            "down.0.attn.to_q.weight": q,
            "down.0.attn.to_q.bias": row("f16", (8,)),
            "mid.proj.weight": row("bf16", (4, 32)),
            "time_embed.weight": row("f16", (6, 48)),
            "mid.norm.weight": row("f32", (8,)),
            "position_ids": ("i64", (1, 16), bytes(8 * 16)),
        },
        "text_encoder": {"embed.weight": row("f16", (16, 32))},
    }


def run(
    root: Path,
    store: Any,
    source: ModelArtifact,
    spec: QuantizePlan,
    *,
    name: str = "quantize",
    epoch: int = 1,
    inspected: bool = False,
    compose: bool = False,
) -> tuple[QuantizeResult | None, ModelArtifact]:
    model = _derive_model(QuantizationSource, source.manifest.digest)
    with NativeExecution(
        store, root, name, {"source": source}, {"model": BOUND}, epoch=epoch
    ) as owner:
        ctx = owner.context()
        tel = fake_telemetry(fake_attempt(name))
        with ctx.tensorfs_source(model) as capability:
            view = capability.inspect()
        if not compose:
            result = quantize(view if inspected else model, spec, ctx=ctx, tel=tel)
            transaction = next(iter(owner.client.opened))
            receipt = store.derived_lookup(transaction)["receipt"]
            return result, owner.client.adopt_model(receipt)
        bf16 = prepare_bf16(view, keep=spec.keep, source_dtypes=spec.bf16_sources)
        selected = (
            None
            if spec.encoding is None
            else prepare_source_quantization(view, components=spec.components)
        )
        excluded = (
            set()
            if selected is None
            else {(tensor.component, tensor.key) for tensor in selected.tensors}
        )
        targets = bf16_targets(bf16, source="source", excluded=excluded)
        if selected is not None and spec.encoding is not None:
            for component in selected.components:
                additions = {
                    **targets[component].add,
                    **quantization_additions(spec.encoding, selected, component),
                }
                targets[component] = Target(
                    source="source",
                    source_component=component,
                    drop=tuple(sorted(additions)),
                    add=additions,
                )
        with ctx.output("model").open(
            Derivation(
                {"source": view.source},
                targets,
                inherited_configs(bf16, source="source"),
                bf16.order,
            )
        ) as writer:
            write_bf16(writer, ctx, tel, plan=bf16, source="source", excluded=excluded)
            if selected is not None and spec.encoding is not None:
                for component in selected.components:
                    quantize_component_into(
                        writer,
                        ctx,
                        ArtifactQuantizationRequest(),
                        tel,
                        encoding=spec.encoding,
                        plan=selected,
                        component=component,
                        source="source",
                        source_component=component,
                        target_component=component,
                    )
            return None, ctx.adopt_model(writer.commit())


def header(store: Any, artifact: ModelArtifact) -> Any:
    return tensorfs.parse_header(store.manifest(artifact.manifest.digest)["header"])


def read_value(
    root: Path, store: Any, artifact: ModelArtifact, component: str, key: str, size: int
) -> bytes:
    with NativeExecution(
        store, root, "read-" + component + "-" + key, {"source": artifact}, {"model": 0}
    ) as owner:
        with owner.client.source(artifact.manifest.digest) as capability:
            view = capability.inspect()
        with owner.client.open_output(
            "model",
            Derivation(
                {"source": view.source},
                {component: Target(source="source", source_component=component)},
                {},
                [(component, name) for name in view.components[component]],
            ),
        ) as writer:
            data = bytearray(size)
            writer.source_read_into("source", component, key, "value", 0, data)
            return bytes(data)


@pytest.mark.parametrize("encoding", [None, "fp8-rowwise/1", "mxfp8/1"])
def test_facade_and_native_helper_composition_produce_identical_artifacts(
    tmp_path: Path, encoding: str | None
) -> None:
    store, source = mint(tmp_path, sdxl_rows())
    spec = plan(("unet",) if encoding else (), encoding)
    result, facade = run(tmp_path, store, source, spec, name="facade")
    _, composed = run(tmp_path, store, source, spec, name="composition", compose=True)
    assert facade.manifest == composed.manifest
    assert result is not None and result.converted_keys == 1
    assert result.encoded_keys == (2 if encoding else 0)
    assert result.source_bytes_read_this_run > 0


def test_model_and_native_inspection_reach_the_same_tensors(tmp_path: Path) -> None:
    store, source = mint(tmp_path, sdxl_rows())
    spec = plan(("unet",), "mxfp8/1", max_relative_frobenius=0.25)
    by_model, first = run(tmp_path, store, source, spec, name="model")
    by_view, second = run(tmp_path, store, source, spec, name="view", inspected=True)
    assert first.manifest == second.manifest
    assert by_model is not None and by_view is not None
    assert (by_model.encoded_keys, by_model.converted_keys, by_model.inherited_keys) == (2, 1, 4)
    assert by_model.worst_relative_frobenius == by_view.worst_relative_frobenius
    assert (
        by_model.worst_relative_frobenius is not None
        and 0 < by_model.worst_relative_frobenius < 0.25
    )
    assert by_model.normalization_worst_relative_frobenius is not None
    assert 0 < by_model.normalization_worst_relative_frobenius <= 2**-8


def test_tripwire_and_nonfinite_values_refuse_before_native_commit(tmp_path: Path) -> None:
    for poison, threshold, code in (
        (False, 1e-9, "quantization_tripwire"),
        (True, None, "unsupported_input"),
    ):
        root = tmp_path / str(poison)
        store, source = mint(root, sdxl_rows(poison=poison))
        with pytest.raises(UnsupportedInput) as refused:
            run(root, store, source, plan(("unet",), "mxfp8/1", max_relative_frobenius=threshold))
        assert refused.value.code == code
        # The source is the only committed native result; the refused output never has a receipt.
        with Workspace(Path(store.root)).locked() as db:
            rows = db.execute("SELECT receipt FROM weights WHERE request='quantize'").fetchall()
        assert len(rows) == 1 and not rows[0][0]
    invalid: tuple[Callable[[], QuantizePlan], ...] = (
        lambda: plan(("unet",)),
        lambda: plan((), "mxfp8/1"),
        lambda: plan(("unet",), "int3/0"),
        lambda: plan(("unet",), "mxfp8/1", max_relative_frobenius=-1),
        lambda: plan(("unet",), "mxfp8/1", keep=("unet",)),
        lambda: plan(keep=("vae", "vae")),
        lambda: plan(bf16_sources=()),
        lambda: plan(bf16_sources=("f16", "f16")),
        lambda: plan(bf16_sources=("fp8",)),
    )
    for invalid_plan in invalid:
        with pytest.raises(UnsupportedInput) as refused:
            invalid_plan()
        assert refused.value.code == "quantization_plan"


def test_native_replay_returns_prior_identity_with_zero_payload_work(tmp_path: Path) -> None:
    store, source = mint(tmp_path, sdxl_rows())
    spec = plan(("unet",), "mxfp8/1")
    first, first_artifact = run(tmp_path, store, source, spec)
    replay, replay_artifact = run(tmp_path, store, source, spec, epoch=2)
    assert first is not None and replay is not None
    assert replay_artifact == first_artifact and replay.replayed
    assert replay.weights_transaction_id == first.weights_transaction_id
    assert (replay.source_bytes_read_this_run, replay.new_bytes_written_this_run) == (0, 0)
    assert (replay.encoded_keys, replay.converted_keys, replay.inherited_keys) == (2, 1, 4)
    assert replay.normalization_worst_relative_frobenius is None


def test_fp32_safetensors_normalizes_to_exact_bf16_with_recorded_stats(tmp_path: Path) -> None:
    rng = np.random.default_rng(82)
    values = {
        "blocks.0.attn.weight": rng.standard_normal((16, 32)).astype(np.float32) * 0.02,
        "blocks.0.norm.weight": rng.standard_normal((16,)).astype(np.float32),
    }
    carrier = st.Writer()
    for key, value in values.items():
        carrier.add(key, "F32", list(value.shape), value.tobytes())
    path = tmp_path / "source.safetensors"
    carrier.write(path)
    source_header, base = st.read_header(path)
    store, source = mint(
        tmp_path,
        {
            "dit": {
                key: (
                    row["dtype"].lower(),
                    tuple(row["shape"]),
                    st.read_raw(path, source_header, base, key),
                )
                for key, row in source_header.items()
            }
        },
    )
    result, artifact = run(tmp_path, store, source, plan())
    assert result is not None and result.canonical
    assert (result.converted_keys, result.kept_keys, result.encoded_keys) == (2, 0, 0)
    assert result.normalization_worst_relative_frobenius is not None
    assert 0 < result.normalization_worst_relative_frobenius <= 2**-8
    for key, value in values.items():
        assert (
            header(store, artifact)["components"]["dit"][key]["logical"]["logical_dtype"] == "bf16"
        )
        assert read_value(tmp_path, store, artifact, "dit", key, value.size * 2) == st.from_f32(
            value, "BF16"
        )


def test_fp16_inherits_exact_original_artifact_without_source_reads(tmp_path: Path) -> None:
    rows = {"unet": {"a.weight": ("f16", (8, 32), np.ones((8, 32), dtype="<f2").tobytes())}}
    store, source = mint(tmp_path, rows)
    result, artifact = run(tmp_path, store, source, plan())
    assert result is not None and result.canonical
    assert artifact.manifest == source.manifest
    assert (result.converted_keys, result.kept_keys, result.inherited_keys) == (0, 0, 1)
    assert result.source_bytes_read_this_run == 0


def test_explicit_fp16_bf16_preparation_commits_exact_bytes_and_replays(tmp_path: Path) -> None:
    # Exactly representable FP16 inputs include BF16 ties, the FP16 maximum and
    # its smallest subnormal. Expected BF16 words are independent of the writer.
    values = np.array([1.00390625, 1.01171875, -1.00390625, 0, 65504, 2**-24], dtype="<f2")
    expected = np.array([0x3F80, 0x3F82, 0xBF80, 0, 0x4780, 0x3380], dtype="<u2").tobytes()
    rows = {
        "adapter": {
            "a.weight": ("f16", (2, 3), values.tobytes()),
            "b.weight": ("f32", (2, 3), values.astype("<f4").tobytes()),
            "ready.weight": ("bf16", (2, 3), expected),
        },
        "frozen": {"original.weight": ("f16", (2, 3), values.tobytes())},
    }
    store, source = mint(tmp_path, rows)
    spec = plan(keep=("frozen",), bf16_sources=("f16", "f32"))
    first, artifact = run(tmp_path, store, source, spec)
    replay, again = run(tmp_path, store, source, spec, epoch=2)
    assert first is not None and replay is not None
    assert (first.converted_keys, first.kept_keys, first.inherited_keys) == (2, 1, 2)
    assert (first.source_bytes_read_this_run, first.new_bytes_written_this_run) == (36, 24)
    assert first.normalization_worst_relative_frobenius is not None
    original, prepared = header(store, source), header(store, artifact)
    assert original["components"]["frozen"] == prepared["components"]["frozen"]
    assert (
        original["components"]["adapter"]["ready.weight"]
        == prepared["components"]["adapter"]["ready.weight"]
    )
    for key in rows["adapter"]:
        assert prepared["components"]["adapter"][key]["logical"]["logical_dtype"] == "bf16"
        assert read_value(tmp_path, store, artifact, "adapter", key, len(expected)) == expected
    assert again == artifact and replay.replayed
    assert (replay.source_bytes_read_this_run, replay.new_bytes_written_this_run) == (0, 0)


@pytest.mark.parametrize("encoding", ["fp8-rowwise/1", "mxfp8/1"])
def test_explicit_fp16_selection_keeps_quantized_parts_and_changes_logical_dtype(
    tmp_path: Path, encoding: str
) -> None:
    values = np.arange(256, dtype="<f2").reshape(8, 32) / 256
    store, source = mint(tmp_path, {"adapter": {"a.weight": ("f16", (8, 32), values.tobytes())}})
    _, original = run(tmp_path, store, source, plan(("adapter",), encoding), name="fp16")
    result, prepared = run(
        tmp_path,
        store,
        source,
        plan(("adapter",), encoding, bf16_sources=("f16", "f32")),
        name="bf16",
    )
    before = header(store, original)["components"]["adapter"]["a.weight"]
    after = header(store, prepared)["components"]["adapter"]["a.weight"]
    assert before["logical"]["logical_dtype"] == "f16"
    assert after["logical"]["logical_dtype"] == "bf16"
    assert before["parts"] == after["parts"]
    assert result is not None and (result.encoded_keys, result.converted_keys) == (1, 0)


def test_keep_listed_fp32_component_preserves_exact_native_objects(tmp_path: Path) -> None:
    store, source = mint(
        tmp_path,
        {
            "dit": {"block.weight": ("f32", (8, 32), np.ones((8, 32), dtype="<f4").tobytes())},
            "audio_vae": {
                "decoder.weight": ("f32", (4, 4), np.ones((4, 4), dtype="<f4").tobytes())
            },
        },
    )
    result, artifact = run(tmp_path, store, source, plan(keep=("audio_vae",)))
    assert result is not None and (
        result.converted_keys,
        result.kept_keys,
        result.inherited_keys,
    ) == (1, 1, 1)
    assert (
        header(store, artifact)["components"]["audio_vae"]
        == header(store, source)["components"]["audio_vae"]
    )
    with pytest.raises(UnsupportedInput):
        run(tmp_path, store, source, plan(keep=("ghost",)), name="missing-keep")
