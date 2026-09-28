"""Checkpoint construction config has one authority: inline CozyTensors JCS bytes."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import tensorfs
from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Source, Target

from cozy_runtime.author import Config, ConformanceError
from cozy_runtime.internal import canonical, model_config
from cozy_runtime.internal.worker.package_prepare import (
    PreparationRefusal,
    _selected_models,
    selections,
)
from test_derived_config_runtime import _released_source
from test_model_runtime_closure import _CONFIG


def test_one_inline_config_is_the_constructor_document() -> None:
    raw = b'{"hidden_size":2048,"layers":32}'

    assert model_config.construction_config({"transformer": raw}) == raw


def test_a_noncanonical_writer_spelling_is_normalized_not_refused() -> None:
    spelled = b'{ "layers": 32,\n  "hidden_size": 2048 }'

    assert model_config.construction_config({"transformer": spelled}) == (
        b'{"hidden_size":2048,"layers":32}'
    )


def test_named_inline_configs_form_one_canonical_constructor_document() -> None:
    document = model_config.construction_config(
        {
            "unet": b'{"channels":[320,640]}',
            "text_encoder": b'{"hidden_size":2048}',
        }
    )

    assert document == (b'{"text_encoder":{"hidden_size":2048},"unet":{"channels":[320,640]}}')


def test_external_config_reference_is_not_a_config() -> None:
    with pytest.raises(model_config.ModelConfigRefusal, match="malformed inline config"):
        model_config.construction_config(
            {
                "unet": {
                    "sha256": "1" * 64,
                    "length": 42,
                    "media_type": "application/json",
                }
            }
        )


# ------------------------------------------- a reserved name is not a construction config
#
# cr-121, and it OUTLIVES the contract it was written for (cr-124). `execution` was a lane's
# execution contract; the runtime no longer reads it, but two checkpoints minted while it did
# are published or retained and still carry it. Merged into the constructor document its
# `weights` key reads as a source carrier and the §1.1 ban refuses the whole lane at prepare —
# measured on an H100: `config.execution.weights: 'weights' names a source carrier … (§1.1)`.
# So the name stays SKIPPED. Inert, not fatal, and these are the real bytes it must be inert
# against: the two `execution` configs of `fp8-fa3-adaln-pruned` and `fp8-attn8-adaln-pruned`,
# by the raw sha256 the producer, the header and the hub's projection all named (se-037).

FA3_CONFIG: dict[str, Any] = {
    "activations": "bf16",
    "attention": {
        "distribution": "flash-attn3",
        "entry": "flash_attn_func",
        "kwargs": {
            "causal": False,
            "deterministic": False,
            "k_descale": None,
            "num_splits": 1,
            "q_descale": None,
            "v_descale": None,
        },
        "revision": "7cb368cf8278b583132eb72cbf312d54586df2e2",
        "scheme": (
            "bf16 q/k/v with no descale (the bf16 path, not fp8); fp32 softmax and PV "
            "accumulation; full non-causal attention, no window, no softcap"
        ),
        "variant": "torch-stable-abi29-cu130-x86_64-linux",
        "version": "1",
    },
    "class": "same",
    "device": "sm90",
    "weights": {"route": "encoded_gemm"},
}

ATTN8_CONFIG: dict[str, Any] = {
    "activations": "bf16",
    "attention": {
        "distribution": "sageattention",
        "entry": "sageattn_qk_int8_pv_fp8_cuda_sm90",
        "kwargs": {
            "is_causal": False,
            "pv_accum_dtype": "fp32+fp32",
            "qk_quant_gran": "per_thread",
            "smooth_k": True,
            "tensor_layout": "NHD",
        },
        "scheme": (
            "qk int8 per-thread (Q64/16, K128/128) K-mean-smoothed; v fp8-e4m3 per-channel; "
            "pv fp32+fp32"
        ),
        "version": "2.2.0",
    },
    "class": "quantized",
    "device": "sm90",
    "weights": {"route": "encoded_gemm"},
}

RETIRED = {
    "sha256:e655674ee1080fdb5f8493e972aac29a296928d1c2d7a67b2a2c2264c40d28a4": FA3_CONFIG,
    "sha256:4dfd9f135c2d0a2d4869a13b515c09edfac6043ead1ef87f3f4c547010256bec": ATTN8_CONFIG,
}


def _lane_checkpoint(
    tmp_path: Path, construction: bytes | None, contract: bytes
) -> tuple[Any, str]:
    """One real lane checkpoint, produced the way se-037's `attention-lane` job produces one:
    the source checkpoint's components and configs by reference, plus the inline `execution`
    config, over the real weights sink and a real TensorFS store."""

    store, source, source_length = _released_source(tmp_path)
    definition = Derivation(
        {"source": Source(source, source_length)},
        {"unet": Target(source="source", source_component="unet")},
        {
            "unet": NativeConfig("copy", "source", "unet")
            if construction is None
            else NativeConfig("add"),
            "execution": NativeConfig("add"),
        },
        (("unet", "weight"),),
    )
    transaction = store.begin_derived(
        tensorfs.object_id(b"inline-config-proof"),
        1,
        *definition.native_arguments(1 << 20),
        work_fingerprint=tensorfs.object_id(b"inline-config-proof"),
    )
    if construction is not None:
        transaction.add_config("unet", BytesIO(construction))
    transaction.add_config("execution", BytesIO(contract))
    receipt = transaction.commit()
    return store, "sha256:" + receipt["manifest"]["sha256"]


def _prepared_config(store: Any, manifest: str) -> bytes:
    """The construction document a worker's prepare reads for that checkpoint."""

    selection = {
        "package": "example/package",
        "slot": "generate.models.model",
        "model": "example/model",
        "manifest": manifest,
    }
    selected = _selected_models(
        "example/package",
        selections([selection]),
        Path(store.root),
        lambda *_args: None,
        construction_slots={"generate.models.model"},
    )
    built = selected["generate.models.model"].construction
    assert built is not None
    return built.config_bytes


def test_the_retired_contracts_leave_the_constructor_document_untouched() -> None:
    for digest, document in RETIRED.items():
        configs = {"transformer": b'{"num_layers":50}', "execution": canonical.write(document)}
        assert "sha256:" + hashlib.sha256(configs["execution"]).hexdigest() == digest

        assert model_config.construction_config(configs) == configs["transformer"]

    with pytest.raises(model_config.ModelConfigRefusal, match="no inline construction configs"):
        model_config.construction_config({"execution": canonical.write(FA3_CONFIG)})


def test_a_lane_checkpoint_prepares_and_constructs(tmp_path: Path) -> None:
    contract = canonical.write(FA3_CONFIG)
    store, lane = _lane_checkpoint(tmp_path, None, contract)
    header = tensorfs.parse_header(bytes(store.manifest(lane)["header"]))
    assert {name: bytes(raw) for name, raw in header["configs"].items()} == {
        "unet": _CONFIG,
        "execution": contract,
    }

    config = _prepared_config(store, lane)

    # The config adds no tensor and no construction fact: the document is the one the
    # config-less checkpoint the lane shares its blobs with carries, byte for byte.
    assert config == _CONFIG
    # The line the derive child runs on it (`internal/derive_child.py`), which refused here.
    assert Config(canonical.parse_canonical(config), "config").keys() == ("hidden_size", "layers")


def test_a_construction_config_naming_a_source_carrier_still_refuses(tmp_path: Path) -> None:
    """§1.1 survives cr-121: only the reserved name is exempt, and only because it is never a
    construction config. A genuine one carrying a path refuses on the same prepare path."""

    carrier = canonical.write({"weights": "/mnt/h3/model.safetensors"})
    store, lane = _lane_checkpoint(tmp_path, carrier, canonical.write(FA3_CONFIG))

    config = _prepared_config(store, lane)

    assert config == carrier
    with pytest.raises(ConformanceError) as refused:
        Config(canonical.parse_canonical(config), "config")
    assert refused.value.code == "config_path"
    assert refused.value.fields == ("config.weights",)


def test_a_selection_serves_its_manifest_after_its_lane_moves(tmp_path: Path) -> None:
    """A release/lane is a label: the selected digest is served even when the lane now
    names another manifest, and an identical repeated selection is one selection."""

    store, checkpoint = _lane_checkpoint(tmp_path, None, canonical.write(FA3_CONFIG))
    current = store.resolve_release("test", "model", "1", "main")["manifest_digest"]
    assert current != checkpoint
    selection = {
        "package": "example/package",
        "slot": "generate.models.model",
        "model": "test/model",
        "release": "1",
        "lane": "main",
        "manifest": checkpoint,
    }
    selected = _selected_models(
        "example/package",
        selections([selection, dict(selection)]),
        Path(store.root),
        lambda *_args: None,
        construction_slots={"generate.models.model"},
    )
    assert selected["generate.models.model"].manifest_length == len(
        bytes(store.manifest(checkpoint)["manifest"])
    )
    with pytest.raises(PreparationRefusal, match="model_selection_mismatch"):
        _selected_models(
            "example/package",
            selections([selection, {**selection, "manifest": current}]),
            Path(store.root),
            lambda *_args: None,
            construction_slots={"generate.models.model"},
        )
