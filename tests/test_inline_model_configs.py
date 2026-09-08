"""Checkpoint construction config has one authority: inline CozyTensors JCS bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.author import Config, ConformanceError
from cozy_runtime.author._services import Attempt
from cozy_runtime.author._weights import (
    WeightsCommit,
    WeightsConfig,
    WeightsTarget,
    WeightsTransaction,
)
from cozy_runtime.internal import canonical, model_config
from cozy_runtime.internal.weights_sink import WeightsTransactionHost
from cozy_runtime.internal.worker.package_prepare import _compatibility_models
from cozy_runtime.internal.worker.session import _placement_profile
from cozy_runtime.protocol import documents
from test_derived_config_runtime import _released_source
from test_model_runtime_closure import _CONFIG


def test_one_inline_config_is_the_constructor_document() -> None:
    raw = b'{"hidden_size":2048,"layers":32}'

    assert model_config.construction_config({"transformer": raw}) == raw


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


def test_placement_slot_refuses_external_config_authority() -> None:
    digest = "sha256:" + "1" * 64
    placement = {
        "placement_id": "placement",
        "development": {
            "package": "example/package",
            "release": "0.2.5",
            "source_digest": digest,
        },
        "package_interface": {"digest": digest, "length": 1},
        "bindings_digest": digest,
        "models": [
            {
                "id": "model-0000",
                "repo": "example-model",
                "version": "1",
                "manifest": {"digest": digest, "length": 1},
            }
        ],
        "entrypoints": [
            {
                "name": "generate",
                "entrypoint_binding_digest": digest,
                "slots": [
                    {
                        "slot": "model",
                        "reference_model_id": "model-0000",
                        "config": {
                            "document": {"digest": digest, "length": 42},
                            "assets": [],
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(documents.DocumentError) as refused:
        _placement_profile(placement)

    assert refused.value.code == "unknown_field"


# ------------------------------------------- a reserved name is not a construction config
#
# cr-121, and it OUTLIVES the contract it was written for (cr-123). `execution` was a lane's
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
    tmp_path: Path, construction: WeightsConfig, contract: bytes
) -> tuple[Any, str]:
    """One real lane checkpoint, produced the way se-037's `attention-lane` job produces one:
    the source checkpoint's components and configs by reference, plus the inline `execution`
    config, over the real weights sink and a real TensorFS store."""

    store, source, source_length = _released_source(tmp_path)
    configs = {
        "unet": construction,
        "execution": WeightsConfig(data=contract, length=len(contract)),
    }
    targets = {"unet": WeightsTarget(source="source", source_component="unet")}
    request = WeightsCommit(
        output_slot="lane",
        sources={"source": source},
        targets=targets,
        configs=configs,
        order=(("unet", "weight"),),
        max_new_bytes=1 << 20,
    )
    host = WeightsTransactionHost(
        store=store,
        owner_scope="retired-contract-lane",
        request_id="request-1",
        invocation_spec_digest="sha256:" + "11" * 32,
        work_fingerprint="sha256:" + "22" * 32,
        writer_session_id=1,
        allowed_sources={source: source_length},
        output_bounds={"lane": 1 << 20},
    )
    transaction = WeightsTransaction(
        attempt=Attempt("request-1", tmp_path / "spool"),
        cancel=lambda: False,
        output="lane",
        sources=request.sources,
        targets=targets,
        configs=configs,
        host=host.open(request),
    )
    if isinstance(construction.data, bytes):
        transaction.add_config("unet", construction.data)
    transaction.add_config("execution", contract)
    receipt = transaction.commit()
    return store, "sha256:" + json.loads(receipt.tensorfs_receipt)["manifest"]["sha256"]


def _prepared_config(store: Any, manifest: str) -> bytes:
    """The construction document a worker's prepare reads for that checkpoint."""

    selection = {
        "package": "example/package",
        "slot": "generate.models.model",
        "model": "example/model",
        "manifest": manifest,
    }
    selected = _compatibility_models([selection], tensorfs_root=Path(store.root))
    return bytes(selected["generate.models.model"]["config"])


def test_the_retired_contracts_leave_the_constructor_document_untouched() -> None:
    for digest, document in RETIRED.items():
        configs = {"transformer": b'{"num_layers":50}', "execution": canonical.write(document)}
        assert "sha256:" + hashlib.sha256(configs["execution"]).hexdigest() == digest

        assert model_config.construction_config(configs) == configs["transformer"]

    with pytest.raises(model_config.ModelConfigRefusal, match="no inline construction configs"):
        model_config.construction_config({"execution": canonical.write(FA3_CONFIG)})


def test_a_lane_checkpoint_prepares_and_constructs(tmp_path: Path) -> None:
    contract = canonical.write(FA3_CONFIG)
    store, lane = _lane_checkpoint(
        tmp_path, WeightsConfig(source="source", source_config="unet"), contract
    )
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
    store, lane = _lane_checkpoint(
        tmp_path, WeightsConfig(data=carrier, length=len(carrier)), canonical.write(FA3_CONFIG)
    )

    config = _prepared_config(store, lane)

    assert config == carrier
    with pytest.raises(ConformanceError) as refused:
        Config(canonical.parse_canonical(config), "config")
    assert refused.value.code == "config_path"
    assert refused.value.fields == ("config.weights",)
