"""Checkpoint construction config has one authority: inline CozyTensors JCS bytes."""

from __future__ import annotations

import hashlib

import pytest

from cozy_runtime.internal import model_config
from cozy_runtime.internal.worker.session import _placement_profile
from cozy_runtime.protocol import documents


def test_one_inline_config_is_the_constructor_document() -> None:
    raw = b'{"hidden_size":2048,"layers":32}'

    digest, document = model_config.construction_document({"transformer": raw})

    assert document == raw
    assert digest == "sha256:" + hashlib.sha256(raw).hexdigest()


def test_named_inline_configs_form_one_canonical_constructor_document() -> None:
    digest, document = model_config.construction_document(
        {
            "unet": b'{"channels":[320,640]}',
            "text_encoder": b'{"hidden_size":2048}',
        }
    )

    assert document == (b'{"text_encoder":{"hidden_size":2048},"unet":{"channels":[320,640]}}')
    assert digest == "sha256:" + hashlib.sha256(document).hexdigest()


def test_external_config_reference_is_not_a_config() -> None:
    with pytest.raises(model_config.ModelConfigRefusal, match="malformed inline config"):
        model_config.construction_document(
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
