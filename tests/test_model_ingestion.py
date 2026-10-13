"""Complete-model ingestion refuses unknown pins and incompatible native sources."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime.author import Invocation, UnsupportedInput, attempt, describe
from cozy_runtime.author._model import _derive_model
from cozy_runtime.derive.operations import QuantizationSource, app
from cozy_runtime.models.birefnet.ingestion import REPOSITORY, REVISION
from cozy_runtime.models.ingestion import huggingface_recipe
from native_weights import NativeExecution
from test_quantize_artifact_native import _source


def test_ingestion_refuses_unreviewed_pins() -> None:
    with pytest.raises(UnsupportedInput) as error:
        huggingface_recipe(REPOSITORY, "0" * 40)
    assert error.value.code == "model_ingestion_revision"
    assert huggingface_recipe("unreviewed/model", REVISION) is None


def test_birefnet_cli_uses_native_identity_ingest_and_its_complete_model_config() -> None:
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "cozy_runtime.cli.main",
            "--json",
            "model-ingestion-plan",
            "ZhengPeng7/BiRefNet",
            "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    recipe = json.loads(command.stdout)["recipe"]
    assert recipe["carriers"] == ["model.safetensors"]
    assert recipe["profile"] == "as-is/1"
    assert recipe["name"] == "birefnet/original/1"
    assert set(recipe["metadata"]) == {"config.json"}


def test_preparation_binds_real_native_source_and_refuses_wrong_census(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "control")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    config = {
        "_name_or_path": REPOSITORY,
        "architectures": ["BiRefNet"],
        "auto_map": {
            "AutoConfig": "BiRefNet_config.BiRefNetConfig",
            "AutoModelForImageSegmentation": "birefnet.BiRefNet",
        },
        "custom_pipelines": {
            "image-segmentation": {
                "pt": ["AutoModelForImageSegmentation"],
                "tf": [],
                "type": "image",
            }
        },
        "bb_pretrained": False,
    }
    (metadata / "config.json").write_text(json.dumps(config, indent=2))
    digest = "sha256:" + "a1" * 32
    with NativeExecution(
        store, tmp_path / "execution", "prepare", {"source": source}, {"model": 8 << 20}
    ) as host:
        describe(app)
        result, outcome, _ = attempt(
            app.get("prepare_model"),
            {
                "source": msgspec.to_builtins(source),
                "metadata": digest,
                "recipe": "birefnet/original/1",
            },
            Invocation(
                "prepare",
                tmp_path / "output",
                time.monotonic() + 30,
                trees={digest: (metadata, digest)},
                models={"source": _derive_model(QuantizationSource, source.manifest.digest)},
                tensorfs_source=host.client.source,
                tensorfs_output=host.client.open_output,
                tensorfs_adopt=host.client.adopt_model,
            ),
        )
    assert result is None and outcome.terminal == "failed"
    assert "BiRefNet source tensor keys do not match" in str(outcome)
