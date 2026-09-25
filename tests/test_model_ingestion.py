"""Complete-model ingestion refuses unknown pins and incompatible native sources."""

from __future__ import annotations

import asyncio
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
from cozy_runtime.author.sources import ingest_huggingface
from cozy_runtime.derive.operations import QuantizationSource, app
from cozy_runtime.models.qwen_image21.ingestion import REPOSITORY, REVISION, configuration
from native_weights import NativeExecution
from test_quantize_artifact_native import _source


def test_ingestion_cli_reports_reviewed_full_metadata_without_inference() -> None:
    command = subprocess.run(
        [
            sys.executable,
            "-m",
            "cozy_runtime.cli.main",
            "--json",
            "model-ingestion-plan",
            REPOSITORY,
            REVISION,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    recipe = json.loads(command.stdout)["recipe"]
    assert recipe["profile"] == "hf/qwen/qwen-image-2.1/original/1"
    assert set(recipe["metadata"]) >= {
        "processor/tokenizer.json",
        "transformer/config.json",
        "LICENSE",
    }
    assert all(
        row["length"] > 0 and len(row["sha256"]) == 64 for row in recipe["metadata"].values()
    )
    for repository, revision, code in (
        (REPOSITORY, "0" * 40, "model_ingestion_revision"),
        ("unreviewed/model", REVISION, "model_ingestion_recipe_unavailable"),
    ):
        with pytest.raises(UnsupportedInput) as error:
            asyncio.run(ingest_huggingface(repository, revision=revision))
        assert error.value.code == code


def test_qwen_configuration_rejects_changed_metadata_before_use(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer/config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Unreviewed Qwen metadata"):
        configuration(tmp_path)


def test_preparation_binds_real_native_source_and_refuses_wrong_census(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    source = _source(store, "control")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
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
                "recipe": "qwen-image-2.1/original/1",
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
    assert "exact original source tensor census" in str(outcome)
