"""Complete-model ingestion refuses unknown pins and incompatible native sources."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    App,
    Context,
    Invocation,
    ObjectRef,
    SourceArtifact,
    UnsupportedInput,
    attempt,
    describe,
)
from cozy_runtime.author._calls import _Broker
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.sources import ingest_huggingface
from cozy_runtime.derive.operations import QuantizationSource, app
from cozy_runtime.internal import source_interfaces
from cozy_runtime.models.birefnet.ingestion import REPOSITORY, REVISION
from durable_seam import wire
from native_weights import NativeExecution
from test_quantize_artifact_native import _source


def test_ingestion_refuses_unreviewed_pins() -> None:
    for repository, revision, code in (
        (REPOSITORY, "0" * 40, "model_ingestion_revision"),
        ("unreviewed/model", REVISION, "model_ingestion_recipe_unavailable"),
    ):
        with pytest.raises(UnsupportedInput) as error:
            asyncio.run(ingest_huggingface(repository, revision=revision))
        assert error.value.code == code


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


class _Empty(msgspec.Struct):
    pass


def test_metadata_download_overlaps_the_conversion(tmp_path: Path) -> None:
    """The Host sees the metadata download before the conversion settles."""
    script = App()

    @script.job
    async def ingest(ctx: Context, payload: _Empty) -> _Empty:
        await ingest_huggingface(REPOSITORY, revision=REVISION)
        return _Empty()

    source = SourceArtifact(
        "download", "source", ObjectRef("sha256:" + "1" * 64, 100), "sha256:" + "2" * 64
    )
    activated: list[tuple[int, str]] = []
    polls: Counter[int] = Counter()

    def host(kind: str, body: dict[str, Any]) -> dict[str, Any]:
        if kind == "child_call":
            activated.append((body["call_index"], body["export"]))
            return {"ok": True}
        if kind != "child_poll":
            return {"ok": True}
        index = body["call_index"]
        polls[index] += 1
        if index == 0:
            result = canonical_json.encode(msgspec.to_builtins(source)).decode()
            return {"ok": True, "state": "succeeded", "result": result}
        if index == 1 and (2, "download_huggingface") in activated:
            return {"ok": False, "code": "conversion_stopped"}
        if index == 1 and polls[index] > 3:
            return {"ok": False, "code": "metadata_waited_for_conversion"}
        return {"ok": True, "state": "pending"}

    bindings = {(b.module, b.export): b for b in source_interfaces.bindings().values()}
    _, outcome, _ = attempt(
        script.get("ingest"),
        {},
        Invocation(
            "parent", tmp_path, time.monotonic() + 60, calls=_Broker("parent", bindings, wire(host))
        ),
    )
    assert outcome.code == "conversion_stopped"
    assert activated == [
        (0, "download_huggingface"),
        (1, "convert_cozytensors"),
        (2, "download_huggingface"),
    ]
