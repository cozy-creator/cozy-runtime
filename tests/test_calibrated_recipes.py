"""A package with a calibrated recipe (xs-066): its `quantize` is `quantize_calibrated`, on a GPU,
with the recipe's model built on demand; the static describe and the imported app agree; its
plain recipes still run with that slot unbound."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import Discovered, discover
from store_fixtures import home
from test_declared_recipes import _run, _source

PROJECT = Path(__file__).resolve().parent / "testdata" / "calibrated"


@pytest.fixture
def described() -> Iterator[tuple[dict[str, Any], Discovered]]:
    document = static_interface.build(PROJECT)
    recipes = package_interface.parse(package_interface.canonical_bytes(document)).recipes
    yield document, discover(PROJECT, recipes=recipes)
    for name in [m for m in sys.modules if m.split(".")[0] == "tiny_calibrated"]:
        sys.modules.pop(name)


def test_a_calibrated_recipe_serves_quantize_with_its_model_on_demand(
    described: tuple[dict[str, Any], Discovered],
) -> None:
    document, found = described
    rows = {row["name"]: row for row in document["recipes"]}
    assert rows["svdq_nvfp4"]["calibrate"] == "tiny_calibrated:run_inputs"
    assert "fingerprint" not in rows["svdq_nvfp4"] and "calibrate" not in rows["int8"]
    entry = {e["name"]: e for e in document["entrypoints"]}["quantize"]
    assert entry.get("accelerator", True) is True  # a GPU: the default, not written
    assert entry["invocable"]["export"] == "quantize_calibrated"
    slots = {slot["path"]: slot for slot in entry["models"]}
    assert sorted(slots) == ["quantize.models.model", "quantize.models.source"]
    model = slots["quantize.models.model"]
    assert model["class"] == "Tiny" and model.get("on_demand") is True and not model.get("weights")
    imported = package_interface.source_facts(package_interface.build(found))
    package_interface.compare(document, imported, ("the source reading", "the imported app"))


@pytest.mark.parametrize("models", [(), ("quantize.models.model",)])
def test_a_plain_recipe_beside_it_runs_without_its_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    described: tuple[dict[str, Any], Discovered],
    models: tuple[str, ...],
) -> None:
    """The int8 lane of the same package: its on-demand model, bound or not, is never built
    (`Tiny.load` would fail the run)."""
    _, found = described
    store = home(tmp_path / "tensorfs", monkeypatch)
    source = _source(store, {"unet": {"a.weight": (8, 256)}})
    reply, shown = _run(
        tmp_path, found, source, {"lane": "int8"}, ["int8", "svdq-nvfp4"], models=models
    )
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    (item,) = shown
    written = tensorfs.open(item.revision, cache_dir=store, local_files_only=True)
    encoding = written.components["unet"].tensors["a.weight"].encoding
    assert encoding is not None and encoding.scheme == "int8.convrot"


def test_the_calibrated_lane_without_its_model_refuses_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, described: tuple[dict[str, Any], Discovered]
) -> None:
    _, found = described
    source = _source(home(tmp_path / "tensorfs", monkeypatch), {"unet": {"a.weight": (8, 256)}})
    reply, shown = _run(tmp_path, found, source, {"lane": "svdq-nvfp4"}, ["svdq-nvfp4"])
    outcome = reply["outcome"]
    assert (outcome["code"], shown) == ("unbound_model", []), outcome
