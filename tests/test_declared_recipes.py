"""Declared recipes, end to end on CPU: the static describe records them and the `quantize`
entrypoint that runs them; a real executor over a real store picks the recipe for the lane and
the source and writes the lane with its provenance; a lane, model or plan that does not fit
refuses by name."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import msgspec
import numpy as np
import pytest
import tensorfs

from cozy_runtime.author import ModelArtifact, describe
from cozy_runtime.author._executor_requests import Answer, CheckpointItem, Reply, Request, refuse
from cozy_runtime.author._recipes import fingerprint
from cozy_runtime.derive import operations
from cozy_runtime.derive import safetensors_io as st
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import Discovered, discover
from cozy_runtime.internal.seam import RESULT_DOCUMENT
from session import session
from store_fixtures import home

PROJECT = Path(__file__).resolve().parent / "testdata" / "recipes"
ASKED = {"package": "acme/tiny-recipes", "release": "0.1.0"}
MAX = (1 << 53) - 1


@pytest.fixture
def described() -> Iterator[tuple[dict[str, Any], Discovered]]:
    document = static_interface.build(PROJECT)
    recipes = package_interface.parse(package_interface.canonical_bytes(document)).recipes
    yield document, discover(PROJECT, recipes=recipes)
    for name in [m for m in sys.modules if m.split(".")[0] == "tiny_recipes"]:
        sys.modules.pop(name)


def _source(store: Path, components: Mapping[str, Mapping[str, tuple[int, ...]]]) -> str:
    writer = tensorfs.Writer(cache_dir=store)
    rng = np.random.default_rng(69)
    for component, tensors in components.items():
        out = writer.component(component, cls=("diffusers", "", "Tiny"), config={})
        for key, shape in tensors.items():
            values = rng.normal(0, 0.05, shape).astype(np.float32)
            bf16 = np.frombuffer(st.from_f32(values, "BF16"), np.uint16).reshape(shape)
            out.put(key, memoryview(bf16), dtype="BF16", shape=list(shape))
    ref = writer.commit().id
    assert ref is not None
    return ref


UNET: Mapping[str, Mapping[str, tuple[int, ...]]] = {
    "unet": {"a.weight": (8, 256), "z.weight": (8, 256), "bias": (8,)}
}


def test_the_static_describe_records_recipes_and_the_quantize_that_runs_them(
    described: tuple[dict[str, Any], Discovered],
) -> None:
    document, found = described
    rows = {row["name"]: row for row in document["recipes"]}
    assert sorted(rows) == ["bf16", "head", "int8", "int8_q", "speaker_int8", "w4a8", "z"]
    int8 = rows["int8"]
    assert (int8["lane"], int8["module"], int8["export"]) == (
        "int8",
        "tiny_recipes.recipes",
        "INT8",
    )
    assert int8["model"] == {"class": "Tiny", "component_use": {"denoise": ["unet"]}}
    assert int8["plan"] == {
        "kind": "quantization",
        "encoding": "int8.convrot",
        "components": ["unet"],
        "include": ["unet.a.weight", "unet.z.weight"],
        "output_precision": "preserve",
    }
    assert int8["fingerprint"] == fingerprint(int8["plan"])
    # `INT8.plan.include` folded into W4A8, its rule after the lane's encoding.
    assert rows["w4a8"]["plan"]["include"] == int8["plan"]["include"]
    assert rows["w4a8"]["plan"]["encodings"] == [
        {"pattern": "unet.a.*", "encoding": "w4a8.convrot"}
    ]
    assert rows["bf16"]["model"] == {"class": "Model", "component_use": {}}
    # A hook is code: named by its function, fingerprinted only with the release that ships it.
    assert rows["z"]["plan"] == {"kind": "derivation", "hook": "tiny_recipes.recipes:z_only"}
    assert "fingerprint" not in rows["z"]
    entry = {e["name"]: e for e in document["entrypoints"]}["quantize"]
    assert entry["accelerator"] is False and entry["result"] == {"input": "model"}
    assert [slot["path"] for slot in entry["models"]] == ["quantize.models.source"]
    assert [field["name"] for field in entry["request"]["fields"]] == [
        "lane",
        "package",
        "release",
        "recipe",
    ]
    lanes = [output["output_id"] for output in entry["weights_outputs"]]
    assert lanes == ["bf16", "head", "int8", "w4a8", "z"]
    assert entry["invocable"]["module"] == "cozy_runtime.derive.recipes"
    # The imported app serves the same entrypoint, from the same declarations.
    imported = package_interface.source_facts(package_interface.build(found))
    package_interface.compare(document, imported, ("the source reading", "the imported app"))


def test_the_runtime_operations_still_take_a_callers_plan() -> None:
    """The builtin `quantize` a job calls with its own plan reads the plan type from the author
    surface, statically as on import (every environment describes it)."""
    document = static_interface.build_builtin("operations")
    module = sys.modules[operations.app.get("quantize").fn.__module__]
    found = Discovered(
        operations.app, document["application"], PROJECT, module, describe(operations.app)
    )
    imported = package_interface.source_facts(package_interface.build(found))
    package_interface.compare(document, imported, ("the source reading", "the imported app"))


def _run(
    root: Path,
    found: Discovered,
    source: str,
    request: dict[str, str],
    lanes: list[str],
    models: tuple[str, ...] = (),
) -> tuple[dict[str, Any], list[CheckpointItem]]:
    shown: list[CheckpointItem] = []

    def machine(asked: Request) -> Reply:
        if not isinstance(asked, CheckpointItem):
            return refuse("no_durable_exchange", type(asked).__name__)
        shown.append(asked)
        return Answer(ok=True)

    spool = root / f"quantize-{request['lane']}"
    with session(root, found, models=models) as host:
        host.handler = machine
        reply = host.run(
            "quantize",
            {**ASKED, **request},
            spool,
            sources={"source": source},
            weights=dict.fromkeys(lanes, MAX),
        )
    if reply["outcome"]["terminal"] == "succeeded":
        reply["result"] = json.loads((spool / RESULT_DOCUMENT).read_bytes())
    return reply, shown


@pytest.mark.parametrize("lane", ["int8", "z"])
def test_quantize_writes_the_recipes_lane_with_its_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    described: tuple[dict[str, Any], Discovered],
    lane: str,
) -> None:
    document, found = described
    store = home(tmp_path / "tensorfs", monkeypatch)
    source = _source(store, UNET)
    reply, shown = _run(tmp_path, found, source, {"lane": lane}, ["int8", "z"])
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    artifact = msgspec.convert(reply["result"], ModelArtifact)
    (item,) = shown
    assert item.output == lane
    written = tensorfs.open(item.revision, cache_dir=store, local_files_only=True)
    assert written.id == artifact.checkpoint
    row = {r["name"]: r for r in document["recipes"]}[lane]
    provenance = written.metadata["provenance"]
    assert provenance == {
        **ASKED,
        "recipe": lane,
        "fingerprint": row.get("fingerprint") or fingerprint(row["plan"], **ASKED),
        "source": source,
        "plan": row["plan"],
    }
    tensors = written.components["unet"].tensors
    if lane == "int8":
        # Of the two int8 recipes for Tiny, INT8_Q names `unet.q.weight`, which this source lacks.
        for key in ("a.weight", "z.weight"):
            encoding = tensors[key].encoding
            assert encoding is not None and encoding.scheme == "int8.convrot", key
        assert tensors["bias"].encoding is None
    else:
        assert sorted(tensors) == ["z.weight"]
        # The hook's fingerprint carries the release: another release is another recipe.
        assert provenance["fingerprint"] != fingerprint(row["plan"], ASKED["package"], "0.2.0")


@pytest.mark.parametrize(
    ("components", "lane", "code", "says"),
    [
        (UNET, "nvfp4", "recipe_not_found", "declares no nvfp4 recipe; its lanes: bf16, head"),
        ({"text": {"a.weight": (8, 256)}}, "head", "recipe_source", "head reads Tiny (unet)"),
        (
            {"unet": {"a.weight": (8, 256), "q.weight": (8, 256), "z.weight": (8, 256)}},
            "int8",
            "recipe_ambiguous",
            "2 of acme/tiny-recipes's int8 recipes (int8, int8_q) select from this checkpoint",
        ),
    ],
)
def test_a_lane_or_model_without_a_recipe_refuses_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    described: tuple[dict[str, Any], Discovered],
    components: Mapping[str, Mapping[str, tuple[int, ...]]],
    lane: str,
    code: str,
    says: str,
) -> None:
    _, found = described
    source = _source(home(tmp_path / "tensorfs", monkeypatch), components)
    reply, shown = _run(tmp_path, found, source, {"lane": lane}, ["int8", "head"])
    outcome = reply["outcome"]
    assert (outcome["code"], shown) == (code, []), outcome
    assert says in outcome["message"], outcome


def test_a_named_recipe_settles_an_ambiguous_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, described: tuple[dict[str, Any], Discovered]
) -> None:
    _, found = described
    both: Mapping[str, Mapping[str, tuple[int, ...]]] = {
        "unet": {"a.weight": (8, 256), "q.weight": (8, 256), "z.weight": (8, 256)}
    }
    source = _source(home(tmp_path / "tensorfs", monkeypatch), both)
    reply, (item,) = _run(tmp_path, found, source, {"lane": "int8", "recipe": "int8-q"}, ["int8"])
    assert reply["outcome"]["terminal"] == "succeeded", reply["outcome"]
    store = tmp_path / "tensorfs"
    written = tensorfs.open(item.revision, cache_dir=store, local_files_only=True)
    assert written.metadata["provenance"]["recipe"] == "int8_q"
    tensors = written.components["unet"].tensors
    assert tensors["q.weight"].encoding is not None and tensors["a.weight"].encoding is None
