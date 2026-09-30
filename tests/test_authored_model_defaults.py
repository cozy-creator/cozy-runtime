"""Authored model ladders have one meaning through import, static read and publication."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.author._model_defaults import default_ladder
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover

VECTORS = json.loads((Path(__file__).parent / "testdata/model-default-ladders.json").read_text())
LADDER = VECTORS["valid"][1]["ladder"]


@pytest.mark.parametrize("vector", VECTORS["valid"], ids=lambda row: row["name"])
def test_ladder_valid_vectors(vector: dict[str, Any]) -> None:
    assert default_ladder(vector["ladder"]) == tuple(
        (row["gpu"], row.get("gpus", 0), row["lane"]) for row in vector["ladder"]
    )


@pytest.mark.parametrize("vector", VECTORS["invalid"], ids=lambda row: row["name"])
def test_ladder_invalid_vectors(vector: dict[str, Any]) -> None:
    with pytest.raises(ConformanceError, match="model_defaults"):
        default_ladder(vector["ladder"])


def project(
    tmp_path: Path,
    *,
    invocable: bool,
    defaults: str = "{'model': H3}",
    ladder: list[dict[str, Any]] = LADDER,
) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.toml").write_text('[application]\nobject = "authored_defaults_fixture:app"\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "authored-defaults"\nversion = "1.0.0"\n'
    )
    declarations = (
        f"@app.entrypoint\n@invocable(defaults={defaults})\n"
        "async def {name}(ctx: Context, *, model: Demo, prompt: str = 'hello') -> Result:\n"
        "    return Result(1)\n"
        if invocable
        else f"@app.entrypoint(defaults={defaults})\n"
        "def {name}(payload: Request, model: Demo) -> Result:\n"
        "    return Result(1)\n"
    )
    body = (
        "import msgspec\n"
        "from cozy_runtime.author import App, Context, Loader, Model, invocable, uses_components\n"
        f"H3 = {ladder!r}\n"
        "app = App()\n"
        "class Request(msgspec.Struct):\n    prompt: str = 'hello'\n"
        "class Result(msgspec.Struct):\n    count: int\n"
        "class Demo(Model):\n"
        "    def load(self, loader: Loader) -> None:\n        pass\n"
        "    @uses_components('dit')\n    def generate(self) -> object:\n        return None\n"
        + declarations.replace("{name}", "fl2va")
        + declarations.replace("{name}", "ref2va")
    )
    (root / "authored_defaults_fixture.py").write_text(body)
    return root


def imported(root: Path) -> dict[str, Any]:
    # Each fixture is an independent package import, as in a fresh publication process.
    previous = sys.modules.pop("authored_defaults_fixture", None)
    try:
        return package_interface.build(discover(root))
    finally:
        sys.modules.pop("authored_defaults_fixture", None)
        if previous is not None:
            sys.modules["authored_defaults_fixture"] = previous


@pytest.mark.parametrize("managed", [False, True], ids=["app_sync", "invocable_async"])
def test_shared_constant_has_identical_imported_static_and_stored_metadata(
    tmp_path: Path, managed: bool
) -> None:
    root = project(tmp_path, invocable=managed)
    direct = imported(root)
    static = static_interface.build(root)
    assert direct == static
    raw = package_interface.canonical_bytes(static)
    assert package_interface.read_bytes(raw) == static
    for row in direct["entrypoints"]:
        assert row["models"][0]["path"] == f"{row['name']}.models.model"
        assert row["models"][0]["default_ladder"] == LADDER
    # Same constant after a top-level failure still describes: the static reader runs nothing.
    source = root / "authored_defaults_fixture.py"
    source.write_text(source.read_text() + '\nraise RuntimeError("not an import")\n')
    assert static_interface.build(root) == static
    changed = deepcopy(direct)
    changed["entrypoints"][0]["models"][0]["default_ladder"][0]["lane"] = (
        "paul/minimax-h3@1.0.0-rc.2/bf16-pruned"
    )
    assert package_interface.package_interface_digest(
        changed
    ) != package_interface.package_interface_digest(direct)


@pytest.mark.parametrize("managed", [False, True], ids=["app_sync", "invocable_async"])
def test_org_relative_ladder_is_recorded_as_written(tmp_path: Path, managed: bool) -> None:
    ladder = next(row["ladder"] for row in VECTORS["valid"] if row["name"] == "org_relative")
    root = project(tmp_path, invocable=managed, ladder=ladder)
    static = static_interface.build(root)
    assert imported(root) == static
    assert package_interface.read_bytes(package_interface.canonical_bytes(static)) == static
    for row in static["entrypoints"]:
        assert row["models"][0]["default_ladder"] == ladder


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("parameter", ["prompt", "ctx", "Demo", "unknown"])
def test_defaults_must_name_actual_injected_models(
    tmp_path: Path, managed: bool, parameter: str
) -> None:
    root = project(tmp_path, invocable=managed, defaults=f"{{{parameter!r}: H3}}")
    with pytest.raises(ConformanceError, match="not injected Models"):
        imported(root)
    with pytest.raises(ConformanceError, match="not injected Models"):
        static_interface.build(root)


def test_artifact_reader_checks_ladders_and_preserves_additive_fields(tmp_path: Path) -> None:
    body = static_interface.build(project(tmp_path, invocable=True))
    assert package_interface.read_bytes(package_interface.canonical_bytes(body)) == body
    for vector in VECTORS["invalid"]:
        invalid = deepcopy(body)
        invalid["entrypoints"][0]["models"][0]["default_ladder"] = vector["ladder"]
        if vector["name"] == "unknown_field":
            # Author declarations reject unknown fields; a published interface
            # preserves additions from newer producers without changing the ladder.
            assert (
                package_interface.read_bytes(package_interface.canonical_bytes(invalid)) == invalid
            )
            continue
        with pytest.raises(ConformanceError) as refused:
            package_interface.read_bytes(package_interface.canonical_bytes(invalid))
        # Structural mistakes can be refused by the typed interface reader before
        # the shared ladder validator checks the model-default semantics.
        assert refused.value.code in ("model_defaults", "malformed_package_interface"), vector
        if refused.value.code == "malformed_package_interface":
            assert "default_ladder" in refused.value.message


def test_defaults_do_not_open_other_fixed_point_locations(tmp_path: Path) -> None:
    body = static_interface.build(project(tmp_path, invocable=True))
    body["entrypoints"][0]["default_ladder"] = LADDER
    with pytest.raises(ConformanceError):
        package_interface.assemble(
            str(body["application"]), [("entrypoint", body["entrypoints"][0])], {}
        )
    # A reader ignores fields another Runtime wrote; only the author surface refuses.
    read = package_interface.read_bytes(package_interface.canonical_bytes(body))
    assert read["entrypoints"][0]["name"] == body["entrypoints"][0]["name"]


def test_conflicting_decorator_declarations_refuse(tmp_path: Path) -> None:
    root = project(tmp_path, invocable=True)
    source = root / "authored_defaults_fixture.py"
    source.write_text(
        source.read_text().replace("@app.entrypoint\n", "@app.entrypoint(defaults={'model': H3})\n")
    )
    with pytest.raises(ConformanceError, match="declare defaults once"):
        imported(root)
    with pytest.raises(ConformanceError, match="declare defaults once"):
        static_interface.build(root)


def test_decorator_copies_the_authors_mutable_constant(tmp_path: Path) -> None:
    root = project(tmp_path, invocable=True)
    source = root / "authored_defaults_fixture.py"
    source.write_text(
        source.read_text() + "\nH3[0]['lane'] = 'paul/minimax-h3@1.0.0-rc.2/bf16-pruned'\n"
    )
    body = imported(root)
    assert body["entrypoints"][0]["models"][0]["default_ladder"] == LADDER
    assert static_interface.build(root) == body


def test_computed_default_is_never_evaluated_by_static_reader(tmp_path: Path) -> None:
    root = project(tmp_path, invocable=True, defaults="make_defaults()")
    with pytest.raises(ConformanceError, match="static"):
        static_interface.build(root)


def test_explicit_gpu_count_is_preserved_and_validated() -> None:
    lane = "paul/minimax-h3@1.0.0/fp8"
    assert default_ladder([{"gpu": "H100", "gpus": 2, "lane": lane}]) == (("H100", 2, lane),)
    for count in [0, -1, True, 1.5, "2"]:
        with pytest.raises(ConformanceError):
            default_ladder([{"gpu": "H100", "gpus": count, "lane": lane}])
    assert default_ladder([{"gpu": "future", "gpus": 17, "lane": lane}]) == (("future", 17, lane),)
    # Older immutable source/captures had no count and never asserted singleton execution.
    assert default_ladder([{"gpu": "H100", "lane": lane}]) == (("H100", 0, lane),)


def test_gpu_count_requires_the_models_declared_group() -> None:
    defaults = default_ladder([{"gpu": "H100", "gpus": 2, "lane": "paul/minimax-h3@1.0.0/fp8"}])

    def slot(sequence_parallel: list[int]) -> Any:
        return package_interface.model_slot(
            "run.models.model", "Demo", "refuse", "refuse", {}, sequence_parallel, defaults=defaults
        )

    assert slot([2, 4])["default_ladder"][0]["gpus"] == 2
    with pytest.raises(ConformanceError):
        slot([4])


def test_counted_artifact_with_malformed_parallel_contract_refuses_typed(tmp_path: Path) -> None:
    body = static_interface.build(project(tmp_path, invocable=True))
    for parallel in (None, {"degrees": None}, {"degrees": [2, True]}):
        invalid = deepcopy(body)
        slot = invalid["entrypoints"][0]["models"][0]
        slot["default_ladder"][0]["gpus"] = 2
        slot["sequence_parallel"] = parallel
        with pytest.raises(ConformanceError):
            package_interface.read_bytes(package_interface.canonical_bytes(invalid))
