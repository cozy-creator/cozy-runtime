"""A callable's `components=` declaration has one meaning through import and static read."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover

MODULE = "callable_components_fixture"


def project(tmp_path: Path, *, invocable: bool, components: str = "{'model': ('dit',)}") -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.toml").write_text(f'[application]\nobject = "{MODULE}:app"\n')
    (root / "pyproject.toml").write_text('[project]\nname = "components"\nversion = "1.0.0"\n')
    declared = (
        f"@app.entrypoint\n@invocable(components={components})\n"
        "async def generate(ctx: Context, *, model: Demo, prompt: str = 'hello') -> Result:\n"
        "    return Result(1)\n"
        if invocable
        else f"@app.entrypoint(components={components})\n"
        "def generate(payload: Request, model: Demo) -> Result:\n"
        "    return Result(1)\n"
    )
    (root / f"{MODULE}.py").write_text(
        "import msgspec\n"
        "from cozy_runtime.author import App, Context, Loader, Model, invocable, uses_components\n"
        "app = App()\n"
        "class Request(msgspec.Struct):\n    prompt: str = 'hello'\n"
        "class Result(msgspec.Struct):\n    count: int\n"
        "class Demo(Model):\n"
        "    def load(self, loader: Loader) -> None:\n        pass\n"
        "    @uses_components('text_encoder', 'dit')\n"
        "    def denoise(self) -> None:\n        pass\n"
        "    @uses_components('vae')\n    def decode(self) -> None:\n        pass\n"
        + declared
        + "@app.entrypoint\ndef everything(payload: Request, model: Demo) -> Result:\n"
        "    return Result(1)\n"
        "@app.job(components={'model': ('vae', 'text_encoder')})\n"
        "def bake(payload: Request, model: Demo) -> Result:\n    return Result(1)\n"
    )
    return root


def imported(root: Path) -> dict[str, Any]:
    previous = sys.modules.pop(MODULE, None)
    try:
        return package_interface.build(discover(root))
    finally:
        sys.modules.pop(MODULE, None)
        if previous is not None:
            sys.modules[MODULE] = previous


def slots(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row["models"][0] for row in (*body["entrypoints"], *body["jobs"])}


@pytest.mark.parametrize("managed", [False, True], ids=["app_sync", "invocable_async"])
def test_declared_components_are_identical_imported_and_static(
    tmp_path: Path, managed: bool
) -> None:
    root = project(tmp_path, invocable=managed, components="{'model': ['dit', 'text_encoder']}")
    static = static_interface.build(root)
    assert imported(root) == static
    assert package_interface.read_bytes(package_interface.canonical_bytes(static)) == static
    by_name = slots(static)
    assert by_name["generate"]["components"] == ["dit", "text_encoder"]
    assert by_name["bake"]["components"] == ["text_encoder", "vae"]
    assert "components" not in by_name["everything"]  # absent: every component


@pytest.mark.parametrize("managed", [False, True], ids=["app_sync", "invocable_async"])
def test_a_component_no_method_declares_refuses(tmp_path: Path, managed: bool) -> None:
    root = project(tmp_path, invocable=managed, components="{'model': ('dit', 'unet')}")
    for describe in (imported, static_interface.build):
        with pytest.raises(ConformanceError, match="unet") as refused:
            describe(root)
        assert refused.value.code == "model_components"


@pytest.mark.parametrize("managed", [False, True], ids=["app_sync", "invocable_async"])
@pytest.mark.parametrize(
    "components",
    [
        "{'prompt': ('dit',)}",
        "{'unknown': ('dit',)}",
        "{'model': ()}",
        "{'model': ('dit', 'dit')}",
        "{'model': 'dit'}",
        "{'model': ('',)}",
        "('dit',)",
    ],
)
def test_malformed_declarations_refuse_on_both_readers(
    tmp_path: Path, managed: bool, components: str
) -> None:
    root = project(tmp_path, invocable=managed, components=components)
    for describe in (imported, static_interface.build):
        # a decorator's refusal reaches import-path describe as the failed import
        with pytest.raises(ConformanceError, match="model_components"):
            describe(root)


def test_declaring_components_on_both_layers_refuses(tmp_path: Path) -> None:
    root = project(tmp_path, invocable=True)
    source = root / f"{MODULE}.py"
    source.write_text(
        source.read_text().replace(
            "@app.entrypoint\n@invocable",
            "@app.entrypoint(components={'model': ('vae',)})\n@invocable",
        )
    )
    for describe in (imported, static_interface.build):
        with pytest.raises(ConformanceError, match="declare components once"):
            describe(root)


def test_the_reader_checks_declared_components(tmp_path: Path) -> None:
    body = static_interface.build(project(tmp_path, invocable=True))
    index = next(i for i, row in enumerate(body["entrypoints"]) if row["name"] == "generate")
    for components in ([], ["dit", "dit"], ["unet"]):
        invalid = deepcopy(body)
        invalid["entrypoints"][index]["models"][0]["components"] = components
        with pytest.raises(ConformanceError, match="components"):
            package_interface.read_bytes(package_interface.canonical_bytes(invalid))
