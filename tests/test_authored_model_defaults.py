"""Authored model defaults and precision tolerances have one meaning through import, static
read and publication."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover

FIXTURE = """\
import msgspec
from cozy_runtime.author import App, Context, Loader, Model, invocable, uses_components

app = App()
# Any case: a name is never refused for its capitals.
DEFAULTS = {"model": "sdxl@1.0.0", "lm": "Fidika/Qwen3.5-9B@0.2.0"}


class Request(msgspec.Struct):
    prompt: str = "hello"


class Result(msgspec.Struct):
    count: int


class Diffusion(Model):
    def load(self, loader: Loader) -> None:
        pass

    @uses_components("dit")
    def generate(self) -> object:
        return None


class Language(Model, min_bits=4, phase="decode"):
    def load(self, loader: Loader) -> None:
        pass

    @uses_components("lm")
    def decode(self) -> object:
        return None


class Drafter(Language):
    pass


class Exact(Drafter, min_bits=16):
    pass


@app.entrypoint(defaults=DEFAULTS)
def generate(payload: Request, model: Diffusion, lm: Language) -> Result:
    return Result(1)


@app.entrypoint
@invocable(defaults={"draft": "qwen3.5-9b@0.2.0"})
async def caption(ctx: Context, *, draft: Drafter, exact: Exact, prompt: str = "") -> Result:
    return Result(1)
"""


def project(tmp_path: Path, source: str = FIXTURE) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.toml").write_text('[application]\nobject = "authored_defaults_fixture:app"\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "authored-defaults"\nversion = "1.0.0"\n'
    )
    (root / "authored_defaults_fixture.py").write_text(source)
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


def slots(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {slot["path"]: slot for row in body["entrypoints"] for slot in row["models"]}


def test_describe_emits_defaults_and_tolerances_identically_by_import_and_source(
    tmp_path: Path,
) -> None:
    root = project(tmp_path)
    static = static_interface.build(root)
    assert imported(root) == static
    assert package_interface.read_bytes(package_interface.canonical_bytes(static)) == static
    precision = {
        path: {key: slot[key] for key in ("default", "min_bits", "phase") if key in slot}
        for path, slot in slots(static).items()
    }
    assert precision == {
        "generate.models.model": {"default": "sdxl@1.0.0"},
        "generate.models.lm": {
            "default": "Fidika/Qwen3.5-9B@0.2.0",
            "min_bits": 4,
            "phase": "decode",
        },
        # Inherited from Language; Exact overrides min_bits and keeps the phase.
        "caption.models.draft": {"default": "qwen3.5-9b@0.2.0", "min_bits": 4, "phase": "decode"},
        "caption.models.exact": {"min_bits": 16, "phase": "decode"},
    }
    # The static reader runs nothing of the package.
    source = root / "authored_defaults_fixture.py"
    source.write_text(source.read_text() + '\nraise RuntimeError("not an import")\n')
    assert static_interface.build(root) == static


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("min_bits=4", "min_bits=6"),
        ("min_bits=4", "min_bits=8.0"),
        ("min_bits=4", "min_bits='8'"),
        ('phase="decode"', 'phase="prefill"'),
        ("min_bits=16", "attention=16"),
    ],
)
def test_out_of_vocabulary_class_keywords_refuse(tmp_path: Path, old: str, new: str) -> None:
    root = project(tmp_path, FIXTURE.replace(old, new))
    with pytest.raises(ConformanceError, match=r"(?i)closed|class_keyword"):
        imported(root)
    with pytest.raises(ConformanceError, match="closed set"):
        static_interface.build(root)


@pytest.mark.parametrize(
    "default",
    [
        "sdxl@1.0.0/int8",
        "sdxl",
        "sdxl@" + "a" * 64,
        "sdxl@",
        "[{'gpu': '*', 'lane': 'sdxl@1.0.0/int8'}]",
    ],
)
def test_a_default_names_a_release_and_nothing_else(tmp_path: Path, default: str) -> None:
    value = default if default.startswith("[") else repr(default)
    root = project(tmp_path, FIXTURE.replace('"sdxl@1.0.0"', value))
    with pytest.raises(ConformanceError, match="model_defaults"):
        imported(root)
    with pytest.raises(ConformanceError, match="model_defaults"):
        static_interface.build(root)


@pytest.mark.parametrize("parameter", ["payload", "Diffusion", "unknown"])
def test_defaults_must_name_actual_injected_models(tmp_path: Path, parameter: str) -> None:
    root = project(tmp_path, FIXTURE.replace('{"model":', f"{{{parameter!r}:"))
    with pytest.raises(ConformanceError, match="not injected Models"):
        imported(root)
    with pytest.raises(ConformanceError, match="not injected Models"):
        static_interface.build(root)


def test_conflicting_decorator_declarations_refuse(tmp_path: Path) -> None:
    root = project(
        tmp_path,
        FIXTURE.replace("@app.entrypoint\n", "@app.entrypoint(defaults={'draft': 'a@1'})\n"),
    )
    with pytest.raises(ConformanceError, match="declare defaults once"):
        imported(root)
    with pytest.raises(ConformanceError, match="declare defaults once"):
        static_interface.build(root)


def test_computed_default_is_never_evaluated_by_static_reader(tmp_path: Path) -> None:
    root = project(tmp_path, FIXTURE.replace("defaults=DEFAULTS", "defaults=make_defaults()"))
    with pytest.raises(ConformanceError, match="static"):
        static_interface.build(root)


def test_published_interface_reader_checks_defaults_and_tolerances(tmp_path: Path) -> None:
    body = static_interface.build(project(tmp_path))
    for member in (
        {"default": "sdxl@1.0.0/int8"},
        {"default": "sdxl"},
        {"min_bits": 6},
        {"min_bits": "8"},
        {"phase": "prefill"},
    ):
        invalid = deepcopy(body)
        invalid["entrypoints"][0]["models"][0].update(member)
        with pytest.raises(ConformanceError) as refused:
            package_interface.read_bytes(package_interface.canonical_bytes(invalid))
        assert refused.value.code in ("model_defaults", "malformed_package_interface"), member
    # A retired member from an older writer is an unknown field: tolerated, never read.
    stale = deepcopy(body)
    stale["entrypoints"][0]["models"][0]["default_ladder"] = [{"gpu": "*", "lane": "a@1/b"}]
    assert package_interface.read_bytes(package_interface.canonical_bytes(stale)) == stale


def test_default_opens_no_other_fixed_point_location(tmp_path: Path) -> None:
    body = static_interface.build(project(tmp_path))
    entry = body["entrypoints"][0]
    entry["default"] = {"model": "sdxl", "release": "1.0.0"}
    with pytest.raises(ConformanceError, match="fixed_point"):
        package_interface.assemble(str(body["application"]), [entry])
