"""`Memoize(requires=...)`: both describe paths state the same memo condition."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import ConformanceError, Memoize
from cozy_runtime.internal import canonical, interface_wheel, package_interface, static_interface
from cozy_runtime.internal.discovery import discover

SOURCE = """import msgspec
import cozy_runtime.author as sdk
from cozy_runtime.author import App, Context, Memoize, invocable

class Request(msgspec.Struct, kw_only=True):
    prompt: str
    seed: int | None = None
    count: int | None = msgspec.field(name="wire_count", default=None)

class Result(msgspec.Struct):
    value: str

app = App()

@app.entrypoint(memoize=sdk.Memoize(requires=["wire_count"]))
def measure(ctx: Context, payload: Request) -> Result:
    raise RuntimeError("never executed")

@invocable(memoize=Memoize(requires=("seed", "seed")))
async def generate(ctx: Context, *, prompt: str, seed: int | None = None) -> Result:
    raise RuntimeError("never executed")

@invocable(memoize=True)
async def plain(ctx: Context, *, prompt: str, seed: int | None = None) -> Result:
    raise RuntimeError("never executed")

@invocable(memoize=Memoize())
async def unconditional(ctx: Context, *, prompt: str, seed: int | None = None) -> Result:
    raise RuntimeError("never executed")

app.entrypoint(generate, accelerator=False)
app.entrypoint(plain, accelerator=False)
app.entrypoint(unconditional, accelerator=False)
"""


def project(tmp_path: Path, source: str = SOURCE) -> Path:
    sys.modules.pop("memo_fixture", None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="memo-fixture"\nversion="0.1.0"\n')
    (root / "package.toml").write_text('[application]\nobject="memo_fixture:app"\n')
    (root / "memo_fixture.py").write_text(source)
    return root


def declarations(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row["invocable"] for row in document["entrypoints"]}


def test_static_and_imported_describe_state_the_same_memo_condition(tmp_path: Path) -> None:
    root = project(tmp_path)
    static = static_interface.build(root)
    imported = package_interface.source_facts(package_interface.build(discover(root)))
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(imported)
    rows = declarations(static)
    assert rows["generate"] == {
        "module": "memo_fixture",
        "export": "generate",
        "context": "ctx",
        "parameters": ["prompt", "seed"],
        "defaults": {"request/seed": None},
        "type_names": {"request": "generateRequest", "result": "Result"},
        "enum_members": {},
        "memoize": True,
        "capabilities": [],
        "memo_requires": ["seed"],
    }
    assert rows["measure"]["parameters"] == ["prompt", "seed", "wire_count"]
    assert rows["measure"]["memoize"] is True
    assert rows["measure"]["memo_requires"] == ["wire_count"]
    for name in ("plain", "unconditional"):
        assert rows[name]["memoize"] is True and "memo_requires" not in rows[name]
    raw = package_interface.canonical_bytes(static)
    assert package_interface.read_bytes(raw) == static
    interface_wheel.generate(raw)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            'Memoize(requires=("seed", "seed"))',
            'Memoize(requires=("sed",))',
            r'memoize requires "sed", which is not a parameter of generate '
            r"\(parameters: prompt, seed\)",
        ),
        (
            'Memoize(requires=("seed", "seed"))',
            'Memoize(requires=["prompt"])',
            r'memoize requires "prompt", a required parameter of generate',
        ),
        (
            'sdk.Memoize(requires=["wire_count"])',
            'sdk.Memoize(requires=["count"])',
            r'memoize requires "count", which is not a parameter of measure '
            r"\(parameters: prompt, seed, wire_count\)",
        ),
        (
            'sdk.Memoize(requires=["wire_count"])',
            'sdk.Memoize(requires=["prompt"])',
            r'memoize requires "prompt", a required parameter of measure',
        ),
        (
            "seed: int | None = None) -> Result",
            "seed: int = 1005) -> Result",
            r'memoize requires "seed", so generate must default it to None: declare '
            r"`seed: int \| None = None` and draw the seed in the function when it's None",
        ),
        (
            "default=None)\n\nclass Result",
            "default=1)\n\nclass Result",
            r'memoize requires "wire_count", so measure must default it to None: declare '
            r"`wire_count: int \| None = None`",
        ),
    ],
)
def test_memo_condition_names_an_optional_parameter(
    tmp_path: Path, before: str, after: str, message: str
) -> None:
    root = project(tmp_path, SOURCE.replace(before, after))
    for describe in (static_interface.build, lambda root: package_interface.build(discover(root))):
        with pytest.raises(ConformanceError, match=message) as caught:
            describe(root)
        assert caught.value.code == "invocable_memoize"


def test_malformed_memo_conditions_refuse() -> None:
    assert Memoize(requires=["seed", "steps", "seed"]).requires == ("seed", "steps")
    assert Memoize() == Memoize(requires=[])
    for requires in ("seed", ("",), (1,)):
        with pytest.raises(ConformanceError, match="non-empty parameter names"):
            Memoize(requires=requires)  # type: ignore[arg-type]


def test_reader_requires_memoize_for_a_memo_condition(tmp_path: Path) -> None:
    document = canonical.parse_canonical(
        package_interface.canonical_bytes(static_interface.build(project(tmp_path)))
    )
    generate = next(row for row in document["entrypoints"] if row["name"] == "generate")
    generate["invocable"]["memoize"] = False
    with pytest.raises(ConformanceError, match="memo_requires needs memoize"):
        package_interface.read_bytes(canonical.write(document))


def test_installed_release_keeps_a_concrete_default_memo_condition(tmp_path: Path) -> None:
    """A release published before the None-default rule still installs (paul/sdxl 2.7.0)."""
    source = SOURCE.replace("seed: int | None = None) -> Result", "seed: int = 1005) -> Result")
    root = project(tmp_path, source)
    with pytest.raises(ConformanceError, match="must default it to None"):
        static_interface.build(root)
    imported = package_interface.build(discover(root), installed=True)
    assert declarations(imported)["generate"]["memo_requires"] == ["seed"]
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    metadata = site / "memo_fixture-0.1.0.dist-info"
    metadata.mkdir(parents=True)
    (tmp_path / "venv" / "bin").mkdir()
    (tmp_path / "venv" / "bin" / "python").touch()
    (tmp_path / "venv" / "pyvenv.cfg").write_text("home = /unused\n")
    (metadata / "METADATA").write_text("Name: memo-fixture\nVersion: 0.1.0\n")
    (metadata / "entry_points.txt").write_text("[cozy.application]\ndefault = memo_fixture:app\n")
    (metadata / "RECORD").write_text("memo_fixture.py,,\n")
    (site / "memo_fixture.py").write_text(source)
    body = static_interface.build_installed(
        "memo-fixture", environment_python=tmp_path / "venv" / "bin" / "python"
    )
    assert declarations(body)["generate"]["memo_requires"] == ["seed"]
