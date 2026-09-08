"""`describe` reads SOURCE: byte-equal to the imported app, in a venv that cannot import torch.

The static reader (`internal/static_interface.py`) is the interface of record (decision #713);
the import path is its oracle, admitted only inside the package's own locked environment.
Both halves are exercised here for real: the rich fixture under `testdata/static_describe`
spells every construct the closed vocabulary admits, and one venv built from the wheel —
with no torch and no diffusers — describes it, refuses the oracle, and then passes the oracle
once the fixture is installed. Refusal arms plant one construct outside the vocabulary each.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.static_interface import StaticRefusal

ROOT = Path(__file__).resolve().parent.parent
RICH = ROOT / "tests" / "testdata" / "static_describe" / "rich"
MARCO = ROOT / "examples" / "marco-polo"
TIMEOUT = 600


def _imported(project: Path) -> bytes:
    return package_interface.canonical_bytes(package_interface.build(discover(project)))


def _static(project: Path) -> bytes:
    return package_interface.canonical_bytes(static_interface.build(project))


def _run(*argv: str, cwd: Path | str = "/", expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    if result.returncode != expect:
        pytest.fail(
            f"{argv} exited {result.returncode}, wanted {expect}\n"
            f"--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}"
        )
    return result


@pytest.mark.parametrize("project", [RICH, MARCO], ids=["rich", "marco-polo"])
def test_source_reading_equals_imported_app(project: Path) -> None:
    assert _static(project) == _imported(project)


def test_rich_fixture_spells_the_whole_vocabulary() -> None:
    """The equality above proves nothing unless the fixture reaches every construct."""
    doc = static_interface.build(RICH)
    text = json.dumps(doc)
    for construct in (
        '"tag_field": "type"',
        '"asset_bound"',
        '"media_types": ["audio/wav"]',
        '"wire": "omissible"',
        '"constraints": {"ge": 256, "le": 2048}',
        '"literal": [30, 40, 50]',
        '"literal": ["16:9", "1:1"]',
        '"map"',
        '"kind_of"',
        '"input": "tree"',
        '"asset": "file"',
        '"encoded_leaves": "accept"',
        '"fusion": "accept"',
        '"sequence_parallel": {"degrees": [2, 4]}',
        '"component_use": {"encode": ["text_encoder"], "render": ["dit", "vae"]}',
        '"normalization_worst_relative_frobenius"',
        '"output_id": "mxfp8"',
        '"publishes": true',
        '"max_count": 2, "max_bytes": 67108864, "max_decoded_bytes": 268435456',
        '"parameter": "stills"',
    ):
        assert construct in text, construct
    assert [e["name"] for e in doc["entrypoints"]] == ["render"], "hidden stays unpublished"
    assert [j["name"] for j in doc["jobs"]] == ["gate", "quantize"]


def _variant(tmp_path: Path, edits: dict[str, tuple[str, str]]) -> Path:
    """The rich fixture with one construct rewritten per file."""
    project = tmp_path / "rich"
    shutil.copytree(RICH, project, ignore=shutil.ignore_patterns("__pycache__"))
    for name, (old, new) in edits.items():
        path = project / "rich_package" / name
        text = path.read_text()
        assert old in text, old
        path.write_text(text.replace(old, new))
    return project


def _refusal(project: Path) -> StaticRefusal:
    with pytest.raises(StaticRefusal) as caught:
        static_interface.build(project)
    return caught.value


def test_refuses_an_annotation_outside_the_vocabulary(tmp_path: Path) -> None:
    project = _variant(
        tmp_path,
        {
            "schemas.py": (
                "from enum import Enum, IntEnum\n",
                "from enum import Enum, IntEnum\nfrom fractions import Fraction\n",
            )
        },
    )
    schemas = project / "rich_package" / "schemas.py"
    schemas.write_text(
        schemas.read_text().replace("replay: str | None = None", "replay: Fraction | None = None")
    )
    line = schemas.read_text().splitlines().index("    replay: Fraction | None = None") + 1
    refusal = _refusal(project)
    assert refusal.code == "static_unresolvable"
    assert f"rich_package/schemas.py:{line}: `Fraction`" in str(refusal)
    assert "fractions.Fraction is outside the closed vocabulary" in str(refusal)


def test_refuses_a_computed_decorator_argument(tmp_path: Path) -> None:
    project = _variant(
        tmp_path,
        {
            "__init__.py": (
                "    weights=(\n"
                '        WeightsOutput("fp8", max_new_bytes=LANE_BYTES),\n'
                '        WeightsOutput("mxfp8", max_new_bytes=LANE_BYTES),\n'
                "    ),\n",
                "    weights=tuple(WeightsOutput(l, max_new_bytes=LANE_BYTES) "
                'for l in ("fp8", "mxfp8")),\n',
            )
        },
    )
    refusal = _refusal(project)
    assert refusal.code == "static_computed"
    assert "rich_package/__init__.py:" in str(refusal)
    assert "`tuple(WeightsOutput(l, max_new_bytes=LANE_BYTES) for l in" in str(refusal)
    assert "a call: its value exists only at run time" in str(refusal)


def test_refuses_a_call_inside_a_literal(tmp_path: Path) -> None:
    project = _variant(
        tmp_path,
        {
            "schemas.py": (
                "Steps = Annotated[Literal[30, 40, 50], ",
                "def _steps() -> tuple[int, ...]:\n    return (30, 40, 50)\n\n\n"
                "Steps = Annotated[Literal[_steps()], ",
            )
        },
    )
    refusal = _refusal(project)
    assert refusal.code == "static_computed"
    assert "rich_package/schemas.py:" in str(refusal) and "`_steps()`" in str(refusal)


def test_refuses_a_type_imported_inside_a_function(tmp_path: Path) -> None:
    project = _variant(
        tmp_path,
        {
            "__init__.py": (
                "from .schemas import (\n    Facts,\n",
                "from .schemas import (\n",
            )
        },
    )
    init = project / "rich_package" / "__init__.py"
    init.write_text(
        init.read_text().replace(
            "app = App()\n",
            "app = App()\n\n\ndef _facts() -> type:\n"
            "    from .schemas import Facts\n\n    return Facts\n",
        )
    )
    refusal = _refusal(project)
    assert refusal.code == "static_unresolvable"
    assert "rich_package/__init__.py:" in str(refusal)
    assert "'Facts' is not bound at module scope" in str(refusal)
    assert "imports it inside `def _facts` — every import goes at the top of the file" in str(
        refusal
    )


def test_refuses_registration_by_call(tmp_path: Path) -> None:
    project = _variant(
        tmp_path,
        {
            "__init__.py": (
                "@app.job(publishes=True)\ndef gate(",
                "def gate(",
            )
        },
    )
    init = project / "rich_package" / "__init__.py"
    init.write_text(init.read_text() + "\n\napp.job(gate, publishes=True)\n")
    refusal = _refusal(project)
    assert refusal.code == "static_unsupported"
    assert "`app.job(gate, publishes=True)`" in str(refusal)


@pytest.fixture(scope="module")
def torch_free() -> Iterator[Path]:
    """The wheel installed into a venv with nothing heavy: no torch, no diffusers."""
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build and install the wheel under test"
    with tempfile.TemporaryDirectory(prefix="cozy-static.") as raw:
        root = Path(raw)
        _run(uv, "build", "--wheel", "--out-dir", str(root / "dist"), str(ROOT))
        _run(uv, "build", "--wheel", "--out-dir", str(root / "dist"), str(RICH))
        _run(uv, "venv", "--python", sys.executable, str(root / "venv"))
        runtime = next((root / "dist").glob("cozy_runtime-*.whl"))
        # numpy is for the ORACLE only: the fixture's job result names
        # `cozy_runtime.derive.QuantizeResult`, and importing that module needs numpy. The
        # static reader parses the same file and never imports it.
        _run(
            uv,
            "pip",
            "install",
            "--python",
            str(root / "venv" / "bin" / "python"),
            str(runtime),
            "numpy",
        )
        yield root


def test_torch_free_venv_describes_refuses_and_conforms(torch_free: Path, tmp_path: Path) -> None:
    python = torch_free / "venv" / "bin" / "python"
    cli = torch_free / "venv" / "bin" / "cozy-runtime"
    _run(str(python), "-c", "import torch", expect=1)
    _run(str(python), "-c", "import diffusers", expect=1)
    _run(
        str(python),
        "-c",
        "import sys, cozy_runtime.cli.main; "
        "assert not [m for m in sys.modules if m.split('.')[0] in ('torch', 'diffusers')]",
    )

    described = _run(str(cli), "--json", "--dir", str(RICH), "describe")
    assert described.stdout.encode() == _imported(RICH) + b"\n"

    # A refusal is exit 3 (`Exit.validation`) with the typed body on stderr, file:line named.
    planted = _variant(
        tmp_path, {"schemas.py": ("replay: str | None = None", "replay: bytes | None = None")}
    )
    refused = _run(str(cli), "--json", "--dir", str(planted), "describe", expect=3)
    error = json.loads(refused.stderr)["error"]
    assert error["code"] == 3
    assert error["message"].startswith("static_unresolvable: rich_package/schemas.py:")
    assert "`bytes`" in error["message"]

    # The oracle imports package code, so it refuses everywhere but the locked environment.
    outside = _run(str(cli), "--json", "--conformance", "--dir", str(RICH), "describe", expect=3)
    assert json.loads(outside.stderr)["error"]["message"].startswith("conformance_environment:")

    uv = shutil.which("uv")
    assert uv is not None
    fixture = next((torch_free / "dist").glob("rich_package-*.whl"))
    _run(uv, "pip", "install", "--python", str(python), str(fixture))
    inside = _run(str(cli), "--full", "--conformance", "--dir", str(RICH), "describe")
    assert "equal: imported rich_package:app (rich-package 0.1.0 in" in inside.stdout
    conformed = _run(str(cli), "--json", "--conformance", "--dir", str(RICH), "describe")
    assert conformed.stdout == described.stdout
