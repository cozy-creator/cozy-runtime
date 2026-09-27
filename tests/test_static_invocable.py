"""Managed interfaces come from ASTs even when importing their implementation must fail."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import interface_wheel, package_interface, static_interface

OPERATIONS = """
from typing import Literal
import msgspec
from cozy_runtime.author import (
    Context, ModelArtifact, Model, Telemetry, invocable, uses_components,
)

raise AssertionError("the static reader must never execute this module")

class Source(Model, encoded_leaves="accept"):
    @uses_components("transformer")
    def tensors(self) -> object:
        raise AssertionError("model code must never execute")

class Detail(msgspec.Struct, frozen=True):
    note: str = "original"

class Result(msgspec.Struct, frozen=True):
    artifact: ModelArtifact
    detail: Detail
    accepted: bool = msgspec.field(default=True)

@invocable(memoize=True)
async def quantize(ctx: Context, *, source: Source, encoding: Literal["fp8", "mxfp8"],
    threshold: float | None = None, tel: Telemetry) -> Result:
    raise AssertionError("implementation must never execute")

@invocable(memoize=True)
async def precompute(ctx: Context, *, model: ModelArtifact, timesteps: int = 50,
    generating_model: ModelArtifact | None = None) -> ModelArtifact:
    raise AssertionError("implementation must never execute")
"""


def project(tmp_path: Path, source: str = OPERATIONS) -> Path:
    root = tmp_path / "project"
    package = root / "ast_operations"
    package.mkdir(parents=True)
    (root / "package.toml").write_text('[application]\nobject = "ast_operations:app"\n')
    (root / "pyproject.toml").write_text('[project]\nname = "ast-operations"\nversion = "1.0.0"\n')
    (package / "__init__.py").write_text(
        "from cozy_runtime.author import App, WeightsOutput\n"
        "from . import operations\n"
        "app = App()\n"
        "BYTES = 1 << 20\n"
        'app.job(operations.quantize, name="quantize-artifact", '
        'weights=(WeightsOutput("model", BYTES),))\n'
        'app.job(name="precompute-adaln")(operations.precompute)\n'
    )
    (package / "operations.py").write_text(source)
    return root


def test_static_invocable_preserves_managed_names_defaults_models_and_capabilities(
    tmp_path: Path,
) -> None:
    document = static_interface.build(project(tmp_path))
    raw = package_interface.canonical_bytes(document)
    assert package_interface.canonical_bytes(package_interface.read_bytes(raw)) == raw
    jobs = {row["name"]: row for row in cast(list[dict[str, Any]], document["jobs"])}
    assert sorted(jobs) == ["precompute-adaln", "quantize-artifact"]
    quantize = jobs["quantize-artifact"]
    assert quantize["models"] == [
        {
            "path": "quantize-artifact.models.source",
            "class": "Source",
            "encoded_leaves": "accept",
            "component_use": {"tensors": ["transformer"]},
        }
    ]
    export = quantize["invocable"]
    assert export["module"] == "ast_operations.operations" and export["export"] == "quantize"
    assert export["memoize"] is True
    assert export["context"] == "ctx"
    assert export["parameters"] == ["source", "encoding", "threshold"]
    assert export["capabilities"] == ["telemetry", "weights"]
    assert export["defaults"] == {
        "request/source": None,
        "request/threshold": None,
        "result/detail/note": "original",
        "result/accepted": True,
    }
    assert export["type_names"] == {
        "request": "quantizeRequest",
        "result": "Result",
        "result/detail": "Detail",
    }
    assert quantize["weights_outputs"][0]["max_bytes"] == 1 << 20
    orchestration = jobs["precompute-adaln"]
    assert orchestration["result"] == {"input": "model"}
    assert "models" not in orchestration
    assert orchestration["invocable"]["defaults"] == {
        "request/timesteps": 50,
        "request/generating_model": None,
    }
    assert orchestration["invocable"]["capabilities"] == []
    # The actual interface generator accepts this result and produces the expected
    # caller functions without requiring any implementation import.
    files = interface_wheel.generate(raw)
    generated = ast.parse(files["ast_operations/operations/__init__.py"])
    names = {
        node.name
        for node in generated.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"quantize", "precompute"} <= names
    assert not any(
        name == "ast_operations" or name.startswith("ast_operations.") for name in sys.modules
    )


def test_invocable_defaults_are_part_of_the_static_interface_identity(tmp_path: Path) -> None:
    original = project(tmp_path)
    before = package_interface.package_interface_digest(static_interface.build(original))
    source = original / "ast_operations/operations.py"
    source.write_text(source.read_text().replace('note: str = "original"', 'note: str = "changed"'))
    after = package_interface.package_interface_digest(static_interface.build(original))
    assert before != after


def test_serving_memo_dependencies_are_installed_metadata_not_executed_source(
    tmp_path: Path,
) -> None:
    root = project(tmp_path)
    (root / "ast_operations/__init__.py").write_text("""
import msgspec
from cozy_runtime.author import App, Context, MemoDistribution, MemoResource
raise AssertionError("the static reader must never execute this module")
class Request(msgspec.Struct):
    value: int
class Result(msgspec.Struct):
    value: int
def helper():
    raise AssertionError("memo helpers must never execute during static inspection")
def measure(ctx: Context, payload: Request) -> Result:
    return Result(payload.value)
app = App()
app.entrypoint(measure, memoize=True, memo_version=helper(), memo_dependencies=(
    helper, MemoResource("uninstalled", "plan.json"), MemoDistribution("uninstalled"),
))
""")
    document = static_interface.build(root)
    rows = cast(list[dict[str, Any]], document["entrypoints"])
    assert len(rows) == 1
    assert rows[0]["invocable"]["memoize"] is True
    assert "operation_identity" not in rows[0]["invocable"]


def test_declared_tensorfs_output_is_a_capability_without_a_weights_sink_parameter(
    tmp_path: Path,
) -> None:
    source = OPERATIONS
    document = static_interface.build(project(tmp_path, source))
    jobs = {row["name"]: row for row in cast(list[dict[str, Any]], document["jobs"])}
    assert jobs["quantize-artifact"]["invocable"]["capabilities"] == ["telemetry", "weights"]
    assert jobs["quantize-artifact"]["weights_outputs"][0]["output_id"] == "model"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("@invocable(memoize=True)", "@invocable(memoize=compute_policy())"),
        ("timesteps: int = 50", "timesteps: int = compute_default()"),
        ("timesteps: int = 50", "timesteps: list[int] = []"),
        ("timesteps: int = 50", 'timesteps: int = "wrong"'),
        ("ctx: Context, *, model:", "ctx: Context, model:"),
        ('note: str = "original"', "note: str = msgspec.field(default_factory=compute_default)"),
        ("@invocable(memoize=True)", "@unknown_decorator\n@invocable(memoize=True)"),
    ],
)
def test_static_invocable_refuses_computed_or_nonportable_contracts(
    tmp_path: Path, old: str, new: str
) -> None:
    with pytest.raises(ConformanceError):
        static_interface.build(project(tmp_path, OPERATIONS.replace(old, new)))


def test_static_memoized_invocable_refuses_secret_capability(tmp_path: Path) -> None:
    source = OPERATIONS.replace("Context, ModelArtifact,", "Context, Secrets, ModelArtifact,")
    source = source.replace("timesteps: int = 50,", "secret: Secrets[Detail], timesteps: int = 50,")
    with pytest.raises(ConformanceError, match="memo"):
        static_interface.build(project(tmp_path, source))


def test_app_decorator_can_register_an_invocable_without_running_it(tmp_path: Path) -> None:
    root = project(tmp_path)
    (root / "ast_operations/__init__.py").write_text("""
from cozy_runtime.author import App, Context, ModelArtifact, invocable
app = App()
@app.job(name="prepare")
@invocable(memoize=True)
async def prepare(ctx: Context, *, model: ModelArtifact) -> ModelArtifact:
    raise AssertionError("never execute")
""")
    document = static_interface.build(root)
    job = cast(list[dict[str, Any]], document["jobs"])[0]
    assert job["name"] == "prepare" and job["invocable"]["memoize"] is True
