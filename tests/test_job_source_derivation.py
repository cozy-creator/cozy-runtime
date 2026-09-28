"""Job source discovery must not fabricate an empty serving construction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.internal import derive_child


@pytest.fixture
def source_project(tmp_path: Path) -> Path:
    (tmp_path / "source_kind.py").write_text("""
import msgspec
from cozy_runtime.author import App, Model, uses_components
app = App()
class Source(Model[object]):
    def load(self, loader):
        raise AssertionError("a job source must not construct a serving model")
class EmptyServing(Model[object]):
    def load(self, loader):
        pass
constructions = 0
class LinearPipeline:
    def __init__(self, config):
        global constructions
        constructions += 1
        if constructions != 1:
            raise AssertionError("compatibility rebuilt the same model")
        import torch
        dtype = {"f32": torch.float32, "f16": torch.float16}[
            config.tensor_dtype("linear.weight", default="f32")
        ]
        self.components = {"linear": torch.nn.Linear(2, 3, dtype=dtype)}
class Serving(Model[LinearPipeline]):
    def load(self, loader):
        self.pipe = loader.construct(LinearPipeline, factory=LinearPipeline)
    @uses_components("linear")
    def sample(self) -> int:
        return 3
class Request(msgspec.Struct):
    value: int = 1
class Result(msgspec.Struct):
    value: int
@app.job
def produce(payload: Request, source: Source) -> Result:
    return Result(payload.value)
@app.entrypoint
def generate(payload: Request, model: EmptyServing) -> Result:
    return Result(payload.value)
@app.entrypoint
def linear(payload: Request, model: Serving) -> Result:
    return Result(model.sample())
""")
    (tmp_path / "package.toml").write_text('[application]\nobject = "source_kind:app"\n')
    return tmp_path


def test_declared_job_source_skips_load(source_project: Path) -> None:
    source = derive_child.DeriveRequest(
        (derive_child.SlotRequest("produce.models.source", b"{}"),), project=source_project
    )
    assert derive_child.derive_in(Path(sys.executable), source) == (
        derive_child.SourceSlot("produce.models.source"),
    )
    unknown = derive_child.DeriveRequest(
        (derive_child.SlotRequest("absent.models.source", b"{}"),), project=source_project
    )
    with pytest.raises(derive_child.DeriveRefusal, match="no model slot"):
        derive_child.derive_in(Path(sys.executable), unknown)


def test_ordinary_torch_model_derives_once_and_empty_serving_refuses(source_project: Path) -> None:
    pytest.importorskip("torch")
    serving = derive_child.DeriveRequest(
        (derive_child.SlotRequest("linear.models.model", b"{}"),), project=source_project
    )
    assert derive_child.derive_in(Path(sys.executable), serving) == (
        derive_child.TensorRequirements(
            "linear.models.model",
            (("linear", "bias", "f32", (3,)), ("linear", "weight", "f32", (3, 2))),
            ("linear",),
        ),
    )
    empty = derive_child.DeriveRequest(
        (derive_child.SlotRequest("generate.models.model", b"{}"),), project=source_project
    )
    with pytest.raises(derive_child.DeriveRefusal, match="requirements shape"):
        derive_child.derive_in(Path(sys.executable), empty)


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_native_dtype_reaches_real_derivation_child(source_project: Path, dtype: str) -> None:
    pytest.importorskip("torch")
    request = derive_child.DeriveRequest(
        (
            derive_child.SlotRequest(
                "linear.models.model", b"{}", tensor_dtypes={"linear.weight": dtype}
            ),
        ),
        project=source_project,
    )
    (result,) = derive_child.derive_in(Path(sys.executable), request)
    assert isinstance(result, derive_child.TensorRequirements)
    # Float16 is the ordinary convertible 16-bit fit contract; float32 is a hard
    # precision island. The constructor census above separately measures exact bytes.
    assert {row[2] for row in result.rows} == ({None} if dtype == "f16" else {"f32"})


@pytest.mark.parametrize("error", ["RuntimeError", "ModuleNotFoundError"])
def test_installed_broken_torch_import_is_not_hidden(tmp_path: Path, error: str) -> None:
    package = tmp_path / "torch"
    package.mkdir()
    (package / "__init__.py").write_text(f'raise {error}("torch import failure")\n')
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from cozy_runtime.internal.derive_child import _warm_torch; _warm_torch()",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"{error}: torch import failure" in result.stderr


def test_source_response_partition_is_exact() -> None:
    request = derive_child.DeriveRequest(
        (derive_child.SlotRequest("produce.models.source", b"{}"),)
    )
    for rows in (
        [{"slot": "another.models.source", "source": True}],
        [{"slot": "produce.models.source", "source": False}],
        [{"slot": "produce.models.source", "source": True}] * 2,
        [{"slot": "produce.models.source", "components": [], "tensors": []}],
    ):
        response = msgspec.convert({"requirements": rows}, derive_child.DeriveResponse)
        with pytest.raises(derive_child.DeriveRefusal):
            derive_child.read_response(response, request)
