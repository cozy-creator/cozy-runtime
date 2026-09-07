"""Job source discovery must not fabricate an empty serving construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import derive_child


def test_declared_job_source_skips_load_but_empty_serving_still_refuses(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    (tmp_path / "source_kind.py").write_text("""
import msgspec
from cozy_runtime.author import App, Model
app = App()
class Source(Model[object]):
    def load(self, loader):
        raise AssertionError("a job source must not construct a serving model")
class EmptyServing(Model[object]):
    def load(self, loader):
        pass
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
""")
    (tmp_path / "package.toml").write_text('[application]\nobject = "source_kind:app"\n')
    source = derive_child.DeriveRequest(
        (derive_child.SlotRequest("produce.models.source", b"{}"),), project=tmp_path
    )
    assert derive_child.derive_in(Path(sys.executable), source) == (
        derive_child.SourceSlot("produce.models.source"),
    )
    serving = derive_child.DeriveRequest(
        (derive_child.SlotRequest("generate.models.model", b"{}"),), project=tmp_path
    )
    with pytest.raises(derive_child.DeriveRefusal, match="requirements shape"):
        derive_child.derive_in(Path(sys.executable), serving)
    unknown = derive_child.DeriveRequest(
        (derive_child.SlotRequest("absent.models.source", b"{}"),), project=tmp_path
    )
    with pytest.raises(derive_child.DeriveRefusal, match="no model slot"):
        derive_child.derive_in(Path(sys.executable), unknown)


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
        with pytest.raises(derive_child.DeriveRefusal):
            derive_child.read_response({"requirements": rows}, request)
