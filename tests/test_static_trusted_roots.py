"""Only cozy-runtime's own derivation surface is read as trusted Runtime source."""

from pathlib import Path

import pytest

from cozy_runtime.internal.static_interface import StaticRefusal, build


def test_similarly_named_runtime_module_is_not_trusted(tmp_path: Path) -> None:
    (tmp_path / "package.toml").write_text('[application]\nobject = "example:app"\n')
    (tmp_path / "example.py").write_text("""
import msgspec
from cozy_runtime.author import App, Context
from cozy_runtime.models_untrusted import FakeModel
app = App()
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    value: int
@app.entrypoint
def render(ctx: Context, payload: Request, model: FakeModel) -> Result:
    return Result(1)
""")
    (tmp_path / "cozy_runtime").mkdir()
    (tmp_path / "cozy_runtime/models_untrusted.py").write_text("""
from cozy_runtime.author import Model
class FakeModel(Model[object]):
    pass
""")
    with pytest.raises(StaticRefusal, match="not an author surface type"):
        build(tmp_path)
