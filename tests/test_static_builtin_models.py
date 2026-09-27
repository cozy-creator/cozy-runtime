"""Resolve builtin declarations through the existing AST reader without ML imports."""

import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal.static_interface import StaticRefusal, build


@pytest.mark.parametrize(
    "model_import, annotation",
    [
        ("from cozy_runtime.models.minimax_h3 import H3Model", "H3Model"),
        ("from cozy_runtime.models.minimax_h3.model import H3TurboBase", "H3TurboBase"),
        ("import cozy_runtime.models.minimax_h3.model as models", "models.H3Model"),
    ],
)
def test_builtin_model_is_described_without_importing_it(
    tmp_path: Path, model_import: str, annotation: str
) -> None:
    (tmp_path / "package.toml").write_text('[application]\nobject = "example:app"\n')
    (tmp_path / "example.py").write_text(f"""
import msgspec
from cozy_runtime.author import App, Context, uses_components
{model_import}
raise AssertionError("the static reader must not execute package code")
app = App()
class Custom({annotation}):
    @uses_components("text_encoder")
    def extra(self):
        return None
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    value: int
@app.entrypoint(defaults={{"model": [{{"gpu": "H100", "lane": "paul/minimax-h3@1.0.0/fp8"}}]}})
def render(ctx: Context, payload: Request, model: Custom) -> Result:
    raise AssertionError("not invoked")
""")
    document = build(tmp_path)
    (model,) = document["entrypoints"][0]["models"]
    assert model["class"] == "Custom"
    assert model["encoded_leaves"] == "accept"
    assert model["fusion"] == "accept"
    assert model["sequence_parallel"] == {"degrees": [2, 4, 7, 8]}
    assert model["component_use"]["sample_fl2va"] == ["fl2va_dit"]
    assert model["component_use"]["extra"] == ["text_encoder"]
    assert model["default_ladder"] == [{"gpu": "H100", "lane": "paul/minimax-h3@1.0.0/fp8"}]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from cozy_runtime.internal.static_interface import build
build(Path(sys.argv[1]))
assert not {'torch', 'diffusers', 'transformers'} & sys.modules.keys()
assert 'cozy_runtime.models.minimax_h3.model' not in sys.modules
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
