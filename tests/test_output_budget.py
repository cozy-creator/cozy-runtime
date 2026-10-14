"""Current author output budgets and static interface parity."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest

from cozy_runtime.author import App, AssetBound, Invocation, Outputs, Tree, attempt
from cozy_runtime.internal import output_budget, package_interface, static_interface
from cozy_runtime.internal.discovery import discover


def spec_for(*sizes: int) -> dict[str, Any]:
    return {
        "outputs": [{"output_id": f"out{i}", "max_bytes": size} for i, size in enumerate(sizes)]
    }


def test_admitted_slots_preserve_defaults_and_ignore_weights_capacity() -> None:
    assert output_budget.admitted({}) == 256 << 20
    assert output_budget.admitted(spec_for(256 << 20, 256 << 20)) == 512 << 20
    assert output_budget.admitted(spec_for(7, 11)) == 18
    assert output_budget.intermediate(spec_for(7, 11)) == 256 << 20
    assert output_budget.intermediate(spec_for(256 << 20, 256 << 20)) == 512 << 20
    assert output_budget.admitted(spec_for(1 << 40)) == 256 << 20
    assert (
        output_budget.admitted(
            {
                "outputs": [
                    {
                        "output_id": "weights",
                        "mime_type": "application/vnd.cozy.model-manifest",
                        "max_bytes": 1 << 40,
                    },
                    {"output_id": "report", "max_bytes": 17},
                ]
            }
        )
        == 17
    )
    with pytest.raises(ValueError):
        output_budget.admitted(spec_for(*([1] * 33)))
    for bad in (-1, True, "100"):
        with pytest.raises(ValueError):
            output_budget.admitted({"outputs": [{"max_bytes": bad}]})
    assert output_budget.effective(spec_for(7, 11), {"max_output_bytes": 13}) == 13
    assert output_budget.effective(spec_for(7, 11), {"max_output_bytes": 256 << 20}) == 18
    for bad in ((256 << 20) + 1, -1, True, "18"):
        with pytest.raises(ValueError):
            output_budget.effective(spec_for(7, 11), {"max_output_bytes": bad})


def test_optional_tree_bound_static_imported_parity(tmp_path: Path) -> None:
    sys.modules.pop("bounded_trees", None)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="bounded-trees"\nversion="0.1.0"\n')
    (tmp_path / "package.toml").write_text('[application]\nobject="bounded_trees:app"\n')
    (tmp_path / "bounded_trees.py").write_text("""from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, MediaDecoder, Tree
class Request(msgspec.Struct):
    prefix: Annotated[Tree | None, AssetBound(max_bytes=268435456)] = None
class Result(msgspec.Struct):
    video: Annotated[Tree, AssetBound(max_bytes=268435456)]
    prefix: Annotated[Tree, AssetBound(max_bytes=268435456)]
app = App()
@app.job(accelerator=False)
def assemble(payload: Request, media: MediaDecoder) -> Result:
    raise RuntimeError("describe only")
""")
    source = static_interface.build(tmp_path)
    imported = package_interface.build(discover(tmp_path))
    assert package_interface.canonical_bytes(source) == package_interface.canonical_bytes(imported)


def test_lower_aggregate_stops_the_author_before_a_second_large_copy(tmp_path: Path) -> None:
    # Small real bytes exercise the same aggregate check without a duplicate large fixture.
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"01234567")
    app = App()

    @app.job(accelerator=False)
    def write(payload: Request, out: Outputs) -> Pair:
        return Pair(out.save_tree(source), out.save_tree(source))

    _, outcome, _ = attempt(
        app.get("write"),
        {},
        Invocation(
            "producer",
            tmp_path / "spool",
            time.monotonic() + 30,
            max_output_bytes=15,
        ),
    )
    assert outcome.code == "output_too_large"


class Request(msgspec.Struct):
    pass


class Pair(msgspec.Struct):
    first: Annotated[Tree, AssetBound(max_bytes=140 << 20)]
    second: Annotated[Tree, AssetBound(max_bytes=140 << 20)]
