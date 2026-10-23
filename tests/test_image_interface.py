"""cr-219: the shared image request and result, as a package declares and serves them."""

from __future__ import annotations

import sys
from pathlib import Path

import msgspec

from cozy_runtime.author import MEGAPIXEL, ImageRequest, held
from cozy_runtime.author.fakes import fake_adjustments
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover


def fixture(root: Path) -> Path:
    (root / "package.toml").write_text('[application]\nobject="painter:app"\n')
    (root / "painter.py").write_text(
        """from typing import Annotated
import msgspec
from cozy_runtime.author import App, Context, ImageRequest, ImageResult, ReferenceImages
class Request(ImageRequest):
    reference_images: Annotated[ReferenceImages, msgspec.Meta(max_length=4)] = msgspec.field(
        default_factory=list
    )
    hires: bool | None = None
class Result(ImageResult):
    sampler: str
app = App()
@app.entrypoint
def generate(ctx: Context, payload: Request) -> ImageResult:
    raise RuntimeError("handler must not execute")
@app.entrypoint
def generate_turbo(ctx: Context, payload: ImageRequest) -> Result:
    raise RuntimeError("handler must not execute")
"""
    )
    return root


def test_a_package_extends_the_shared_types_and_static_describe_reads_them(tmp_path: Path) -> None:
    root = fixture(tmp_path)
    sys.modules.pop("painter", None)
    imported = package_interface.build(discover(root))
    static = static_interface.build(root)
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(imported)
    entries = {entry["name"]: entry for entry in static["entrypoints"]}  # type: ignore[union-attr,index]
    fields = [field["name"] for field in entries["generate"]["request"]["fields"]]
    assert fields == [
        "prompt",
        "reference_images",
        "aspect_ratio",
        "megapixels",
        "steps",
        "guidance",
        "negative_prompt",
        "seed",
        "prompt_upsampling",
        "hires",
    ]
    references = entries["generate"]["request"]["fields"][1]
    assert references["constraints"]["max_length"] == 4, references
    turbo = [field["name"] for field in entries["generate_turbo"]["request"]["fields"]]
    assert turbo == fields[:-1]
    assert [field["name"] for field in entries["generate"]["result"]["fields"]] == [
        "image",
        "width",
        "height",
        "seed",
        "prompt",
    ]
    assert [field["name"] for field in entries["generate_turbo"]["result"]["fields"]][
        -1
    ] == "sampler"


def test_every_knob_left_out_is_none_the_models_default() -> None:
    request = msgspec.json.decode(b'{"prompt": "a lighthouse", "unknown": 1}', type=ImageRequest)
    assert request.reference_images == []
    assert (request.aspect_ratio, request.megapixels, request.steps, request.guidance) == (
        None,
    ) * 4
    assert (request.negative_prompt, request.seed, request.prompt_upsampling) == (None,) * 3


def test_a_missing_size_is_the_first_references_else_the_models() -> None:
    request = ImageRequest("x")
    assert request.target(None, megapixels=4).ratio == (1, 1)
    assert request.target(None, megapixels=4, ratio="16:9").pixels == 4 * MEGAPIXEL
    edit = request.target((640, 366), megapixels=4)
    assert (edit.ratio, edit.pixels) == ((640, 366), 640 * 366)
    given = ImageRequest("x", aspect_ratio="2:1", megapixels=0.5).target((640, 366), megapixels=4)
    assert (given.ratio, given.pixels) == ((2, 1), 0.5 * MEGAPIXEL)
    ratio_only = ImageRequest("x", aspect_ratio="1:1").target((640, 366), megapixels=4)
    assert (ratio_only.ratio, ratio_only.pixels) == ((1, 1), 640 * 366)


def test_out_of_range_clamps_and_unused_knobs_warn_never_refuse() -> None:
    adj = fake_adjustments()
    assert held(adj, "steps", 150, 1, 100) == 100
    assert held(adj, "guidance", 3.5, 1.0, 20.0) == 3.5
    request = ImageRequest("x", guidance=4.0, negative_prompt="", prompt_upsampling=False)
    request.ignored(adj, "guidance", "negative_prompt", "prompt_upsampling", reason="no guidance")
    rows = [(row.kind, row.field, row.requested, row.applied) for row in adj.rows]
    assert rows == [("clamp", "steps", 150, 100), ("warn", "guidance", None, None)]
    assert ImageRequest("x", seed=7).drawn_seed() == 7
    assert 0 <= ImageRequest("x").drawn_seed() < 1 << 53
