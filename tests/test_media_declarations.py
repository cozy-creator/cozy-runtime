"""Media declaration rules hold identically for describe and both static readers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError, Invocation, attempt, fakes
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS, host_decode_limits
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover
from media_sources import mp4, steady

HEADER = """from typing import Annotated
import msgspec
from cozy_runtime.author import (
    App, AssetBound, Context, ImageAsset, VideoAsset, VideoPreparation, invocable,
)
class Done(msgspec.Struct):
    seconds: str
app = App()
"""
DIRECT = """
@invocable
async def render(
    ctx: Context,
    *,
    clip: Annotated[
        VideoAsset,
        AssetBound(max_bytes=1 << 30, max_decoded_bytes=1 << 30),
        VideoPreparation(max_seconds=1, excess="truncate"),
    ],
) -> Done:
    return Done(str(clip.read().duration))
app.entrypoint(render, accelerator=False)
"""


def _package(root: Path, name: str, body: str) -> Path:
    sys.modules.pop(name, None)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname="{name.replace("_", "-")}"\nversion="0.1.0"\n'
    )
    (root / "package.toml").write_text(f'[application]\nobject="{name}:app"\n')
    (root / f"{name}.py").write_text(HEADER + body)
    return root


def _both_refuse(root: Path, code: str) -> None:
    with pytest.raises(ConformanceError) as statically:
        static_interface.build(root)
    with pytest.raises(ConformanceError) as imported:
        package_interface.build(discover(root))
    assert statically.value.code == imported.value.code == code


def test_a_preparation_on_a_direct_invocable_parameter_truncates_it(tmp_path: Path) -> None:
    root = _package(tmp_path / "pkg", "direct_prep", DIRECT)
    imported = package_interface.build(discover(root))
    assert package_interface.canonical_bytes(imported) == package_interface.canonical_bytes(
        static_interface.build(root)
    )
    clip = next(f for f in imported["entrypoints"][0]["request"]["fields"] if f["name"] == "clip")
    assert clip["asset_bound"]["prepare"] == {
        "profile": "video-trim/1",
        "excess": "truncate",
        "max_seconds": 1,
    }
    starts, end = steady(30, 60)
    grant = fakes.granted_input(
        mp4(tmp_path / "clip.mp4", size=(64, 48), starts=starts, end=end), input_id="clip"
    )
    app = sys.modules["direct_prep"].app
    result, outcome, _ = attempt(
        app.get("render"),
        {"clip": grant.digest},
        Invocation("direct", tmp_path / "spool", time.monotonic() + 60, assets={"clip": grant}),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.seconds == "1"
    assert [(row.field, row.source) for row in result.adjustments] == [("clip", "policy")]


@pytest.mark.parametrize(
    "body",
    [
        """
@invocable
async def render(
    ctx: Context,
    *,
    still: Annotated[
        ImageAsset, AssetBound(max_decoded_bytes=1 << 20), VideoPreparation(max_seconds=1)
    ],
) -> Done:
    raise RuntimeError("source only")
app.entrypoint(render, accelerator=False)
""",
    ],
    ids=["image-parameter"],
)
def test_a_misplaced_parameter_preparation_refuses_in_every_reader(
    tmp_path: Path, body: str
) -> None:
    _both_refuse(_package(tmp_path, "misplaced_prep", body), "preparation_target")


@pytest.mark.parametrize(
    "body",
    [
        """
class Request(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, media_types=("video/mp4",))]
@app.entrypoint(accelerator=False)
def upscale(payload: Request) -> Done:
    raise RuntimeError("source only")
""",
        """
@invocable
async def upscale(
    ctx: Context, *, video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30)]
) -> Done:
    raise RuntimeError("source only")
app.entrypoint(upscale, accelerator=False)
""",
    ],
    ids=["entrypoint", "invocable"],
)
def test_a_media_input_with_no_decoded_bound_describes_in_every_reader(
    tmp_path: Path, body: str
) -> None:
    """Owner, 2026-10-17: memory is Runtime's job, so no package is refused for a bound it did
    not declare; both readers describe it the same way."""
    root = _package(tmp_path, "unbounded_media", body)
    imported = package_interface.build(discover(root))
    assert package_interface.canonical_bytes(imported) == package_interface.canonical_bytes(
        static_interface.build(root)
    )


@pytest.mark.parametrize(
    "body",
    [
        """
class Request(msgspec.Struct):
    prompt: str
class Clip(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30), VideoPreparation(max_seconds=1)]
@app.entrypoint(accelerator=False)
def render(payload: Request) -> Clip:
    raise RuntimeError("source only")
""",
        """
class Clip(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30), VideoPreparation(max_seconds=1)]
@invocable
async def render(ctx: Context, *, prompt: str) -> Clip:
    raise RuntimeError("source only")
app.entrypoint(render, accelerator=False)
""",
    ],
    ids=["entrypoint", "invocable"],
)
def test_a_result_preparation_refuses_in_every_reader(tmp_path: Path, body: str) -> None:
    _both_refuse(_package(tmp_path, "result_prep", body), "preparation_target")


BUDGETS = """
from cozy_runtime.author import AssetLimits, Assets, AudioAsset
References = Annotated[
    Assets[Annotated[VideoAsset, AssetBound(max_decoded_bytes=8 << 30)] | AudioAsset],
    AssetLimits(videos=1, audio=1),
]
class Shot(msgspec.Struct):
    clip: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=12 << 30)]
@invocable
async def render(ctx: Context, *, shot: Shot, refs: References) -> Done:
    raise RuntimeError("source only")
app.entrypoint(render, accelerator=False)
"""


def test_declared_bounds_of_any_size_describe_and_undeclared_kinds_publish_the_default(
    tmp_path: Path,
) -> None:
    root = _package(tmp_path, "budgets", BUDGETS)
    imported = package_interface.build(discover(root))
    assert package_interface.canonical_bytes(imported) == package_interface.canonical_bytes(
        static_interface.build(root)
    )
    kinds = {row["kind"]: row for row in imported["entrypoints"][0]["assets"]["kinds"]}
    # Machine-independent: a declared bound as declared, an undeclared one as the default.
    assert kinds["video"]["max_decoded_bytes"] == 8 << 30
    assert kinds["audio"]["max_decoded_bytes"] == DEFAULT_DECODE_LIMITS.max_decoded_bytes
    assert DEFAULT_DECODE_LIMITS.max_decoded_bytes == 2 << 30
    assert fakes.fake_attempt("host").media._budget.limit == host_decode_limits().max_decoded_bytes
    assert host_decode_limits().max_decoded_bytes >= 2 << 30
