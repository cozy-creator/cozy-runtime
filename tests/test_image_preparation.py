from __future__ import annotations

import binascii
import io
import json
import struct
import sys
from pathlib import Path
from typing import Annotated

import msgspec
import pytest
from PIL import Image

from cozy_runtime.author import (
    App,
    AssetLimits,
    Assets,
    ConformanceError,
    ImagePreparation,
    describe,
)
from cozy_runtime.author import Image as InputImage
from cozy_runtime.author._decode import (
    DEFAULT_DECODE_LIMITS,
    IMAGE_SOURCE_MAX_BYTES,
    DecodedImage,
    MediaDecodeError,
    _Budget,
    decode_image_path,
    image_target_size,
)
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import Discovered


class PrepRequest(msgspec.Struct):
    prompt: str


class PrepResult(msgspec.Struct):
    width: int


def decode(path: Path, preparation: ImagePreparation | None = None) -> DecodedImage:
    return decode_image_path(
        path, DEFAULT_DECODE_LIMITS, _Budget(DEFAULT_DECODE_LIMITS.max_decoded_bytes), preparation
    )


def test_source_limits_refuse_before_pixel_decode(tmp_path: Path) -> None:
    too_large = tmp_path / "large.jpg"
    with too_large.open("wb") as file:
        file.truncate(IMAGE_SOURCE_MAX_BYTES + 1)
    with pytest.raises(MediaDecodeError, match=r"source.*byte"):
        decode(too_large, ImagePreparation(max_edge=128))
    encoded = io.BytesIO()
    Image.new("RGB", (1, 1)).save(encoded, format="PNG")
    body = bytearray(encoded.getvalue())
    body[16:24] = struct.pack(">II", 9000, 9000)
    body[29:33] = struct.pack(">I", binascii.crc32(body[12:29]))
    oversized_header = tmp_path / "oversized.png"
    oversized_header.write_bytes(body)
    with pytest.raises(MediaDecodeError, match="pixel"):
        decode(oversized_header, ImagePreparation(max_edge=128))


def test_preparation_descriptor_is_argument_metadata(tmp_path: Path) -> None:
    app = App()

    @app.entrypoint
    def render(
        payload: PrepRequest,
        assets: Annotated[
            Assets[InputImage], AssetLimits(images=2), ImagePreparation(max_edge=512)
        ],
    ) -> PrepResult:
        return PrepResult(assets[0].width)

    found = Discovered(
        app, "image_preparation:app", tmp_path, sys.modules[__name__], describe(app), {}
    )
    body = package_interface.build(found)
    encoded = json.dumps(body).encode()
    read = package_interface.read_bytes(encoded)
    assert read["entrypoints"][0]["assets"]["kinds"][0]["prepare"] == {
        "profile": "image-fit/1",
        "max_edge": 512,
    }
    for policy in (
        {"profile": "unknown", "max_edge": 512},
        {"profile": "image-fit/1"},
        {"profile": "image-fit/1", "max_edge": 0},
        {"profile": "image-fit/1", "max_edge": True},
        {"profile": "image-fit/1", "max_pixels": 1.5},
        {"profile": "image-fit/1", "max_pixels": None},
    ):
        invalid = json.loads(encoded)
        invalid["entrypoints"][0]["assets"]["kinds"][0]["prepare"] = policy
        with pytest.raises(ConformanceError):
            package_interface.read_bytes(json.dumps(invalid).encode())
    additive = json.loads(encoded)
    additive["entrypoints"][0]["assets"]["kinds"][0]["prepare"]["crop"] = True
    read = package_interface.read_bytes(json.dumps(additive).encode())
    assert read["entrypoints"][0]["assets"]["kinds"][0]["prepare"]["max_edge"] == 512
    for cap in (0, -1, True, 1.5):
        with pytest.raises(ConformanceError, match="positive"):
            ImagePreparation(max_edge=cap)  # type: ignore[arg-type]


def test_pixel_fit_preserves_hard_caps_for_thin_images() -> None:
    assert image_target_size(10001, 1, ImagePreparation(max_pixels=100)) == (100, 1)
    assert image_target_size(1, 10001, ImagePreparation(max_pixels=100)) == (1, 100)
    assert image_target_size(20, 30, ImagePreparation(max_edge=128)) == (20, 30)


def test_image_preparation_static_alias_never_imports_package(tmp_path: Path) -> None:
    (tmp_path / "package.toml").write_text('[application]\nobject="image_preparation:app"\n')
    (tmp_path / "image_preparation.py").write_text("""
from typing import Annotated
import msgspec
from cozy_runtime.author import App, Assets, Image, ImagePreparation, AssetLimits
raise AssertionError("package analysis executed user code")
PictureInputs = Annotated[Assets[Image], AssetLimits(images=2), ImagePreparation(max_edge=512)]
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    width: int
app = App()
@app.entrypoint
def render(
    payload: Request,
    assets: PictureInputs,
) -> Result:
    return Result(assets[0].width)
""")
    body = static_interface.build(tmp_path)
    assert body["entrypoints"][0]["assets"]["kinds"][0]["prepare"] == {
        "profile": "image-fit/1",
        "max_edge": 512,
    }


@pytest.mark.parametrize(
    "format,orientation,order,size",
    [
        ("JPEG", None, (0, 1, 2, 3, 4, 5), (3, 2)),
        ("PNG", 1, (0, 1, 2, 3, 4, 5), (3, 2)),
        ("PNG", 2, (2, 1, 0, 5, 4, 3), (3, 2)),
        ("WEBP", 6, (3, 0, 4, 1, 5, 2), (2, 3)),
    ],
    ids=["jpeg-exif-without-orientation", "png-upright", "png-mirrored", "webp-rotated"],
)
def test_exif_images_decode_to_expected_pixels(
    tmp_path: Path,
    format: str,
    orientation: int | None,
    order: tuple[int, ...],
    size: tuple[int, int],
) -> None:
    # PyAV 18.1 rejected EXIF side data even without an orientation tag. Exercise
    # each image carrier and identity/mirror/rotation, not Pillow's full matrix.
    source = tmp_path / ("exif." + format.lower())
    image = Image.frombytes("RGB", (3, 2), bytes(range(18)))
    exif = Image.Exif()
    exif[305] = "synthetic orientation regression"
    if orientation is not None:
        exif[274] = orientation
    image.save(source, format=format, exif=exif, lossless=True)
    original = source.read_bytes()
    with Image.open(source) as saved:
        pixels = saved.convert("RGB").tobytes()  # Account for JPEG's lossy encoding.
    decoded = decode(source)
    assert (decoded.width, decoded.height) == size
    assert decoded.rgb == b"".join(pixels[3 * i : 3 * i + 3] for i in order)
    assert source.read_bytes() == original


@pytest.mark.parametrize("format", ["WEBP", "TIFF"], ids=["animation", "multipage"])
def test_still_image_refuses_animation(tmp_path: Path, format: str) -> None:
    source = tmp_path / ("animated." + format.lower())
    Image.new("RGB", (7, 5), "red").save(
        source,
        format=format,
        save_all=True,
        append_images=[Image.new("RGB", (7, 5), "blue")],
        duration=100,
    )
    with pytest.raises(MediaDecodeError) as error:
        decode(source)
    assert error.value.code == "animated_image"


def test_still_image_refuses_truncated_pixels(tmp_path: Path) -> None:
    source = tmp_path / "truncated.png"
    encoded = io.BytesIO()
    Image.frombytes("RGB", (7, 5), bytes(range(105))).save(encoded, format="PNG")
    source.write_bytes(encoded.getvalue()[:45])
    with pytest.raises(MediaDecodeError):
        decode(source)


def test_still_image_retained_budget_precedes_pixel_decode(tmp_path: Path) -> None:
    source = tmp_path / "image.png"
    Image.new("RGB", (7, 5)).save(source)
    source.write_bytes(source.read_bytes()[:45])  # Header fits, but pixel loading would fail.
    with pytest.raises(MediaDecodeError, match="decoded"):
        decode_image_path(source, DEFAULT_DECODE_LIMITS, _Budget(104))
