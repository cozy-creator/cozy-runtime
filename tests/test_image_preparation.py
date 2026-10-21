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
from PIL import Image, ImageOps

from cozy_runtime.author import (
    App,
    AssetLimits,
    Assets,
    ConformanceError,
    ImageAsset,
    ImagePreparation,
    describe,
    fakes,
)
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
            Assets[ImageAsset], AssetLimits(images=2), ImagePreparation(max_edge=512)
        ],
    ) -> PrepResult:
        return PrepResult(assets[0].probe().width)

    found = Discovered(app, "image_preparation:app", tmp_path, sys.modules[__name__], describe(app))
    body = package_interface.build(found)
    encoded = json.dumps(body).encode()
    read = package_interface.read_bytes(encoded)
    assert read["entrypoints"][0]["assets"]["kinds"][0]["prepare"] == {
        "profile": "image-fit/1",
        "max_edge": 512,
    }
    unknown = json.loads(encoded)
    unknown["entrypoints"][0]["assets"]["kinds"][0]["prepare"] = {"profile": "fit/9", "x": []}
    package_interface.read_bytes(json.dumps(unknown).encode())  # a newer writer's profile
    for policy in (
        {"profile": "image-fit/1"},
        {"profile": "video-trim/1", "max_seconds": 5, "excess": "refuse"},
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
    with pytest.raises(ConformanceError):  # a known profile admits exactly its grammar
        package_interface.read_bytes(json.dumps(additive).encode())
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
from cozy_runtime.author import App, Assets, ImageAsset, ImagePreparation, AssetLimits
raise AssertionError("package analysis executed user code")
PictureInputs = Annotated[
    Assets[ImageAsset], AssetLimits(images=2), ImagePreparation(max_edge=512)
]
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
    return Result(assets[0].probe().width)
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


@pytest.mark.parametrize(
    ("format", "options"),
    [
        ("MPO", {}),
        ("PNG", {}),
        ("PNG", {"default_image": True}),
        ("GIF", {}),
        ("WEBP", {"lossless": True}),
        ("TIFF", {}),
    ],
    ids=["iphone-mpo", "apng", "apng-default-image", "gif", "webp", "tiff-pages"],
)
def test_multi_frame_image_reads_its_first_frame(
    tmp_path: Path, format: str, options: dict[str, object]
) -> None:
    source = tmp_path / ("frames." + format.lower())
    first = Image.frombytes("RGB", (8, 6), bytes(range(144)))
    if format == "MPO":  # An iPhone JPEG: the rotated photo, then a half-size gray gain map.
        exif = Image.Exif()
        exif[274] = 6
        options, second = {"exif": exif}, Image.new("L", (4, 3), 128)
    else:
        second = Image.new("RGB", (8, 6), "blue")
    first.save(
        source, format=format, save_all=True, append_images=[second], duration=100, **options
    )
    with Image.open(source) as saved:
        assert getattr(saved, "n_frames", 1) == 2
        expected = ImageOps.exif_transpose(saved).convert("RGB")
    asset = fakes.file_input(ImageAsset, source)
    assert (asset.probe().width, asset.probe().height) == expected.size
    assert asset.read().tobytes() == decode(source).rgb == expected.tobytes()
    if format == "MPO":
        assert expected.size == (6, 8)  # The rotated primary, not the 4x3 gain map.
    else:
        assert expected.tobytes() == first.tobytes()
    claim = 8 * 6 * 3  # The retained budget is the decoded frame's, nothing more.
    decode_image_path(source, DEFAULT_DECODE_LIMITS, _Budget(claim))
    with pytest.raises(MediaDecodeError, match="decoded"):
        decode_image_path(source, DEFAULT_DECODE_LIMITS, _Budget(claim - 1))


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


def _rgba() -> Image.Image:
    """Opaque red, transparent green, half-transparent blue and a clear pixel over black."""
    return Image.frombytes(
        "RGBA", (2, 2), bytes([255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 128, 0, 0, 0, 0])
    )


@pytest.mark.parametrize("format", ["png", "webp"])
def test_rgba_saves_and_reads_with_its_alpha(tmp_path: Path, format: str) -> None:
    source = _rgba()
    saved = fakes.fake_outputs().save_image(source, format=format)
    assert saved._local is not None
    with Image.open(saved._local) as encoded:
        assert encoded.mode == "RGBA"
        assert encoded.tobytes() == source.tobytes()  # Exact, colour under alpha 0 included.
    asset = fakes.fake_input(saved)
    rgba = asset.read(mode="RGBA")
    assert rgba.mode == "RGBA" and rgba.tobytes() == source.tobytes()
    rgb = asset.read()
    assert rgb.mode == "RGB"
    # Over white: opaque stays, transparent is white, half-transparent blends.
    assert rgb.tobytes() == bytes([255, 0, 0, 255, 255, 255, 127, 127, 255, 255, 255, 255])


def test_save_image_converts_every_mode(tmp_path: Path) -> None:
    out = fakes.fake_outputs()
    gray = Image.frombytes("L", (2, 1), bytes([10, 200]))
    shaded = Image.frombytes("LA", (2, 1), bytes([10, 255, 200, 0]))
    keyed = Image.new("P", (2, 1))
    keyed.putpalette([0, 0, 0, 9, 8, 7])
    keyed.putpixel((1, 0), 1)
    keyed.info["transparency"] = 1
    expected = [
        ("RGB", bytes([10, 10, 10, 200, 200, 200])),
        ("RGBA", bytes([10, 10, 10, 255, 200, 200, 200, 0])),
        ("RGBA", bytes([0, 0, 0, 255, 9, 8, 7, 0])),
    ]
    for image, (mode, pixels) in zip((gray, shaded, keyed), expected, strict=True):
        saved = out.save_image(image)
        assert saved._local is not None
        with Image.open(saved._local) as encoded:
            assert (encoded.mode, encoded.tobytes()) == (mode, pixels)


def test_rgba_read_is_prepared_and_charged_four_bytes_a_pixel(tmp_path: Path) -> None:
    source = tmp_path / "large.png"
    Image.new("RGBA", (64, 32), (0, 0, 255, 0)).save(source)
    asset = fakes.file_input(ImageAsset, source, preparation=ImagePreparation(max_edge=16))
    image = asset.read(mode="RGBA")
    assert (image.mode, image.size) == ("RGBA", (16, 8))
    assert image.tobytes()[3::4] == bytes(16 * 8)
    assert asset.read().tobytes() == b"\xff" * (16 * 8 * 3)
    assert asset._media is not None and asset._media.retained_bytes == 16 * 8 * (4 + 3)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        asset.read(mode="CMYK")  # type: ignore[arg-type]
