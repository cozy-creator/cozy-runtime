from __future__ import annotations

import binascii
import io
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import av
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
from cozy_runtime.author._codec import encode_png, encode_webp
from cozy_runtime.author._decode import (
    DEFAULT_DECODE_LIMITS,
    IMAGE_SOURCE_MAX_BYTES,
    DecodedImage,
    MediaDecodeError,
    _Budget,
    _decode_image_frame,
    _jpeg_rotation,
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


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_preparation_lossless_handoff_and_idempotence(tmp_path: Path, orientation: int) -> None:
    source = tmp_path / "photo.jpg"
    image = Image.linear_gradient("L").resize((1001, 733)).convert("RGB")
    exif = Image.Exif()
    exif[274] = orientation
    image.save(source, quality=93, exif=exif)
    original = source.read_bytes()
    preparation = ImagePreparation(max_edge=333, max_pixels=100_000)
    prepared = decode(source, preparation)
    expected = (243, 333) if orientation in (6, 8) else (333, 243)
    assert (prepared.width, prepared.height) == expected
    for extension, encode in (("png", encode_png), ("webp", encode_webp)):
        derivative = tmp_path / f"prepared.{extension}"
        derivative.write_bytes(encode(prepared.width, prepared.height, prepared.rgb))
        # All pixels, not only dimensions: canonical lossless transport must be
        # identical to applying the same preparation to the raw Runtime upload.
        assert decode(derivative, preparation) == prepared
        assert decode(derivative) == prepared
    assert source.read_bytes() == original


def test_preparation_accepts_any_codec_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs 1563/1564: an exact PyAV/Pillow/libav closure refused every image input on a pod
    that resolved a newer codec. Preparation must work on whatever codec is installed."""
    source = tmp_path / "photo.png"
    Image.linear_gradient("L").resize((900, 600)).convert("RGB").save(source)
    monkeypatch.setattr(av, "__version__", "99.0.0")
    prepared = decode(source, ImagePreparation(max_edge=300))
    assert (prepared.width, prepared.height) == (300, 200)


def test_jpeg_orientation_reads_only_bounded_header_segments(tmp_path: Path) -> None:
    source = tmp_path / "orientation.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.linear_gradient("L").resize((1024, 768)).save(source, quality=100, exif=exif)

    class HeaderReader(io.BytesIO):
        count = 0

        def read(self, size: int | None = -1) -> bytes:
            assert size is not None and 0 <= size <= 65535
            data = super().read(size)
            self.count += len(data)
            return data

    encoded = source.read_bytes()
    reader = HeaderReader(encoded)
    assert _jpeg_rotation(reader) == -90
    assert reader.count < 1024 < len(encoded)


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


def test_source_pixels_do_not_hide_behind_small_prepared_output(tmp_path: Path) -> None:
    limits = replace(DEFAULT_DECODE_LIMITS, max_pixels_per_frame=128 * 128)
    prep = ImagePreparation(max_edge=64)
    photo = tmp_path / "source.png"
    Image.new("RGB", (512, 512)).save(photo)
    with pytest.raises(MediaDecodeError, match="pixel"):
        decode_image_path(photo, limits, _Budget(limits.max_decoded_bytes), prep)
    # Real small native frame: the second check must hold even if a codec were
    # to expose geometry different from its initial header.
    frame = av.VideoFrame(512, 512, "rgb24")
    with pytest.raises(MediaDecodeError, match="pixel"):
        _decode_image_frame(
            frame, limits, _Budget(limits.max_decoded_bytes), rotation=0, preparation=prep
        )


@pytest.mark.parametrize("named_alias", [False, True])
def test_image_preparation_static_reader_never_imports_package(
    tmp_path: Path, named_alias: bool
) -> None:
    (tmp_path / "package.toml").write_text('[application]\nobject="image_preparation:app"\n')
    (tmp_path / "image_preparation.py").write_text("""
from typing import Annotated
import msgspec
from cozy_runtime.author import App, Assets, Image, ImagePreparation, AssetLimits
raise AssertionError("package analysis executed user code")
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    width: int
app = App()
@app.entrypoint
def render(
    payload: Request,
    assets: Annotated[Assets[Image], AssetLimits(images=2), ImagePreparation(max_edge=512)],
) -> Result:
    return Result(assets[0].width)
""")
    if named_alias:
        module = tmp_path / "image_preparation.py"
        annotation = (
            "Annotated[Assets[Image], AssetLimits(images=2), ImagePreparation(max_edge=512)]"
        )
        source = module.read_text().replace(
            "class Request(msgspec.Struct):",
            "PictureInputs = " + annotation + "\nclass Request(msgspec.Struct):",
        )
        module.write_text(source.replace("assets: " + annotation, "assets: PictureInputs"))
    body = static_interface.build(tmp_path)
    assert body["entrypoints"][0]["assets"]["kinds"][0]["prepare"] == {
        "profile": "image-fit/1",
        "max_edge": 512,
    }
