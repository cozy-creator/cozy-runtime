"""Self and generated callers preserve returned-media decode capability bounds."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from cozy_runtime.author._call_results import decode
from cozy_runtime.author._decode import MediaDecoder
from cozy_runtime.author._errors import CapabilityError, RuntimeFailure
from cozy_runtime.author._invoke import asset_inputs
from cozy_runtime.author._services import Attempt
from cozy_runtime.internal import (
    canonical,
    interface_wheel,
    package_interface,
    schema,
    static_interface,
)
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import Executor

SOURCE = """from typing import Annotated
import msgspec
from cozy_runtime.author import App, AssetBound, Context, ImageAsset, Outputs, invocable
class Request(msgspec.Struct):
    reference: Annotated[ImageAsset | None, AssetBound(max_bytes=1000, max_decoded_bytes=99)] = None
class Result(msgspec.Struct):
    wire_frame: Annotated[ImageAsset, AssetBound(max_bytes=1000, max_decoded_bytes=36,
        media_types=("image/png",))]
@invocable
async def segment(ctx: Context, *, request: Request, out: Outputs) -> Result:
    raise RuntimeError("source only")
app = App()
app.job(segment, emits_media=True)
"""


@pytest.mark.parametrize("generated", [False, True])
def test_returned_image_keeps_bounds_and_request_interface_identity(
    tmp_path: Path,
    generated: bool,
) -> None:
    sys.modules.pop("native_result_media", None)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="native-result-media"\nversion="0.1.0"\n'
    )
    (tmp_path / "package.toml").write_text('[application]\nobject="native_result_media:app"\n')
    (tmp_path / "native_result_media.py").write_text(SOURCE)
    interface = package_interface.build(discover(tmp_path))
    raw = package_interface.canonical_bytes(interface)
    assert raw == package_interface.canonical_bytes(static_interface.build(tmp_path))
    source = sys.modules["native_result_media"]
    # Decoded limits historically stay out of request wire schemas. Changing only
    # the output transport must not change published input descriptor identities.
    request = schema.render(source.Request)
    assert isinstance(request, dict)
    request_fields = request["fields"]
    assert isinstance(request_fields, list)
    assert request_fields[0]["asset_bound"] == {"max_bytes": 1000}
    result_schema = interface["jobs"][0]["result"]
    assert result_schema["fields"][0]["asset_bound"]["max_decoded_bytes"] == 36
    executor = object.__new__(Executor)
    executor.schema_digests = {}
    assert executor._result_schema_digest(source.app.get("segment").surface) == canonical.digest(
        result_schema
    )
    result_type = source.Result
    if generated:
        module = ModuleType("generated_native_result_media")
        sys.modules[module.__name__] = module
        code = interface_wheel.generate(raw)["native_result_media/__init__.py"]
        exec(compile(code, "generated_native_result_media.py", "exec"), module.__dict__)
        binding = module.__dict__["__cozy_bindings__"]["segment"]
        result_type = binding.result
        assert schema.render(binding.request) == interface["jobs"][0]["request"]
    image = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "red").save(image)
    data = image.read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    value = {
        "wire_frame": {
            "asset_ref": digest,
            "digest": digest,
            "kind": "image",
            "size_bytes": len(data),
            "media_type": "image/png",
        }
    }
    grants = [
        {
            "output_id": "wire_frame",
            "kind": "image",
            "digest": digest,
            "length": len(data),
            "content_bytes": len(data),
            "media_type": "image/png",
            "local": str(image),
        }
    ]
    closed = False

    def guard() -> None:
        if closed:
            raise CapabilityError("result escaped parent", code="escaped_handle")

    result, _ = decode(
        value, result_type, grants, request_id="parent", guard=guard, observation=None
    )
    asset = asset_inputs(result)[0].asset
    assert asset._max_decoded_bytes == 36 and asset._input_id == "wire_frame"
    decoder = MediaDecoder(Attempt("parent", tmp_path), active=lambda: False)
    decoded = decoder.decode_image(asset)  # type: ignore[arg-type]
    assert (decoded.width, decoded.height, len(decoded.rgb)) == (4, 3, 36)
    closed = True
    with pytest.raises(CapabilityError, match="escaped_handle"):
        decoder.decode_image(asset)  # type: ignore[arg-type]
    closed = False
    image.write_bytes(data + b"changed")
    with pytest.raises(RuntimeFailure, match="input_changed"):
        decoder.decode_image(asset)  # type: ignore[arg-type]


def test_generated_only_job_can_request_decoder_without_payload_assets(tmp_path: Path) -> None:
    """Generated child results supply decode capabilities after the parent starts."""
    source = """import msgspec
from cozy_runtime.author import App, Context, MediaDecoder
app = App()
class Request(msgspec.Struct):
    prompt: str
class Result(msgspec.Struct):
    width: int
@app.job
async def compose(ctx: Context, payload: Request, media: MediaDecoder) -> Result:
    raise RuntimeError("discovery only")
"""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="generated-only"\nversion="0.1.0"\n')
    (tmp_path / "package.toml").write_text('[application]\nobject="generated_only:app"\n')
    (tmp_path / "generated_only.py").write_text(source)
    imported = package_interface.build(discover(tmp_path))
    assert package_interface.canonical_bytes(imported) == package_interface.canonical_bytes(
        static_interface.build(tmp_path)
    )
    assert imported["jobs"][0]["name"] == "compose"


def test_generated_job_does_not_waive_explicit_input_decode_bounds(tmp_path: Path) -> None:
    from cozy_runtime.author import ConformanceError

    (tmp_path / "package.toml").write_text('[application]\nobject="unbounded_decode:app"\n')
    (tmp_path / "unbounded_decode.py").write_text("""import msgspec
from cozy_runtime.author import App, Context, ImageAsset, MediaDecoder
app = App()
class Request(msgspec.Struct):
    image: ImageAsset
class Result(msgspec.Struct):
    width: int
@app.job
async def compose(ctx: Context, payload: Request, media: MediaDecoder) -> Result:
    raise RuntimeError("discovery only")
""")
    with pytest.raises(ConformanceError, match="asset_decode_unbounded"):
        package_interface.build(discover(tmp_path))
