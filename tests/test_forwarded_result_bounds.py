"""A returned asset meets its own result field's bound, saved or forwarded from a child."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Annotated, Any, cast

import msgspec
import pytest

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    Invocation,
    Outputs,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._services import ImageFrame, encode_png
from durable_seam import wire


class Empty(msgspec.Struct):
    pass


class ImageReport(msgspec.Struct, frozen=True):
    image: Annotated[
        ImageAsset, AssetBound(max_bytes=4096, max_decoded_bytes=768, media_types=("image/png",))
    ]


@invocable(memoize=True)
async def produce_image(ctx: Context, *, out: Outputs) -> ImageReport:
    return ImageReport(out.save_image(ImageFrame(16, 16, b"\x0a\x14\x1e" * 256), format="png"))


def _result(decoded: int, media: str = "image/png") -> type:
    bound = AssetBound(max_bytes=4096, max_decoded_bytes=decoded, media_types=(media,))
    return msgspec.defstruct("Forwarded", [("value", cast(type, Annotated[ImageAsset, bound]))])


def _forwarding_parent(tmp_path: Path, result_type: type) -> tuple[Any, Any, list[str]]:
    child = App()
    child.job(produce_image)
    surface = describe(child)[0]
    parent = App()

    async def main(ctx: Context, payload: Empty) -> Any:
        return result_type((await cast(Any, produce_image)()).image)

    main.__annotations__["return"] = result_type
    parent.job(main)
    response: dict[str, Any] = {}
    kinds: list[str] = []

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        kinds.append(kind)
        if kind == "child_call":
            # The child's worker post phase encodes its registered frame; this is that PNG.
            image = tmp_path / "child-image.png"
            image.write_bytes(encode_png(ImageFrame(16, 16, b"\x0a\x14\x1e" * 256)))
            data = image.read_bytes()
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            response["result"] = msgspec.json.encode(
                {
                    "image": {
                        "asset_ref": digest,
                        "digest": digest,
                        "kind": "image",
                        "size_bytes": len(data),
                        "media_type": "image/png",
                    }
                }
            ).decode()
            response["byte_grants"] = [
                {
                    "output_id": "image",
                    "kind": "image",
                    "digest": digest,
                    "length": len(data),
                    "content_bytes": len(data),
                    "media_type": "image/png",
                    "local": str(image),
                }
            ]
            return {"ok": True}
        if kind == "child_forget":
            return {"ok": True}
        assert kind == "child_poll"
        return {"ok": True, "state": "succeeded", **response}

    broker = _Broker(
        "parent",
        {
            (__name__, "produce_image"): _CallType(
                "sha256:" + "21" * 32,
                __name__,
                "produce_image",
                cast(type[msgspec.Struct], surface.payload_type),
                ImageReport,
            )
        },
        wire(exchange),
    )
    result, outcome, _ = attempt(
        parent.get("main"),
        {},
        Invocation("parent", tmp_path / "parent", time.monotonic() + 30, calls=broker),
    )
    return result, outcome, kinds


def test_forwarded_image_within_its_decoded_bound_is_returned(tmp_path: Path) -> None:
    result, outcome, kinds = _forwarding_parent(tmp_path, _result(768))
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.value.media_type == "image/png"
    assert "child_call" in kinds


@pytest.mark.parametrize(
    ("result_type", "code"),
    [(_result(767), "result_decoded_bound"), (_result(768, "image/webp"), "result_bound")],
)
def test_forwarded_image_over_the_parents_bound_is_refused(
    tmp_path: Path, result_type: type, code: str
) -> None:
    result, outcome, kinds = _forwarding_parent(tmp_path, result_type)
    assert result is None and outcome.terminal != "succeeded", outcome
    assert outcome.code.endswith(code), outcome
    if code == "result_decoded_bound":
        assert "768 B, over its captured decoded bound of 767 B" in outcome.message
    assert "child_call" in kinds


@pytest.mark.parametrize("decoded", [11, 12])
def test_saved_image_meets_its_result_decoded_bound(tmp_path: Path, decoded: int) -> None:
    result_type = _result(decoded)
    app = App()

    async def main(ctx: Context, payload: Empty, out: Outputs) -> Any:
        return result_type(out.save_image(ImageFrame(2, 2, b"\xff\x00\x00" * 4), format="png"))

    main.__annotations__["return"] = result_type
    app.job(main)
    result, outcome, _ = attempt(
        app.get("main"), {}, Invocation("saved", tmp_path / "out", time.monotonic() + 30)
    )
    if decoded == 12:
        assert outcome.terminal == "succeeded" and result is not None, outcome
    else:
        assert result is None and outcome.code.endswith("result_decoded_bound"), outcome
