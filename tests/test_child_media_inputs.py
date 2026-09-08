"""Nested admitted media inputs cross the broker only by verified content identity."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest

from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    Invocation,
    MediaDecoder,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._services import ImageFrame, encode_png


class ImageInput(msgspec.Struct, tag="image"):
    image: Annotated[ImageAsset, AssetBound(max_bytes=4096, max_decoded_bytes=4096)]


class DigestResult(msgspec.Struct):
    digest: str


@invocable(memoize=True)
async def score(ctx: Context, *, media: ImageInput, decoder: MediaDecoder) -> DigestResult:
    decoded = decoder.decode_image(media.image)
    return DigestResult(hashlib.sha256(decoded.rgb).hexdigest())


class ParentInput(msgspec.Struct):
    image: Annotated[ImageAsset, AssetBound(max_bytes=4096, max_decoded_bytes=4096)]
    forge: bool = False


@pytest.mark.parametrize("forged", [False, True])
def test_broker_serializes_verified_nested_media_and_refuses_unhydrated_forgery(
    tmp_path: Path,
    forged: bool,
) -> None:
    raw = encode_png(ImageFrame(16, 16, bytes(range(256)) * 3))
    media = tmp_path / "pixels.png"
    media.write_bytes(raw)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    child = App()
    child.job(score)
    surface = describe(child)[0]
    assert isinstance(surface.payload_type, type) and issubclass(
        surface.payload_type, msgspec.Struct
    )
    parent = App()

    @parent.job
    async def main(ctx: Context, payload: ParentInput) -> DigestResult:
        asset = ImageAsset(digest) if payload.forge else payload.image
        return await score(media=ImageInput(asset))

    calls: list[dict[str, Any]] = []
    response = ""

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal response
        if kind == "child_call":
            calls.append(value)
            document = json.loads(value["payload"])
            assert document == {"media": {"type": "image", "image": digest}}
            assert (
                "pixels.png" not in value["payload"] and "opaque-original" not in value["payload"]
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                result, outcome, _ = executor.submit(
                    attempt,
                    child.get("score"),
                    document,
                    Invocation(
                        "child",
                        tmp_path / "child",
                        time.monotonic() + 30,
                        assets={
                            "media.image": GrantedInput(
                                "media.image",
                                media,
                                "image/png",
                                digest,
                                len(raw),
                            )
                        },
                    ),
                ).result()
            assert outcome.terminal == "succeeded", outcome
            assert result is not None
            response = msgspec.json.encode(result.result).decode()
            return {"ok": True}
        assert kind == "child_poll"
        return {"ok": True, "state": "succeeded", "result": response}

    broker = _Broker(
        "parent",
        {
            ("", __name__, "score"): _CallType(
                "sha256:" + "21" * 32,
                __name__,
                "score",
                surface.payload_type,
                DigestResult,
            )
        },
        exchange,
    )
    result, outcome, _ = attempt(
        parent.get("main"),
        {"image": "opaque-original", "forge": forged},
        Invocation(
            "parent",
            tmp_path / str(forged),
            time.monotonic() + 30,
            assets={"image": GrantedInput("image", media, "image/png", digest, len(raw))},
            calls=broker,
        ),
    )
    if forged:
        assert result is None and outcome.code == "child.asset_ungranted", outcome
        assert calls == []
    else:
        assert outcome.terminal == "succeeded", outcome
        assert (
            result is not None
            and result.result.digest == hashlib.sha256(bytes(range(256)) * 3).hexdigest()
        )
        assert len(calls) == 1
