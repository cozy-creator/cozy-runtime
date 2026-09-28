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
    Tree,
    attempt,
    describe,
    invocable,
)
from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._services import ImageFrame, encode_png
from durable_seam import wire


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
        # Generated client interfaces omit injected capabilities; this fixture shares
        # the implementation module so the type checker still sees its decoder.
        return await score(media=ImageInput(asset))  # type: ignore[call-arg]

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
                                input_id="media.image",
                                local=media,
                                media_type="image/png",
                                digest=digest,
                                length=len(raw),
                            )
                        },
                    ),
                ).result()
            assert outcome.terminal == "succeeded", outcome
            assert result is not None
            response = msgspec.json.encode(result.result).decode()
            return {"ok": True}
        if kind == "child_forget":
            return {"ok": True}
        assert kind == "child_poll"
        return {"ok": True, "state": "succeeded", "result": response}

    broker = _Broker(
        "parent",
        {
            (__name__, "score"): _CallType(
                "sha256:" + "21" * 32,
                __name__,
                "score",
                surface.payload_type,
                DigestResult,
            )
        },
        wire(exchange),
    )
    result, outcome, _ = attempt(
        parent.get("main"),
        {"image": "opaque-original", "forge": forged},
        Invocation(
            "parent",
            tmp_path / str(forged),
            time.monotonic() + 30,
            assets={
                "image": GrantedInput(
                    input_id="image",
                    local=media,
                    media_type="image/png",
                    digest=digest,
                    length=len(raw),
                )
            },
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


class TreeInput(msgspec.Struct):
    corpus: Tree
    forwarding: str = "granted"


@invocable(memoize=True)
async def score_tree(ctx: Context, *, corpus: Tree) -> DigestResult:
    return DigestResult(hashlib.sha256((corpus.path / "sample.txt").read_bytes()).hexdigest())


@pytest.mark.parametrize("forwarding", ["granted", "unhydrated", "foreign", "malformed"])
def test_broker_forwards_only_this_attempts_verified_tree(tmp_path: Path, forwarding: str) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    content = b"retained evaluation input"
    (corpus / "sample.txt").write_bytes(content)
    digest = "sha256:" + "ab" * 32
    child = App()
    child.job(score_tree)
    surface = describe(child)[0]
    assert isinstance(surface.payload_type, type) and issubclass(
        surface.payload_type, msgspec.Struct
    )
    parent = App()

    @parent.job
    async def main(ctx: Context, payload: TreeInput) -> DigestResult:
        tree = payload.corpus
        if payload.forwarding == "unhydrated":
            tree = Tree(digest)
        elif payload.forwarding == "foreign":
            tree = Tree(digest, digest=digest, root=corpus, attempt="another-parent")
        elif payload.forwarding == "malformed":
            tree = Tree(digest, digest="sha256:invalid", root=corpus, attempt=ctx.request_id)
        return await score_tree(corpus=tree)

    calls: list[dict[str, Any]] = []
    response = ""

    def exchange(kind: str, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal response
        if kind == "child_call":
            calls.append(value)
            document = json.loads(value["payload"])
            assert document == {"corpus": digest}
            assert "opaque-original" not in value["payload"]
            assert str(corpus) not in value["payload"]
            with ThreadPoolExecutor(max_workers=1) as executor:
                result, outcome, _ = executor.submit(
                    attempt,
                    child.get("score_tree"),
                    document,
                    Invocation(
                        "child",
                        tmp_path / "child",
                        time.monotonic() + 30,
                        trees={digest: (corpus, digest)},
                    ),
                ).result()
            assert outcome.terminal == "succeeded", outcome
            assert result is not None
            response = msgspec.json.encode(result.result).decode()
            return {"ok": True}
        if kind == "child_forget":
            return {"ok": True}
        assert kind == "child_poll"
        return {"ok": True, "state": "succeeded", "result": response}

    broker = _Broker(
        "parent",
        {
            (__name__, "score_tree"): _CallType(
                "sha256:" + "21" * 32,
                __name__,
                "score_tree",
                surface.payload_type,
                DigestResult,
            )
        },
        wire(exchange),
    )
    result, outcome, _ = attempt(
        parent.get("main"),
        {"corpus": "opaque-original", "forwarding": forwarding},
        Invocation(
            "parent",
            tmp_path / "parent",
            time.monotonic() + 30,
            trees={"opaque-original": (corpus, digest)},
            calls=broker,
        ),
    )
    if forwarding == "granted":
        assert outcome.terminal == "succeeded", outcome
        assert result is not None and result.result.digest == hashlib.sha256(content).hexdigest()
        assert len(calls) == 1
    else:
        assert result is None and outcome.code == "child.asset_ungranted", outcome
        assert calls == []
