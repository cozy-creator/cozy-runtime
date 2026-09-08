from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest
from PIL import Image

from cozy_runtime.author import (
    App,
    AssetBound,
    Assets,
    AuthorError,
    CapabilityError,
    ConformanceError,
    Context,
    FileAsset,
    ImageAsset,
    Invocation,
    MediaDecoder,
    Preflight,
    PreflightViolation,
    attempt,
    describe,
    invocable,
    prepare,
)
from cozy_runtime.author._assets import GrantedInput, InputMetadata
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._invoke import run_prepared
from cozy_runtime.internal import interface_wheel, package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.worker import grants

Pictures = Annotated[
    Assets[Annotated[ImageAsset, AssetBound(max_bytes=1024, max_decoded_bytes=1024)]],
    msgspec.Meta(min_length=1, max_length=4),
]


class Request(msgspec.Struct):
    prompt: str
    named: FileAsset


class Facts(msgspec.Struct, frozen=True):
    labels: tuple[str, ...]
    positions: tuple[int, ...]


class Result(msgspec.Struct):
    labels: list[str]
    ids: list[str]
    positions: list[int]
    digests: list[str]
    rgb: list[str]
    named: str


def preflight(payload: Request, assets: Pictures) -> Facts:
    assert type(payload) is Request
    assert not any(asset.hydrated for asset in assets)
    return Facts(tuple(asset.label for asset in assets), tuple(asset.position for asset in assets))


app = App()


@app.entrypoint(preflight=preflight)
def collect(
    payload: Request, assets: Pictures, facts: Preflight[Facts], decoder: MediaDecoder
) -> Result:
    assert type(payload) is Request
    assert tuple(asset.label for asset in assets) == facts.labels
    assert assets.by_id(assets[0].id) is assets[0]
    if assets[0].label:
        assert assets.by_label(assets[0].label) is assets[0]
    return Result(
        [a.label for a in assets],
        [a.id for a in assets],
        [a.position for a in assets],
        [a.digest for a in assets],
        [decoder.decode_image(a).rgb.hex() for a in assets],
        payload.named.read_bytes().decode(),
    )


@invocable(memoize=True)
async def child(ctx: Context, *, assets: Pictures) -> Result:
    return Result(
        [a.label for a in assets],
        [a.id for a in assets],
        [a.position for a in assets],
        [a.digest for a in assets],
        [a.read_bytes().hex() for a in assets],
        "",
    )


app.job(child)


def inputs(root: Path) -> tuple[dict[str, object], dict[str, GrantedInput]]:
    image = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image, format="PNG")
    raw = image.getvalue()
    photo = root / "photo.png"
    photo.write_bytes(raw)
    named = root / "note.txt"
    named.write_bytes(b"unrelated named field")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    other = "sha256:" + hashlib.sha256(named.read_bytes()).hexdigest()
    rows = {}
    for input_id, path, mime, identity, order in (
        ("assets.0.asset", photo, "image/png", digest, 0),
        ("named", named, "", other, 0),
        ("assets.1.asset", photo, "image/png", digest, 1),
    ):
        rows[input_id] = GrantedInput(
            input_id=input_id,
            local=path,
            media_type=mime,
            digest=identity,
            length=path.stat().st_size,
            order=order,
        )
    return {
        "prompt": "hello",
        "named": other,
        "assets": [{"asset": digest, "label": "alice"}, {"asset": digest}],
    }, rows


def test_actual_grant_hydration_preserves_collection_and_named_inputs(tmp_path: Path) -> None:
    wire, rows = inputs(tmp_path)
    bound = grants.BoundGrant(
        inputs={
            key: grants.BoundInput(
                key,
                row.local.as_uri(),
                bytes.fromhex(row.digest[7:]),
                row.length,
                row.media_type,
                row.order,
            )
            for key, row in rows.items()
        }
    )
    verified = grants.hydrate_inputs(
        bound, grants.Authorizer(roots=(str(tmp_path),)), spool=tmp_path / "spooled"
    )
    prepared = prepare(app.get("collect"), wire, input_metadata=rows)
    result, outcome, _ = run_prepared(
        prepared, Invocation("collect", tmp_path / "result", time.monotonic() + 10, assets=verified)
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    assert result.result == Result(
        ["alice", ""],
        ["assets.0.asset", "assets.1.asset"],
        [0, 1],
        [rows["assets.0.asset"].digest] * 2,
        ["ff0000" * 4] * 2,
        "unrelated named field",
    )
    escaped = prepared.overlay.payload.assets[0].asset.with_label("renamed")
    with pytest.raises(CapabilityError, match="escaped_handle"):
        escaped.read_bytes()


@pytest.mark.parametrize(
    "change", ["missing", "order", "digest", "mime", "length", "label", "count"]
)
def test_assets_preflight_refuses_bad_binding_before_hydration(tmp_path: Path, change: str) -> None:
    wire, rows = inputs(tmp_path)
    if change == "missing":
        rows.pop("assets.0.asset")
    elif change == "order":
        rows["assets.0.asset"] = replace(rows["assets.0.asset"], order=1)
    elif change == "digest":
        rows["assets.0.asset"] = replace(rows["assets.0.asset"], digest="sha256:" + "a" * 64)
    elif change == "mime":
        rows["assets.0.asset"] = replace(rows["assets.0.asset"], media_type="audio/wav")
    elif change == "length":
        rows["assets.0.asset"] = replace(rows["assets.0.asset"], length=1025)
    elif change == "label":
        wire["assets"] = [{"asset": rows["assets.0.asset"].digest, "label": "alice"}] * 2
    else:
        wire["assets"] = []
    with pytest.raises((ValueError, AuthorError)):
        prepare(app.get("collect"), wire, input_metadata=rows)


def test_assets_metadata_has_no_bytes_and_hydration_rechecks_it(tmp_path: Path) -> None:
    wire, rows = inputs(tmp_path)
    metadata = {
        key: InputMetadata(
            input_id=row.input_id,
            media_type=row.media_type,
            digest=row.digest,
            length=row.length,
            order=row.order,
        )
        for key, row in rows.items()
    }
    prepared = prepare(app.get("collect"), wire, input_metadata=metadata)
    with pytest.raises(PreflightViolation):
        prepared.overlay.payload.assets[0].asset.read_bytes()
    rows["assets.0.asset"] = replace(rows["assets.0.asset"], media_type="image/jpeg")
    _, outcome, _ = run_prepared(
        prepared, Invocation("collect", tmp_path / "result", time.monotonic() + 10, assets=rows)
    )
    assert outcome.code == "asset_binding_changed"


def test_assets_descriptor_and_generated_interface_keep_explicit_parameter(tmp_path: Path) -> None:
    found = Discovered(app, __name__ + ":app", tmp_path, sys.modules[__name__], describe(app), {})
    raw = package_interface.canonical_bytes(package_interface.build(found))
    body = package_interface.read_bytes(raw)
    entry = body["entrypoints"][0]
    assert entry["assets"]["parameter"] == "assets"
    assert entry["assets"]["kinds"][0]["max_decoded_bytes"] == 1024
    call = body["jobs"][0]
    assert call["invocable"]["parameters"] == ["assets"]
    generated = interface_wheel.generate(raw)
    assert any(
        b"assets: Assets[" in value for key, value in generated.items() if key.endswith(".py")
    )
    for key, value in generated.items():
        if key.endswith(".py"):
            compile(value, key, "exec")


def test_managed_subset_labels_and_duplicates_are_explicit_and_rebind(tmp_path: Path) -> None:
    _wire, rows = inputs(tmp_path)
    surface = next(s for s in describe(app) if s.name == "child")
    assert isinstance(surface.payload_type, type)

    def no_exchange(_kind: str, _body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("reservation alone does not contact the owner")

    broker = _Broker(
        "parent",
        {
            ("", __name__, "child"): _CallType(
                "sha256:" + "b" * 64, __name__, "child", surface.payload_type, Result
            )
        },
        no_exchange,
    )
    from cozy_runtime.author.fakes import fake_context

    broker.bind(fake_context(request_id="parent"))
    photo = ImageAsset(
        rows["assets.0.asset"].digest,
        local=rows["assets.0.asset"].local,
        attempt="parent",
        digest=rows["assets.0.asset"].digest,
    )
    original = Assets([photo.with_label("original")])
    forwarded = Assets([original[0].with_label("second"), original[0].with_label("first")])
    reservation = broker.reserve("", __name__, "child", {"assets": forwarded})
    payload = json.loads(reservation.payload)
    assert payload == {
        "assets": [
            {"asset": photo.digest, "label": "second"},
            {"asset": photo.digest, "label": "first"},
        ]
    }
    assert original[0].label == "original" and original[0].id == "assets.0.asset"
    selected = {
        key: replace(row, input_id=key) for key, row in rows.items() if key.startswith("assets.")
    }
    result, outcome, _ = attempt(
        app.get("child"),
        payload,
        Invocation("child", tmp_path / "child", time.monotonic() + 10, assets=selected),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.labels == ["second", "first"]
    assert result.result.ids == ["assets.0.asset", "assets.1.asset"]
    reordered = broker.reserve(
        "", __name__, "child", {"assets": Assets([forwarded[1], forwarded[0]])}
    )
    assert reordered.payload != reservation.payload
    relabelled = broker.reserve(
        "", __name__, "child", {"assets": Assets([forwarded[0].with_label("woman"), forwarded[1]])}
    )
    assert relabelled.payload != reservation.payload
    assert [row["asset"] for row in json.loads(relabelled.payload)["assets"]] == [
        row["asset"] for row in payload["assets"]
    ]
    unlabelled = broker.reserve("", __name__, "child", {"assets": Assets([photo, photo])})
    assert unlabelled.payload != reservation.payload
    with pytest.raises(CapabilityError, match=r"child\.asset_ungranted"):
        broker.reserve("", __name__, "child", {"assets": Assets([ImageAsset(photo.digest)])})


@dataclass
class DataclassRequest:
    prompt: str


def test_dataclass_payload_keeps_existing_signature_refusal() -> None:
    other = App()

    @other.entrypoint
    def unsupported(payload: DataclassRequest, assets: Pictures) -> Result:
        raise AssertionError("an unsupported author signature never executes")

    with pytest.raises(ConformanceError, match="not an author surface type"):
        describe(other)


TinyPictures = Annotated[
    Assets[Annotated[ImageAsset, AssetBound(max_bytes=1024, max_decoded_bytes=8)]],
    msgspec.Meta(min_length=1, max_length=4),
]


def test_assets_decode_uses_existing_per_item_limit(tmp_path: Path) -> None:
    wire, rows = inputs(tmp_path)
    limited = App()

    @limited.entrypoint
    def decode(payload: Request, assets: TinyPictures, decoder: MediaDecoder) -> Result:
        decoder.decode_image(assets[0])
        raise AssertionError("12 decoded RGB bytes cannot fit an8-byte field bound")

    _, outcome, _ = attempt(
        limited.get("decode"),
        wire,
        Invocation("limited", tmp_path / "limited", time.monotonic() + 10, assets=rows),
    )
    assert outcome.terminal == "refused" and outcome.code == "decoded_bytes_limit"


def test_collection_labels_are_exact_and_do_not_mutate_source() -> None:
    source = ImageAsset("ref", media_type="image/png", size_bytes=12)
    collection = Assets(
        [source.with_label("女人"), source.with_label("Woman"), source.with_label("woman")]
    )
    assert source.label == "" and source.id == "" and source.position == -1
    assert collection.by_label("女人").id == "assets.0.asset"
    assert [a.position for a in collection] == [0, 1, 2]
    with pytest.raises(KeyError):
        collection.by_label("")
    with pytest.raises(KeyError):
        collection.by_label("WOMAN")
    with pytest.raises(ValueError):
        Assets([source.with_label("same"), source.with_label("same")])
    with pytest.raises(TypeError):
        source.with_label(None)  # type: ignore[arg-type]
