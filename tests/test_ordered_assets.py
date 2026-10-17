from __future__ import annotations

import hashlib
import inspect
import io
import json
import sys
import time
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Annotated, Any

import msgspec
import pytest
from PIL import Image

from cozy_runtime.author import (
    App,
    AssetBound,
    AssetLimits,
    Assets,
    AudioAsset,
    AuthorError,
    CapabilityError,
    ConformanceError,
    Context,
    DecodedVideo,
    FileAsset,
    ImageAsset,
    Invocation,
    Preflight,
    PreflightViolation,
    VideoAsset,
    attempt,
    describe,
    invocable,
    prepare,
)
from cozy_runtime.author._assets import GrantedInput, InputMetadata
from cozy_runtime.author._calls import _Broker, _CallType
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS, host_decode_limits
from cozy_runtime.author._invoke import run_prepared
from cozy_runtime.author.fakes import fake_context
from cozy_runtime.internal import interface_wheel, package_interface
from cozy_runtime.internal.discovery import Discovered
from cozy_runtime.internal.executor_commands import InputFile
from durable_seam import wire

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
def collect(payload: Request, assets: Pictures, facts: Preflight[Facts]) -> Result:
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
        [a.read().tobytes().hex() for a in assets],
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


app.entrypoint(child, accelerator=False)


def inputs(
    root: Path, size: tuple[int, int] = (2, 2)
) -> tuple[dict[str, Any], dict[str, GrantedInput]]:
    image = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(image, format="PNG")
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


def test_native_input_projection_preserves_collection_and_named_inputs(tmp_path: Path) -> None:
    wire, rows = inputs(tmp_path)
    verified = {
        key: InputFile(
            local=str(row.local),
            media_type=row.media_type,
            digest=row.digest,
            length=row.length,
            order=row.order,
        ).granted(key)
        for key, row in rows.items()
    }
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
    found = Discovered(app, __name__ + ":app", tmp_path, sys.modules[__name__], describe(app))
    raw = package_interface.canonical_bytes(package_interface.build(found))
    body = package_interface.read_bytes(raw)
    entries = {entry["name"]: entry for entry in body["entrypoints"]}
    entry = entries["collect"]
    assert entry["assets"]["parameter"] == "assets"
    assert entry["assets"]["kinds"][0]["max_decoded_bytes"] == 1024
    assert entries["child"]["invocable"]["parameters"] == ["assets"]
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
            (__name__, "child"): _CallType(
                "sha256:" + "b" * 64, __name__, "child", surface.payload_type, Result
            )
        },
        wire(no_exchange),
    )
    broker.bind(fake_context(request_id="parent"))
    photo = ImageAsset(
        rows["assets.0.asset"].digest,
        local=rows["assets.0.asset"].local,
        attempt="parent",
        digest=rows["assets.0.asset"].digest,
    )
    original: Assets[ImageAsset] = Assets([photo.with_label("original")])
    forwarded: Assets[ImageAsset] = Assets([photo.with_label("second"), photo.with_label("first")])
    reservation = broker.reserve(__name__, "child", {"assets": forwarded})
    payload = json.loads(reservation.payload)
    assert payload == {
        "assets": [
            {"asset": photo.digest, "label": "second", "fidelity": "auto"},
            {"asset": photo.digest, "label": "first", "fidelity": "auto"},
        ]
    }
    assert original.info(0).label == "original" and original.info(0).id == "assets.0.asset"
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
    reordered = broker.reserve(__name__, "child", {"assets": forwarded.select(1, 0)})
    assert reordered.payload != reservation.payload
    fidelity = broker.reserve(
        __name__,
        "child",
        {
            "assets": Assets(
                [photo.with_label("second").with_fidelity("low"), photo.with_label("first")]
            )
        },
    )
    assert fidelity.payload != reservation.payload
    assert [item["asset"] for item in json.loads(fidelity.payload)["assets"]] == [photo.digest] * 2
    assert photo.fidelity == "auto"
    relabelled = broker.reserve(
        __name__,
        "child",
        {"assets": Assets([photo.with_label("woman"), photo.with_label("first")])},
    )
    assert relabelled.payload != reservation.payload
    assert [row["asset"] for row in json.loads(relabelled.payload)["assets"]] == [
        row["asset"] for row in payload["assets"]
    ]
    unlabelled = broker.reserve(__name__, "child", {"assets": Assets([photo, photo])})
    assert unlabelled.payload != reservation.payload
    with pytest.raises(CapabilityError, match=r"child\.asset_ungranted"):
        broker.reserve(__name__, "child", {"assets": Assets([ImageAsset(photo.digest)])})


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
    def decode(payload: Request, assets: TinyPictures) -> Result:
        assets[0].read()
        raise AssertionError("12 decoded RGB bytes cannot fit an 8-byte field bound")

    _, outcome, _ = attempt(
        limited.get("decode"),
        wire,
        Invocation("limited", tmp_path / "limited", time.monotonic() + 10, assets=rows),
    )
    assert outcome.terminal == "refused" and outcome.code == "decoded_bytes_limit"


LargePictures = Annotated[
    Assets[Annotated[ImageAsset, AssetBound(max_decoded_bytes=8 << 30)]],
    msgspec.Meta(min_length=1),
]


def test_a_declared_bound_above_the_old_2_gib_ceiling_describes() -> None:
    """Retained media is budgeted by the machine's memory, not a flat ceiling."""
    large = App()

    @large.entrypoint
    def decode(payload: Request, assets: LargePictures) -> Result:
        raise AssertionError("describe never executes")

    describe(large)
    assert host_decode_limits().max_decoded_bytes >= DEFAULT_DECODE_LIMITS.max_decoded_bytes


def test_collection_labels_are_exact_and_do_not_mutate_source() -> None:
    source = ImageAsset("ref", media_type="image/png", size_bytes=12)
    collection: Assets[ImageAsset] = Assets(
        [source.with_label("女人"), source.with_label("Woman"), source.with_label("woman")]
    )
    assert source.label == "" and source.id == "" and source.position == -1
    assert collection.info("女人").id == "assets.0.asset"
    assert [collection.info(i).position for i in range(len(collection))] == [0, 1, 2]
    with pytest.raises(KeyError):
        collection.by_label("")
    with pytest.raises(KeyError):
        collection.by_label("WOMAN")
    with pytest.raises(ValueError):
        Assets([source.with_label("same"), source.with_label("same")])
    with pytest.raises(TypeError):
        source.with_label(None)  # type: ignore[arg-type]


SimplePictures = Annotated[Assets[ImageAsset], msgspec.Meta(min_length=1, max_length=4)]


@invocable(memoize=True)
async def decoded_child(ctx: Context, *, assets: SimplePictures) -> Result:
    return Result(
        [assets.info(i).label for i in range(len(assets))],
        [assets.info(i).id for i in range(len(assets))],
        [assets.info(i).position for i in range(len(assets))],
        [assets.info(i).digest for i in range(len(assets))],
        [image.read().tobytes().hex() for image in assets],
        "",
    )


decoded_app = App()
decoded_app.entrypoint(decoded_child, accelerator=False)


def test_handles_read_lazily_through_one_attempt_budget(tmp_path: Path) -> None:
    wire, rows = inputs(tmp_path)
    escaped: list[Assets[ImageAsset]] = []
    inspected: list[str] = []

    def inspect_inputs(payload: Request, assets: SimplePictures) -> Facts:
        assert assets.info("alice").media_type == "image/png"
        inspected.append(assets.info(0).digest)
        with pytest.raises(FrozenInstanceError):
            assets.info(0).label = "changed"  # type: ignore[misc]
        with pytest.raises(PreflightViolation, match="hydration"):
            getattr(assets[0], "re" + "ad")()  # the runtime backstop behind the static fence
        assert assets.get("missing") is None
        return Facts(("alice", ""), (0, 1))

    local = App()

    @local.entrypoint(preflight=inspect_inputs)
    def decode(payload: Request, assets: SimplePictures) -> Result:
        media = assets[0]._media
        assert media is not None and media.retained_bytes == 0  # preflight decoded nothing
        assert assets[0] is assets["alice"] and assets.get("alice") is assets[0]
        first = assets["alice"].read()
        assert type(first) is Image.Image and first.mode == "RGB" and first.size == (2, 2)
        assert media.retained_bytes == 12
        second = assets[1].read()
        assert second is not first and media.retained_bytes == 24
        first.putpixel((0, 0), (0, 0, 255))
        assert second.getpixel((0, 0)) == (255, 0, 0)
        escaped.append(assets)
        return Result([], [], [], [], [second.tobytes().hex()], payload.named.read_bytes().decode())

    prepared = prepare(local.get("decode"), wire, input_metadata=rows)
    assert inspected == [rows["assets.0.asset"].digest]
    result, outcome, _ = run_prepared(
        prepared, Invocation("decoded", tmp_path / "decoded", time.monotonic() + 10, assets=rows)
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.rgb == ["ff0000" * 4]
    with pytest.raises(CapabilityError, match="escaped_handle"):
        escaped[0]["alice"].read()  # A handle cannot outlive its attempt.


def test_decoded_selection_forwards_custody_without_decoding(tmp_path: Path) -> None:
    _, rows = inputs(tmp_path)
    photo = ImageAsset(
        rows["assets.0.asset"].digest,
        local=rows["assets.0.asset"].local,
        digest=rows["assets.0.asset"].digest,
        attempt="parent",
    )
    original: Assets[ImageAsset] = Assets([photo.with_label("one"), photo.with_label("two")])
    forwarded = original.select("two", "one")
    assert forwarded.info(0).label == "two" and original.info(0).label == "one"
    assert forwarded[:1].info(0).label == "two"
    surface = describe(decoded_app)[0]
    assert isinstance(surface.payload_type, type)
    broker = _Broker(
        "parent",
        {
            (__name__, "decoded_child"): _CallType(
                "sha256:" + "b" * 64, __name__, "decoded_child", surface.payload_type, Result
            )
        },
        wire(lambda _operation, _payload: {}),
    )
    broker.bind(fake_context(request_id="parent"))
    call = broker.reserve(__name__, "decoded_child", {"assets": forwarded})
    payload = json.loads(call.payload)
    assert [row["label"] for row in payload["assets"]] == ["two", "one"]
    selected = {key: row for key, row in rows.items() if key.startswith("assets.")}
    result, outcome, _ = attempt(
        decoded_app.get("decoded_child"),
        payload,
        Invocation(
            "decoded-child", tmp_path / "decoded-child", time.monotonic() + 10, assets=selected
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None and result.result.labels == ["two", "one"]
    assert result.result.rgb == ["ff0000" * 4] * 2


def test_handle_descriptor_and_generated_client_preserve_the_handle_type(tmp_path: Path) -> None:
    found = Discovered(
        decoded_app,
        __name__ + ":decoded_app",
        tmp_path,
        sys.modules[__name__],
        describe(decoded_app),
    )
    document = package_interface.build(found)
    slot = document["entrypoints"][0]["assets"]
    assert "view" not in slot
    assert slot["kinds"][0]["max_decoded_bytes"] == DEFAULT_DECODE_LIMITS.max_decoded_bytes
    assert "media_decode" in document["entrypoints"][0]["invocable"]["capabilities"]
    raw = package_interface.canonical_bytes(document)
    package_interface.read_bytes(raw)
    generated = interface_wheel.generate(raw)
    assert any(b"Assets[Annotated[ImageAsset," in value for value in generated.values())
    for key, value in generated.items():
        if key.endswith(".py"):
            compile(value, key, "exec")


def test_assets_elements_are_handles_only() -> None:
    local = App()

    @local.entrypoint
    def bad(payload: Request, assets: Assets[ImageAsset | DecodedVideo]) -> Result:
        raise AssertionError("a decoded value type must refuse during describe")

    with pytest.raises(ConformanceError, match="assets_element"):
        describe(local)


LimitedMixed = Annotated[
    Assets[ImageAsset | VideoAsset | AudioAsset],
    AssetLimits(images=1, videos=1, audio=1, total=3),
]
SimpleLimited = Annotated[Assets[ImageAsset], AssetLimits(images=2)]


def test_kind_count_refuses_before_hydration(tmp_path: Path) -> None:
    wire, rows = inputs(tmp_path)
    local = App()

    @local.entrypoint
    def limited(payload: Request, assets: LimitedMixed) -> Result:
        raise AssertionError("two images exceed the kind limit despite fitting total3")

    with pytest.raises(AuthorError, match="asset_count"):
        prepare(local.get("limited"), wire, input_metadata=rows)


def test_fidelity_is_metadata_and_does_not_resize(tmp_path: Path) -> None:
    fidelity = "low"
    wire, rows = inputs(tmp_path, (513, 257))
    wire["assets"][0]["fidelity"] = fidelity
    local = App()

    @local.entrypoint
    def render(payload: Request, assets: SimpleLimited) -> Result:
        assert assets.info("alice").fidelity == fidelity
        assert assets.info(1).fidelity == "auto"
        assert assets["alice"].read().size == (513, 257)
        assert assets[1].read().size == (513, 257)
        assert assets.select(0).info(0).fidelity == fidelity
        return Result([], [], [], [], [], "")

    result, outcome, _ = attempt(
        local.get("render"),
        wire,
        Invocation("fidelity", tmp_path / "fidelity", time.monotonic() + 10, assets=rows),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None


@pytest.mark.parametrize("value", ["unknown", 1])
def test_invalid_fidelity_refuses_before_input_acquisition(tmp_path: Path, value: object) -> None:
    wire, rows = inputs(tmp_path)
    wire["assets"][0]["fidelity"] = value
    with pytest.raises(AuthorError):
        prepare(app.get("collect"), wire, input_metadata=rows)


def test_count_metadata_comes_from_argument_and_conflicts_refuse(tmp_path: Path) -> None:
    local = App()

    @local.entrypoint
    def limited(payload: Request, assets: SimpleLimited) -> Result:
        raise AssertionError("describe only")

    found = Discovered(local, "example:app", tmp_path, sys.modules[__name__], describe(local))
    doc = package_interface.build(found)
    entry = doc["entrypoints"][0]
    assert entry["assets"]["kinds"][0]["max_count"] == 2
    assert (
        next(field for field in entry["request"]["fields"] if field["name"] == "assets")[
            "constraints"
        ]["max_length"]
        == 2
    )
    for count in (-1, True, 1.5, "2"):
        bad = json.loads(json.dumps(doc))
        bad["entrypoints"][0]["assets"]["kinds"][0]["max_count"] = count
        with pytest.raises(ConformanceError):
            package_interface.read_bytes(json.dumps(bad).encode())
    conflict = App()

    @conflict.entrypoint
    def wrong(
        payload: Request,
        assets: Annotated[Assets[ImageAsset], AssetLimits(total=2), msgspec.Meta(max_length=3)],
    ) -> Result:
        raise AssertionError("conflicting declarations must refuse")

    with pytest.raises(ConformanceError, match="conflicting Assets total"):
        describe(conflict)
    misplaced = App()

    @misplaced.entrypoint
    def wrong_field(payload: Annotated[Request, AssetLimits(images=2)]) -> Result:
        raise AssertionError("count marker belongs only on Assets")

    with pytest.raises(ConformanceError, match="AssetLimits annotates"):
        describe(misplaced)
    assert "assets" not in inspect.signature(invocable).parameters


def test_assets_kinds_and_named_media_describe_with_no_declared_bound() -> None:
    class WithRaw(msgspec.Struct):
        photo: ImageAsset

    def handler(payload: Request, assets: SimpleLimited) -> Result:
        raise AssertionError("describe only")

    local = App()
    local.entrypoint(handler)
    surface = describe(local)[0]
    binding = surface.assets_binding
    assert binding is not None and "media_decode" in surface.capabilities
    handler.__annotations__["payload"] = WithRaw
    named = App()
    named.entrypoint(handler)
    assert "media_decode" in describe(named)[0].capabilities
